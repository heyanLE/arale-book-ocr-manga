//! 图像预处理：检测器 letterbox、识别器裁剪、ViT 224 归一化。
//!
//! ## 口径必须与 Python 完全一致，否则框会整体偏
//!
//! 1. `comic_text_detector.inference.preprocess_img()` 干的事是：
//!    BGR→RGB → `letterbox(auto=False, stride=64)` → HWC→CHW → **`[::-1]`（又变回 BGR）**
//!    → `/255`。所以喂给检测器的张量其实还是 **BGR** 顺序。这里等价地：解出来是 RGB，
//!    交换 R/B 即可。
//! 2. `letterbox` 的补边**只加在右/下**（源码里 `dw /= 2` 被注释掉了）：
//!    `copyMakeBorder(im, 0, dh, 0, dw)`。所以 `resize_ratio = (w/(1024-dw), h/(1024-dh))`
//!    用的是**未补边**的尺寸。补成居中会让所有框水平偏 ~dw/2。
//! 3. 识别器：`MangaOcr.__call__` 先 `convert("L")` 再 `convert("RGB")`（**先转灰度**），
//!    然后 ViTImageProcessor：缩放到 224×224（**BILINEAR**，`resample: 2`）、/255、mean=std=0.5 归一化。

use anyhow::{Context, Result};

use image::{DynamicImage, GenericImageView};
use ndarray::Array4;

/// letterbox 结果：张量 + 缩放比 + 补边量（都是 Python 那套口径）。
pub struct Letterboxed {
    pub tensor: Array4<f32>,
    /// 补边量（只加在右/下）。乘回原图时要除以 `1024 - dw` / `1024 - dh`。
    pub dw: u32,
    pub dh: u32,
}

/// 检测器输入：resize 到 1024×1024（保持比例，右下补 0），BGR、/255。
pub fn letterbox_for_detector(image: &DynamicImage, size: u32) -> Letterboxed {
    let (w, h) = image.dimensions();
    let r = f64::min(size as f64 / h as f64, size as f64 / w as f64);
    let new_w = (w as f64 * r).round() as u32;
    let new_h = (h as f64 * r).round() as u32;
    let dw = size.saturating_sub(new_w);
    let dh = size.saturating_sub(new_h);
    let resized = resize_rgb_bilinear_cv(&image.to_rgb8(), new_w.max(1), new_h.max(1));
    let mut tensor = Array4::<f32>::zeros((1, 3, size as usize, size as usize));
    for y in 0..new_h as usize {
        for x in 0..new_w as usize {
            let px = resized.get_pixel(x as u32, y as u32);
            // RGB → BGR（见文件头第 1 条）
            tensor[[0, 0, y, x]] = px[2] as f32 / 255.0;
            tensor[[0, 1, y, x]] = px[1] as f32 / 255.0;
            tensor[[0, 2, y, x]] = px[0] as f32 / 255.0;
        }
    }
    Letterboxed { tensor, dw, dh }
}

/// `cv2.resize(..., INTER_LINEAR)` 口径的双线性：**不抗锯齿**。
///
/// 这是框误差的主要来源：`letterbox` 把 1441×2048 缩到 720×1024（2× 降采样），
/// cv2 的 `INTER_LINEAR` 按像素中心直接双线性取样（等价于点采样），而 `image` crate 的
/// `Triangle`/`CatmullRom` 在缩小时会**按比例张开 support 做抗锯齿**。概率图因此不同，
/// DB 域边界差 1–3 px（原图 2–6 px），倾斜行的外接框跟着变，识别就换字。
fn resize_rgb_bilinear_cv(img: &image::RgbImage, dw: u32, dh: u32) -> image::RgbImage {
    let (sw, sh) = (img.width() as usize, img.height() as usize);
    let (dw, dh) = (dw as usize, dh as usize);
    let mut out = image::RgbImage::new(dw as u32, dh as u32);
    let (sx, sy) = (sw as f64 / dw as f64, sh as f64 / dh as f64);
    for y in 0..dh {
        let fy = ((y as f64 + 0.5) * sy - 0.5).clamp(0.0, (sh - 1) as f64);
        let y0 = fy.floor() as usize;
        let y1 = (y0 + 1).min(sh - 1);
        let wy = (fy - y0 as f64) as f32;
        for x in 0..dw {
            let fx = ((x as f64 + 0.5) * sx - 0.5).clamp(0.0, (sw - 1) as f64);
            let x0 = fx.floor() as usize;
            let x1 = (x0 + 1).min(sw - 1);
            let wx = (fx - x0 as f64) as f32;
            let p00 = img.get_pixel(x0 as u32, y0 as u32);
            let p01 = img.get_pixel(x1 as u32, y0 as u32);
            let p10 = img.get_pixel(x0 as u32, y1 as u32);
            let p11 = img.get_pixel(x1 as u32, y1 as u32);
            let mut px = image::Rgb([0u8; 3]);
            for c in 0..3 {
                let top = p00[c] as f32 * (1.0 - wx) + p01[c] as f32 * wx;
                let bottom = p10[c] as f32 * (1.0 - wx) + p11[c] as f32 * wx;
                px[c] = (top * (1.0 - wy) + bottom * wy).round().clamp(0.0, 255.0) as u8;
            }
            out.put_pixel(x as u32, y as u32, px);
        }
    }
    out
}

/// 识别器输入：裁剪 → **灰度** → 拉伸成 RGB → 224×224 → /255 → (x-0.5)/0.5。
///
/// 缩放内核不是想当然的：`preprocessor_config.json` 里 `resample: 2` 是 **BILINEAR**
/// （不是 BICUBIC），而且 `MangaOcr.__call__` 是 `img.convert("L").convert("RGB")`
/// ——**先转灰度再复制成三通道**。两点都要照抄：
///
/// - 灰度用 PIL 的定点系数 `(19595R + 38470G + 7471B + 32768) >> 16`；
/// - 缩放用 PIL `Image.resize` 的双线性（`Resampling.c`：support=1.0、
///   `filterscale = max(1, insize/outsize)`、逐输出点权重归一化）。
///
/// 实测与 HF `ViTImageProcessor` 的输出差 **≤1 个灰阶**（max|Δ| = 0.0078，
/// mean|Δ| = 2e-5）。之前用 `image` crate 的 CatmullRom（a=-0.5）时 max|Δ| = 0.1176
/// ——难图上足以换词。
pub fn crop_for_recognizer(image: &DynamicImage, box_: [u32; 4], size: u32) -> Result<Array4<f32>> {
    let (w, h) = image.dimensions();
    let x1 = box_[0].min(w.saturating_sub(1));
    let y1 = box_[1].min(h.saturating_sub(1));
    let x2 = box_[2].clamp(x1 + 1, w);
    let y2 = box_[3].clamp(y1 + 1, h);
    let crop = image.crop_imm(x1, y1, x2 - x1, y2 - y1).to_rgb8();
    let (cw, ch) = (crop.width() as usize, crop.height() as usize);
    let gray = to_luma_pil(&crop);
    let resized = resize_bilinear_pil(&gray, cw, ch, size as usize, size as usize);

    let mut tensor = Array4::<f32>::zeros((1, 3, size as usize, size as usize));
    for y in 0..size as usize {
        for x in 0..size as usize {
            // (v/255 - 0.5) / 0.5
            let normalized = resized[y * size as usize + x] / 127.5 - 1.0;
            for c in 0..3 {
                tensor[[0, c, y, x]] = normalized;
            }
        }
    }
    Ok(tensor)
}

/// PIL `convert("L")`：Rec.601 定点系数 + 四舍五入（`ImagingConvertRGB2L`）。
fn to_luma_pil(rgb: &image::RgbImage) -> Vec<f32> {
    rgb.pixels()
        .map(|p| {
            let v = 19595u32 * p[0] as u32 + 38470u32 * p[1] as u32 + 7471u32 * p[2] as u32 + 32768;
            (v >> 16) as f32
        })
        .collect()
}

/// 一个输出坐标对应的源区间与权重（PIL `precompute_coeffs` 的口径）。
fn coeffs(insize: usize, outsize: usize, i: usize) -> (usize, Vec<f32>) {
    let scale = insize as f64 / outsize as f64;
    // 放大时 filterscale = 1（不抗锯齿）；缩小时按比例张开 support（抗锯齿）。
    let filterscale = if scale < 1.0 { 1.0 } else { scale };
    let support = filterscale; // 双线性 support = 1.0
    let ss = 1.0 / filterscale;
    let center = (i as f64 + 0.5) * scale;
    let mut xmin = (center - support + 0.5) as i64; // C 的 (int) 截断
    if xmin < 0 {
        xmin = 0;
    }
    let mut xmax = (center + support + 0.5) as i64;
    if xmax > insize as i64 {
        xmax = insize as i64;
    }
    let xmin = xmin as usize;
    let xmax = (xmax as usize).max(xmin);
    let mut weights: Vec<f32> = (xmin..xmax)
        .map(|x| {
            let arg = ((x as f64 - center + 0.5) * ss).abs();
            if arg < 1.0 { (1.0 - arg) as f32 } else { 0.0 }
        })
        .collect();
    let sum: f32 = weights.iter().sum();
    if sum != 0.0 {
        for w in weights.iter_mut() {
            *w /= sum;
        }
    } else if weights.is_empty() {
        // 理论上到不了；兜底成"取最近的源点"而不是全 0。
        return (xmin.min(insize.saturating_sub(1)), vec![1.0]);
    }
    (xmin, weights)
}

/// PIL `Image.resize(..., BILINEAR)`：先横后纵两趟分离卷积。
fn resize_bilinear_pil(src: &[f32], sw: usize, sh: usize, ow: usize, oh: usize) -> Vec<f32> {
    let mut tmp = vec![0.0f32; ow * sh];
    for y in 0..sh {
        for x in 0..ow {
            let (start, weights) = coeffs(sw, ow, x);
            let mut acc = 0.0f32;
            for (k, w) in weights.iter().enumerate() {
                acc += src[y * sw + start + k] * w;
            }
            tmp[y * ow + x] = acc;
        }
    }
    let mut out = vec![0.0f32; ow * oh];
    for y in 0..oh {
        let (start, weights) = coeffs(sh, oh, y);
        for x in 0..ow {
            let mut acc = 0.0f32;
            for (k, w) in weights.iter().enumerate() {
                acc += tmp[(start + k) * ow + x] * w;
            }
            out[y * ow + x] = acc;
        }
    }
    out
}

pub fn load_image(path: &str) -> Result<DynamicImage> {
    image::open(path).with_context(|| format!("读图失败：{path}"))
}
