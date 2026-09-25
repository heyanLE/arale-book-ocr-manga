//! arale_onnx_v1：comic-text-detector + manga-ocr 的 Rust/ONNX 实现。
//!
//! 用法（应用就是这么 spawn 它的）：
//! ```text
//! arale_onnx_v1 --pages-file pages.json      # {"pages":[{rel,absPath,width,height}]}
//! arale_onnx_v1 --pages a.jpg b.jpg          # 手工调试
//! arale_onnx_v1 --probe                      # 自检：只报告能不能跑，不识别任何图
//! arale_onnx_v1 --crop a.jpg --box x1,y1,x2,y2   # 调试：只识别一块
//! ```
//!
//! 输出是 NDJSON（见 `protocol.rs`），日志走 stderr。

mod decode;
mod detect;
mod imageproc;
mod models;
mod protocol;
mod tokenizer;

use anyhow::{Context, Result};
use ndarray::{Array3, Array4};
use ort::value::Tensor;
use serde::Deserialize;
use std::time::Instant;

#[derive(Deserialize)]
struct PageSpec {
    rel: String,
    #[serde(rename = "absPath")]
    abs_path: String,
}

#[derive(Deserialize)]
struct PagesFile {
    pages: Vec<PageSpec>,
}

struct Args {
    pages: Vec<PageSpec>,
    probe: bool,
    crop: Option<(String, [u32; 4])>,
    tokenize: Option<String>,
    /// 调试：把裁剪后的 224×224×3 张量按 f32 原始字节写到文件（与 HF 处理器逐值对比用）。
    dump_tensor: Option<String>,
    /// 调试：把检测器输出的 1024×1024 分割图（channel 0）按 f32 原始字节写到文件。
    dump_prob: Option<String>,
    /// 调试：把解码后的原图按 RGB u8 原始字节写到文件（与 PIL 解码逐值对比用）。
    dump_image: Option<String>,
}

fn parse_args() -> Result<Args> {
    let mut argv = std::env::args().skip(1);
    let mut pages_file: Option<String> = None;
    let mut pages: Vec<PageSpec> = Vec::new();
    let mut probe = false;
    let mut crop: Option<(String, [u32; 4])> = None;
    let mut tokenize: Option<String> = None;
    let mut dump_tensor: Option<String> = None;
    let mut dump_prob: Option<String> = None;
    let mut dump_image: Option<String> = None;
    while let Some(arg) = argv.next() {
        match arg.as_str() {
            "--pages-file" => {
                pages_file = Some(argv.next().context("--pages-file 后面要跟路径")?);
            }
            "--pages" => {
                for path in argv.by_ref() {
                    pages.push(PageSpec { rel: path.clone(), abs_path: path });
                }
            }
            "--probe" => probe = true,
            // 复核分词：打印 id 序列（与 HF 的 BertJapaneseTokenizer 逐条对比用）
            "--tokenize" => tokenize = Some(argv.next().context("--tokenize 后面要跟文本")?),
            "--dump-image" => {
                dump_image = Some(argv.next().context("--dump-image 后面要跟输出路径")?)
            }
            "--dump-prob" => {
                dump_prob = Some(argv.next().context("--dump-prob 后面要跟输出路径")?)
            }
            "--dump-tensor" => {
                dump_tensor = Some(argv.next().context("--dump-tensor 后面要跟输出路径")?)
            }
            "--crop" => {
                let path = argv.next().context("--crop 后面要跟图片路径")?;
                let _ = argv.next(); // --box
                let raw = argv.next().context("--box 后面要跟 x1,y1,x2,y2")?;
                let parts: Vec<u32> = raw
                    .split(',')
                    .map(|part| part.trim().parse::<u32>())
                    .collect::<Result<_, _>>()
                    .context("--box 要写成 x1,y1,x2,y2（原图像素）")?;
                anyhow::ensure!(parts.len() == 4, "--box 需要 4 个数");
                crop = Some((path, [parts[0], parts[1], parts[2], parts[3]]));
            }
            other => anyhow::bail!("未知参数：{other}"),
        }
    }
    if probe {
        return Ok(Args {
            pages: Vec::new(),
            probe: true,
            crop: None,
            tokenize,
            dump_tensor,
            dump_prob: None,
            dump_image: None,
        });
    }
    if let Some(file) = pages_file {
        let text = std::fs::read_to_string(&file).with_context(|| format!("读页清单失败：{file}"))?;
        // 两种形状都认：`{"pages":[…]}` 与裸数组（历史原因，扩展清单两种都出现过）。
        let parsed: Vec<PageSpec> = match serde_json::from_str::<PagesFile>(&text) {
            Ok(file) => file.pages,
            Err(_) => serde_json::from_str(&text).context("页清单既不是 {pages:[…]} 也不是数组")?,
        };
        pages = parsed;
    }
    Ok(Args { pages, probe, crop, tokenize, dump_tensor, dump_prob, dump_image })
}

/// 一页：检测 → 逐个裁切识别 → 一行一条。
fn recognize_page(
    models: &mut models::Models,
    tokenizer: &tokenizer::Tokenizer,
    beam: &decode::BeamConfig,
    page: &PageSpec,
) -> Result<(u32, u32, Vec<ProtocolLine>)> {
    let image = imageproc::load_image(&page.abs_path)?;
    let (width, height) = (image.width(), image.height());

    // ① 检测：letterbox → ONNX → DB 分割图 → 行框
    let letterboxed = imageproc::letterbox_for_detector(&image, 1024);
    let outputs = models.detector.run(ort::inputs![
        "image" => Tensor::from_array(letterboxed.tensor.clone())?,
    ])?;
    let (shape, lines_map) = outputs[2].try_extract_tensor::<f32>()?;
    let map = Array4::from_shape_vec(
        (1, shape[1] as usize, shape[2] as usize, shape[3] as usize),
        lines_map.to_vec(),
    )?;
    // 乘回原图：用**未补边**的尺寸（Python 同口径；补边只加在右下）。
    let resize_ratio = (
        width as f32 / (1024 - letterboxed.dw) as f32,
        height as f32 / (1024 - letterboxed.dh) as f32,
    );
    let boxes = detect::lines_from_map(&map, resize_ratio);

    // ② 识别：每个框裁一块（按同口径外扩）→ 编码器 → 束搜索
    let mut lines = Vec::with_capacity(boxes.len());
    for line in boxes {
        let padded = detect::pad_box(line.rect, line.font_size, width, height);
        let crop = imageproc::crop_for_recognizer(&image, padded, 224)?;
        let encoded = models.encoder.run(ort::inputs![
            "pixel_values" => Tensor::from_array(crop)?,
        ])?;
        let (es, hidden) = encoded[0].try_extract_tensor::<f32>()?;
        let hidden = Array3::from_shape_vec(
            (es[0] as usize, es[1] as usize, es[2] as usize),
            hidden.to_vec(),
        )?;
        let text = decode::decode(&mut models.decoder, tokenizer, &hidden, beam)?;
        if text.trim().is_empty() {
            // 空文本没有意义：在文字层里就是一个点不到的空块。
            continue;
        }
        lines.push(ProtocolLine {
            text,
            confidence: line.score.clamp(0.0, 1.0),
            box_: [
                padded[0] as f32,
                padded[1] as f32,
                padded[2] as f32,
                padded[3] as f32,
            ],
            vertical: line.vertical,
        });
    }
    Ok((width, height, lines))
}

struct ProtocolLine {
    text: String,
    confidence: f32,
    box_: [f32; 4],
    vertical: bool,
}

fn main() -> Result<()> {
    let args = parse_args()?;
    let root = models::bundle_root();
    protocol::log(&format!("引擎目录：{}", root.display()));

    if args.probe {
        // 自检：只证明「模型能载入」。真正的可用性由应用侧的 status() 决策。
        match models::load(&root) {
            Ok(_) => protocol::emit(&serde_json::json!({ "kind": "probe", "ok": true, "error": null })),
            Err(error) => protocol::emit(&serde_json::json!({
                "kind": "probe", "ok": false, "error": error.to_string(),
            })),
        }
        return Ok(());
    }

    let loaded = match models::load(&root) {
        Ok(loaded) => loaded,
        Err(error) => {
            protocol::fatal(&format!("载入模型失败：{error:#}"));
            std::process::exit(1);
        }
    };
    let tokenizer = match tokenizer::Tokenizer::from_vocab_txt(&loaded.vocab) {
        Ok(tokenizer) => tokenizer,
        Err(error) => {
            protocol::fatal(&format!("载入词表失败：{error:#}"));
            std::process::exit(1);
        }
    };
    // 调试用：只打印分词 id（验证「NFKC + 逐字查表」与 HF 分词器一致）
    if let Some(text) = args.tokenize {
        println!(
            "{}",
            serde_json::to_string(&tokenizer.encode(&text)).unwrap_or_else(|_| "[]".into())
        );
        return Ok(());
    }

    let beam = decode::BeamConfig::default();
    let mut models = loaded;

    // 调试用：只识别一块，直接打印文本（不吐 NDJSON）。
    if let Some((path, box_)) = args.crop {
        if let Some(dest) = &args.dump_image {
            let image = imageproc::load_image(&path)?.to_rgb8();
            std::fs::write(dest, image.as_raw()).with_context(|| format!("写原图失败：{dest}"))?;
            protocol::log(&format!("原图已写入 {dest}（{}×{}）", image.width(), image.height()));
        }
        if let Some(dest) = &args.dump_prob {
            let image = imageproc::load_image(&path)?;
            let letterboxed = imageproc::letterbox_for_detector(&image, 1024);
            let outputs = models.detector.run(ort::inputs![
                "image" => Tensor::from_array(letterboxed.tensor.clone())?,
            ])?;
            let (shape, lines_map) = outputs[2].try_extract_tensor::<f32>()?;
            let plane = (shape[2] as usize) * (shape[3] as usize);
            let channel0 = &lines_map[..plane];
            let mut bytes = Vec::with_capacity(plane * 4);
            for value in channel0 {
                bytes.extend_from_slice(&value.to_le_bytes());
            }
            std::fs::write(dest, &bytes).with_context(|| format!("写分割图失败：{dest}"))?;
            protocol::log(&format!("分割图已写入 {dest}（{plane} 个值）"));
        }
        let image = imageproc::load_image(&path)?;
        let crop = imageproc::crop_for_recognizer(&image, box_, 224)?;
        if let Some(dest) = &args.dump_tensor {
            let mut bytes = Vec::with_capacity(224 * 224 * 3 * 4);
            for value in crop.iter() {
                bytes.extend_from_slice(&value.to_le_bytes());
            }
            std::fs::write(dest, &bytes).with_context(|| format!("写张量失败：{dest}"))?;
            protocol::log(&format!("张量已写入 {dest}（CHW f32，{} 个值）", crop.len()));
        }
        let encoded = models.encoder.run(ort::inputs![
            "pixel_values" => Tensor::from_array(crop)?,
        ])?;
        let (es, hidden) = encoded[0].try_extract_tensor::<f32>()?;
        let hidden = Array3::from_shape_vec(
            (es[0] as usize, es[1] as usize, es[2] as usize),
            hidden.to_vec(),
        )?;
        println!("{}", decode::decode(&mut models.decoder, &tokenizer, &hidden, &beam)?);
        return Ok(());
    }

    protocol::meta("comictextdetector+manga-ocr(onnx)");
    let total = args.pages.len();
    let started = Instant::now();
    let mut failed = 0usize;
    for (index, page) in args.pages.iter().enumerate() {
        let page_started = Instant::now();
        let name = if page.rel.is_empty() { page.abs_path.clone() } else { page.rel.clone() };
        match recognize_page(&mut models, &tokenizer, &beam, page) {
            Ok((width, height, lines)) => {
                let payload: Vec<serde_json::Value> = lines
                    .iter()
                    .map(|line| {
                        serde_json::json!({
                            "text": line.text,
                            "confidence": line.confidence,
                            "box": line.box_,
                            "vertical": line.vertical,
                        })
                    })
                    .collect();
                protocol::emit(&serde_json::json!({
                    "kind": "page",
                    "file": name,
                    "ok": true,
                    "width": width,
                    "height": height,
                    "lines": payload,
                }));
                protocol::log(&format!(
                    "第 {}/{} 页 {}：{} 行（{:.1}s）",
                    index + 1,
                    total,
                    name,
                    lines.len(),
                    page_started.elapsed().as_secs_f32()
                ));
            }
            Err(error) => {
                // 单页失败不能毁掉整本：报这一页失败，继续下一页（应用会保留该页原文字层）。
                failed += 1;
                protocol::emit(&serde_json::json!({
                    "kind": "page",
                    "file": name,
                    "ok": false,
                    "error": format!("{error:#}"),
                }));
                protocol::log(&format!("第 {} 页失败：{error:#}", index + 1));
            }
        }
    }
    protocol::log(&format!(
        "完成：{} 页，失败 {} 页，用时 {:.1}s",
        total,
        failed,
        started.elapsed().as_secs_f32()
    ));
    Ok(())
}
