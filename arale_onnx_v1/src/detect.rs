//! 检测器后处理：`lines_map`（DB 分割图）→ 一个个文字行/列的框。
//!
//! ## 为什么不用 YOLO 头（实测）
//!
//! 检测器吐三样东西：`blks`（YOLO 的线框，2 类）、`mask`（区域掩码）、`lines_map`（DB 行分割）。
//! 试过只用 `blks`（NMS 之后）省掉 DB 那套后处理——**不行**：与现在管线产出的行框
//! IoU 中位数只有 0.41/0.59，而且第三条更致命：它**少找行**（001 页 3 个 vs 5 个）。
//! 漏行就是漏文字，不能接受。所以按 Python 的口径走 DB：
//!
//! ```text
//! lines_map[:,0] > thresh(0.3) → 连通域 → 每域一个框
//!   → box_score(域内概率均值) ≥ box_thresh(0.7) → 保留
//!   → 最短边 ≥ min_size+2(5px) → 保留
//!   → 乘回原图比例（未补边尺寸）
//! ```
//!
//! ## 两个实测出来的坑（都不是"优化"，是必须这么做）
//!
//! 1. **必须求最小外接旋转矩形，不能取轴对齐 AABB。** 第一版用 AABB，022 页（大量倾斜
//!    手写体）整整 30 行全被算成「宽 351×高 108」这类横排大块，`vertical` 全错、识别
//!    串行。原因：倾斜一行文字的 AABB 又宽又扁（165×46），而它的**最小外接旋转矩形**是
//!    19×129（窄高）——那才是真实的一列/一行。官方 representer 也是取最小外接矩形。
//! 2. **外扩要按 DB 的 unclip 公式来**：`distance = area × 1.5 / perimeter`，再把矩形四边
//!    各外推这个距离。少了它，每一行都会比 Python 那边瘦一圈，框的 IoU 白掉几个点。
//!
//! ## 与 Python 的**有意差异**
//!
//! - Python 用 `cv2.findContours` + `approxPolyDP` + `pyclipper` 求多边形再取最小外接
//!   矩形；这里直接对连通域求最小外接旋转矩形（自己写旋转卡壳），省掉 OpenCV 与 pyclipper
//!   两个依赖，几何等价。
//! - Python 还做 `mask refine` / `refine_undetected_mask` 并 `group_output` 把行归到块里；
//!   这里不做：应用要的是**一个文字行/列一条**，块的语义归应用侧（`blocksFromLines`）。
//!   顺带还消掉了旧管线里那批「粗框重复」（raw + refined 两份，171 页里 23 页有）。

use ndarray::Array4;

pub struct LineBox {
    /// 原图像素 `[x1,y1,x2,y2]`（字段不叫 `box`：那是 Rust 的保留字）。
    pub rect: [u32; 4],
    /// 竖排（按框的宽高比判断，与应用的 `isVerticalBox` 同一条规则 1.25）。
    pub vertical: bool,
    /// 字格边长（原图像素）——旋转矩形的**短边**。
    ///
    /// 这是参考实现里 `block.font_size` 的作用：外扩量 `pad = max(2, int(font_size*0.10))`。
    /// 曾经拿 AABB 的高度当字格边长，倾斜行会算大 3 倍（27 → 64），框因此多出 15 px，
    /// 相邻的字被吃进来，识别直接换词（022 页 `いやっ！！` → `いいの！！`）。
    pub font_size: f32,
    /// 域内平均概率，作为置信度。
    pub score: f32,
}

const THRESH: f32 = 0.3;
/// `inference.py` 里真正的分阈值是 **0.6**（`box_thresh = 0.6` 覆盖了 representer 的
/// 默认 0.7）。用 0.7 会整行整列地漏（实测 001 页参考 5 行、我们只出 2 行）。
const BOX_THRESH: f32 = 0.6;
const MIN_SIDE: u32 = 5; // Python: sside < min_size(3) + 2
const MAX_CANDIDATES: usize = 1000;
/// DB 的 unclip 比例（与 `SegDetectorRepresenter(unclip_ratio=1.5)` 一致）。
const UNCLIP_RATIO: f32 = 1.5;
/// 竖排判定阈值。与 `core/ocr/types.ts` 的 `VERTICAL_ASPECT_THRESHOLD` **同一个数**：
/// 引擎这边定了朝向，应用那边就不必再猜（两条规则不一致会让同一页在两处得到不同结论）。
const VERTICAL_ASPECT: f32 = 1.25;

/// `lines_map` → 行框。`resize_ratio` 是把 1024 空间乘回原图的比例。
pub fn lines_from_map(map: &Array4<f32>, resize_ratio: (f32, f32)) -> Vec<LineBox> {
    let height = map.shape()[2];
    let width = map.shape()[3];
    // 第 0 通道是文字行分割（第 1 通道是别的东西，Python 也只取 `pred[:, 0]`）。
    let mut binary = vec![false; width * height];
    let mut prob = vec![0f32; width * height];
    for y in 0..height {
        for x in 0..width {
            let value = map[[0, 0, y, x]];
            prob[y * width + x] = value;
            binary[y * width + x] = value > THRESH;
        }
    }

    let mut boxes = Vec::new();
    let mut visited = vec![false; width * height];
    let mut stack: Vec<usize> = Vec::with_capacity(1024);
    for start in 0..width * height {
        if visited[start] || !binary[start] {
            continue;
        }
        // 8 邻接连通域（Python 的 findContours 也是 8 邻接语义）。
        stack.clear();
        stack.push(start);
        visited[start] = true;
        let mut points: Vec<(f32, f32)> = Vec::new();
        let mut sum = 0f32;
        let mut count = 0usize;
        while let Some(index) = stack.pop() {
            let (x, y) = (index % width, index / width);
            points.push((x as f32, y as f32));
            sum += prob[index];
            count += 1;
            for dy in -1i32..=1 {
                for dx in -1i32..=1 {
                    if dx == 0 && dy == 0 {
                        continue;
                    }
                    let nx = x as i32 + dx;
                    let ny = y as i32 + dy;
                    if nx < 0 || ny < 0 || nx >= width as i32 || ny >= height as i32 {
                        continue;
                    }
                    let nindex = ny as usize * width + nx as usize;
                    if !visited[nindex] && binary[nindex] {
                        visited[nindex] = true;
                        stack.push(nindex);
                    }
                }
            }
        }
        if count == 0 {
            continue;
        }
        // ★ 域内**概率**均值 ≈ Python 的 `box_score_fast`（轮廓内概率均值）。
        //   注意是概率不是二值：拿二值算的话每个域都接近 1，这个阈值就形同虚设。
        let score = sum / count as f32;
        if score < BOX_THRESH {
            continue;
        }
        let Some(rect) = min_area_rect(&points) else { continue };
        // DB 的 unclip：四边各外推 area × ratio / perimeter
        let (mut rw, mut rh) = (rect.width, rect.height);
        let perimeter = 2.0 * (rw + rh);
        if perimeter > 0.0 {
            let distance = (rw * rh) * UNCLIP_RATIO / perimeter;
            rw += distance * 2.0;
            rh += distance * 2.0;
        }
        if rw.round().max(1.0) < MIN_SIDE as f32 || rh.round().max(1.0) < MIN_SIDE as f32 {
            continue;
        }
        // 1024 空间 → 原图（用**未补边**尺寸的比例，见 imageproc 的文件头）。
        //
        // ★ 输出的是旋转矩形的**四个角**在原图里的 AABB，不是"中心 ± 旋转后的宽高"。
        //   倾斜的一行，后者会把框压扁（实测 141×33，而正确的外接矩形是 136×70），
        //   裁出来的图只剩一条横带，识别必然错。参考实现同样取外接矩形来裁切。
        let to_orig = |x: f32, y: f32| -> (f32, f32) { (x * resize_ratio.0, y * resize_ratio.1) };
        let (ux, uy) = rect.axis;
        let (vx, vy) = (-uy, ux);
        let (hw, hh) = (rw / 2.0, rh / 2.0);
        let (cx, cy) = rect.center;
        let corners = [
            (cx + hw * ux + hh * vx, cy + hw * uy + hh * vy),
            (cx - hw * ux + hh * vx, cy - hw * uy + hh * vy),
            (cx - hw * ux - hh * vx, cy - hw * uy - hh * vy),
            (cx + hw * ux - hh * vx, cy + hw * uy - hh * vy),
        ];
        let (mut min_x, mut min_y) = (f32::INFINITY, f32::INFINITY);
        let (mut max_x, mut max_y) = (f32::NEG_INFINITY, f32::NEG_INFINITY);
        for (px, py) in corners {
            let (px, py) = to_orig(px, py);
            min_x = min_x.min(px);
            min_y = min_y.min(py);
            max_x = max_x.max(px);
            max_y = max_y.max(py);
        }
        if max_x - min_x < 1.0 || max_y - min_y < 1.0 {
            continue;
        }
        let box_ = [
            min_x.max(0.0).round() as u32,
            min_y.max(0.0).round() as u32,
            max_x.max(1.0).round() as u32,
            max_y.max(1.0).round() as u32,
        ];
        // 朝向看**旋转后**的长轴：倾斜的行长轴是横的（= 横排），窄高的列才是竖排。
        // 用 AABB 判断会把倾斜的行误判成竖排（AABB 又宽又扁也会被别的规则判反）。
        let vertical = rh > rw * VERTICAL_ASPECT;
        let font_size = (rw * resize_ratio.0).min(rh * resize_ratio.1);
        boxes.push(LineBox { rect: box_, vertical, font_size, score });
        if boxes.len() >= MAX_CANDIDATES {
            break;
        }
    }
    boxes
}

/// 一个旋转矩形：中心 + 边长（宽 = 沿 `axis` 方向，高 = 垂直方向）+ 宽方向的单位向量。
struct RotatedRect {
    center: (f32, f32),
    width: f32,
    height: f32,
    /// 宽方向的单位向量，用来还原四个角（再取 AABB）。
    axis: (f32, f32),
}

/// 最小外接**旋转**矩形（旋转卡壳）：凸包上每条边都当一次"底"，取面积最小的那个。
///
/// 为什么不用轴对齐：倾斜的一行文字 AABB 又宽又扁（实测 165×46），而它的最小外接矩形是
/// 19×129——朝向、尺寸、后面的裁切全都不一样。这是 022 页整整 30 行全错的根因。
fn min_area_rect(points: &[(f32, f32)]) -> Option<RotatedRect> {
    let hull = convex_hull(points);
    if hull.len() < 3 {
        // 退化成一条线/一个点：用包围盒兜底（下面还会被尺寸阈值过滤掉）。
        let min_x = points.iter().map(|p| p.0).fold(f32::INFINITY, f32::min);
        let max_x = points.iter().map(|p| p.0).fold(f32::NEG_INFINITY, f32::max);
        let min_y = points.iter().map(|p| p.1).fold(f32::INFINITY, f32::min);
        let max_y = points.iter().map(|p| p.1).fold(f32::NEG_INFINITY, f32::max);
        return Some(RotatedRect {
            center: ((min_x + max_x) / 2.0, (min_y + max_y) / 2.0),
            width: max_x - min_x,
            height: max_y - min_y,
            axis: (1.0, 0.0),
        });
    }
    let mut best: Option<(f32, RotatedRect)> = None;
    for i in 0..hull.len() {
        let (x1, y1) = hull[i];
        let (x2, y2) = hull[(i + 1) % hull.len()];
        let (dx, dy) = (x2 - x1, y2 - y1);
        let length = (dx * dx + dy * dy).sqrt();
        if length <= f32::EPSILON {
            continue;
        }
        let (ux, uy) = (dx / length, dy / length);
        // 把点投影到这条边（u 方向）与它的法向（v 方向）
        let mut min_u = f32::INFINITY;
        let mut max_u = f32::NEG_INFINITY;
        let mut min_v = f32::INFINITY;
        let mut max_v = f32::NEG_INFINITY;
        for (px, py) in &hull {
            let u = (px - x1) * ux + (py - y1) * uy;
            let v = -(px - x1) * uy + (py - y1) * ux;
            min_u = min_u.min(u);
            max_u = max_u.max(u);
            min_v = min_v.min(v);
            max_v = max_v.max(v);
        }
        let width = max_u - min_u;
        let height = max_v - min_v;
        let area = width * height;
        if best.as_ref().map(|(a, _)| area < *a).unwrap_or(true) {
            let center_u = (min_u + max_u) / 2.0;
            let center_v = (min_v + max_v) / 2.0;
            let center = (
                x1 + center_u * ux + center_v * -uy,
                y1 + center_u * uy + center_v * ux,
            );
            best = Some((area, RotatedRect { center, width, height, axis: (ux, uy) }));
        }
    }
    best.map(|(_, rect)| rect)
}

/// Andrew 单调链凸包（按 x,y 排序后左右两遍扫描）。
fn convex_hull(points: &[(f32, f32)]) -> Vec<(f32, f32)> {
    if points.len() < 3 {
        return points.to_vec();
    }
    let mut sorted: Vec<(f32, f32)> = points.to_vec();
    sorted.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal).then(
        a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal),
    ));
    let cross = |o: (f32, f32), a: (f32, f32), b: (f32, f32)| -> f32 {
        (a.0 - o.0) * (b.1 - o.1) - (a.1 - o.1) * (b.0 - o.0)
    };
    let mut lower: Vec<(f32, f32)> = Vec::new();
    for &point in &sorted {
        while lower.len() >= 2 && cross(lower[lower.len() - 2], lower[lower.len() - 1], point) <= 0.0 {
            lower.pop();
        }
        lower.push(point);
    }
    let mut upper: Vec<(f32, f32)> = Vec::new();
    for &point in sorted.iter().rev() {
        while upper.len() >= 2 && cross(upper[upper.len() - 2], upper[upper.len() - 1], point) <= 0.0 {
            upper.pop();
        }
        upper.push(point);
    }
    lower.pop();
    upper.pop();
    lower.extend(upper);
    lower
}

/// 与旧桥同口径的外扩：`max(2, 字格边长 × 0.10)`。
///
/// 「字格边长」取与排版方向垂直的那条边（竖排看宽度、横排看高度）——识别器裁得太紧会把
/// 边缘笔画切掉，裁得太松会把邻居吃进来。
pub fn pad_box(box_: [u32; 4], font_size: f32, width: u32, height: u32) -> [u32; 4] {
    // 与参考实现逐字一致：`pad = max(2, int(font_size * 0.10))`（`int()` 是截断）。
    let pad = ((font_size * 0.10) as u32).max(2);
    [
        box_[0].saturating_sub(pad),
        box_[1].saturating_sub(pad),
        (box_[2] + pad).min(width),
        (box_[3] + pad).min(height),
    ]
}
