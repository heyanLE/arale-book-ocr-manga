//! NDJSON 输出：与旧 Python 桥**逐字段同口径**，应用侧一行都不用改。
//!
//! ```
//! {"kind":"meta","engine":"comictextdetector+manga-ocr(onnx)","languages":["ja-JP"],"requested":["ja-JP"]}
//! {"kind":"page","file":"<rel>","ok":true,"width":W,"height":H,"lines":[{text,confidence,box,vertical}]}
//! {"kind":"page","file":"<rel>","ok":false,"error":"…"}
//! {"kind":"fatal","error":"…"}
//! ```
//!
//! 两条硬规矩（见主仓库 `src/shared/ocr-protocol.ts`）：
//! - **stdout 只有 NDJSON**：日志一律走 stderr。混进一行日志，客户端整本都失败。
//! - `box` 是**原图像素、左上原点**的 `[x1,y1,x2,y2]`。

use std::io::Write;

pub fn emit(value: &serde_json::Value) {
    let stdout = std::io::stdout();
    let mut lock = stdout.lock();
    // 序列化失败在结构上是不可达的（我们只喂 serde_json 自己的 Value / 结构体）；
    // 真走到这里也不许 panic —— 那会把已经识别的页白扔。
    if let Ok(line) = serde_json::to_string(value) {
        let _ = writeln!(lock, "{line}");
        let _ = lock.flush();
    }
}

/// 一句给人看的日志 → **stderr**（不是 stdout）。
pub fn log(message: &str) {
    eprintln!("[arale-ocr] {message}");
}

pub fn meta(engine: &str) {
    emit(&serde_json::json!({
        "kind": "meta",
        "engine": engine,
        "languages": ["ja-JP"],
        "requested": ["ja-JP"],
    }));
}

pub fn fatal(error: &str) {
    emit(&serde_json::json!({ "kind": "fatal", "error": error }));
}
