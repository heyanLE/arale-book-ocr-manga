//! ONNX 会话：三个图（检测器 / ViT 编码器 / BERT 解码器）与它们的路径解析。
//!
//! ONNX Runtime 用 `ort` 的 **load-dynamic**：构建时不下载运行时，由归档自带的
//! `libonnxruntime.*` 提供（应用 spawn 时通过 `extension.json` 的 `runner.env` 传
//! `ORT_DYLIB_PATH`）。这样构建可离线、也避免构建机与目标机版本不一致。

use anyhow::{Context, Result};
use ort::session::Session;
use std::path::{Path, PathBuf};

pub struct Models {
    pub detector: Session,
    pub encoder: Session,
    pub decoder: Session,
    pub vocab: String,
}

/// 归档根目录：可执行文件在 `<root>/bin/` 下。
///
/// 用**可执行文件的位置**而不是 cwd 来定位模型：应用把 cwd 设成安装目录、手工跑时
/// 可能是别处。两者都试，先看 exe 旁边。
pub fn bundle_root() -> PathBuf {
    if let Ok(exe) = std::env::current_exe() {
        if let Some(bin_dir) = exe.parent() {
            let root = bin_dir.parent().unwrap_or(bin_dir);
            if root.join("models").is_dir() {
                return root.to_path_buf();
            }
        }
    }
    std::env::current_dir().unwrap_or_else(|_| PathBuf::from("."))
}

pub fn load(root: &Path) -> Result<Models> {
    let models = root.join("models");
    let open = |name: &str| -> Result<Session> {
        let path = models.join(name);
        Session::builder()?
            .commit_from_file(&path)
            .with_context(|| format!("载入 ONNX 失败：{}", path.display()))
    };
    let vocab_path = models.join("vocab.txt");
    let vocab = std::fs::read_to_string(&vocab_path)
        .with_context(|| format!("读词表失败：{}", vocab_path.display()))?;
    Ok(Models {
        detector: open("detector.onnx")?,
        encoder: open("manga-ocr-encoder.onnx")?,
        decoder: open("manga-ocr-decoder.onnx")?,
        vocab,
    })
}
