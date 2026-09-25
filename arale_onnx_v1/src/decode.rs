//! 束搜索：**照抄 `generate()` 的默认参数**，不是贪心。
//!
//! `generation_config` 实测是 `num_beams=4, no_repeat_ngram_size=3, length_penalty=2.0,
//! early_stopping=True, max_length=300`。第一版探针用裸 argmax，同一块给出
//! `いいのよ` 而参考是 `．．．`——**难图会给出完全不同的文字**。所以这一段必须一起移植。

use crate::tokenizer::Tokenizer;
use ndarray::{Array2, Array3};
use ort::session::Session;
use ort::value::Tensor;

pub struct BeamConfig {
    pub beams: usize,
    pub no_repeat_ngram: usize,
    pub length_penalty: f32,
    pub max_len: usize,
}

impl Default for BeamConfig {
    fn default() -> Self {
        Self { beams: 4, no_repeat_ngram: 3, length_penalty: 2.0, max_len: 128 }
    }
}

fn log_softmax(row: &[f32]) -> Vec<f32> {
    let max = row.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let sum: f32 = row.iter().map(|v| (v - max).exp()).sum();
    let log_sum = sum.ln();
    row.iter().map(|v| v - max - log_sum).collect()
}

fn ngram_repeats(seq: &[i32], next: i32, n: usize) -> bool {
    if n == 0 || seq.len() + 1 < n {
        return false;
    }
    let mut candidate: Vec<i32> = seq[seq.len() + 1 - n..].to_vec();
    candidate.push(next);
    // 生成序列里是否已出现过同一个 n-gram
    if seq.len() + 1 < n {
        return false;
    }
    let full: Vec<i32> = seq.iter().copied().chain(std::iter::once(next)).collect();
    full.windows(n).filter(|w| *w == candidate.as_slice()).count() > 1
}

/// 一次解码：给定编码器输出，返回识别文本（已 `post_process`）。
pub fn decode(
    session: &mut Session,
    tokenizer: &Tokenizer,
    encoder_hidden: &Array3<f32>,
    config: &BeamConfig,
) -> anyhow::Result<String> {
    let mut beams: Vec<(Vec<i32>, f32)> = vec![(vec![tokenizer.cls], 0.0)];
    let mut finished: Vec<(Vec<i32>, f32)> = Vec::new();
    let penalty = |ids: &[i32], score: f32| -> f32 {
        let len = (ids.len().saturating_sub(1)).max(1) as f32;
        score / len.powf(config.length_penalty)
    };

    for _ in 0..config.max_len {
        if beams.is_empty() {
            break;
        }
        let batch = beams.len();
        let mut hidden = Array3::<f32>::zeros((batch, encoder_hidden.shape()[1], encoder_hidden.shape()[2]));
        for (i, _) in beams.iter().enumerate() {
            hidden.index_axis_mut(ndarray::Axis(0), i).assign(encoder_hidden.index_axis(ndarray::Axis(0), 0));
        }
        let max_len = beams.iter().map(|(ids, _)| ids.len()).max().unwrap_or(1);
        let mut ids = Array2::<i64>::zeros((batch, max_len));
        for (i, (seq, _)) in beams.iter().enumerate() {
            for (j, id) in seq.iter().enumerate() {
                ids[[i, j]] = *id as i64;
            }
        }

        let outputs = session.run(ort::inputs![
            "encoder_hidden_states" => Tensor::from_array(hidden)?,
            "input_ids" => Tensor::from_array(ids)?,
        ])?;
        let (shape, logits) = outputs[0].try_extract_tensor::<f32>()?;
        let vocab = shape[2] as usize;
        let seq_len = shape[1] as usize;

        let mut candidates: Vec<(Vec<i32>, f32)> = Vec::new();
        for (i, (seq, score)) in beams.iter().enumerate() {
            let row = &logits[(i * seq_len + (seq_len - 1)) * vocab..(i * seq_len + seq_len) * vocab];
            let probs = log_softmax(row);
            let mut order: Vec<usize> = (0..vocab).collect();
            order.sort_by(|a, b| probs[*b].partial_cmp(&probs[*a]).unwrap_or(std::cmp::Ordering::Equal));
            for token in order.into_iter().take(config.beams * 2) {
                let token = token as i32;
                if ngram_repeats(seq, token, config.no_repeat_ngram) {
                    continue;
                }
                let mut next = seq.clone();
                next.push(token);
                candidates.push((next, score + probs[token as usize]));
            }
        }
        candidates.sort_by(|a, b| penalty(&b.0, b.1).partial_cmp(&penalty(&a.0, a.1)).unwrap_or(std::cmp::Ordering::Equal));

        beams.clear();
        for (seq, score) in candidates {
            if *seq.last().unwrap() == tokenizer.sep {
                finished.push((seq, score));
            } else {
                beams.push((seq, score));
            }
            if beams.len() >= config.beams || finished.len() >= config.beams {
                break;
            }
        }
        if finished.len() >= config.beams {
            break;
        }
    }

    let pool = if finished.is_empty() { beams } else { finished };
    let best = pool
        .into_iter()
        .max_by(|a, b| penalty(&a.0, a.1).partial_cmp(&penalty(&b.0, b.1)).unwrap_or(std::cmp::Ordering::Equal))
        .map(|(ids, _)| ids)
        .unwrap_or_default();
    let body: Vec<i32> = best.into_iter().filter(|id| *id != tokenizer.cls && *id != tokenizer.sep).collect();
    Ok(crate::tokenizer::post_process(&tokenizer.decode(&body)))
}
