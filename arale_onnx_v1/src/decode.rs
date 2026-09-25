//! 束搜索：**逐行照抄 transformers 5.x 的 `_beam_search()`**，不是贪心。
//!
//! ## 为什么必须抄到这种程度
//!
//! 参考实现（manga-ocr 0.1.16 + transformers 5.17）走的是 `generate()` 默认参数：
//! `num_beams=4, no_repeat_ngram_size=3, length_penalty=2.0, early_stopping=True,
//! max_length=300`。第一版用"束搜索 + 每步取 top-2K 候选、碰到 EOS 就记 finished、
//! finished 满 num_beams 就停"的经典写法（transformers 4.x 的 `BeamSearchScorer`
//! 语义），**在难图上会给出完全不同的文字**：
//!
//! | 裁切 | 参考 | 4.x 语义 | 本文 |
//! |---|---|---|---|
//! | `(1082,214,1218,284)` | `いやっ！！` | `いいのよ` | `いやっ！！` |
//!
//! 差异不在预处理也不在 ONNX 图：把 **HF 自己的编码器输出**喂给两边，PyTorch 解码器与
//! ONNX 解码器的 step-0 分布 max|Δ| = 1.3e-5、argmax 同为 `[CLS]`，也就是说网络是一致的，
//! 分叉完全来自搜索策略。transformers 5.x 改成了：
//!
//! - 每步只在 `beams_to_keep = 2 * num_beams` 个候选里取 `num_beams` 条**没结束**的继续跑
//!   （结束的候选加 `-1e9`，但它们仍占 topK 名额）；
//! - 已完成的假设按 **beam 槽位**存进 `sequences/beam_scores`，每槽只留该槽最高分，
//!   而不是"凑够 num_beams 条 finished 就收工"；
//! - 停止条件是三条同时成立：`early_stop_heuristic` 还认为可能被超越、还有没结束的槽、
//!   以及 topK 里还有没撞停止条件的候选。
//!
//! 于是同一条假设可以一直活到自然生成 `[SEP]` 为止，参考答案
//! `[CLS] いやっ！！`（归一化 -0.1117）就是这么出来的，而 4.x 语义在第 4 步就判"够了"、
//! 拿一个更短的假设收场。
//!
//! 移植正确性由 4 组真实裁切逐条对齐（文字与 `sequences_scores` 全等），
//! 见 `docs/roadmap.md` 的验收记录。

use crate::tokenizer::Tokenizer;
use ndarray::{Array2, Array3};
use ort::session::Session;
use ort::value::Tensor;

/// `-inf` 的有限替身：HF 用 `-1e9` 的地方一模一样用它，保证与浮点比较行为一致。
const NEG: f32 = -1.0e9;

pub struct BeamConfig {
    pub beams: usize,
    pub no_repeat_ngram: usize,
    pub length_penalty: f32,
    pub max_len: usize,
}

impl Default for BeamConfig {
    fn default() -> Self {
        Self { beams: 4, no_repeat_ngram: 3, length_penalty: 2.0, max_len: 300 }
    }
}

fn log_softmax(row: &[f32]) -> Vec<f32> {
    let max = row.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let sum: f32 = row.iter().map(|v| (v - max).exp()).sum();
    let log_sum = sum.ln();
    row.iter().map(|v| v - max - log_sum).collect()
}

/// `NoRepeatNGramLogitsProcessor`：前缀 = 序列最后 `n-1` 个 token，凡是以它为前缀的
/// 历史窗口，其尾 token 一律禁掉（`-inf`）。注意 `seq` 含开头的 `[CLS]`，与 HF 一致。
fn ngram_banned(seq: &[i32], n: usize) -> Vec<i32> {
    let len = seq.len();
    if n == 0 || len < n {
        return Vec::new();
    }
    let prefix = &seq[len + 1 - n..];
    let mut banned = Vec::new();
    for start in 0..=(len - n) {
        if &seq[start..start + n - 1] == prefix {
            banned.push(seq[start + n - 1]);
        }
    }
    banned
}

/// 取前 `k` 个下标，按 `(分数降序, 下标降序)`——与 torch.topk 的并列行为对齐。
/// 这里用全排序，`4 * 6144` 个元素的规模完全可以忽略。
fn top_k(scores: &[f32], k: usize) -> Vec<usize> {
    let mut order: Vec<usize> = (0..scores.len()).collect();
    order.sort_by(|a, b| {
        scores[*b]
            .partial_cmp(&scores[*a])
            .unwrap_or(std::cmp::Ordering::Equal)
            .then(b.cmp(a))
    });
    order.truncate(k);
    order
}

/// 一次解码：给定编码器输出，返回识别文本（已 `post_process`）。
pub fn decode(
    session: &mut Session,
    tokenizer: &Tokenizer,
    encoder_hidden: &Array3<f32>,
    config: &BeamConfig,
) -> anyhow::Result<String> {
    let num_beams = config.beams;
    let beams_to_keep = 2 * num_beams;
    let decoder_prompt_len = 1usize;
    let max_len = config.max_len;

    // running_sequences：num_beams 条正在跑的假设，定长 max_len，尾部填 pad。
    let mut running: Vec<Vec<i32>> = vec![vec![tokenizer.pad; max_len]; num_beams];
    for row in running.iter_mut() {
        row[0] = tokenizer.cls;
    }
    let mut sequences = running.clone();
    let mut running_scores = vec![NEG; num_beams];
    running_scores[0] = 0.0;
    let mut beam_scores = vec![NEG; num_beams];
    let mut is_finished = vec![false; num_beams];
    let mut unsat = true;
    let mut running_idx: Vec<Vec<i32>> = vec![vec![-1; max_len - decoder_prompt_len]; num_beams];
    let mut beam_idx: Vec<Vec<i32>> = running_idx.clone();

    let mut cur_len = 1usize;
    // top_num_beam_mask：只有排名 < num_beams 的候选才有资格被登记为"完成"。
    while cur_len + 1 < max_len {
        // a. 前向：num_beams 条假设共享同一份编码器输出，复制 num_beams 份。
        let mut hidden =
            Array3::<f32>::zeros((num_beams, encoder_hidden.shape()[1], encoder_hidden.shape()[2]));
        let single = encoder_hidden.index_axis(ndarray::Axis(0), 0);
        for mut plane in hidden.axis_iter_mut(ndarray::Axis(0)) {
            plane.assign(&single);
        }
        let mut ids = Array2::<i64>::zeros((num_beams, cur_len));
        for (i, row) in running.iter().enumerate() {
            for j in 0..cur_len {
                ids[[i, j]] = row[j] as i64;
            }
        }
        let outputs = session.run(ort::inputs![
            "encoder_hidden_states" => Tensor::from_array(hidden)?,
            "input_ids" => Tensor::from_array(ids)?,
        ])?;
        let (shape, logits) = outputs[0].try_extract_tensor::<f32>()?;
        let vocab = shape[2] as usize;
        let n_calls = shape[1] as usize;

        // b. log_softmax 之后再走 no_repeat_ngram（HF 5.x 也是把处理器作用在 log_probs 上）。
        let mut accumulated = vec![f32::NEG_INFINITY; num_beams * vocab];
        for i in 0..num_beams {
            let row =
                &logits[(i * n_calls + (n_calls - 1)) * vocab..(i * n_calls + n_calls) * vocab];
            let mut probs = log_softmax(row);
            for token in ngram_banned(&running[i][..cur_len], config.no_repeat_ngram) {
                probs[token as usize] = f32::NEG_INFINITY;
            }
            for t in 0..vocab {
                accumulated[i * vocab + t] = probs[t] + running_scores[i];
            }
        }

        // c. 全局取 topK 候选。
        let top = top_k(&accumulated, beams_to_keep);
        let top_scores: Vec<f32> = top.iter().map(|i| accumulated[*i]).collect();
        let parent: Vec<usize> = top.iter().map(|i| i / vocab).collect();
        let tokens: Vec<i32> = top.iter().map(|i| (i % vocab) as i32).collect();

        let mut top_seq: Vec<Vec<i32>> = Vec::with_capacity(beams_to_keep);
        let mut top_idx: Vec<Vec<i32>> = Vec::with_capacity(beams_to_keep);
        for c in 0..beams_to_keep {
            let mut seq = running[parent[c]].clone();
            seq[cur_len] = tokens[c];
            top_seq.push(seq);
            let mut idx = running_idx[parent[c]].clone();
            idx[cur_len - decoder_prompt_len] = parent[c] as i32;
            top_idx.push(idx);
        }
        // d. 停止条件：EOS 命中（MaxLength 也在里面，这里用上面的 while 边界覆盖）。
        let hits: Vec<bool> = tokens.iter().map(|t| *t == tokenizer.sep).collect();
        let hits_all = hits.iter().all(|h| *h);

        // e. 选出还能继续跑的 num_beams 条（撞停止条件的置 -1e9 后被刷掉）。
        let masked: Vec<f32> =
            top_scores.iter().zip(&hits).map(|(s, h)| if *h { s + NEG } else { *s }).collect();
        let next = top_k(&masked, num_beams);
        running = next.iter().map(|i| top_seq[*i].clone()).collect();
        running_scores = next.iter().map(|i| masked[*i]).collect();
        running_idx = next.iter().map(|i| top_idx[*i].clone()).collect();

        // f. 登记已完成的假设：按槽位取最高分（未完成的候选在这一列是 -1e9，不会被选中）。
        let mut final_scores: Vec<f32> = top_scores
            .iter()
            .map(|s| s / ((cur_len + 1 - decoder_prompt_len) as f32).powf(config.length_penalty))
            .collect();
        if is_finished.iter().all(|f| *f) {
            for s in final_scores.iter_mut() {
                *s += NEG;
            }
        }
        if !unsat {
            for s in final_scores.iter_mut() {
                *s += NEG;
            }
        }
        for c in 0..beams_to_keep {
            // did_top_num_beams_just_finished
            if !(hits[c] && c < num_beams) {
                final_scores[c] += NEG;
            }
        }
        let mut merged_scores = beam_scores.clone();
        merged_scores.extend_from_slice(&final_scores);
        let mut merged_seqs = sequences.clone();
        merged_seqs.extend(top_seq.iter().cloned());
        let mut merged_idx = beam_idx.clone();
        merged_idx.extend(top_idx.iter().cloned());
        let mut merged_fin = is_finished.clone();
        merged_fin.extend((0..beams_to_keep).map(|c| hits[c] && c < num_beams));

        let keep = top_k(&merged_scores, num_beams);
        sequences = keep.iter().map(|i| merged_seqs[*i].clone()).collect();
        beam_scores = keep.iter().map(|i| merged_scores[*i]).collect();
        beam_idx = keep.iter().map(|i| merged_idx[*i].clone()).collect();
        is_finished = keep.iter().map(|i| merged_fin[*i]).collect();

        // g. 停止条件三条。
        cur_len += 1;
        let hypothetical = running_scores[0] / ((cur_len - decoder_prompt_len) as f32).powf(config.length_penalty);
        let worst_finished = beam_scores.iter().cloned().fold(f32::INFINITY, f32::min);
        let can_improve = is_finished.iter().any(|fin| {
            let bound = if *fin { worst_finished } else { NEG };
            hypothetical > bound
        });
        unsat = unsat && can_improve;
        let exists_open = !is_finished.iter().all(|f| *f);
        let valid_continuations = !hits_all;
        if !(unsat && exists_open && valid_continuations) {
            break;
        }
    }

    // 输出：槽位 0 的序列，长度按 beam_idx 记录的真实生成步数截断。
    let generated = beam_idx.iter().map(|row| row.iter().filter(|i| **i != -1).count()).max().unwrap_or(0);
    let out_len = (decoder_prompt_len + generated).min(sequences[0].len());
    let ids: Vec<i32> = sequences[0][..out_len].to_vec();
    Ok(crate::tokenizer::post_process(&tokenizer.decode(&ids)))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ngram_prefix_bans_matching_window_tail() {
        // [CLS]=2, 序列 2,7,8,9,7,8 → 前缀 (7,8)，历史窗口 (7,8,9) 命中 → 禁 9。
        assert_eq!(ngram_banned(&[2, 7, 8, 9, 7, 8], 3), vec![9]);
    }

    #[test]
    fn ngram_shorter_than_n_bans_nothing() {
        assert!(ngram_banned(&[2, 7], 3).is_empty());
    }
}
