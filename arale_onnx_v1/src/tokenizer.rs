//! manga-ocr 的分词与文本后处理。
//!
//! ## 实测依据（这条决定了整个移植的可行性）
//!
//! HuggingFace 那边配的是 `BertJapaneseTokenizer` + `word_tokenizer_type: mecab` +
//! `mecab_dic: unidic_lite`，看起来必须带上 248 MiB 的 UniDic 与 fugashi（C 扩展）。
//! 但它的 `subword_tokenizer_type` 是 **character**、词表 6144 全是单字，实测下来它
//! **等价于「NFKC 归一化 + 逐字符查表」**：
//!
//! - 真实语料 2241 条（整本 171 页的 2232 个文字块 + 9 条边界样本）逐条比对 id 序列，
//!   **零差异**；
//! - 差异只出现在需要归一化的地方：`髙神`→`髙 神`、`ＡＢ１２３`→`A B 1 2 3`、`𝕏`→`X`，
//!   实测都与 `NFKC` 逐字一致（NFD/NFKD 都有反例）。
//!
//! 所以 Rust 侧不需要 MeCab、不需要 UniDic：`unicode-normalization` 的 NFKC + 一张
//! 24 KB 的 `vocab.txt` 就够。

use std::collections::HashMap;

pub struct Tokenizer {
    vocab: HashMap<String, i32>,
    pub cls: i32,
    pub sep: i32,
    /// `[PAD]` 的 id：束搜索里"还没写到的位置"要填它。
    pub pad: i32,
    unk: i32,
}

impl Tokenizer {
    /// 从 `vocab.txt`（每行一个 token，行号即 id）载入。
    pub fn from_vocab_txt(text: &str) -> anyhow::Result<Self> {
        let mut vocab = HashMap::new();
        for (id, line) in text.split('\n').enumerate() {
            let token = line.trim_end_matches('\r');
            if token.is_empty() {
                continue;
            }
            vocab.entry(token.to_string()).or_insert(id as i32);
        }
        let get = |name: &str| -> anyhow::Result<i32> {
            vocab
                .get(name)
                .copied()
                .ok_or_else(|| anyhow::anyhow!("词表里缺少 {name}"))
        };
        let (cls, sep, unk, pad) = (get("[CLS]")?, get("[SEP]")?, get("[UNK]")?, get("[PAD]")?);
        Ok(Self { vocab, cls, sep, pad, unk })
    }

    /// 文本 → id 序列（含 `[CLS]` / `[SEP]`）。
    pub fn encode(&self, text: &str) -> Vec<i32> {
        use unicode_normalization::UnicodeNormalization;
        let normalized: String = text.nfkc().collect();
        let mut ids = Vec::with_capacity(normalized.chars().count() + 2);
        ids.push(self.cls);
        for ch in normalized.chars() {
            let key = ch.to_string();
            ids.push(self.vocab.get(&key).copied().unwrap_or(self.unk));
        }
        ids.push(self.sep);
        Ok::<(), ()>(()).ok();
        ids
    }

    /// id → 文本（跳过特殊 token）。
    pub fn decode(&self, ids: &[i32]) -> String {
        let mut by_id: Vec<Option<&str>> = vec![None; self.vocab.len()];
        for (token, id) in &self.vocab {
            if let Some(slot) = by_id.get_mut(*id as usize) {
                *slot = Some(token.as_str());
            }
        }
        let mut out = String::new();
        for id in ids {
            match by_id.get(*id as usize).and_then(|slot| *slot) {
                Some(token) if !token.starts_with('[') => out.push_str(token),
                _ => {}
            }
        }
        out
    }
}

/// `manga_ocr.post_process()` 的等价实现（顺序与它完全一致）。
///
/// 1. 去掉**所有**空白（Python 那边是 `"".join(text.split())`，连换行一起去掉）；
/// 2. `…` → `...`；
/// 3. 连续两个以上的 `・` 或 `.` 折叠成同样个数的 `.`；
/// 4. `jaconv.h2z(ascii=True, digit=True)`：半角 ASCII/数字 → 全角。
pub fn post_process(text: &str) -> String {
    let mut out: String = text.chars().filter(|c| !c.is_whitespace()).collect();
    out = out.replace('…', "...");
    out = collapse_dots(&out);
    h2z_ascii(&out)
}

fn collapse_dots(text: &str) -> String {
    let chars: Vec<char> = text.chars().collect();
    let mut out = String::with_capacity(text.len());
    let mut i = 0;
    while i < chars.len() {
        let c = chars[i];
        if c == '・' || c == '.' {
            let mut j = i;
            while j < chars.len() && (chars[j] == '・' || chars[j] == '.') {
                j += 1;
            }
            let count = j - i;
            if count >= 2 {
                for _ in 0..count {
                    out.push('.');
                }
                i = j;
                continue;
            }
        }
        out.push(c);
        i += 1;
    }
    out
}

fn h2z_ascii(text: &str) -> String {
    text.chars()
        .map(|c| match c {
            // jaconv.h2z(ascii=True, digit=True)：U+0021..U+007E → +0xFEE0
            '!'..='~' => char::from_u32(c as u32 + 0xFEE0).unwrap_or(c),
            _ => c,
        })
        .collect()
}
