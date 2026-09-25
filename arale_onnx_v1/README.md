# arale_onnx_v1（引擎）

**あられブック（ARaLeBook）的 OCR 引擎**：把 `comic-text-detector`（检测气泡/文字行）与
`manga-ocr`（识别文字）这两个模型**导出成 ONNX**，用 **Rust** 跑起来，自带 ONNX Runtime。

它是 [arale-book-ocr](../README.md) 引擎库里的引擎之一，**也是当前唯一发布的**。
原来的 Python 实现（CPython + PyTorch + UniDic，归档 741 MiB）已经弃用，
移到了 [`legacy/python-manga-anki/`](../legacy/python-manga-anki/) 留档。

| | Python 版（弃用） | 本引擎（目标） |
|---|---|---|
| 运行时 | CPython 3.12 + PyTorch + OpenCV + transformers | **一个 Rust 二进制** + ONNX Runtime |
| 模型 | `pytorch_model.bin` 444 MB + `comictextdetector.pt` 80 MB | 三个 ONNX 图（int8 合计 **170 MB**，gzip 132 MB） |
| 分词 | `BertJapaneseTokenizer` + **MeCab + UniDic 248 MB** | **NFKC + 逐字查表**（不需要 MeCab） |
| 归档 | 741 MiB（macOS）/ 745 MiB（Win） | 目标 **≈170 MB** |

---

## 术语：MeCab / UniDic / fugashi，以及为什么这里一个都不需要

- **MeCab**：日语的**形态素解析器**（切词 + 词性标注）。日语不写空格，「词从哪到哪」只能靠
  **词典 + 维特比动态规划**算出来。它是一个 C++ 库，外加一份词典。
- **UniDic**：喂给 MeCab 的**词典**之一（国立国語研究所出品），几十万词条，带读音、词性、活用。
  PyPI 上的 `unidic_lite` 是精简再分发，**解包 248 MiB**——实测构成：
  `sys.dic` 188 MB（词条 trie）+ `matrix.bin` 72 MB（词与词之间的连接代价表）+ 若干定义文件。
- **fugashi**：MeCab 的 Python 绑定（C 扩展）。HuggingFace 的 `BertJapaneseTokenizer` 就是靠
  它 + UniDic 做**预切词**的——这就是那 248 MiB 出现在 OCR 归档里的**唯一原因**。
- **同类还有**：IPADIC（老、小）、NEGoLd/NEologd（网络新词多）、SudachiDict（配 Sudachi 用）。

**为什么本引擎不需要它们**：manga-ocr 用的那个模型，`tokenizer_config.json` 里
`subword_tokenizer_type` 是 **character**、词表 6144 **全是单字**，于是切词边界根本进不了输出——
token 序列就是「`[CLS]` + 每个字 + `[SEP]`」。实测（见下）「**NFKC 归一化 + 逐字查表**」与
真实的 `BertJapaneseTokenizer` 在**真实语料 2241 条**（整本 171 页的 2232 个文字块 + 9 条边界样本）
上 **id 序列零差异**；差异只出现在需要归一化的字符上（`髙神`→`髙 神`、`ＡＢ１２３`→`A B 1 2 3`、
`𝕏`→`X`），且实测都与 **NFKC** 逐字一致（NFD/NFKD 都有反例）。

所以 Rust 侧的分词就是 `unicode-normalization` 的 NFKC + 一张 24 KB 的 `vocab.txt`：
`src/tokenizer.rs`。**248 MiB 的辞书、一个 C 扩展、一份带再分发条款的字典，全都省掉。**

> 顺带一提：**应用自己的「分词」**（生成一本书的词表那件事）也从来不用 MeCab——
> 那是「词典扫描 + 去屈折」做的（见主仓库 `src/core/dict/scanner.ts`）。

---

## 解码：不是贪心，参数要照抄

`generation_config` 实测是 `num_beams=4, no_repeat_ngram_size=3, length_penalty=2.0,
early_stopping=True, max_length=300`。第一版探针用裸 argmax，同一块图给出 `いいのよ`
而参考是 `．．．`——**难图会给出完全不同的文字**。所以 `src/decode.rs` 照抄了这套束搜索。

## 文本后处理

`manga_ocr.post_process()` 的等价实现（`src/tokenizer.rs::post_process`），顺序一致：
去掉所有空白 → `…`→`...` → 连续 2 个以上的 `・`/`.` 折叠成 `.` → `jaconv.h2z(ascii, digit)`（半角→全角）。

## ONNX Runtime 怎么带

用 `ort` crate 的 **`load-dynamic`**：构建时**不下载**运行时，运行时由归档里自带的
`libonnxruntime.dylib` / `onnxruntime.dll` 提供（`ORT_DYLIB_PATH`）。
好处：构建可离线、可复现，也不会出现「构建机与目标机的 ORT 版本不一致」。

## 状态（诚实版）

已完成并实测：

- 模型导出 ONNX，数值与 PyTorch 一致（编码器 `max|Δ|=1.1e-5`；检测器 mask `5.6e-6`）；
- 识别端在真实裁剪上与 PyTorch **逐字一致 3/3**（含裸贪心会答错的那例），int8 也 3/3；
- 分词等价性：2241/2241；
- Rust 侧地基：crates.io 可达、`ort` + 本地 ORT dylib **已真跑通一次推理**；
- `src/tokenizer.rs`、`src/decode.rs` 已落地。

待做：

- `main.rs`：NDJSON 协议（`meta`/`page`/`fatal`）+ `--pages-file`，与 Python 版同口径；
- 图像预处理（检测器 letterbox、识别器「灰度→RGB→224 双三次→归一化」）；
- **检测器后处理**（唯一的大块）：实测**不能只用 YOLO 头**——它与现有 DB 分割产出的框
  IoU 中位只有 0.41/0.59，必须移植 DB representer + 行分组那套逻辑；
- `build.mjs`：装配二进制 + ONNX 模型 + ORT 库 + `extension.json`，并更新库根 `catalog.json`；
- **验收**：171 页 golden 语料逐页比对（文字逐字一致率 ≥99%、框 IoU 中位 ≥0.90）。
