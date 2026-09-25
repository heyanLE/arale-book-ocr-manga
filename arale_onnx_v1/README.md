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

跑得通，端到端可用。分项如下，每条都有实测依据（细节写在对应源文件的文件头里）。

**已对齐参考实现：**

- 分词：NFKC + 逐字查表，真实语料 2241/2241 与 `BertJapaneseTokenizer` 零差异
  （**不需要** MeCab / 248 MiB 的 UniDic）；
- 解码：**照抄 transformers 5.x 的 `_beam_search()`**。4 组真实裁切（含倾斜难图）
  文字与 `sequences_scores` 全部与 `MangaOcr` 相同——注意 4.x 的
  `BeamSearchScorer` 语义在难图上会给出完全不同的文字（`いいのよ` vs `いやっ！！`）；
- 预处理：与 HF `ViTImageProcessor` 逐值差 ≤1 个灰阶（`max|Δ| = 0.0078`）；
- 检测器网络：ONNX 分割图与 torch **逐值相同**（`max|Δ| = 0.0`）；
- 检测器后处理：最小外接旋转矩形 → unclip(1.5) → 四角 AABB → `pad = max(2, int(font_size*0.10))`，
  分阈值 0.6（与 `inference.py` 一致）。

**还差的一次验收：**

171 页 golden 语料逐页比对只做了 5 页：行数基本对齐，**框 IoU 中位 0.905**，
**文字逐字一致率 63.8%**（离自定的 ≥99% 很远）。根因已定位：引擎自己解码 JPEG
（`image` crate）与参考用的 libjpeg-turbo 差 ±1 LSB，概率图 `max|Δ|` 0.748 →
DB 域边界移 1–3 px（原图 2–6 px）；manga-ocr 在贴边裁切上对这几 px 极敏感。
要逐像素对齐得换 libjpeg-turbo 兼容解码器（C 依赖），见 `../docs/roadmap.md` 第 2 节。

**还没做：**

- Windows 归档（需要 Windows 的 `onnxruntime.dll` 与 x64 二进制，macOS 上交叉不了）；
- 模型的 int8 量化：当前归档用的是 fp32 ONNX（未压缩 563 MiB），量化后约 170 MiB、gzip 约 132 MiB；
- 传 release 并把清单里的 sha256/bytes 换成真值（`build.mjs` 已经会算）；
- 性能：新的束搜索比旧的早停版慢约 3×，171 页实测约 4.4 s/页（中位 4.25 s）。

## 打包

```bash
# 本机（darwin-arm64）：cargo 构建 + 装配 + 打 zip + 写 catalog 条目
node build.mjs --target darwin-arm64 \
  --models /path/to/models --ort /path/to/libonnxruntime.1.30.0.dylib

# 交叉/已构建好的二进制
node build.mjs --target win32-x64 --skip-build --bin …/arale_onnx_v1.exe \
  --models /path/to/models --ort /path/to/onnxruntime.dll
```

归档根 = `extension.json` + `bin/` + `models/` + `lib/` + `LICENSE`；
`--models` 目录里要有 `detector.onnx` / `manga-ocr-encoder.onnx` / `manga-ocr-decoder.onnx` / `vocab.txt`。
