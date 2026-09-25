# 已 mark 的后续项

给「先记下来、以后再动」的东西一个固定位置——不然它们只存在于对话里。
每项都写清 **现状 / 要做什么 / 触发条件**，以及为什么现在不做。

---

## 1. 分词引擎也要单独分发（已 mark）

**现状**：分词是**应用内置**的（`src/core/dict/scanner.ts` + `deinflect.ts`，纯 TypeScript：
词典扫描 + 去屈折，**零下载**）。`src/main/segment/service.ts` 直接调用它生成一本书的词表。
引擎库这边只有 `kind: "ocr-engine"` 一种引擎；下载/安装/校验那套基础设施是通用的，但
协议与 `kind` 是 OCR 专用的。

**要做**：把分词也做成「可下载的引擎」，与 OCR 同构：

| 事项 | 说明 |
|---|---|
| 新 `ExtensionKind` | 加 `'segment-engine'`。**注意**：应用侧 `parseCatalog` 现在把 kind 写死成 `kind === 'ocr-engine' ? 'ocr-engine' : 'ocr-engine'`（等于无条件当 OCR），加第二种 kind 时**必须先改这一行**，否则第二类引擎会被静默当成 OCR。 |
| 协议 | 与 OCR 一样走 NDJSON，但字段不同：输入是**一段文本**（或一本书的纯文本），输出是 **token 序列**（surface / start / end / lemma / 词性 / 去屈折轨迹）。放在引擎库 README 的「说同一套 NDJSON」里，按 `kind` 分节。 |
| 应用侧抽象 | `main/segment/` 需要一个与 `main/ocr/provider.ts` 同构的 `SegmentEngine` 接口（输入一本书 / 输出进度与结果），内置实现 = 现在的词典扫描。 |
| 默认不下载 | 与 OCR 的「系统 OCR 随包走 + 扩展可选」同构：**内置分词永远是默认**，可下载的引擎是可选替换。 |

**为什么值得做**：内置分词是**词典驱动**的——查不到的词就切不开，也没有词性/读音。
真正需要形态分析的场景（未知词、活用还原、词性筛选）恰恰是 **MeCab/Sudachi + UniDic** 的强项。
有意思的是：**那 248 MiB 的辞书在这里才是真正需要的**，而在 OCR 里是纯冗余（见
[`arale_onnx_v1/README.md`](../arale_onnx_v1/README.md) 的术语一节：OCR 那份是 character 子词，
实测与 NFKC+逐字查表零差异）。所以「分词引擎」很可能是**第一个真的需要下载辞书的引擎**。

**触发条件**：Rust OCR 引擎（`arale_onnx_v1`）完成并发布之后再做；顺序上别插队。

---

## 2. `arale_onnx_v1` 的状态

见 [`arale_onnx_v1/README.md`](../arale_onnx_v1/README.md) 的「状态」一节。

**已完成（都有实测依据，见 `arale_onnx_v1/src/*.rs` 的文件头）：**

1. `main.rs`：NDJSON 协议 + `--pages-file`（与旧 Python 版同口径，应用侧一行都不用改）+
   `--probe` / `--crop` / `--tokenize` / `--dump-tensor` / `--dump-prob` / `--dump-image`；
2. 图像预处理：检测器 letterbox（cv2 `INTER_LINEAR` 口径、右/下补边、BGR）+
   识别器（PIL 灰度 → 224 BILINEAR → mean/std 0.5）。与 HF `ViTImageProcessor`
   逐值差 ≤1 个灰阶（max|Δ| = 0.0078）；
3. 分词：NFKC + 逐字查表（等价于 `BertJapaneseTokenizer`，2241/2241 真实语料零差异），
   不需要 MeCab / UniDic；
4. 解码：**照抄 transformers 5.x 的 `_beam_search()`**（不是 4.x 的 `BeamSearchScorer` 语义，
   也不是贪心）；4 组真实裁切文字与 `sequences_scores` 全部与 MangaOcr 相同；
5. 检测器后处理：DB 分割图 → 8 连通域 → 最小外接**旋转**矩形 → unclip(1.5) →
   外接矩形四角 AABB + `pad = max(2, int(font_size*0.10))` + 分阈值 0.6；
6. `build.mjs`：装配 `bin/` + `models/*.onnx` + ONNX Runtime + `extension.json`，
   写 `dist/catalog-entry-<target>.json` 与库根 `catalog.json`。

**还没做完：**

1. **171 页 golden 语料的逐页验收**。现在做了其中 **30 页**（每 6 页取 1 页，385 行）：
   行数 383 vs 385、**框 IoU 中位 0.908**、**文字逐字一致 323/375 = 86.1%**
   ——IoU ≥0.90 达标，文字 ≥99% 差得远。（更早的 5 页小样本只有 63.8%，
   因为抽中了 022 这种倾斜难页；30 页才是可信口径。）
2. 这个差距的**根因已经定位，不是"没对齐某个参数"**：
   - 检测器 ONNX 的分割图与 torch 参考**逐值相同**（max|Δ| = 0.0）；
   - 同一份 PIL 解码下，本文的 letterbox 复刻出来的概率图与参考也几乎一致
     （max|Δ| = 0.053、>0.3 像素 15790 vs 15791）；
   - 但**引擎自己解码** JPEG（`image` crate）与 PIL（libjpeg-turbo）差 ±1 LSB，
     概率图 max|Δ| 就到 0.748、二值域差 1%，DB 域边界移 1–3 px（原图 2–6 px）。
   - 倾斜行的外接框跟着变，而 manga-ocr 在**贴边的难裁切**上对 1–6 px 极敏感
     （022 页：参考框 `(1082,214,1218,284)` → `いやっ！！`，我们的框 `(1081,216,1224,285)`
     → `いやいや`；把参考框喂给引擎则逐字一致）。30 页样本按 IoU 分桶也印证：
     IoU ≥0.95 的 140 条一致 93%，IoU 0.8–0.9 的 41 条只有 76%。
   - 要逐像素对齐，得换成 libjpeg-turbo 兼容的解码器（C 依赖，Windows 交叉构建变复杂）。
     这是**需要拍板**的事，不是继续调参能解决的。
3. Windows 归档：需要一份 Windows 的 `onnxruntime.dll` 与 x64 二进制（本机 macOS 交叉不了），
   所以 `--target all` 现在只能在本机生产 darwin 那一份。
4. 传 release → 清单里的 sha256 从空变成真值（现在 sha 为空 = 未发布，安装会被明确拒绝）。
5. **性能**：新束搜索（一直跑自然 `[SEP]`）比旧的早停版慢约 3×，
   171 页实测约 4.4 s/页（中位 4.25 s）。可以在后续版本里做 KV cache 或早停优化。

---

## 3. 模型权重怎么分发（待决定）

现在：模型在**构建时**从一份本地 `manga_anki` 检出拷进归档（只查存在性、**不校验 sha256**），
用户在**安装扩展时**一次性下载整个归档。四种候选形态与体积账见 [`models.md`](models.md)。

Rust 版换了模型格式（PyTorch → ONNX，int8 后 132 MB gzip），所以形态要重新选：
跟着归档走（≈170 MB 一次下完）还是首次 OCR 按需下载（扩展本体 ~35 MB）。

---

## 4. NDJSON 协议缺一个版本字段

清单有 `schemaVersion`、`manga.json` 有 `engineSignature`，但 runner 协议本身**没有版本号**：
旧归档配新应用只能靠「认不出的行忽略掉」自然退化，不会崩，但也没有显式协商。
建议在 `meta` 行加 `protocol: 1`，应用不认就明确报错。**与分词引擎一起改更划算**（协议要扩两处）。

---

## 5. 随 Python 弃用而消失的旧问题（复核即可，不必再做）

`tools/audit-win-deps.mjs` 当初在 Windows 归档里抓到的两条真风险——`torch_cpu.dll` 需要
`vcruntime140_threads.dll`、`torch_python.dll` 需要 `msvcp140_atomic_wait.dll`（包里没有），
以及 `cv2.pyd` 需要 Media Foundation（Windows N/KN 版没有）——**都是 Python 侧的依赖**。
Rust 引擎没有 torch/cv2，Windows 上只剩「1 个 exe + 1 个 ORT dll」。
待 Rust 版打包后跑一次体检复核，然后把这条从待办里划掉。

---

## 6. 清理

- `vendor/ocr-manga-anki/`（3.5 GB 的**旧 Python** 构建缓存）与演示实例
  `.arale-demo/extensions/ocr-manga-anki`（指向它的软链）：Python 弃用后没有任何用途，
  删之前把演示实例里的扩展卸掉即可。
- `engines/legacy/python-manga-anki/` 保留（留档），但**不进任何发布流程**。
