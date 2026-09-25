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

## 2. `arale_onnx_v1` 本身还没写完

见 [`arale_onnx_v1/README.md`](../arale_onnx_v1/README.md) 的「状态」一节。剩下：

1. `main.rs`：NDJSON 协议 + `--pages-file`（与旧 Python 版同口径，应用侧一行都不用改）；
2. 图像预处理（检测器 letterbox / 识别器 灰度→RGB→224→归一化）；
3. **检测器后处理**（最大一块）：实测**不能只用 YOLO 头**（与现有 DB 分割产出的框 IoU 中位
   只有 0.41/0.59），要移植 DB representer + 行分组；
4. `build.mjs`：装配 `bin/` + `models/*.onnx` + ONNX Runtime + `extension.json`，
   并更新库根 `catalog.json`；
5. **验收**：171 页 golden 语料逐页比对（文字逐字一致率 ≥99%、框 IoU 中位 ≥0.90）；
6. 传 release → 清单里的 sha256 从空变成真值（现在 sha 为空 = 未发布，安装会被明确拒绝）。

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
