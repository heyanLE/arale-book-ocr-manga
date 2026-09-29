# 当前 OCR 引擎：Python + ONNX Runtime + KV cache

核对日期：2026-09-30。引擎功能基线 `59eed53`；实现版本 `0.2.0`。后续文档提交不会改变这个功能基线。

## 1. 已确定的实现

- provider id：`arale_onnx_v1`；扩展安装 id：`ocr-arale_onnx_v1`。
- 用户包自带 CPython 3.12、ONNX Runtime 1.30.0、OpenCV/NumPy 等依赖和模型，不依赖用户自己的 Python，不包含 PyTorch。
- 保留 Mokuro/comic-text-detector 的图像几何、DB/YOLO 后处理、分组、透视裁切与文本后处理，神经网络前向用 ORT。
- 旧 Rust OCR 和旧 PyTorch Python 桥源码已删除；应用自己的 Rust 解包器不属于本仓库，也未删除。
- 运行时默认 CPU EP、**4 个 intra-op 线程**。`ARALE_OCR_THREADS` 可覆盖；8 线程只用于部分性能测试，未改成默认值。
- 缓存版默认开启；`ARALE_OCR_KV_CACHE=0` 只用于有旧无缓存图的本地对照。用户 ZIP 不带旧图，设置此开关会给出错误。

## 2. 图、源码和缓存

运行包包含四张 fp32/opset17 ONNX 图及词表，文件 SHA 以 [model-manifest.json](../arale_onnx_v1/model-manifest.json) 为准。

| 文件 | 职责 |
|---|---|
| `detector.onnx` | 页面文字/块/掩码检测 |
| `manga-ocr-encoder.onnx` | 每个文字裁切编码一次 |
| `manga-ocr-decoder-init.onnx` | 首 token 的 logits，生成 self/cross-attention KV |
| `manga-ocr-decoder-step.onnx` | 新 token + past KV → logits + 新 self KV；复用 cross KV |
| `vocab.txt` | 字符词表，运行包不需要 UniDic |

[python/ocr_run.py](../arale_onnx_v1/python/ocr_run.py) 维护 beam search，并按选中的父 beam 重排 self 和 cross cache。两层 BERT 的输入输出名目前显式写在代码中，更换 checkpoint 结构必须同步导出器和 runtime。
两张解码图分别包含权重；尚未合并图或共享外部权重，这使 ZIP 比无缓存版大约 95 MiB。

## 3. 应用契约

归档根的 `extension.json` 声明 runner：macOS 为 `python/bin/python3`，Windows 为 `python/python.exe`。参数运行 `ocr/ocr_run.py --pages-file <json>`，`PYTHONPATH=engine`；Windows 的 `_pth` 已由 `prepare-runtime.mjs` 补入 `..\ocr`。

页清单接受 `{ "pages": [{ "rel": "001.jpg", "absPath": "...", "width": 100, "height": 200 }] }` 或裸数组。
stdout 是 NDJSON：`meta`、逐页 `page`、自检 `probe`、整体错误 `fatal`；诊断写 stderr。
每行文字含 `text`、`confidence`、原图像素 `[x1,y1,x2,y2]`、`vertical`。应用负责排序成块、队列取消和落盘，不应把这些行为重新塞入引擎。

## 4. 构建输入与产物

```text
arale_onnx_v1/
  python/                       # 运行源码，进 Git
  tools/                        # 构建期导出/对照工具，进 Git
  model-manifest.json            # 模型锁定哈希，进 Git
  models/                       # 大权重，gitignore
  runtime/<platform>-<arch>/     # 包内 python/ 和 engine/ 依赖，gitignore
  build/dev-<platform>-<arch>/   # 完整开发引擎，gitignore
  dist/*.zip                    # 发布归档，gitignore
  dist/catalog-entry-*.json      # 构建记录，进 Git
repositories/default.jsonl       # 一行一个扩展的应用仓库，进 Git
```

clone 不会带回 ignored 大文件。Windows 可以从当前 Windows ZIP 恢复 `models/` 和 `runtime/win32-x64/{python,engine}/`；详细 PowerShell 步骤在父仓库 `docs/windows-handoff.md`。

从**引擎库根**运行：

```bash
node arale_onnx_v1/prepare-runtime.mjs --target darwin-arm64 \
  --python-dir /path/to/self-contained/python \
  --site-packages /path/to/site-packages \
  --ort-packages /path/to/unpacked-ort-wheel
node arale_onnx_v1/build.mjs --target darwin-arm64 --debug
node arale_onnx_v1/build.mjs --target darwin-arm64
node arale_onnx_v1/build.mjs --target win32-x64
```

Windows runtime 准备可加 `--opencv-packages /path/to/unpacked-headless-wheel`，以替换普通 OpenCV。脚本会排除 PyTorch/transformers/UniDic 等旧依赖，但输入解释器和 wheel 必须是目标平台架构。
构建校验模型 SHA，打 ZIP 并更新 JSONL；解包器可能不恢复 Unix 可执行位，应用安装器负责对 runner `chmod`。

**安装开放策略的实际边界**：非本机平台交叉构建默认将 JSONL 中该平台 SHA 留空；`--index-cross-build` 可覆盖。本机平台构建会自动填 SHA，脚本不会自动证明真机验收已通过。因此在 Windows 运行 build 或使用覆盖参数后，发布者仍须独立检查验收结果。

## 5. 构建期工具

- [export-models.py](../arale_onnx_v1/tools/export-models.py)：从本地原始权重导出检测器/编码器、词表和开发对照用的无缓存解码图。
- [export-decoder-cache.py](../arale_onnx_v1/tools/export-decoder-cache.py)：导出缓存首步/续步，校验首步和连续两步的 ONNX/PyTorch 张量差；已观察最大差约 3e-5，工具阈值为 1e-3。
- [compare-mokuro.py](../arale_onnx_v1/tools/compare-mokuro.py)：相同图片对照 Mokuro，按框 IoU ≥0.5 匹配并比较文字。
- [audit-win-deps.mjs](../tools/audit-win-deps.mjs)：扫描 Windows PE 导入依赖，静态检查不能替代真机执行。

导出和对照环境需要 PyTorch/transformers（本轮为 2.14.0/5.17.0），这些仅在构建机使用。验证新图后才更新模型哈希；不要只看导出成功就替换运行包。

## 6. 证据与限制

| 验证 | 2026-09-26 已记录结果 |
|---|---|
| 无缓存 vs KV cache | 同一批 30 页，388 行文字和框逐项一致 |
| 性能 | 同批 30 页、8 线程：126.4 秒 → 96.5 秒，约少 24% 时间；不是默认 4 线程保证 |
| Mokuro 一致率 | 参考 387 行、本引擎 388 行；配对 387 行中 380 行文字一致（98.2%），差异集中 001/169 页；不是人工准确率 |
| macOS 运行 | 包内解释器、开发目录、解压的 ZIP、应用扩展 provider 单页 OCR 已通过 |
| ZIP | 两平台完整性和 SHA 检查通过；Mac 与 Windows 归档均已生成 |
| Windows | Windows 11 build 26200 上修复 `_pth` 后，debug 构建、包内 `--probe` 和 30 页 OCR 均通过；默认 4 线程耗时 118.948 秒，30/30 页、388 行 |

Windows 首轮使用嵌入式 Python 3.12.10、ONNX Runtime 1.30.0、OpenCV 5.0.0、NumPy 2.5.3。与迁移包中的 Mac 缓存版预期输出按同页同行比较，文字 379/388、框 343/388、方向 387/388 精确一致，8/30 页完全一致。该数据是平台独立验收记录，不取代 Mac 8 线程性能数据；识别与框差异仍需评估。应用 provider/队列与 GUI 真 OCR 已通过，干净 Windows 的 VC++ 条件依赖仍未验证。

性能与质量测试使用私有测试漫画，图像没有提交到 Git；原始日志曾位于构建机 `/tmp`，不能假定在新设备存在。复测应使用自行迁移的同一套页图并记录样本清单，不能把别的书的时间直接并表。

## 7. 当前分发资产

以生成的 [macOS 记录](../arale_onnx_v1/dist/catalog-entry-darwin-arm64.json) 和 [Windows 记录](../arale_onnx_v1/dist/catalog-entry-win32-x64.json) 为机器可读真相；下表是本次核对快照。

| 平台 | ZIP 字节数 | 约 MiB | 验证/安装 |
|---|---:|---:|---|
| macOS arm64 | 722309909 | 688.8 | 包内 OCR 已测；最低 macOS 14；索引有 SHA |
| Windows x64 | 724064723 | 690.5 | 交叉打包；索引 SHA 空，当前拒绝安装 |

macOS ZIP SHA-256：`9d83ceca86f6c5e29f635eba2b0b9ab03d06d5c9ff7bc9d09b7f0eb9de244539`。
Windows ZIP 真实 SHA-256：`df1371702cf27cb457d613e6edc5e992511f766198deb11ffb26418e58bd7107`。它只在单平台构建记录中保留，不应误填成“已验证可安装”。
2026-09-30：macOS ZIP 已上传到 [v0.2.0 草稿 Release](https://github.com/heyanLE/arale-book-ocr-manga/releases/tag/untagged-f9d76c87bef99f6d63e5)，GitHub 页面显示 Draft、Assets 1、`arale_onnx_v1-macos-arm64.zip`（688.85 MB）。本地 `shasum -a 256` 与上表 SHA 一致，`unzip -tqq` 通过。草稿尚未公开，正式 tag 尚未创建，应用内计划下载 URL 仍不可用。
Windows ZIP 尚未上传；JSONL 的 Windows SHA 保持空值，应用继续拒绝安装。待 Windows 设备完成干净环境验收后，再将经验证的 Windows 资产加入同一 Release，并更新 JSONL；不能把交叉构建记录的 SHA 直接视作发布许可。

## 8. 下一步

Windows 优先：`python312._pth`、包内 `--probe`、30 页进程 OCR、应用 provider/队列和 GUI 真 OCR 已完成；接着评估跨平台输出差异 → 干净系统的 VC++ DLL → 实际安装/卸载 NSIS。headless OpenCV 已去除静态 Media Foundation 依赖，仍需在干净环境确认。
模型侧后续可以考虑合并两张解码图的重复权重、全量页验收、CoreML/量化实验；这些尚未完成，也不是迁移时默认要改的方向。
旧路线资料都在[归档](archive/2026-09-26/README.md)，不得把旧 Rust 的速度、旧 int8 体积目标或旧 PyTorch 安装方法当成当前方案。
