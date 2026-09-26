# arale_onnx_v1：Mokuro 几何 + ONNX Runtime

引擎用包内 CPython 调用 ONNX Runtime，不导入 PyTorch，也不要求用户安装 Python。保留 Mokuro/comic-text-detector 的 OpenCV/NumPy 几何、文字行分组、透视裁切和后处理。运行时使用四张 fp32 ONNX 图：检测器、编码器、首步解码器、带 KV cache 的续步解码器。

## 验证范围

- macOS arm64：包内 Python 单页真实 OCR 已跑通，输出应用的逐页 NDJSON；引擎目录没有 `torch`、`torchvision`、`transformers` 或 UniDic。
- 当前 macOS ONNX Runtime 1.30.0 wheel 标记的最低系统版本是 **macOS 14**。应用本体仍支持 macOS 11+；较早系统可使用保留的系统 OCR，引擎仓库会把本扩展标为不适用。
- 与 Mokuro 0.2.5 CPU 基准比对：30 页抽样，参考 387 行、本引擎 388 行，按框 IoU ≥0.5 配对 387 行，其中 **380/387 = 98.2% 文字逐字一致**；28 页全部一致，差异集中于 001 页封面字与 169 页。测试图 7/7 行一致。该抽样还不是 171 页全量验收。
- KV cache 与旧无缓存图在同一批 30 页的 **388 行文字和 388 个框完全一致**。8 线程下，无缓存 126.4 秒、缓存 96.5 秒（时间缩短约 24%）；导出的首步与连续两步张量对 PyTorch 最大差约 3e-5。
- Windows x64：已用嵌入式 Python、官方 Windows ONNX Runtime wheel 和 headless OpenCV wheel 交叉打出 ZIP；186 个 PE 文件静态审计没有硬缺失，仍有 `msvcp140.dll` 条件依赖。尚未在 Windows 真机运行，因此 JSONL 里 Windows 资产的 sha256 留空，应用会拒绝安装。

`python/ocr_run.py` 保留 `--pages-file` 与 `--probe`。输出中的每个 `line` 对应 Mokuro 的一行/列；应用继续负责阅读顺序和文字层。`python/mokuro_compat/` 来源于 Mokuro 使用的 comic-text-detector 工具模块，只删去 PyTorch 类型分支，许可见库根 `LICENSE`。
可用 `tools/compare-mokuro.py` 在构建机上重跑逐页对照；这个工具需要 PyTorch，仅用于开发，不进入用户归档。

## 本地素材

以下两处都在 submodule 内，且被 `.gitignore` 排除：

```text
models/
  detector.onnx
  manga-ocr-encoder.onnx
  manga-ocr-decoder-init.onnx
  manga-ocr-decoder-step.onnx
  vocab.txt
runtime/darwin-arm64/
  python/bin/python3
  python/lib/...
  engine/onnxruntime/...
  engine/numpy/...
  engine/cv2/...
  ...
```

模型目前由已校验的本地 PyTorch 权重离线导出。`tools/export-models.py` 导出检测器/编码器与开发对照用的无缓存解码器；`tools/export-decoder-cache.py` 导出两张缓存图并验算连续步骤。验证后将五个运行时文件的 sha256 更新到 `model-manifest.json`，打包脚本会逐一校验。旧 `manga-ocr-decoder.onnx` 可留在本地模型目录，用 `ARALE_OCR_KV_CACHE=0` 对照，但不会进入用户 ZIP。构建脚本只读取本地文件，不在用户机器上下载权重。准备自带运行时可使用：

```bash
node prepare-runtime.mjs --target darwin-arm64 \
  --python-dir /path/to/self-contained/python \
  --site-packages /path/to/site-packages \
  --ort-packages /path/to/onnxruntime-wheel-unpacked
```

该脚本排除 PyTorch、transformers、manga-ocr Python 实现和 UniDic。务必使用目标平台/架构的解释器与 wheel；Windows 包须在 Windows 上验证。
Windows 建议再传 `--opencv-packages /path/to/unpacked-opencv-python-headless-wheel`：headless wheel 避免普通 OpenCV 对 Media Foundation 的依赖。使用的 Windows CPython 3.12 x64 ONNX Runtime 1.30.0 wheel sha256 为 `f3501472571f1b1eee50e017851e7929f5ea37312d2d8c2494a19e8fc58b4a38`；headless OpenCV 5.0.0.93 wheel sha256 为 `829717b6a95554f273e49e357cee3b3a2a26b6f4842fbc1bed2b45bdd8f87e0e`。

## 打包

```bash
node build.mjs --target darwin-arm64 --debug  # build/dev-darwin-arm64/，供应用直接加载
node build.mjs --target darwin-arm64          # dist/*.zip + repositories/default.jsonl
node build.mjs --target win32-x64             # 交叉构建：JSONL 中 sha256 留空，暂不可安装
```

`build.mjs` 在归档中放入 `extension.json`、包内解释器、Python 依赖、ONNX 模型、桥接脚本和许可证；计算 ZIP 的 sha256/bytes 并更新 JSONL。开发目录、运行时、模型和 ZIP 都不进 git。发布前将 ZIP 上传到 JSONL 的 Release 地址，并在干净机器上验证下载、解压和 OCR。
Windows 真机验收后，用 `--index-cross-build` 重建或更新 Windows 资产的 sha256；未验收的交叉构建始终保持空 SHA，应用会拒绝安装。

KV cache 版归档：macOS arm64 **722,309,909 字节（688.8 MiB）**，解压约 **1011 MiB**；Windows x64 交叉构建 **724,064,723 字节（690.5 MiB）**，解压约 **992.9 MiB**。两张解码图各带一份权重，所以相对无缓存版 ZIP 增加约 95 MiB；Windows 仍待真机验收。两个 ZIP 均未上传 Release。
