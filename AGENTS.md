# OCR 引擎开发入口

先读 [docs/README.md](docs/README.md) 与 [当前引擎说明](docs/current.md)。
若本仓库作为应用的 `engines/` submodule 使用，Windows 接续工作还要阅读父仓库 `docs/windows-handoff.md`。

`docs/archive/` 是历史原文，不代表当前方案。当前用户运行包是 Python + ONNX Runtime + KV cache，PyTorch 仅用于构建时导出和参考对照；系统 OCR 由父仓库维护。
修改模型、运行时或归档时同步锁定清单、JSONL 和当前验证文档。不要把归档、模型、运行时或真实书籍提交到 Git，也不要把交叉打包当作目标平台运行通过。
