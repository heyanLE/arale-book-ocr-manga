# arale_onnx_v1

当前实现：包内 CPython + ONNX Runtime，四张 fp32 ONNX 图，束搜索使用 self/cross-attention KV cache，用户包不包含 PyTorch。

核对日期：2026-10-02。支持已实测的 macOS arm64（14+）和 Windows x64。Windows 11 已验证包内自检、30 页 OCR、应用队列与正式 Release 下载/安装后识别；不是仅完成交叉构建。干净 Windows 的 VC++ 条件依赖和更多真实书籍仍未验收，详细记录见下方当前文档。

Windows runner 为 `python/python.exe`，源码入口为 `ocr/ocr_run.py`；用户不需要系统 Python。从引擎库根构建使用 `node arale_onnx_v1/build.mjs --target win32-x64`，开发目录加 `--debug`。源码克隆不包含模型/runtime，需先恢复这些构建输入；作为应用 submodule 时见父仓库 [Windows 交接](../../docs/windows-handoff.md)。

从 [引擎当前文档](../docs/current.md) 读取运行布局、导出/打包命令、性能与文字对照、Windows 兼容性待办。所有当前细节集中维护在该文档。

- [模型锁定清单](model-manifest.json)
- [构建脚本](build.mjs)
- [运行源码](python/ocr_run.py)
- [第三方声明](THIRD_PARTY.md)
- [旧说明快照](../docs/archive/2026-09-26/arale_onnx_v1/README.md)
