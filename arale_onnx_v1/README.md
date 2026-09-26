# arale_onnx_v1

当前实现：包内 CPython + ONNX Runtime，四张 fp32 ONNX 图，束搜索使用 self/cross-attention KV cache，用户包不包含 PyTorch。

从 [引擎当前文档](../docs/current.md) 读取运行布局、导出/打包命令、性能与文字对照、Windows 兼容性待办。所有当前细节集中维护在该文档。

- [模型锁定清单](model-manifest.json)
- [构建脚本](build.mjs)
- [运行源码](python/ocr_run.py)
- [第三方声明](THIRD_PARTY.md)
- [旧说明快照](../docs/archive/2026-09-26/arale_onnx_v1/README.md)
