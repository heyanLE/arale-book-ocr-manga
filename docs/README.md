# 引擎文档入口

核对日期：2026-09-26。新上下文先读 [当前引擎说明](current.md)。

| 问题 | 当前依据 |
|---|---|
| 实现、运行时、验证和待办 | [current.md](current.md) |
| 模型内容与哈希 | [model-manifest.json](../arale_onnx_v1/model-manifest.json) |
| 应用展示的仓库 | [default.jsonl](../repositories/default.jsonl) |
| 已生成归档的真实哈希/大小 | [macOS 构建记录](../arale_onnx_v1/dist/catalog-entry-darwin-arm64.json)、[Windows 构建记录](../arale_onnx_v1/dist/catalog-entry-win32-x64.json) |
| 第三方来源与许可 | [THIRD_PARTY.md](../arale_onnx_v1/THIRD_PARTY.md)、[LICENSE](../LICENSE) |
| 旧 Rust/PyTorch 路线 | [历史归档](archive/2026-09-26/README.md)，只用于追溯 |

若作为 `engines/` submodule 工作，先同步父仓库的当前文档和精确 gitlink，再改引擎。两个仓库分别提交、先推引擎后推应用。
