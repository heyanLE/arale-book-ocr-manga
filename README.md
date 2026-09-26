# ARaLeBook OCR 引擎库

当前开发资料从 [docs/README.md](docs/README.md) 进入，再读 [当前引擎说明](docs/current.md)。独立开发或作为应用 submodule 使用时，都以当前检出的源码和模型清单为准。

## 引擎清单

| provider | 实现 | 当前边界 |
|---|---|---|
| `arale_onnx_v1` | 包内 Python + ONNX Runtime + KV cache，复用 Mokuro 几何 | macOS arm64 已实测；Windows x64 交叉包待真机验收 |

[repositories/default.jsonl](repositories/default.jsonl) 是默认 OCR 仓库。归档用 `extension.json` 声明 runner，逐页输出 NDJSON；系统 OCR 由应用仓库提供。
模型、运行时与 ZIP 留在本地忽略目录；Git 保存源码、哈希清单、小型 JSONL 和构建记录。

构建、协议、性能证据、Windows 待办统一维护在 [docs/current.md](docs/current.md)。
以前的 README、Rust/PyTorch 模型和路线文档已归入 [2026-09-26 历史快照](docs/archive/2026-09-26/README.md)，不再作为当前开发依据。

许可见 [LICENSE](LICENSE) 与 [第三方声明](arale_onnx_v1/THIRD_PARTY.md)。
