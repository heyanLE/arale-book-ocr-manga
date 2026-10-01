# ARaLeBook OCR 引擎库

当前开发资料从 [docs/README.md](docs/README.md) 进入，再读 [当前引擎说明](docs/current.md)。独立开发或作为应用 submodule 使用时，都以当前检出的源码和模型清单为准。

核对日期：2026-10-02。本页同步已有验证记录，本轮仅修改文档，没有重新执行 OCR。

## 引擎清单

| provider | 实现 | 当前边界 |
|---|---|---|
| `arale_onnx_v1` | 包内 Python + ONNX Runtime + KV cache，复用 Mokuro 几何 | macOS arm64（14+）与 Windows x64 已实测；干净 Windows VC++ 依赖待验收 |

[repositories/default.jsonl](repositories/default.jsonl) 是默认 OCR 仓库。归档用 `extension.json` 声明 runner，逐页输出 NDJSON；系统 OCR 由应用仓库提供。
模型、运行时与 ZIP 留在本地忽略目录；Git 保存源码、哈希清单、小型 JSONL 和构建记录。

## Windows x64

Windows 11 build 26200 已通过包内 `--probe`、30 页进程 OCR、应用 provider/队列/取消与 GUI 真 OCR。2026-10-01 最终 ZIP 在 Windows 本机构建，并完成正式 Release 的真实网络下载、SHA 校验、应用安装及下载后单页识别。`v0.2.0` 两平台归档已公开；本地 Windows 索引 SHA 已补入，待推送的索引状态见[当前记录](docs/current.md#7-当前分发资产)。

从引擎库根运行（需先恢复被忽略的模型与 Windows runtime）：

```powershell
node arale_onnx_v1/build.mjs --target win32-x64 --debug
Push-Location arale_onnx_v1/build/dev-win32-x64
.\python\python.exe -s -u ocr\ocr_run.py --probe
Pop-Location
node arale_onnx_v1/build.mjs --target win32-x64
```

用户无需另装 Python；包内 `_pth` 已补入 `..\ocr`。Windows ZIP 约 691.3 MiB。开发机运行通过不代表干净系统可直接运行，`msvcp140.dll` 条件依赖、跨平台识别差异仍需验证。NSIS 属于父应用仓库，其安装/卸载尚未验收。运行记录、命令与边界集中在[当前引擎说明](docs/current.md)；作为 submodule 恢复构建输入时另见父仓库 [Windows 交接](../docs/windows-handoff.md)。

构建、协议、性能证据、Windows 待办统一维护在 [docs/current.md](docs/current.md)。
以前的 README、Rust/PyTorch 模型和路线文档已归入 [2026-09-26 历史快照](docs/archive/2026-09-26/README.md)，不再作为当前开发依据。

许可见 [LICENSE](LICENSE) 与 [第三方声明](arale_onnx_v1/THIRD_PARTY.md)。
