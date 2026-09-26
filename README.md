# ARaLeBook OCR 引擎库

这个 submodule 保存可下载 OCR 引擎的源码和仓库索引。应用本体只保留 macOS Vision / Windows 系统 OCR；可选引擎按 `repositories/default.jsonl` 下载，校验 sha256 后安装。

## 引擎清单

| id | 实现 | 状态 |
|---|---|---|
| `arale_onnx_v1` | 包内 CPython + ONNX Runtime，Mokuro 的图像几何与文字处理 | macOS arm64 已本机验证；Windows x64 已交叉打包，待真机验证 |

旧 PyTorch 和 Rust OCR 引擎源码已删除。`arale_onnx_v1` 保留稳定的 provider id，已安装旧版本需要升级到新归档。

## 仓库与协议

一个 OCR 仓库是一个 HTTPS JSONL 文件，每行是一条扩展记录。`arale_onnx_v1/build.mjs` 每次打包后更新 `repositories/default.jsonl` 中该引擎的资产、sha256 与体积。ZIP 放在 `arale_onnx_v1/dist/`，这两个大目录均不进 git；ZIP 以后上传到 JSONL 指定的 Release。应用设置页可以添加、删除仓库。

引擎归档根部的 `extension.json` 声明 runner，应用用 `--pages-file` 传入整本页清单。runner 逐页输出 NDJSON：`meta`、`page`、`probe` 或 `fatal`。`page.lines` 的 `box` 坐标是原图像素。应用负责排序、成块和写文字层。

## 本地构建

将检测器、编码器、首步解码器、续步缓存解码器四张 ONNX 图和 `vocab.txt` 放入 `arale_onnx_v1/models/`，平台匹配的 Python 与 ONNX Runtime 依赖放入 `arale_onnx_v1/runtime/<target>/`。这两处均已 gitignore。准备运行时的辅助脚本、目录布局和模型校验见 [引擎说明](arale_onnx_v1/README.md)。

```bash
node arale_onnx_v1/build.mjs --target darwin-arm64 --debug  # 直接加载的开发目录
node arale_onnx_v1/build.mjs --target darwin-arm64          # 发布 ZIP + JSONL
```

应用的 `npm run pack:debug` 会在开发目录存在时把它带进调试包；`npm run pack:release` 不带 OCR 扩展。
