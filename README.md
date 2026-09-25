# arale-book-ocr —— 引擎库

**あられブック（ARaLeBook）的 OCR 引擎库**。一个引擎 = 一个「把页图变成文字」的实现；
应用只认两份契约（一套 NDJSON 协议 + 一份 `extension.json`），所以引擎怎么实现、用什么语言、
自带多少东西，应用都不知道。

主应用仓库：[heyanLE/arale-book](https://github.com/heyanLE/arale-book)
（它是通过 **git submodule** 把这个库挂在 `engines/` 上的）。

---

## 引擎清单

| 引擎 | 目录 | 语言/运行时 | 归档大小 | 平台 | 状态 |
|---|---|---|---|---|---|
| **arale_onnx_v1** | [`arale_onnx_v1/`](arale_onnx_v1/) | **Rust** + ONNX Runtime（自带） | 目标 ≈170 MB（int8 模型 132 + ORT 33 + 二进制） | macOS arm64、Windows x64 | **重写中**：模型导出、分词、解码已验证；检测器后处理与打包待做 |
| *(已弃用)* manga-anki | [`legacy/python-manga-anki/`](legacy/python-manga-anki/) | Python 3.12 + PyTorch | 741 MiB（macOS）/ 745 MiB（Win） | — | **不再发布**：保留留档。整套 Python 运行时（CPython + torch + UniDic）在新实现里被彻底去掉 |

> 引擎改名/换实现的代价很低（应用只认协议），所以 `arale_onnx_v1` 与旧的 Python 版可以并存于
> 仓库、由清单决定发布哪个。**当前只发布 `arale_onnx_v1`**；Python 版整体弃用、不参与构建与发布。

---

## 一个引擎必须满足什么

### 1. 说同一套 NDJSON（协议冻结）

引擎被应用 spawn 起来，往 stdout 逐行吐 JSON（**一行一个 JSON 对象**，别的什么都不能混进去，
日志走 stderr）：

```json
{"kind":"meta","engine":"…","languages":["ja-JP"],"requested":["ja-JP"]}
{"kind":"page","file":"<页清单里回显的 rel 或 absPath>","ok":true,"width":1441,"height":2048,
 "lines":[{"text":"…","confidence":1.0,"box":[x1,y1,x2,y2],"vertical":true}]}
{"kind":"probe","ok":true}
{"kind":"fatal","error":"…"}
```

- 输入：`--pages-file <json>`（`{"pages":[{rel,absPath,width,height}]}`，也接受裸数组）；
- 输出：**一行一行文字**，不是段落也不是 mokuro 文字块；
- 阅读顺序与成块**不归引擎管**（那是应用侧 `core/ocr/blocks.ts` 的事）；
- `probe` 是自检模式：不加载模型、不跑页，只回答「这台机器能不能用」。

### 2. 有一份 `extension.json`（归档内自描述）

```json
{
  "id": "ocr-<引擎名>", "version": "1.0.0", "kind": "ocr-engine", "provides": "<OcrProviderId>",
  "engine": { "label": "…", "requirement": "…", "downloadSizeMb": 0 },
  "license": "…", "homepage": "…",
  "runner": { "program": "…", "args": ["…", "{pagesFile}"], "env": { } }
}
```

`runner.program` **必须落在归档解压目录内**（应用安装时会校验，防清单被篡改指向系统别处的可执行文件）。
`{pagesFile}` 会被替换成页清单路径。应用把 `cwd` 设成安装目录后 spawn。

> ⚠️ 应用解压归档时会**丢掉 unix 权限位**，而且只给 `runner.program` 这一个路径补 `+x`。
> 所以 macOS 上 `program` 要指向**解释器/可执行文件本身**，不要指向一个还要 `exec` 别人的脚本
> （manga-anki 就是这么踩过：见它的 README）。

### 3. 目录约定

**库根下的一级目录 = 一个引擎**（共享的东西只有 `tools/`、`docs/`、`LICENSE`、`README.md`）。
这样应用把这个库挂成 submodule 之后，路径正好是 `engines/<引擎名>/`——不用多一层 `engines/engines/`。

```
<库根>/
├── README.md          引擎清单 + 「一个引擎必须满足什么」（本文件）
├── <引擎名>/          一个引擎的全部：README + 构建脚本 + 源码
│   ├── README.md      这个引擎是什么、怎么构建、体积、已知限制
│   ├── build.mjs      构建脚本（装配运行时+模型，写 extension.json，打 zip）
│   └── …              桥 / 二进制 / 模型清单…
├── tools/             跨引擎复用的工具（如 Windows 依赖体检）
└── docs/              跨引擎的文档（如模型获取）
```

构建产物落在本引擎目录下的 `build/` 与 `dist/`（`.gitignore` 已排除）；
`dist/catalog-entry-*.json` 是给应用 `catalog.json` 用的**发布条目**（含真实 bytes/sha256），
它很小、发布时要粘贴，所以允许进仓库——但记住它是**上一次构建**的结果。

### 4. 许可要写清楚

归档里分发第三方运行时要让用户看得到出处与许可（在引擎 README 的「许可与出处」里列表），
`extension.json` 的 `license` 填整包最严格的那一档。

---

## 仓库里还有什么

| 路径 | 作用 |
|---|---|
| `tools/audit-win-deps.mjs` | **Windows 依赖体检**：扫归档里所有 PE 文件的导入表，指出「既不在包里、也不是系统 DLL」的依赖。构建机往往是 macOS，跑不了 Windows，只能靠它把关。已接进 manga-anki 的 win32 构建收尾 |
| `LICENSE` | GPL-3.0（comic-text-detector 与 mokuro 都是 GPL，所以这一整库也是） |
| `docs/models.md` | **模型权重是哪儿来的**、现在怎么获取、几种可选形态（供决策） |
| `docs/roadmap.md` | **已 mark 的后续项**：分词引擎也要单独分发（+ 设计约束）、Rust 引擎剩余工作、模型分发形态、协议版本字段 |

---

## 怎么用（在主应用里）

```bash
git clone --recursive git@github.com:heyanLE/arale-book.git     # submodule 一起下来
cd arale-book/engines/arale_onnx_v1
cargo build --release            # Rust 引擎（ONNX Runtime 用 load-dynamic，运行时指过去）
node build.mjs --target all      # 装配归档 + 更新库根 catalog.json（release{repo,tag,asset}）
```
