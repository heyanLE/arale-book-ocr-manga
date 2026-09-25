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
| **manga-anki** | [`engines/manga-anki/`](engines/manga-anki/) | Python 3.12（自带解释器）+ comic-text-detector / manga-ocr | **741 MiB**（macOS）/ **745 MiB**（Windows） | macOS arm64、Windows x64 | macOS 已实机跑通整本 171 页；Windows 已打包并做过依赖体检，**未在 Windows 实机运行** |

> 目前只有这一个。清单是**给未来留的位**：第二个引擎（比如 Rust/ONNX 版、或别的语言的实现）
> 只要满足下面的约定，就能和它并存，应用侧一行代码都不用改。

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

### 3. 目录形状

```
engines/<引擎名>/
├── README.md          这个引擎是什么、怎么构建、体积、已知限制
├── build.mjs          构建脚本（把运行时+模型装配成归档，写 extension.json，打 zip）
└── …                  引擎自己的源码（桥/二进制/模型清单…）
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

---

## 怎么用（在主应用里）

```bash
git clone --recursive git@github.com:heyanLE/arale-book.git     # submodule 一起下来
cd arale-book/engines/manga-anki
node build.mjs --target all --manga-anki-root /path/to/manga_anki
# → dist/ocr-manga-anki-macos-arm64.zip、dist/ocr-manga-anki-windows-x64.zip
# → dist/catalog-entry-*.json（粘进应用的 resources/extensions/catalog.json）
```
