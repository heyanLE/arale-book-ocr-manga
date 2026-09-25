# manga-anki（引擎）

**あられブック（ARaLeBook）的 manga-anki OCR 引擎**，是 [`arale-book-ocr`](../README.md) 引擎库里的
一个引擎（目前唯一一个）。它和主应用的发版节奏、平台、体积都不是一回事，所以不跟主应用一起发版。

它把 `comic-text-detector`（检测）+ `manga-ocr`（识别）这套 Python 管线重新组装成两个
**自带解释器、自带依赖、自带模型**的归档；应用把它当扩展安装，用户机器上
**不需要装 Python、不需要联网**。

```
manga-anki/                ← 引擎目录（库里的一级目录 = 一个引擎）
├── ocr-bridge.py      NDJSON 桥：应用 spawn 它，它把页图变成一行行文字与框（唯一真相源）
├── build.mjs          打包脚本：装配 python/ + engine/ + .models/，写 extension.json，打 zip
├── launcher/ocr-run   macOS 下的手工启动器（排障用，不是正式 runner）
├── dist/              归档与发布条目（`**/dist/*.zip` 不进 git，走 Release）
└── build/             中间产物（不进 git）
```

## 为什么单独一条线

| | 主应用 | 这个引擎 |
|---|---|---|
| 发版节奏 | 跟功能走 | 跟上游模型/依赖走 |
| 平台 | macOS arm64（也支持 Windows/Linux 的界面） | 每个平台一个归档（darwin-arm64 / win32-x64） |
| 体积 | 安装包几十 MB | 归档 ~0.75 GB / 解包 1.5 GB |
| 许可 | GPL-3.0 | GPL-3.0（comic-text-detector、mokuro 都是 GPL） |

应用侧只认一份 `extension.json` 和一套 NDJSON 协议，所以**这里换实现、应用一行都不用动**。

> 换成 Rust/ONNX 这件事已经实测过：两个模型都能导出 ONNX（编码器 343→87 MB、解码器 117→30 MB、
> 检测器 95→54 MB，int8 合计 **170 MB**，数值差异 ~1e-5），识别结果与 PyTorch 在真实裁剪上逐字一致
> （**解码必须照抄 `num_beams=4 / no_repeat_ngram_size=3 / length_penalty=2.0`**，裸贪心在难图上会给出不同文字）；
> 分词器实测等价于「NFKC + 逐字查表」，**不需要 MeCab 与 248 MiB 的 UniDic**（真实语料 2241 条零差异）。
> 剩下的风险集中在检测器那 ~500 行 OpenCV 后处理。完整数据与验收标准见主应用仓库的
> `docs/rust-engine-feasibility.md`。

## 源码很小，**不需要 LFS**

仓库里进 git 的东西加起来 **~90 KB**：

| 文件 | 大小 |
|---|---|
| `build.mjs` | 50 KB |
| `ocr-bridge.py` | 23 KB |
| `README.md` | 14 KB |
| `launcher/ocr-run` | 0.8 KB |

**大东西一个都不进 git**：

| 东西 | 大小 | 去哪儿 |
|---|---|---|
| `dist/*.zip`（两个归档） | 741 MiB + 745 MiB | **GitHub Release 资源**（每个上限 2 GiB，不计 LFS 配额，`git clone` 不会拉它们） |
| `build/`（中间产物：解释器 + 依赖 + 模型） | 1.6 GB | 本地，`.gitignore` |
| 模型权重（识别 424 MiB + 检测 76 MiB） | 500 MB | 构建前置的 manga_anki 检出里（或自备） |

> 顺带一句：这两个 zip **也不可能**用普通 git 提交——GitHub 对单文件超过 100 MiB 的普通
> 提交是直接拒绝的。放 Release 资源才是正路；LFS 反而会吃掉免费额度
> （1 GiB 存储 / 1 GiB 月流量，两个归档 1.5 GB 一上来就爆），而且每次 clone 都在烧流量。

## 构建

前置：Node ≥ 20，加上一份 **manga_anki 检出**（提供 `.ocr-venv` 与 `.models`）：

> 模型现在是从那份检出里**拷**的，脚本只查存在性、**不校验 sha256**。
> 来龙去脉与几种可选形态（自建源 / 按需下载 / 构建时下载）见 [`docs/models.md`](../docs/models.md)。

```bash
git clone --recursive <arale-book>     # 引擎在 submodule 里
cd arale-book/engines/manga-anki

node build.mjs --target darwin-arm64 --manga-anki-root ../manga_anki
node build.mjs --target win32-x64    --manga-anki-root ../manga_anki
node build.mjs --target all          --manga-anki-root ../manga_anki   # 两个都打
# 也可以 export ARALE_MANGA_ANKI_ROOT=../manga_anki 之后省略这个参数
```

| 参数 | 说明 |
| --- | --- |
| `--target <darwin-arm64\|win32-x64\|all>` | 必填 |
| `--manga-anki-root <path>` | **必填**（或 `$ARALE_MANGA_ANKI_ROOT`）：manga_anki 检出根 |
| `--version <v>` | 写进 `extension.json` 与清单条目，默认 `1.0.0` |
| `--skip-zip` | 只装配 `build/<target>/`，不打归档 |
| `--python-archive <zip>` | Windows 用本地 embeddable CPython（默认自动下载 3.12.10） |
| `--keep` / `--no-keep` | 是否保留中间产物，默认保留 |

脚本是幂等的、输出可复现：同样的输入重跑得到**同样的 sha256**（zip 用固定 DOS 时间戳）。

- macOS：解释器从 venv 的基础解释器拷贝并裁剪；依赖直接从 venv 的 site-packages 拷；不需要联网。
- Windows：`python.org` 的 embeddable CPython + `pip install --platform win_amd64` 交叉安装
  （torch/torchvision 走 `download.pytorch.org/whl/cpu`），所以**构建机可以就是 macOS 一台**。

## 发布

```bash
# 1) 打归档（会打印 bytes / sha256，并写出 dist/catalog-entry-<target>.json）
node build.mjs --target all --manga-anki-root ../manga_anki

# 2) 传到 Release（每个平台一个资源）
#    macOS: ocr-manga-anki-macos-arm64.zip
#    Windows: ocr-manga-anki-windows-x64.zip

# 3) 把 dist/catalog-entry-<target>.json 粘进应用的
#    resources/extensions/catalog.json 的 extensions 数组（urls 改成本仓库的 release 地址，
#    sha256/bytes 是构建时写出的真值，不用改）
```

应用侧安装时会**强制校验 sha256**，所以「换了归档忘了改 sha」只会安装失败，不会装进一个坏东西。

## 归档内部布局

```
extension.json                 # 自描述 + runner（平台不同）
ocr-bridge.py
bin/ocr-run                    # 仅 macOS，排障用
python/                        # 自带解释器（不是 venv）
engine/                        # 第三方包（裁剪后的 site-packages）
.models/manga-ocr-base/…       # 识别模型
.models/mokuro-cache/manga-ocr/comictextdetector.pt
```

两个平台各自的 `runner`：

```json
// macOS
{ "program": "python/bin/python3",
  "args": ["-s","-u","ocr-bridge.py","--manga-anki-root",".","--pages-file","{pagesFile}"],
  "env": { "PYTHONPATH": "engine" } }
// Windows
{ "program": "python/python.exe",
  "args": ["ocr-bridge.py","--manga-anki-root",".","--pages-file","{pagesFile}"] }
```

**为什么 macOS 的 runner 是解释器而不是 `bin/ocr-run`**：应用解压归档时会丢掉 unix 权限位，
而它只给 `runner.program` 这一个路径补 `+x`。指向脚本的话会得到「脚本能跑、它 exec 的
解释器还是 0644」→ EACCES。zip 的 external attributes 救不了（问题在解压端）。

## 已验证 / 未验证

已验证（macOS arm64，本机）：

- 归档解压后只给 `runner.program` 补 0755，再按 `extension.json` 原样 spawn：真实漫画 022.jpg
  32 行 / 095.jpg 21 行，退出码 0，文字与框都在图内；
- 用**应用真正的解压器**（`arale-native extract`）解归档 → 解出的文件数与字节数与中间产物
  逐条一致（`skipped=0`，构建脚本自己也会做这一步自检）；
- 装进应用后走应用自己的 OCR 队列跑完整本书（171 页 / 2232 个文字块）成功；
- 重跑构建 sha256 不变（可复现）：本仓库第一次构建就得到了与归档发布时**逐字节相同**的
  `7e4c4e44…`，也就是说「把桥搬进这个仓库、删掉个人路径默认值」没有改变任何产物。

未验证：

- **Windows 归档没有在 Windows 机器上运行过**：只做了交叉安装 + 打包 + 静态核对
  （150 个 `.pyd` / 22 个 dll / `python312._pth` / 各平台 `extension.json`）。
  第一次上 Windows 建议先跑
  `python\python.exe ocr-bridge.py --manga-anki-root . --pages-file pages.json`，看有没有缺 DLL。
- 为此加了一个**依赖体检**：`node tools/audit-win-deps.mjs build/win32-x64`
  （已接进 `build.mjs` 的 win32 收尾，`--audit <dir>` 可单独跑，`--strict-audit` 更严）。
  它扫包里 204 个 PE 文件的导入表，结论是**没有硬缺失**，但有 6 种「装了才有」的系统依赖，
  其中两条是真风险：`torch_cpu.dll` 要 `vcruntime140_threads.dll`、`torch_python.dll` 要
  `msvcp140_atomic_wait.dll`（**包里没有** → 没装 VC++ 2015–2022 运行库的机器上 torch 加载失败）；
  `cv2.pyd` 要 Media Foundation（**Windows N/KN 版**默认没有）。补法见主应用仓库
  `docs/rust-engine-feasibility.md` §7。
- 没有在 macOS 14 以下或 Intel Mac 上试过。
- 没有做代码签名/公证；走 Apple 公证时需要单独处理归档里的解释器与 `.so`/`.dylib`
  （hardened runtime 下通常要给应用加 `com.apple.security.cs.disable-library-validation`）。

## 许可与出处

本项目 GPL-3.0。归档里分发的是第三方运行时与模型，必须让用户看得到：

| 组件 | 许可 | 出处 |
| --- | --- | --- |
| manga-ocr | MIT | https://github.com/kha-white/manga-ocr |
| mokuro（管线思路来源） | GPL-3.0 | https://github.com/kha-white/mokuro |
| comic-text-detector | GPL-3.0 | https://github.com/dmMaze/comic-text-detector |
| PyTorch / torchvision | BSD-3-Clause | https://pytorch.org |
| CPython 运行时 | PSF-2.0 | https://www.python.org |
| transformers / tokenizers / huggingface_hub | Apache-2.0 | https://huggingface.co |
| unidic-lite（UniDic 辞书，约 248 MiB，解包体积大头） | UniDic 系许可（含再分发条款） | https://pypi.org/project/unidic-lite/ |
| OpenCV | Apache-2.0 | https://opencv.org |
| numpy / scipy / sympy / networkx / Pillow / shapely / pyclipper | BSD / MIT 系 | 各自项目 |
