# 模型权重：现在是怎么拿到的

结论先说：**我们的构建脚本从不下载模型**——它只是从本机的一份 `manga_anki` 检出里**拷**，
而且只查「文件在不在」，**不校验 sha256**。模型真正的下载发生在**上游 manga_anki 项目**里
（它有一套带 sha256 的清单 + 镜像开关），我们是白拿了它的结果。

下面把两条链路分开写清楚，再列可选形态。

---

## 一、事实链条（都实测过）

### 1. 模型最初从哪来

上游 `manga_anki` 有一份**钉死 sha256 的清单**：
`factory/skills/manga-anki/assets/model-files.json`（7 个文件，本地合计 **524.2 MB**）
与 `scripts/bootstrap.py`（下载 → 校验 sha256 → 原子改名；`--endpoint` 可换镜像，强制 HTTPS）。

| 文件 | 来源 | 体积 | sha256（前 16 位） |
|---|---|---|---|
| `manga-ocr-base/pytorch_model.bin` | `https://huggingface.co/kha-white/manga-ocr-base/resolve/**aa6573bd…**/pytorch_model.bin` | 444 MB | `c63e0bb5b3ff798c` |
| `manga-ocr-base/{config,vocab.txt,tokenizer_config,special_tokens_map,preprocessor_config}` | 同上（同一个 commit-pinned 版本） | < 0.1 MB | … |
| `mokuro-cache/manga-ocr/comictextdetector.pt` | `https://github.com/zyddnys/manga-image-translator/releases/download/beta-0.2.1/comictextdetector.pt` | 80 MB | `1f90fa60aeeb1eb8` |

> ✅ 我把本机这 7 个文件逐个算了一遍 sha256，**与清单全部一致**——所以「按 hash 钉住」这条路是通的。
> ✅ 识别模型的 URL 带 **commit hash**（`aa6573bd…`），是不可变地址；检测器上游只发 GitHub Release，
> 但清单里有 `mirror_url`（HF 镜像 `hhuggg/comictextdetector`）。

### 2. 我们的构建脚本干了什么

`engines/manga-anki/build.mjs`（移植前是 `vendor/ocr-manga-anki/build.mjs`）：

```
--manga-anki-root <dir>            ← 必填（或 $ARALE_MANGA_ANKI_ROOT）
  <dir>/factory/.models/manga-ocr-base/        → 拷进归档 .models/（去掉 .cache/）
     检查：pytorch_model.bin 存在？        ← 只有存在性检查
  <dir>/factory/.models/mokuro-cache/manga-ocr/comictextdetector.pt
     检查：存在？                          ← 只有存在性检查
  <dir>/factory/.ocr-venv/lib/python3.12/site-packages/ → 拷成归档 engine/
```

**没有 sha256、没有版本号、没有下载**。所以今天「构建出一份可复现的归档」的前提是：
那台机器上恰好有一份**内容正确**的 manga_anki 检出。内容错了（换了模型、被改过）脚本不会发现——
而用户侧只校验**归档**的 sha256（应用安装时），不校验模型。

### 3. 用户侧：模型是**跟着归档一次下完**的

```
用户在应用里点「安装扩展」
  → 读 catalog.json（urls + bytes + sha256）
  → 下载整个归档 741 MiB（macOS）/ 745 MiB（Windows）
  → 校验 sha256 → 解包到 <userData>/extensions/ocr-manga-anki/
  → 之后永久离线可用，OCR 时不再有任何网络请求
```

实测：`.models` 压缩后 **456.1 MiB**，占 741.2 MiB 归档的 **62%**；其余 ≈285 MiB 是
Python 运行时（CPython + site-packages 裁剪后的 engine/）。

---

## 二、可选形态（供决策）

### A. 维持现状：模型进归档

- 用户：一次 741 MiB，装完永久离线；
- 构建：需要一份正确的本地检出（人工准备）；
- 优点：最简单，离线语义最好；缺点：构建不可复现、构建机依赖别人的仓库。

### B. 模型拆出来，首次 OCR 时按需下载

- 归档只剩运行时：**≈285 MiB**（现在是 741 MiB），模型 456 MiB 另下；
- 应用**已经有**带 sha256 校验 + 进度条 + 断点友好的下载器（就是扩展安装器本身），
  只差「把模型当第二类可下载资源」这件事；
- 优点：不想用 OCR 的人少下 456 MiB；缺点：第一次识别要等一次下载，
  「装着就离线」变成「下过才离线」。

### C. 构建脚本自己下载（带 sha256 钉住）

- `build.mjs --fetch-models`：按 §1 的 URL+sha256 拉这 7 个文件（可选 `--model-mirror` 换源）；
- 好处：**源码仓库 + 一条命令就能构建**，不再需要别人的 manga_anki 检出（这正好修掉
  「构建前置不体面」那条）；
- 风险（实测过）：**这台机器 huggingface.co 超时、GitHub Release 也超时**——
  本机跑不通，只能在能出网/有镜像的环境里验证。所以这条路必须**同时**支持
  「本地目录已有就跳过」与「镜像地址」两个开关。

### D. 模型放进本仓库的 Release（自建源）

- 把 7 个文件（或 Rust 版的 ONNX）作为 Release 资源，`build.mjs` 从这里下；
- 优点：不依赖 HF/GitHub 上游，sha256 自己钉，404 风险最小；
- 缺点：多一次人工上传（~520 MB，或 Rust int8 后 ~140 MB）。

### 组合建议（如果按我的判断）

**C 的形态 + D 的源 + A/B 作为用户侧策略**：构建脚本「先看本地目录 → 再按钉住的 sha256 从
自建 Release 下 → 最后才试上游 URL（HF 带 commit hash，检测器带 GitHub Release + HF 镜像）」。
这样构建可复现、离线可构建、也不依赖别人的检出。
用户侧再单独决定：模型跟随归档（A）还是按需下载（B）——Rust 版把这两个数字改成
170 MB / 35 MB 之后，A 的负担小很多，**B 的必要性会下降**。

> 未验证：C/D 两条路我都没能实跑（本机到 HF 与 GitHub Release 都不通）。要落地得在能出网的
> 机器或 CI 上跑一次，并以 sha256 为准。
