#!/usr/bin/env node
/**
 * build.mjs —— 把本机的 `manga_anki` OCR 管线打成**可下载的扩展归档**。
 *
 * 产物（全部落在 `vendor/ocr-manga-anki/` 下）：
 *
 *     dist/ocr-manga-anki-macos-arm64.zip
 *     dist/ocr-manga-anki-windows-x64.zip
 *     dist/catalog-entry-<target>.json
 *
 * 归档根目录的布局（应用把它解到 `<userData>/extensions/ocr-manga-anki/` 后直接跑）：
 *
 *     extension.json                 # 自描述 + runner
 *     ocr-bridge.py                  # 构建时从 scripts/ocr-bridge.py 拷贝（单一真相源）
 *     bin/ocr-run                    # 仅 macOS；人工排障用，不是正式 runner
 *     python/                        # 自带解释器（不是 venv！）
 *     engine/                        # 第三方包（site-packages 的内容）
 *     .models/…                      # 两个模型，平台无关
 *
 * ## 为什么不能直接打包 venv
 *
 * `<factory>/.ocr-venv/bin/python3` 是一个**符号链接**，指向
 * `<codex runtime>/dependencies/python/bin/python3`，而且 `pyvenv.cfg` 里写的是绝对
 * `home =`。换台机器、换个路径就废。所以这里拷贝**真实解释器安装目录**，自己拼出
 * 一份可搬家的 `python/`。
 *
 * ## 为什么 macOS 的 runner 不是 `bin/ocr-run`
 *
 * 应用安装扩展时走 `src/main/native/sidecar.ts → extractArchive`，落到 Rust 的
 * `native/arale-native/src/archive/zip.rs`。那里写文件只用 `File::create` + `write_all`，
 * **完全不还原 unix 权限位**（`grep -rn "set_permissions" src/` 一条都没有，只有
 * `tar.rs` 在*写* tar 时设 0o644）。唯一补权限的地方是
 * `src/main/extensions/service.ts:597 chmodExecutable()`，而它**只给
 * `extension.json → runner.program` 那一个路径 chmod 0755**。
 *
 * 于是：`program = "bin/ocr-run"` 时会得到「脚本可执行、但它 exec 的
 * `python/bin/python3` 仍是 0644」→ EACCES。zip 的 external attributes 救不了
 * ——解压端根本不读它。
 *
 * 所以 macOS 直接用解释器当 runner：
 *
 *     program = "python/bin/python3"
 *     args    = ["-s","-u","ocr-bridge.py","--manga-anki-root",".","--pages-file","{pagesFile}"]
 *     env     = {"PYTHONPATH":"engine"}
 *
 * 这条路只依赖应用明确实现的那一个 chmod，不依赖任何归档元数据。`runner.env` 的合并
 * 见 `src/main/ocr/providers/extension.ts`（`...process.env, …offline 变量,
 * ...(manifest.runner.env ?? {})`——runner.env 最后展开，所以能覆盖）。
 *
 * 双保险：另外在 `python/lib/python3.12/site-packages/arale-engine.pth` 里写一行
 * 相对路径把 `engine/` 塞进 `sys.path`。`.pth` 的相对路径是相对 **site-packages 自己**
 * 解析的，与 cwd 无关；而 `PYTHONPATH=engine` 是相对 cwd（应用把 cwd 设成安装目录，
 * 所以也对）。两条都留着，坏一条不至于整包报废。
 *
 * ## 为什么 Windows 的 runner 是 `python/python.exe`
 *
 * Windows 没有执行位这回事，`python.exe` 在 embeddable 包里本来就是可执行文件；
 * 而 zip 解压丢失权限位对 Windows 无影响。`python312._pth` 会**替换**整个 `sys.path`，
 * 所以必须把 `engine` 与包根显式写进去（见 `writePth`）。
 *
 * ## zip 是自己写的
 *
 * 用 `zlib` 流式 deflate + 手写 ZIP 记录。要点：
 * - 不用 data descriptor（本地头里直接写真实 crc/压缩后大小/原始大小，靠回填
 *   `pwrite` 打补丁）。有些读端对 data descriptor 支持不完整，少一个变量少一个坑。
 * - external attributes 里**写上 unix mode**（0755），万一哪个解压端认呢。
 * - 固定 DOS 时间戳 ⇒ 同样的输入产生同样的 sha256，构建可复现。
 *
 * 用法：
 *
 *     node build.mjs --target all
 *     node build.mjs --target darwin-arm64 --keep
 *     node build.mjs --target win32-x64 --python-archive /path/to/python-3.12.10-embed-amd64.zip
 */

import { spawnSync } from 'node:child_process';
import crypto from 'node:crypto';
import fs from 'node:fs';
import fsp from 'node:fs/promises';
import os from 'node:os';
import path from 'node:path';
import { Readable, Transform, Writable } from 'node:stream';
import { pipeline } from 'node:stream/promises';
import { fileURLToPath } from 'node:url';
import zlib from 'node:zlib';

// ---------------------------------------------------------------------------
// 常量
// ---------------------------------------------------------------------------

const HERE = path.dirname(fileURLToPath(import.meta.url));
const REPO = HERE; // 这个仓库自己就是根：桥、启动器模板、LICENSE 都在这里

/**
 * 构建前置：一份 **manga_anki 检出**（它提供 `.ocr-venv` 与 `.models`）。
 *
 * 刻意**不留任何个人路径做默认值**：这个仓库是独立分发用的，默认值指向某个人的家目录
 * 只会让别人的第一次构建失败在一句莫名其妙的 ENOENT 上。必须显式给 `--manga-anki-root`
 * 或设 `ARALE_MANGA_ANKI_ROOT`，缺了就当场报错并说清要什么。
 */
const DEFAULT_ROOT = null;
/** 桥就在本仓库根目录——**唯一真相源**，构建时拷进归档。 */
const BRIDGE_SRC = path.join(REPO, 'ocr-bridge.py');
const BRIDGE_REL = 'ocr-bridge.py';
const LAUNCHER_TPL = path.join(HERE, 'launcher', 'ocr-run');

const PYTHON_EMBED_URL =
  'https://www.python.org/ftp/python/3.12.10/python-3.12.10-embed-amd64.zip';
const PYTHON_EMBED_NAME = 'python-3.12.10-embed-amd64.zip';

const DETECTOR_REL = '.models/mokuro-cache/manga-ocr/comictextdetector.pt';
const RECOGNIZER_REL = '.models/manga-ocr-base';

const TARGETS = {
  'darwin-arm64': {
    platform: 'darwin',
    arch: 'arm64',
    archive: 'ocr-manga-anki-macos-arm64.zip',
    requirement: '需要 macOS 14+ / Apple Silicon',
  },
  'win32-x64': {
    platform: 'win32',
    arch: 'x64',
    archive: 'ocr-manga-anki-windows-x64.zip',
    requirement: '需要 64 位 Windows 10+',
  },
};

/** 从 `pip freeze` 里剔掉的东西：构建/交互工具，运行时不需要。 */
const REQUIRE_EXCLUDE = [
  /^pip$/i,
  /^setuptools$/i,
  /^wheel$/i,
  /^jupyter/i,
  /^ipython/i,
];

/**
 * `unidic-lite` 与 `yattag` 在 PyPI 上**只有 sdist**（没有 wheel），而
 * `comic-text-detector` 压根不在 PyPI（上游是 GitHub，本次网络不可达）。
 * 这三者都是平台无关的纯 Python/数据包，所以构建时从 macOS 的 site-packages
 * 现场打成 `py3-none-any` 本地 wheel，交给 pip 正常安装 —— 这样
 * `--only-binary=:all:` 不用开洞，依赖解析也照常。
 */
const PURE_WHEELS = [
  { name: 'unidic-lite', version: '1.0.8', dirs: ['unidic_lite'], distInfo: 'unidic_lite-1.0.8.dist-info' },
  { name: 'yattag', version: '1.16.1', dirs: ['yattag'], distInfo: 'yattag-1.16.1.dist-info' },
  { name: 'comic-text-detector', version: '1.0.0', dirs: ['comic_text_detector'], distInfo: null },
];

/**
 * 这几个 `testing` 目录是包自己的公开 API / 被包内部 import，删了会当场炸，明确保留：
 * - `torch/testing`：`torch/autograd/gradcheck.py` 第 10 行就是 `import torch.testing`，
 *   而 `torch.autograd` 在 `import torch` 时就被拉起；
 * - `sympy/testing`：`sympy/__init__.py` 用它实现 `test` / `doctest`；
 * - `numpy/testing`：`np.testing.*` 是公开 API，下游库大量使用。
 * 三个加起来 ~11 MiB，不值得为它冒 `import torch` 直接失败的风险。
 */
const KEEP_TEST_DIRS = ['torch/testing', 'sympy/testing', 'numpy/testing'];

// ---------------------------------------------------------------------------
// 日志 / 体积
// ---------------------------------------------------------------------------

const mib = (bytes) => `${(bytes / 1048576).toFixed(1)} MiB`;

function log(scope, message) {
  process.stdout.write(`[${scope}] ${message}\n`);
}

function fail(scope, message) {
  process.stderr.write(`[${scope}] 错误：${message}\n`);
  process.exit(1);
}

/** 递归统计一个路径下的文件数与字节数（符号链接按链接本身算 0）。 */
function measure(target) {
  let files = 0;
  let bytes = 0;
  const walk = (abs) => {
    let st;
    try {
      st = fs.lstatSync(abs);
    } catch {
      return;
    }
    if (st.isDirectory()) {
      let entries;
      try {
        entries = fs.readdirSync(abs);
      } catch {
        return;
      }
      for (const name of entries) walk(path.join(abs, name));
      return;
    }
    files += 1;
    bytes += st.size;
  };
  walk(target);
  return { files, bytes };
}

const sizeMiB = (target) => mib(measure(target).bytes);

function listFiles(root) {
  const out = [];
  const walk = (abs, rel) => {
    for (const entry of fs.readdirSync(abs, { withFileTypes: true })) {
      const childAbs = path.join(abs, entry.name);
      const childRel = rel === '' ? entry.name : `${rel}/${entry.name}`;
      if (entry.isDirectory()) {
        walk(childAbs, childRel);
      } else if (entry.isSymbolicLink()) {
        log('zip', `跳过符号链接：${childRel}`);
      } else {
        out.push({ rel: childRel, abs: childAbs, size: fs.statSync(childAbs).size });
      }
    }
  };
  walk(root, '');
  out.sort((a, b) => (a.rel < b.rel ? -1 : a.rel > b.rel ? 1 : 0));
  return out;
}

const sha256File = (file) =>
  crypto.createHash('sha256').update(fs.readFileSync(file)).digest('hex');

// ---------------------------------------------------------------------------
// ZIP：写
// ---------------------------------------------------------------------------

const CRC_TABLE = (() => {
  const table = new Int32Array(256);
  for (let n = 0; n < 256; n += 1) {
    let c = n;
    for (let k = 0; k < 8; k += 1) c = c & 1 ? 0xedb88320 ^ (c >>> 1) : c >>> 1;
    table[n] = c;
  }
  return table;
})();

function crc32Update(crc, buf) {
  let c = crc ^ 0xffffffff;
  for (let i = 0; i < buf.length; i += 1) c = CRC_TABLE[(c ^ buf[i]) & 0xff] ^ (c >>> 8);
  return (c ^ 0xffffffff) >>> 0;
}

/** 固定的 DOS 时间戳（2024-01-01 00:00:00），让归档可复现。 */
const DOS_TIME = 0;
const DOS_DATE = ((2024 - 1980) << 9) | (1 << 5) | 1;

/** 本地头 / 中央目录里都不含 data descriptor：先占位、写完回填。 */
async function writeOneEntry(handle, state, entry, level) {
  const nameBuf = Buffer.from(entry.name, 'utf8');
  if (nameBuf.length > 0xffff) throw new Error(`文件名过长：${entry.name}`);

  const localOffset = state.offset;
  const header = Buffer.alloc(30);
  header.writeUInt32LE(0x04034b50, 0);
  header.writeUInt16LE(20, 4); // version needed
  header.writeUInt16LE(0x0800, 6); // flags：UTF-8 文件名
  header.writeUInt16LE(8, 8); // method：deflate
  header.writeUInt16LE(DOS_TIME, 10);
  header.writeUInt16LE(DOS_DATE, 12);
  header.writeUInt32LE(0, 14); // crc 占位
  header.writeUInt32LE(0, 18); // csize 占位
  header.writeUInt32LE(0, 22); // usize 占位
  header.writeUInt16LE(nameBuf.length, 26);
  header.writeUInt16LE(0, 28);

  await writeAt(handle, state, header);
  await writeAt(handle, state, nameBuf);

  let crc = 0;
  let usize = 0;
  let csize = 0;

  const counter = new Transform({
    transform(chunk, _enc, cb) {
      crc = crc32Update(crc, chunk);
      usize += chunk.length;
      cb(null, chunk);
    },
  });
  const sink = new Writable({
    write(chunk, _enc, cb) {
      const position = state.offset;
      state.offset += chunk.length;
      csize += chunk.length;
      handle.write(chunk, 0, chunk.length, position).then(
        () => cb(),
        (error) => cb(error),
      );
    },
  });

  if (entry.data !== undefined) {
    const source = Readable.from([entry.data]);
    await pipeline(source, counter, zlib.createDeflateRaw({ level }), sink);
  } else {
    await pipeline(
      fs.createReadStream(entry.abs, { highWaterMark: 1 << 20 }),
      counter,
      zlib.createDeflateRaw({ level }),
      sink,
    );
  }

  const patch = Buffer.alloc(12);
  patch.writeUInt32LE(crc, 0);
  patch.writeUInt32LE(csize, 4);
  patch.writeUInt32LE(usize, 8);
  await handle.write(patch, 0, 12, localOffset + 14);

  return { crc, csize, usize, localOffset, nameBuf, mode: entry.mode };
}

async function writeAt(handle, state, buf) {
  const position = state.offset;
  state.offset += buf.length;
  await handle.write(buf, 0, buf.length, position);
}

/**
 * 把一组条目写成 zip。
 *
 * @param {string} outFile
 * @param {{name:string, abs?:string, data?:Buffer, mode?:number}[]} entries
 * @param {{level?:number}} [options]
 */
async function writeZip(outFile, entries, options = {}) {
  const level = options.level ?? 9;
  const handle = await fsp.open(outFile, 'w');
  const state = { offset: 0 };
  const central = [];
  try {
    for (const entry of entries) {
      const result = await writeOneEntry(handle, state, entry, level);
      central.push(result);
    }

    const centralOffset = state.offset;
    for (const item of central) {
      const head = Buffer.alloc(46);
      head.writeUInt32LE(0x02014b50, 0);
      head.writeUInt16LE(0x031e, 4); // made by：unix + spec 3.0
      head.writeUInt16LE(20, 6);
      head.writeUInt16LE(0x0800, 8);
      head.writeUInt16LE(8, 10);
      head.writeUInt16LE(DOS_TIME, 12);
      head.writeUInt16LE(DOS_DATE, 14);
      head.writeUInt32LE(item.crc, 16);
      head.writeUInt32LE(item.csize, 20);
      head.writeUInt32LE(item.usize, 24);
      head.writeUInt16LE(item.nameBuf.length, 28);
      head.writeUInt16LE(0, 30); // extra
      head.writeUInt16LE(0, 32); // comment
      head.writeUInt16LE(0, 34); // disk
      head.writeUInt16LE(0, 36); // internal attrs
      // external attrs 高 16 位是 unix mode —— 万一解压端认权限位呢。
      head.writeUInt32LE(((item.mode & 0xffff) << 16) >>> 0, 38);
      head.writeUInt32LE(item.localOffset, 42);
      await writeAt(handle, state, head);
      await writeAt(handle, state, item.nameBuf);
    }
    const centralSize = state.offset - centralOffset;

    if (central.length > 0xffff) throw new Error('条目数超过 65535，需要 ZIP64（未实现）');
    if (centralOffset > 0xffffffff || centralSize > 0xffffffff || state.offset > 0xffffffff) {
      throw new Error('归档超过 4 GiB，需要 ZIP64（未实现）');
    }

    const end = Buffer.alloc(22);
    end.writeUInt32LE(0x06054b50, 0);
    end.writeUInt16LE(0, 4);
    end.writeUInt16LE(0, 6);
    end.writeUInt16LE(central.length, 8);
    end.writeUInt16LE(central.length, 10);
    end.writeUInt32LE(centralSize, 12);
    end.writeUInt32LE(centralOffset, 16);
    end.writeUInt16LE(0, 20);
    await writeAt(handle, state, end);
  } finally {
    await handle.close();
  }
  return state.offset;
}

/** 把目录整棵打成 zip（归档根 = 目录本身）。 */
async function zipTree(rootDir, outFile) {
  const files = listFiles(rootDir);
  return writeZip(
    outFile,
    files.map((file) => {
      // 真实权限位写进 external attributes：`bin/ocr-run`、`python/bin/python3`
      // 因此会被 Info-ZIP / Finder 之类还原成 0755。应用那条解压链路不读它
      // （所以 runner 才指向解释器本身），但多一层保险没坏处。
      const mode = fs.statSync(file.abs).mode & 0o111 ? 0o755 : 0o644;
      return { name: file.rel, abs: file.abs, mode };
    }),
  );
}

/**
 * 把 0 字节文件补成一个换行。
 *
 * 为什么必须做：应用的解压器（`native/arale-native/src/archive/zip.rs:34`）把
 * `has_stream` 定义成 `!is_dir && entry.size() > 0`，也就是**把 0 字节条目整条丢掉**
 * （对漫画页图合理：空文件不是页；对 Python 包是灾难：`comic_text_detector/__init__.py`、
 * `anyio/_core/__init__.py`、成堆的 `py.typed` 都是 0 字节）。实测不补的话
 * 10553 条里会被静默丢掉 174 条。
 *
 * 补一个换行对 `.py` / `py.typed` / `REQUESTED` 都无语义影响，但足以让条目
 * `size() > 0` 从而真的落盘。**这是构建期规范化**，改的是归档里的副本，不是上游源码。
 */
function padEmptyFiles(root, scope) {
  let padded = 0;
  const walk = (abs) => {
    for (const entry of fs.readdirSync(abs, { withFileTypes: true })) {
      const childAbs = path.join(abs, entry.name);
      if (entry.isDirectory()) {
        walk(childAbs);
        continue;
      }
      if (entry.isSymbolicLink()) continue;
      if (fs.statSync(childAbs).size === 0) {
        fs.writeFileSync(childAbs, '\n', 'utf8');
        padded += 1;
      }
    }
  };
  walk(root);
  log(scope, `0 字节文件补换行（否则会被应用解压器整条丢掉）：${padded} 个`);
  return padded;
}

// ---------------------------------------------------------------------------
// ZIP：读（校验 + 解 embeddable）
// ---------------------------------------------------------------------------

function readZipEntries(file) {
  const buf = fs.readFileSync(file);
  let eocd = -1;
  for (let i = buf.length - 22; i >= 0 && i >= buf.length - 22 - 0xffff; i -= 1) {
    if (buf.readUInt32LE(i) === 0x06054b50) {
      eocd = i;
      break;
    }
  }
  if (eocd < 0) throw new Error(`不是 zip（找不到 EOCD）：${file}`);
  const count = buf.readUInt16LE(eocd + 10);
  let offset = buf.readUInt32LE(eocd + 16);
  if (offset === 0xffffffff) throw new Error(`需要 ZIP64（未实现）：${file}`);

  const entries = [];
  for (let index = 0; index < count; index += 1) {
    if (buf.readUInt32LE(offset) !== 0x02014b50) throw new Error('中央目录损坏');
    const method = buf.readUInt16LE(offset + 10);
    const csize = buf.readUInt32LE(offset + 20);
    const usize = buf.readUInt32LE(offset + 24);
    const nameLen = buf.readUInt16LE(offset + 28);
    const extraLen = buf.readUInt16LE(offset + 30);
    const commentLen = buf.readUInt16LE(offset + 32);
    const localOffset = buf.readUInt32LE(offset + 42);
    const name = buf.toString('utf8', offset + 46, offset + 46 + nameLen);
    entries.push({ name, method, csize, usize, localOffset });
    offset += 46 + nameLen + extraLen + commentLen;
  }

  return {
    entries,
    read(entry) {
      const base = entry.localOffset;
      if (buf.readUInt32LE(base) !== 0x04034b50) throw new Error(`本地头损坏：${entry.name}`);
      const nameLen = buf.readUInt16LE(base + 26);
      const extraLen = buf.readUInt16LE(base + 28);
      const start = base + 30 + nameLen + extraLen;
      const raw = buf.subarray(start, start + entry.csize);
      if (entry.method === 0) return Buffer.from(raw);
      if (entry.method === 8) return zlib.inflateRawSync(raw);
      throw new Error(`不支持的压缩方法 ${entry.method}：${entry.name}`);
    },
  };
}

function extractZip(file, outDir) {
  const zip = readZipEntries(file);
  let written = 0;
  for (const entry of zip.entries) {
    if (entry.name.endsWith('/')) continue;
    const target = path.join(outDir, entry.name);
    if (!path.resolve(target).startsWith(path.resolve(outDir) + path.sep)) {
      throw new Error(`条目越界：${entry.name}`);
    }
    fs.mkdirSync(path.dirname(target), { recursive: true });
    fs.writeFileSync(target, zip.read(entry));
    written += 1;
  }
  return written;
}

// ---------------------------------------------------------------------------
// 本地纯 Python wheel
// ---------------------------------------------------------------------------

function buildPureWheel(sitePackages, spec, outDir) {
  const files = [];
  for (const dir of spec.dirs) {
    const abs = path.join(sitePackages, dir);
    if (!fs.existsSync(abs)) throw new Error(`本地 wheel 源目录不存在：${abs}`);
    for (const file of listFiles(abs)) {
      if (file.rel.split('/').includes('__pycache__')) continue;
      if (/\.py[co]$/.test(file.rel)) continue;
      files.push({ name: `${dir}/${file.rel}`, abs: file.abs, mode: 0o644 });
    }
  }

  // PEP 427：wheel **文件名**里的发行版名必须把非字母数字串换成下划线
  // （`unidic-lite` → `unidic_lite-1.0.8-py3-none-any.whl`）。pip 解析 wheel 文件名
  // 的正则不允许 name/version 段里出现 `-`，写成 `unidic-lite-1.0.8-…whl` 会被
  // 静默跳过，表现为「Could not find a version that satisfies unidic-lite==1.0.8
  // (from versions: none)」。dist-info 目录名同规则。
  const escaped = spec.name.replace(/[^A-Za-z0-9.]+/g, '_');
  const distInfo = spec.distInfo ?? `${escaped}-${spec.version}.dist-info`;
  const distInfoAbs = spec.distInfo === null ? null : path.join(sitePackages, spec.distInfo);
  if (distInfoAbs !== null) {
    if (!fs.existsSync(distInfoAbs)) throw new Error(`缺 dist-info：${distInfoAbs}`);
    for (const file of listFiles(distInfoAbs)) {
      if (file.rel === 'WHEEL' || file.rel === 'RECORD' || file.rel === 'INSTALLER') continue;
      if (file.rel === 'direct_url.json' || file.rel === 'REQUESTED') continue;
      files.push({ name: `${distInfo}/${file.rel}`, abs: file.abs, mode: 0o644 });
    }
  } else {
    files.push({
      name: `${distInfo}/METADATA`,
      mode: 0o644,
      data: Buffer.from(
        [
          'Metadata-Version: 2.1',
          `Name: ${spec.name}`,
          `Version: ${spec.version}`,
          'Summary: Vendored from the local manga_anki OCR venv (pure Python)',
          'License: GPL-3.0',
          '',
          'comic-text-detector 上游只发 GitHub 仓库、不发 PyPI wheel；本次构建网络里',
          'github.com 不可达，所以直接从本机 venv 的 site-packages 打成本地 wheel。',
          '该包是纯 Python（无 .so/.pyd），因此 py3-none-any 标签成立。',
          '',
        ].join('\n'),
        'utf8',
      ),
    });
  }

  files.push({
    name: `${distInfo}/WHEEL`,
    mode: 0o644,
    data: Buffer.from(
      ['Wheel-Version: 1.0', 'Generator: aralebook-ocr-build', 'Root-Is-Purelib: true', 'Tag: py3-none-any', ''].join('\n'),
      'utf8',
    ),
  });

  const recordLines = files.map((file) => {
    const bytes = file.data ?? fs.readFileSync(file.abs);
    const hash = crypto.createHash('sha256').update(bytes).digest('base64url');
    return `${file.name},sha256=${hash},${bytes.length}`;
  });
  recordLines.push(`${distInfo}/RECORD,,`);
  files.push({
    name: `${distInfo}/RECORD`,
    mode: 0o644,
    data: Buffer.from(`${recordLines.join('\n')}\n`, 'utf8'),
  });

  files.sort((a, b) => (a.name < b.name ? -1 : a.name > b.name ? 1 : 0));
  const outFile = path.join(outDir, `${escaped}-${spec.version}-py3-none-any.whl`);
  return writeZip(outFile, files).then(() => outFile);
}

// ---------------------------------------------------------------------------
// 裁剪
// ---------------------------------------------------------------------------


function prune(root, match) {
  const before = measure(root);
  const walk = (abs, rel) => {
    let entries;
    try {
      entries = fs.readdirSync(abs, { withFileTypes: true });
    } catch {
      return;
    }
    for (const entry of entries) {
      const childAbs = path.join(abs, entry.name);
      const childRel = rel === '' ? entry.name : `${rel}/${entry.name}`;
      const dir = entry.isDirectory();
      if (match(childRel, entry.name, dir, childAbs)) {
        fs.rmSync(childAbs, { recursive: true, force: true });
        continue;
      }
      if (dir) walk(childAbs, childRel);
    }
  };
  walk(root, '');
  const after = measure(root);
  return { removedBytes: before.bytes - after.bytes, removedFiles: before.files - after.files, after };
}

function trimSteps(platform) {
  const steps = [
    {
      // 只删 `.pyc` / `.pyo`。**绝不能把 `.pyd` 一起删掉**：在 Windows 上 `.pyd` 就是
      // 扩展模块本体（相当于 Unix 的 `.so`），`torch/_C.pyd`、`cv2/cv2.pyd`、
      // `numpy/_core/_multiarray_umath.pyd` 全在这一类里。照抄 `*.py[cod]` 这种
      // 通用裁剪写法会把整个 engine 打成废包（实测删掉 178 MiB 的 .pyd）。
      label: '字节码缓存（__pycache__ / *.pyc / *.pyo；保留 Windows 的 *.pyd）',
      match: (rel, name, dir) => (dir ? name === '__pycache__' : /\.py[co]$/.test(name)),
    },
    {
      label: '静态库与调试符号（*.a / *.pdb / *.exp / *.lib / *.o / *.obj）',
      match: (rel, name, dir) => !dir && /\.(a|pdb|exp|lib|o|obj)$/i.test(name),
    },
    {
      // `--target` 安装偶尔会在目标目录里留下一个 wheel 残片（实测 Windows 侧有一个
      // 1 字节的 `scipy-…-win_amd64.whl`）。engine 必须是「可直接 import 的包目录」，
      // 夹带 .whl 只会让审核的人以为打包错了——对本地的 `--find-links` 目录（在
      // `build/shared/wheels/`，不在归档里）没有任何影响。
      label: 'wheel 残片（engine 里不该出现 *.whl）',
      match: (rel, name, dir) => !dir && /\.whl$/i.test(name),
    },
    {
      label: '打包/构建工具（pip / setuptools / wheel / pkg_resources / _distutils_hack）',
      match: (rel, name, dir) =>
        dir
          ? /^(pip|setuptools|wheel|pkg_resources|_distutils_hack)$/.test(name) ||
            /^(pip|setuptools|wheel)-[0-9][^/]*\.dist-info$/.test(name)
          : name === 'distutils-precedence.pth', // setuptools 的 .pth；包删了它就成了死引用
    },
    {
      label: 'torch 编译头与自带测试（torch/include、torch/test）',
      match: (rel, name, dir) =>
        dir && (rel === 'torch/include' || rel.endsWith('/torch/include') || rel === 'torch/test' || rel.endsWith('/torch/test')),
    },
    {
      label: `测试目录（test / tests / testing；保留 ${KEEP_TEST_DIRS.join('、')}）`,
      match: (rel, name, dir) =>
        dir &&
        /^(test|tests|testing)$/.test(name) &&
        !KEEP_TEST_DIRS.some((keep) => rel === keep || rel.endsWith(`/${keep}`)),
    },
    {
      label: '文档与非 Python 包的 include/（share/doc、没有 __init__.py 的 include）',
      match: (rel, name, dir, abs) =>
        dir &&
        ((name === 'doc' && path.basename(path.dirname(abs)) === 'share') ||
          (name === 'include' && !fs.existsSync(path.join(abs, '__init__.py')))),
    },
    {
      label: '标准库里运行不需要的部分（test / ensurepip / idlelib / tkinter / turtledemo / lib2to3 / _tkinter）',
      match: (rel, name, dir) =>
        rel.startsWith('python/lib/python3.12/') &&
        (dir
          ? /^(test|tests|ensurepip|idlelib|tkinter|turtledemo|lib2to3)$/.test(name)
          : /^_tkinter\..*\.so$/.test(name)),
    },
  ];

  if (platform === 'darwin') {
    steps.push({
      label: '解释器里用不到的附属命令（python/bin/ 下除 python3 之外的全部）',
      match: (rel, name, dir) =>
        rel.startsWith('python/bin/') && (dir || rel !== 'python/bin/python3'),
    });
  }

  if (platform === 'win32') {
    steps.push({
      label: 'pip --target 生成的 POSIX 启动脚本（engine/bin）',
      match: (rel, name, dir) => rel === 'engine/bin',
    });
  }

  return steps;
}

function applyTrim(root, platform, scope) {
  for (const step of trimSteps(platform)) {
    const result = prune(root, step.match);
    log(
      scope,
      `裁剪：${step.label} → 删除 ${result.removedFiles} 项 / ${mib(result.removedBytes)}，剩余 ${mib(result.after.bytes)}`,
    );
  }
}

// ---------------------------------------------------------------------------
// 资源解析
// ---------------------------------------------------------------------------

/** 与 `scripts/ocr-bridge.py:factory_dir()` 同口径。 */
function resolveFactory(root) {
  const candidates = [path.join(root, 'factory'), root];
  for (const candidate of candidates) {
    if (fs.existsSync(path.join(candidate, '.ocr-venv')) || fs.existsSync(path.join(candidate, '.models'))) {
      return candidate;
    }
  }
  return path.join(root, 'factory');
}

function pipFreeze(venvPython) {
  const result = spawnSync(venvPython, ['-m', 'pip', 'freeze'], { encoding: 'utf8' });
  if (result.status !== 0) {
    fail('build', `pip freeze 失败：${result.stderr?.trim() ?? result.error?.message}`);
  }
  return result.stdout
    .split('\n')
    .map((line) => line.trim())
    .filter((line) => line !== '' && !line.startsWith('#'));
}

/** macOS 真实解释器安装目录（venv 的 bin/python3 是符号链接，先解析再回溯两层）。 */
function resolveMacPythonInstall(factory) {
  const venvPython = path.join(factory, '.ocr-venv', 'bin', 'python3');
  if (!fs.existsSync(venvPython)) fail('mac', `venv 里没有 python3：${venvPython}`);
  const real = fs.realpathSync(venvPython);
  const install = path.dirname(path.dirname(real));
  log('mac', `venv python3 → ${real}；真实安装目录 ${install}`);
  return { venvPython, install };
}

// ---------------------------------------------------------------------------
// 公共：写 extension.json / 拷桥接脚本 / 拷模型
// ---------------------------------------------------------------------------

function copyBridge(staging) {
  if (!fs.existsSync(BRIDGE_SRC)) fail('build', `找不到桥接脚本：${BRIDGE_SRC}`);
  const dest = path.join(staging, BRIDGE_REL);
  fs.copyFileSync(BRIDGE_SRC, dest);
  log('build', `${BRIDGE_REL} ← scripts/ocr-bridge.py（${mib(fs.statSync(dest).size)}）`);
}

function copyModels(factory, staging) {
  const recognizer = path.join(factory, RECOGNIZER_REL);
  const detector = path.join(factory, DETECTOR_REL);
  if (!fs.existsSync(path.join(recognizer, 'pytorch_model.bin'))) {
    fail('build', `识别模型不完整：${recognizer}/pytorch_model.bin 不存在`);
  }
  if (!fs.existsSync(detector)) fail('build', `检测模型不存在：${detector}`);

  const modelDest = path.join(staging, RECOGNIZER_REL);
  fs.mkdirSync(path.dirname(modelDest), { recursive: true });
  fs.cpSync(recognizer, modelDest, {
    recursive: true,
    force: true,
    dereference: true,
    // `.cache/huggingface` 只是下载缓存（含 .lock 文件），运行不需要。
    filter: (src) => path.basename(src) !== '.cache',
  });

  const detectorDest = path.join(staging, DETECTOR_REL);
  fs.mkdirSync(path.dirname(detectorDest), { recursive: true });
  fs.copyFileSync(detector, detectorDest);
  log('build', `模型已就位：${sizeMiB(path.join(staging, '.models'))}`);
}

function writeManifest(staging, target, version) {
  const info = TARGETS[target];
  const runner =
    info.platform === 'darwin'
      ? {
          // 见文件头：不能指向 bin/ocr-run（它的解释器拿不到执行位）。
          program: 'python/bin/python3',
          args: ['-s', '-u', BRIDGE_REL, '--manga-anki-root', '.', '--pages-file', '{pagesFile}'],
          env: { PYTHONPATH: 'engine' },
        }
      : {
          program: 'python/python.exe',
          args: [BRIDGE_REL, '--manga-anki-root', '.', '--pages-file', '{pagesFile}'],
        };

  const manifest = {
    id: 'ocr-manga-anki',
    version,
    kind: 'ocr-engine',
    provides: 'manga-anki',
    engine: {
      label: 'manga-anki OCR（mokuro 管线）',
      requirement: info.requirement,
      downloadSizeMb: 0,
    },
    license: 'GPL-3.0',
    homepage: 'https://github.com/kha-white/mokuro',
    runner,
  };
  fs.writeFileSync(path.join(staging, 'extension.json'), `${JSON.stringify(manifest, null, 2)}\n`, 'utf8');
  return manifest;
}

function writeCatalogEntry(target, version, archiveFile, installedBytes) {
  const info = TARGETS[target];
  const bytes = fs.statSync(archiveFile).size;
  const entry = {
    id: 'ocr-manga-anki',
    name: 'manga-anki OCR（mokuro 管线）',
    summary: '自带 Python 运行时与模型的漫画/小说 OCR 引擎（comic-text-detector + manga-ocr）',
    version,
    kind: 'ocr-engine',
    provides: 'manga-anki',
    platforms: [info.platform],
    arch: [info.arch],
    urls: [`https://github.com/aralebook/extensions/releases/latest/download/${path.basename(archiveFile)}`],
    bytes,
    sha256: sha256File(archiveFile),
    installedBytes,
    license: 'GPL-3.0',
    homepage: 'https://github.com/kha-white/mokuro',
    requires: [],
    notes: `离线可用，不需要用户装 Python。解包后约 ${(installedBytes / 1048576).toFixed(0)} MiB，首次识别会加载约 500 MiB 模型`,
  };
  const out = path.join(HERE, 'dist', `catalog-entry-${target}.json`);
  fs.mkdirSync(path.dirname(out), { recursive: true });
  fs.writeFileSync(out, `${JSON.stringify(entry, null, 2)}\n`, 'utf8');
  return entry;
}

// ---------------------------------------------------------------------------
// macOS
// ---------------------------------------------------------------------------

async function buildDarwin(factory, options) {
  const scope = 'mac';
  const staging = path.join(HERE, 'build', 'darwin-arm64');
  fs.rmSync(staging, { recursive: true, force: true });
  fs.mkdirSync(staging, { recursive: true });
  log(scope, `清空并重建 ${staging}`);

  const { install } = resolveMacPythonInstall(factory);

  // 1. 解释器：拷贝真实安装目录并裁掉 venv 无关的东西。
  const pythonDir = path.join(staging, 'python');
  fs.cpSync(install, pythonDir, {
    recursive: true,
    force: true,
    // dereference：`bin/python3` 是指向 `python3.12` 的符号链接；zip 里符号链接
    // 在我们这条解压链路上会变成「内容是目标路径的普通文件」，必须就地展开。
    dereference: true,
    filter: (src) => {
      const rel = path.relative(install, src);
      if (rel === '') return true;
      if (rel === 'bin') return true;
      if (rel === 'bin/python3') return true;
      if (rel.startsWith('bin/')) return false; // 其它命令（pip/2to3/idle3…）不要
      if (rel === 'lib') return true;
      if (rel === 'lib/python3.12') return true;
      if (rel.startsWith('lib/python3.12/')) {
        // 基础解释器自己的 site-packages（356 MiB 的 pandas/artifact_tool 之类）
        // 必须整棵丢掉：那是构建机的杂质，且会和 engine/ 里的 numpy 打架。
        return !rel.startsWith('lib/python3.12/site-packages');
      }
      if (rel === 'lib/libpython3.12.dylib') return true;
      return false; // include / share / tcl / tk / pkgconfig
    },
  });
  log(scope, `解释器（${path.basename(install)}）已拷贝：${sizeMiB(pythonDir)}`);

  // 2. engine
  const sitePackages = path.join(factory, '.ocr-venv', 'lib', 'python3.12', 'site-packages');
  if (!fs.existsSync(sitePackages)) fail(scope, `找不到 site-packages：${sitePackages}`);
  const engineDir = path.join(staging, 'engine');
  fs.cpSync(sitePackages, engineDir, { recursive: true, force: true, dereference: true });
  log(scope, `engine（site-packages 全量）已拷贝：${sizeMiB(engineDir)}`);

  copyBridge(staging);
  copyModels(factory, staging);
  const manifest = writeManifest(staging, 'darwin-arm64', options.version);

  // 3. 人工排障用的启动器
  fs.mkdirSync(path.join(staging, 'bin'), { recursive: true });
  fs.copyFileSync(LAUNCHER_TPL, path.join(staging, 'bin', 'ocr-run'));
  fs.chmodSync(path.join(staging, 'bin', 'ocr-run'), 0o755);
  fs.chmodSync(path.join(pythonDir, 'bin', 'python3'), 0o755);

  // 4. 裁剪
  applyTrim(staging, 'darwin', scope);
  padEmptyFiles(staging, scope);

  // 5. `.pth` 兜底：把 engine/ 塞进 sys.path，不依赖 cwd。
  //    相对路径是相对 site-packages 目录解析的：python/lib/python3.12/site-packages
  //    → 上溯四层 = 包根，于是 ../../../../engine = <bundle>/engine。
  const pthDir = path.join(pythonDir, 'lib', 'python3.12', 'site-packages');
  fs.mkdirSync(pthDir, { recursive: true });
  fs.writeFileSync(path.join(pthDir, 'arale-engine.pth'), '../../../../engine\n', 'utf8');
  log(scope, '已写 python/lib/python3.12/site-packages/arale-engine.pth（engine 进 sys.path 的兜底）');

  const installed = measure(staging);
  log(scope, `最终解包体积：${installed.files} 个文件 / ${mib(installed.bytes)}`);
  log(scope, `runner.program = ${manifest.runner.program}`);

  return { staging, installedBytes: installed.bytes, files: installed.files };
}

// ---------------------------------------------------------------------------
// Windows
// ---------------------------------------------------------------------------

async function fetchPythonEmbed(options) {
  const cacheDir = path.join(HERE, 'build', 'cache');
  fs.mkdirSync(cacheDir, { recursive: true });

  if (options.pythonArchive !== null) {
    if (!fs.existsSync(options.pythonArchive)) fail('win', `--python-archive 不存在：${options.pythonArchive}`);
    log('win', `使用本地 embeddable 包：${options.pythonArchive}`);
    return options.pythonArchive;
  }
  const cached = path.join(cacheDir, PYTHON_EMBED_NAME);
  if (fs.existsSync(cached) && fs.statSync(cached).size > 1_000_000) {
    log('win', `复用已下载的 embeddable 包：${cached}（${mib(fs.statSync(cached).size)}）`);
    return cached;
  }
  log('win', `下载 ${PYTHON_EMBED_URL}`);
  const response = await fetch(PYTHON_EMBED_URL, { redirect: 'follow' });
  if (!response.ok) fail('win', `下载失败：HTTP ${response.status}`);
  await pipeline(Readable.fromWeb(response.body), fs.createWriteStream(cached));
  log('win', `已下载 ${mib(fs.statSync(cached).size)}（sha256 ${sha256File(cached)}）`);
  return cached;
}

/** `python312._pth` **替换**整个 sys.path，必须把 engine 与包根写进去。 */
function writePth(pythonDir) {
  const file = path.join(pythonDir, 'python312._pth');
  const content = ['python312.zip', '.', '..\\engine', '..\\', ''].join('\r\n');
  fs.writeFileSync(file, content, 'utf8');
  log('win', `已重写 python/python312._pth：${JSON.stringify(content.split('\r\n').filter(Boolean))}`);
}

/** 把 `python312.zip` 里的 lib2to3 之流剔掉（embeddable 里本来就没有 test/tkinter）。 */
async function slimStdlibZip(pythonDir) {
  const file = path.join(pythonDir, 'python312.zip');
  if (!fs.existsSync(file)) return;
  const before = fs.statSync(file).size;
  const zip = readZipEntries(file);
  const files = zip.entries.filter((entry) => !entry.name.endsWith('/'));
  const keep = files.filter(
    (entry) =>
      !/^(lib2to3|test|tests|turtledemo|idlelib)\//.test(entry.name) &&
      !entry.name.includes('__pycache__'),
  );
  if (keep.length === files.length) {
    log('win', `python312.zip 无需裁剪（${mib(before)}）`);
    return;
  }
  const tmp = `${file}.tmp`;
  await writeZip(
    tmp,
    keep.map((entry) => ({ name: entry.name, data: zip.read(entry), mode: 0o644 })),
  );
  fs.rmSync(file, { force: true });
  fs.renameSync(tmp, file);
  log('win', `python312.zip 裁剪：${mib(before)} → ${mib(fs.statSync(file).size)}（删了 ${files.length - keep.length} 项）`);
}

/**
 * **裁剪之后**核对整棵 staging：该有的包一个都不能少，该有的 Windows 二进制也得在。
 *
 * 为什么不靠 pip 的退出码就完事：pip 成功只说明「它认为自己装好了」，而我们要的是
 * 「这份 bundle 真能在 Windows 上 import」。少了任何一个都必须在这里**大声失败**，
 * 而不是打一个跑不起来的包出去。
 *
 * 这一步刻意放在 `applyTrim()` **之后**：第一版把裁剪正则写成 `\.py[cod]$`，
 * 顺手把 Windows 的 `.pyd` 全删了（178 MiB），而那时检查还跑在裁剪之前，
 * 于是一路绿灯打出一个废包。检查点在裁剪之后就不会再漏这类错。
 */
function verifyWindowsBundle(staging, scope) {
  const engineDir = path.join(staging, 'engine');
  const required = [
    'torch',
    'torchvision',
    'cv2',
    'numpy',
    'transformers',
    'tokenizers',
    'safetensors',
    'fugashi',
    'unidic_lite',
    'manga_ocr',
    'comic_text_detector',
    'scipy',
    'sympy',
    'networkx',
    'PIL',
    'jaconv',
    'loguru',
    'shapely',
    'pyclipper',
    'yaml',
    'huggingface_hub',
    'tqdm',
  ];
  const missing = required.filter((name) => !fs.existsSync(path.join(engineDir, name)));
  if (missing.length > 0) fail(scope, `engine 里缺少必需的包：${missing.join('、')}`);

  // 解释器与 runner：少一个整包就是废的。
  for (const rel of ['python/python.exe', 'python/python312._pth', 'python/python312.dll', 'ocr-bridge.py', 'extension.json']) {
    if (!fs.existsSync(path.join(staging, rel))) fail(scope, `bundle 里缺少 ${rel}`);
  }

  const countBySuffix = (rel, re) => {
    let found = 0;
    const walk = (abs) => {
      let entries;
      try {
        entries = fs.readdirSync(abs, { withFileTypes: true });
      } catch {
        return;
      }
      for (const entry of entries) {
        if (entry.isDirectory()) walk(path.join(abs, entry.name));
        else if (re.test(entry.name)) found += 1;
      }
    };
    walk(path.join(staging, rel));
    return found;
  };

  // Windows 的扩展模块是 `.pyd`（不是 `.so`），少一个就 import 不了。
  const mustHavePyd = ['torch', 'torchvision', 'cv2', 'numpy', 'tokenizers', 'safetensors', 'fugashi', 'pyclipper'];
  const noPyd = mustHavePyd.filter((name) => countBySuffix(`engine/${name}`, /\.pyd$/i) === 0);
  if (noPyd.length > 0) fail(scope, `这些包下一个 .pyd 都没有，engine 是废的：${noPyd.join('、')}`);

  const torchDlls = countBySuffix('engine/torch/lib', /\.dll$/i);
  if (torchDlls === 0) fail(scope, 'engine/torch/lib 下没有任何 .dll，torch 在 Windows 上跑不起来');

  // 不该出现的：构建工具、被删掉又漏回来的东西。
  for (const rel of ['engine/pip', 'engine/setuptools', 'engine/wheel', 'requirements-win.txt']) {
    if (fs.existsSync(path.join(staging, rel))) fail(scope, `不该出现在归档里：${rel}`);
  }

  const stat = measure(staging);
  const pydTotal = countBySuffix('engine', /\.pyd$/i);
  log(
    scope,
    `静态核对通过：${required.length} 个必需包齐全，.pyd ${pydTotal} 个 / torch/lib dll ${torchDlls} 个，总计 ${stat.files} 个文件 / ${mib(stat.bytes)}`,
  );
}

async function buildWindows(factory, options) {
  const scope = 'win';
  const staging = path.join(HERE, 'build', 'win32-x64');
  fs.rmSync(staging, { recursive: true, force: true });
  fs.mkdirSync(staging, { recursive: true });
  log(scope, `清空并重建 ${staging}`);

  const { venvPython } = resolveMacPythonInstall(factory);
  const sitePackages = path.join(factory, '.ocr-venv', 'lib', 'python3.12', 'site-packages');

  // 1. embeddable CPython
  const embed = await fetchPythonEmbed(options);
  const pythonDir = path.join(staging, 'python');
  fs.mkdirSync(pythonDir, { recursive: true });
  const count = extractZip(embed, pythonDir);
  log(scope, `embeddable CPython 解出 ${count} 个文件：${mib(measure(pythonDir).bytes)}`);
  await slimStdlibZip(pythonDir);
  writePth(pythonDir);

  // 2. 本地纯 Python wheel（PyPI 没有 wheel 或压根没有的三个包）
  const wheelsDir = path.join(HERE, 'build', 'shared', 'wheels');
  fs.rmSync(wheelsDir, { recursive: true, force: true });
  fs.mkdirSync(wheelsDir, { recursive: true });
  for (const spec of PURE_WHEELS) {
    const file = await buildPureWheel(sitePackages, spec, wheelsDir);
    log(scope, `本地 wheel：${path.basename(file)}（${mib(fs.statSync(file).size)}）`);
  }

  // 3. 由 macOS venv 的 pip freeze 推导的依赖表
  const frozen = pipFreeze(venvPython);
  const kept = frozen.filter((line) => !REQUIRE_EXCLUDE.some((re) => re.test(line.split('==')[0])));
  const dropped = frozen.filter((line) => !kept.includes(line));
  log(scope, `pip freeze ${frozen.length} 条 → 保留 ${kept.length} 条，剔除 ${dropped.length} 条：${dropped.join(', ')}`);
  const requirements = [...kept, 'comic-text-detector==1.0.0'];
  // 依赖表放在 build/shared/ 而不是 staging 里：staging 会整棵变成归档根，
  // 构建用的 requirements 不该跟着发给用户（第一版就漏进去过一次）。
  const reqFile = path.join(HERE, 'build', 'shared', 'requirements-win32-x64.txt');
  fs.mkdirSync(path.dirname(reqFile), { recursive: true });
  fs.writeFileSync(reqFile, `${requirements.join('\n')}\n`, 'utf8');

  // 4. pip 交叉安装 win_amd64 wheel
  const engineDir = path.join(staging, 'engine');
  fs.mkdirSync(engineDir, { recursive: true });
  const cacheDir = path.join(HERE, 'build', 'cache', 'pip');
  fs.mkdirSync(cacheDir, { recursive: true });
  const pipArgs = [
    '-m',
    'pip',
    'install',
    '--target',
    engineDir,
    '--platform',
    'win_amd64',
    '--python-version',
    '3.12',
    '--implementation',
    'cp',
    '--only-binary=:all:',
    '--no-compile',
    '--upgrade',
    '--no-warn-script-location',
    '--index-url',
    'https://download.pytorch.org/whl/cpu',
    '--extra-index-url',
    'https://pypi.org/simple',
    '--find-links',
    wheelsDir,
    '-r',
    reqFile,
  ];
  log(scope, `pip install --platform win_amd64 --only-binary=:all: …（日志见下，可能要几分钟）`);
  const pip = spawnSync(venvPython, pipArgs, {
    stdio: 'inherit',
    env: { ...process.env, PIP_CACHE_DIR: cacheDir, PIP_DISABLE_PIP_VERSION_CHECK: '1' },
  });
  if (pip.status !== 0) {
    fail(
      scope,
      'pip 交叉安装失败。若是某个包没有 win_amd64 wheel，请把它加进 PURE_WHEELS（纯 Python）或换版本——不要静默降级。',
    );
  }
  log(scope, `engine 安装完成：${sizeMiB(engineDir)}`);

  copyBridge(staging);
  copyModels(factory, staging);
  const manifest = writeManifest(staging, 'win32-x64', options.version);

  applyTrim(staging, 'win32', scope);
  padEmptyFiles(staging, scope);
  verifyWindowsBundle(staging, scope);

  const installed = measure(staging);
  log(scope, `最终解包体积：${installed.files} 个文件 / ${mib(installed.bytes)}`);
  log(scope, `runner.program = ${manifest.runner.program}`);
  return { staging, installedBytes: installed.bytes, files: installed.files };
}

// ---------------------------------------------------------------------------
// CLI
// ---------------------------------------------------------------------------

function parseArgs(argv) {
  const options = {
    target: null,
    root: process.env.ARALE_MANGA_ANKI_ROOT ?? DEFAULT_ROOT ?? '',
    version: '1.0.0',
    skipZip: false,
    keep: true,
    pythonArchive: null,
  };
  for (let i = 0; i < argv.length; i += 1) {
    const arg = argv[i];
    const next = () => {
      i += 1;
      if (i >= argv.length) fail('cli', `${arg} 缺少取值`);
      return argv[i];
    };
    if (arg === '--target') options.target = next();
    else if (arg === '--manga-anki-root') options.root = next();
    else if (arg === '--version') options.version = next();
    else if (arg === '--python-archive') options.pythonArchive = path.resolve(next());
    else if (arg === '--skip-zip') options.skipZip = true;
    else if (arg === '--keep') options.keep = true;
    else if (arg === '--no-keep') options.keep = false;
    else if (arg === '--help' || arg === '-h') options.help = true;
    else fail('cli', `不认识的参数：${arg}`);
  }
  return options;
}

function usage() {
  process.stdout.write(
    [
      '用法：node build.mjs --target <darwin-arm64|win32-x64|all> [选项]',
      '',
      '  --manga-anki-root <path>   manga_anki 检出根（必填，也可用 $ARALE_MANGA_ANKI_ROOT）',
      '  --version <v>              写进 extension.json 的版本（默认 1.0.0）',
      '  --skip-zip                 只构建 build/<target>/，不打包 dist/',
      '  --python-archive <zip>     Windows embeddable CPython 的本地 zip（默认自动下载）',
      '  --keep / --no-keep         是否保留 build/<target>/ 中间产物（默认保留）',
      '',
    ].join('\n'),
  );
}

async function buildOne(target, options) {
  if (options.root === '') {
    fail(
      'build',
      '缺少 --manga-anki-root（或用 $ARALE_MANGA_ANKI_ROOT 指定）。\n' +
        '  构建需要一份 **manga_anki 检出**：它提供 .ocr-venv（site-packages）与 .models\n' +
        '  （manga-ocr-base 识别模型 + comictextdetector.pt 检测模型）。\n' +
        '  例：node build.mjs --target darwin-arm64 --manga-anki-root /path/to/manga_anki',
    );
  }
  const factory = resolveFactory(options.root);
  log('build', `target=${target} 版本=${options.version}`);
  log('build', `manga_anki root=${options.root} → factory=${factory}`);
  if (!fs.existsSync(factory)) fail('build', `factory 目录不存在：${factory}`);

  const result =
    target === 'darwin-arm64'
      ? await buildDarwin(factory, options)
      : await buildWindows(factory, options);

  if (options.skipZip) {
    log('build', `--skip-zip：跳过打包（中间产物留在 ${result.staging}）`);
    return;
  }

  const archive = path.join(HERE, 'dist', TARGETS[target].archive);
  fs.mkdirSync(path.dirname(archive), { recursive: true });
  fs.rmSync(archive, { force: true });
  const started = Date.now();
  log('zip', `打包 ${result.staging} → ${archive}`);
  const bytes = await zipTree(result.staging, archive);
  log('zip', `完成：${bytes} 字节 / ${mib(bytes)}，用时 ${((Date.now() - started) / 1000).toFixed(1)}s`);

  // 校验：列条目 + 解到临时目录比对
  const zip = readZipEntries(archive);
  log('zip', `条目数 ${zip.entries.length}（目录项 ${zip.entries.filter((e) => e.name.endsWith('/')).length}）`);
  const tmp = fs.mkdtempSync(path.join(os.tmpdir(), 'arale-zip-check-'));
  try {
    const written = extractZip(archive, tmp);
    const listing = measure(tmp);
    if (listing.files !== written) throw new Error(`解出的文件数对不上：${listing.files} != ${written}`);
    if (listing.bytes !== result.installedBytes) {
      throw new Error(`解出的字节数对不上：${listing.bytes} != ${result.installedBytes}`);
    }
    log('zip', `自检通过：解出 ${written} 个文件 / ${mib(listing.bytes)}，与中间产物一致`);
  } finally {
    fs.rmSync(tmp, { recursive: true, force: true });
  }

  const sha = sha256File(archive);
  log('zip', `bytes=${bytes} sha256=${sha}`);
  const entry = writeCatalogEntry(target, options.version, archive, result.installedBytes);
  log('build', `已写 dist/catalog-entry-${target}.json（bytes=${entry.bytes} installedBytes=${entry.installedBytes}）`);

  if (!options.keep) {
    fs.rmSync(result.staging, { recursive: true, force: true });
    log('build', `已删除中间产物 ${result.staging}`);
  }
}

async function main() {
  const options = parseArgs(process.argv.slice(2));
  if (options.help || options.target === null) {
    usage();
    if (options.target === null && options.help !== true) process.exit(1);
    return;
  }
  const targets =
    options.target === 'all' ? ['darwin-arm64', 'win32-x64'] : [options.target];
  for (const target of targets) {
    if (!(target in TARGETS)) fail('cli', `不认识的 target：${target}（可选 darwin-arm64 / win32-x64 / all）`);
  }
  if (!fs.existsSync(BRIDGE_SRC)) fail('build', `找不到 ${BRIDGE_SRC}`);

  const started = Date.now();
  for (const target of targets) await buildOne(target, options);
  log('build', `全部完成，总用时 ${((Date.now() - started) / 1000).toFixed(1)}s`);
}

await main();
