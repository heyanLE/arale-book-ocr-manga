#!/usr/bin/env node
/**
 * build.mjs —— 把 Rust/ONNX 引擎打成**可下载的扩展归档**。
 *
 *     node build.mjs --target darwin-arm64 --models /path/to/models --ort /path/to/libonnxruntime.dylib
 *     node build.mjs --target win32-x64 --bin dist/arale_onnx_v1.exe --models … --ort …/onnxruntime.dll
 *     node build.mjs --target all --skip-build
 *
 * 产物：
 *
 *     dist/arale_onnx_v1-macos-arm64.zip
 *     dist/arale_onnx_v1-windows-x64.zip
 *     dist/catalog-entry-<target>.json      # 单个平台条目（含 sha256/bytes）
 *     ../catalog.json                       # 引擎库根的清单：一个能力一条，平台差异在 assets 里
 *
 * 归档根的布局（应用解到 `<userData>/extensions/ocr-arale_onnx_v1/` 后直接跑）：
 *
 *     extension.json                 # 自描述 + runner
 *     bin/arale_onnx_v1[.exe]        # 唯一的可执行文件（应用只给这一个 chmod）
 *     models/{detector,manga-ocr-encoder,manga-ocr-decoder}.onnx + vocab.txt
 *     lib/libonnxruntime.<ver>.dylib | onnxruntime.dll
 *     LICENSE
 *
 * ## 为什么 ORT 是运行时加载的
 *
 * 引擎用 `ort` 的 **load-dynamic** 编译：构建时**不下载**运行时（构建可离线），
 * 运行时由归档自带的 `lib/` 提供。`extension.json → runner.env.ORT_DYLIB_PATH`
 * 写**相对路径**——应用 spawn 时 `cwd` = 安装目录（`src/main/ocr/providers/extension.ts`），
 * 所以相对路径可解析，也不怕用户搬目录。
 *
 * ## 为什么只有一个可执行文件
 *
 * 应用解压走 `native/arale-native` 的 zip 解包，**不还原 unix 权限位**；唯一补权限的
 * 地方是 `src/main/extensions/service.ts` 的 `chmodExecutable()`，而它只给
 * `extension.json → runner.program` 那一个路径 chmod 0755。所以 runner 必须是**那一个**
 * 原生二进制，不能是 shell 脚本去 exec 别的东西。（上一版 Python 引擎因此把 runner
 * 指成 `python/bin/python3`；Rust 版没这个问题。）
 *
 * ## zip 是自己写的
 *
 * 用 `zlib` 流式 deflate + 手写 ZIP 记录，不引第三方依赖；本地头里直接回填
 * crc/大小（不写 data descriptor），时间戳固定，归档可复现。
 */

import fs from 'node:fs';
import fsp from 'node:fs/promises';
import path from 'node:path';
import crypto from 'node:crypto';
import zlib from 'node:zlib';
import { spawnSync } from 'node:child_process';
import { Readable, Transform, Writable, pipeline } from 'node:stream';
import { fileURLToPath } from 'node:url';
import { promisify } from 'node:util';

const pipe = promisify(pipeline);
const here = path.dirname(fileURLToPath(import.meta.url));
const root = path.resolve(here, '..');

const ENGINE_ID = 'arale_onnx_v1';
const EXTENSION_ID = `ocr-${ENGINE_ID}`;
const ENGINE_NAME = 'arale_onnx_v1（ONNX 版 OCR）';
const ENGINE_SUMMARY =
  'comic-text-detector 检测 + manga-ocr 识别，Rust 实现 + 自带 ONNX Runtime。竖排质量好，装完不需要任何外部依赖。';
const REPO = 'heyanLE/arale-book-ocr-manga';
const LICENSE = 'GPL-3.0';
const HOMEPAGE = `https://github.com/${REPO}`;

/** 三个模型 + 词表：模型文件太大，不进 git，构建时从本机检出拷。 */
const MODEL_FILES = [
  'detector.onnx',
  'manga-ocr-encoder.onnx',
  'manga-ocr-decoder.onnx',
  'vocab.txt',
];

const TARGETS = {
  'darwin-arm64': { platform: 'darwin', arch: 'arm64', asset: 'arale_onnx_v1-macos-arm64.zip', exe: ENGINE_ID, ort: 'libonnxruntime.1.30.0.dylib' },
  'darwin-x64': { platform: 'darwin', arch: 'x64', asset: 'arale_onnx_v1-macos-x64.zip', exe: ENGINE_ID, ort: 'libonnxruntime.1.30.0.dylib' },
  'win32-x64': { platform: 'win32', arch: 'x64', asset: 'arale_onnx_v1-windows-x64.zip', exe: `${ENGINE_ID}.exe`, ort: 'onnxruntime.dll' },
};

function parseArgs(argv) {
  const args = {
    target: null,
    models: null,
    ort: null,
    bin: null,
    version: null,
    out: path.join(here, 'dist'),
    level: 6,
    skipBuild: false,
    keepStaging: false,
  };
  for (let i = 0; i < argv.length; i += 1) {
    const flag = argv[i];
    const next = () => {
      i += 1;
      if (i >= argv.length) throw new Error(`${flag} 后面要跟值`);
      return argv[i];
    };
    if (flag === '--target') args.target = next();
    else if (flag === '--models') args.models = next();
    else if (flag === '--ort') args.ort = next();
    else if (flag === '--bin') args.bin = next();
    else if (flag === '--version') args.version = next();
    else if (flag === '--out') args.out = path.resolve(next());
    else if (flag === '--level') args.level = Number(next());
    else if (flag === '--skip-build') args.skipBuild = true;
    else if (flag === '--keep-staging') args.keepStaging = true;
    else if (flag === '--help' || flag === '-h') args.help = true;
    else throw new Error(`未知参数：${flag}`);
  }
  return args;
}

const USAGE = `用法：
  node build.mjs --target <darwin-arm64|darwin-x64|win32-x64|all> [选项]

选项：
  --models <dir>    含 detector.onnx / manga-ocr-*.onnx / vocab.txt 的目录
  --ort <file>      ONNX Runtime 共享库（darwin: libonnxruntime.*.dylib，win32: onnxruntime.dll）
  --bin <file>      预编译好的引擎可执行文件（跳过 cargo；交叉构建时用）
  --version <v>     扩展版本（缺省读 Cargo.toml 的 version）
  --out <dir>       产物目录（缺省 <引擎目录>/dist）
  --level <0-9>     deflate 级别（缺省 6）
  --skip-build      不跑 cargo
  --keep-staging    保留中间目录（排障）`;

function versionFromCargo() {
  const text = fs.readFileSync(path.join(here, 'Cargo.toml'), 'utf8');
  const match = /^version\s*=\s*"([^"]+)"/m.exec(text);
  if (match === null) throw new Error('Cargo.toml 里找不到 version');
  return match[1];
}

function cargoBuild() {
  const cargo = process.env.CARGO ?? 'cargo';
  console.log(`[build] ${cargo} build --release`);
  const result = spawnSync(cargo, ['build', '--release'], { cwd: here, stdio: 'inherit' });
  if (result.status !== 0) throw new Error(`cargo build 失败（退出码 ${result.status}）`);
  return path.join(here, 'target', 'release', TARGETS['darwin-arm64'].exe);
}

// ---------------------------------------------------------------------------
// 自描述
// ---------------------------------------------------------------------------

function writeManifest(staging, target, version) {
  const spec = TARGETS[target];
  const manifest = {
    id: EXTENSION_ID,
    name: ENGINE_NAME,
    version,
    kind: 'ocr-engine',
    provides: ENGINE_ID,
    license: LICENSE,
    homepage: HOMEPAGE,
    description: ENGINE_SUMMARY,
    runner: {
      program: `bin/${spec.exe}`,
      args: ['--pages-file', '{pagesFile}'],
      // 相对安装目录（应用 spawn 时 cwd = 安装目录）
      env: { ORT_DYLIB_PATH: `lib/${spec.ort}` },
    },
  };
  fs.writeFileSync(path.join(staging, 'extension.json'), `${JSON.stringify(manifest, null, 2)}\n`);
}

// ---------------------------------------------------------------------------
// zip（自写：zlib + 手写记录，不引依赖）
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

async function writeAt(handle, state, buf) {
  const position = state.offset;
  state.offset += buf.length;
  await handle.write(buf, 0, buf.length, position);
}

async function writeOneEntry(handle, state, entry, level) {
  const nameBuf = Buffer.from(entry.name, 'utf8');
  const localOffset = state.offset;
  const header = Buffer.alloc(30);
  header.writeUInt32LE(0x04034b50, 0);
  header.writeUInt16LE(20, 4);
  header.writeUInt16LE(0x0800, 6); // UTF-8 文件名
  header.writeUInt16LE(8, 8); // deflate
  header.writeUInt16LE(DOS_TIME, 10);
  header.writeUInt16LE(DOS_DATE, 12);
  header.writeUInt16LE(nameBuf.length, 26);
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
      handle.write(chunk, 0, chunk.length, position).then(() => cb(), (error) => cb(error));
    },
  });
  await pipe(fs.createReadStream(entry.abs, { highWaterMark: 1 << 20 }), counter, zlib.createDeflateRaw({ level }), sink);

  const patch = Buffer.alloc(12);
  patch.writeUInt32LE(crc, 0);
  patch.writeUInt32LE(csize, 4);
  patch.writeUInt32LE(usize, 8);
  await handle.write(patch, 0, 12, localOffset + 14);
  return { crc, csize, usize, localOffset, nameBuf, mode: entry.mode ?? 0o644 };
}

async function writeZip(outFile, entries, level) {
  const handle = await fsp.open(outFile, 'w');
  const state = { offset: 0 };
  const central = [];
  try {
    for (const entry of entries) central.push(await writeOneEntry(handle, state, entry, level));

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
      head.writeUInt32LE(((item.mode & 0xffff) << 16) >>> 0, 38);
      head.writeUInt32LE(item.localOffset, 42);
      await writeAt(handle, state, head);
      await writeAt(handle, state, item.nameBuf);
    }
    const centralSize = state.offset - centralOffset;
    if (central.length > 0xffff) throw new Error('条目数超过 65535，需要 ZIP64（未实现）');
    if (state.offset > 0xffffffff) throw new Error('归档超过 4 GiB，需要 ZIP64（未实现）');
    const end = Buffer.alloc(22);
    end.writeUInt32LE(0x06054b50, 0);
    end.writeUInt16LE(central.length, 8);
    end.writeUInt16LE(central.length, 10);
    end.writeUInt32LE(centralSize, 12);
    end.writeUInt32LE(centralOffset, 16);
    await writeAt(handle, state, end);
  } finally {
    await handle.close();
  }
  return state.offset;
}

function listFiles(dir, prefix = '') {
  const out = [];
  for (const name of fs.readdirSync(dir).sort()) {
    const abs = path.join(dir, name);
    const rel = prefix === '' ? name : `${prefix}/${name}`;
    const stat = fs.statSync(abs);
    if (stat.isDirectory()) out.push(...listFiles(abs, rel));
    else out.push({ name: rel, abs, bytes: stat.size });
  }
  return out;
}

// ---------------------------------------------------------------------------
// 单个平台
// ---------------------------------------------------------------------------

async function buildTarget(target, args, version) {
  const spec = TARGETS[target];
  const staged = path.join(here, 'build', `staging-${target}`);
  await fsp.rm(staged, { recursive: true, force: true });
  await fsp.mkdir(path.join(staged, 'bin'), { recursive: true });
  await fsp.mkdir(path.join(staged, 'models'), { recursive: true });
  await fsp.mkdir(path.join(staged, 'lib'), { recursive: true });

  // ① 可执行文件
  let bin = args.bin;
  if (bin === null) {
    if (args.skipBuild) throw new Error('--skip-build 时必须给 --bin');
    if (target !== hostTarget()) {
      throw new Error(`本机是 ${hostTarget()}，不能直接构建 ${target}；请用 --bin 给交叉编译产物`);
    }
    bin = cargoBuild();
  }
  await fsp.copyFile(bin, path.join(staged, 'bin', spec.exe));
  await fsp.chmod(path.join(staged, 'bin', spec.exe), 0o755);

  // ② 模型（平台无关，两个归档各一份）
  if (args.models === null) throw new Error('必须给 --models（模型目录）');
  for (const name of MODEL_FILES) {
    const from = path.join(path.resolve(args.models), name);
    if (!fs.existsSync(from)) throw new Error(`模型目录里缺 ${name}：${from}`);
    await fsp.copyFile(from, path.join(staged, 'models', name));
  }

  // ③ ONNX Runtime
  if (args.ort === null) throw new Error('必须给 --ort（ONNX Runtime 共享库）');
  await fsp.copyFile(path.resolve(args.ort), path.join(staged, 'lib', spec.ort));

  // ④ 自描述 + 许可
  writeManifest(staged, target, version);
  await fsp.copyFile(path.join(root, 'LICENSE'), path.join(staged, 'LICENSE'));

  // ⑤ 打包
  await fsp.mkdir(args.out, { recursive: true });
  const outFile = path.join(args.out, spec.asset);
  const files = listFiles(staged);
  const total = files.reduce((sum, file) => sum + file.bytes, 0);
  console.log(`[zip] ${spec.asset}：${files.length} 个文件，未压缩 ${mb(total)}`);
  const bytes = await writeZip(outFile, files, args.level);
  const sha256 = crypto.createHash('sha256').update(await fsp.readFile(outFile)).digest('hex');

  const entry = {
    asset: spec.asset,
    sha256,
    bytes,
    installedBytes: total,
  };
  const record = {
    id: EXTENSION_ID,
    name: ENGINE_NAME,
    summary: ENGINE_SUMMARY,
    version,
    kind: 'ocr-engine',
    provides: ENGINE_ID,
    platforms: [spec.platform],
    arch: [spec.arch],
    urls: [`https://github.com/${REPO}/releases/download/v${version}/${spec.asset}`],
    bytes,
    sha256,
    installedBytes: total,
    license: LICENSE,
    homepage: HOMEPAGE,
    requires: [],
    notes: `离线可用，不需要用户装 Python 或 ONNX Runtime。解包约 ${Math.round(total / 1024 / 1024)} MiB`,
  };
  const entryFile = path.join(args.out, `catalog-entry-${target}.json`);
  await fsp.writeFile(entryFile, `${JSON.stringify(record, null, 2)}\n`);
  console.log(`[ok ] ${outFile}`);
  console.log(`      sha256 ${sha256}`);
  console.log(`      bytes ${bytes}（未压缩 ${total}）`);
  console.log(`      条目 ${entryFile}`);

  if (!args.keepStaging) await fsp.rm(staged, { recursive: true, force: true });
  return entry;
}

function mb(bytes) {
  return `${(bytes / 1024 / 1024).toFixed(1)} MiB`;
}

function hostTarget() {
  if (process.platform === 'darwin') return process.arch === 'arm64' ? 'darwin-arm64' : 'darwin-x64';
  if (process.platform === 'win32') return 'win32-x64';
  return 'linux-x64';
}

/**
 * 引擎库根的 `catalog.json`：**一个能力一条**，平台差异在 `assets` 里。
 * 重复 id 会让整份清单作废（应用的 `parseCatalog` 拒绝），所以这里是合并而不是追加。
 */
async function writeCatalog(version, built) {
  const catalogPath = path.join(root, 'catalog.json');
  let catalog = { schemaVersion: 1, extensions: [] };
  if (fs.existsSync(catalogPath)) {
    try {
      catalog = JSON.parse(fs.readFileSync(catalogPath, 'utf8'));
    } catch {
      console.warn('[warn] 现有 catalog.json 读不动，重建');
    }
  }
  const assets = {};
  for (const { target, entry } of built) assets[`${TARGETS[target].platform}-${TARGETS[target].arch}`] = entry;
  const existing = (catalog.extensions ?? []).find((item) => item.id === EXTENSION_ID);
  const merged = {
    id: EXTENSION_ID,
    name: ENGINE_NAME,
    summary: ENGINE_SUMMARY,
    version,
    kind: 'ocr-engine',
    provides: ENGINE_ID,
    release: { repo: REPO, tag: `v${version}`, assets: { ...(existing?.release?.assets ?? {}), ...assets } },
    license: LICENSE,
    homepage: HOMEPAGE,
    requires: [],
  };
  const others = (catalog.extensions ?? []).filter((item) => item.id !== EXTENSION_ID);
  const next = { schemaVersion: 1, generatedAt: new Date().toISOString(), extensions: [...others, merged] };
  await fsp.writeFile(catalogPath, `${JSON.stringify(next, null, 2)}\n`);
  console.log(`[ok ] ${catalogPath}`);
}

async function main() {
  const args = parseArgs(process.argv.slice(2));
  if (args.help || args.target === null) {
    console.log(USAGE);
    if (args.help) return;
    process.exitCode = 2;
    return;
  }
  const version = args.version ?? versionFromCargo();
  const targets = args.target === 'all' ? ['darwin-arm64', 'win32-x64'] : args.target.split(',');
  for (const target of targets) {
    if (TARGETS[target] === undefined) throw new Error(`未知 --target：${target}`);
  }
  const built = [];
  for (const target of targets) built.push({ target, entry: await buildTarget(target, args, version) });
  if (targets.length > 0) await writeCatalog(version, built);
}

main().catch((error) => {
  console.error(`[fail] ${error.message}`);
  process.exitCode = 1;
});
