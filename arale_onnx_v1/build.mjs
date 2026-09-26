#!/usr/bin/env node
/**
 * build.mjs —— 把包内 Python + ONNX Runtime 引擎打成可下载的扩展归档。
 *
 *     node build.mjs --target darwin-arm64
 *     node build.mjs --target win32-x64
 *
 * 产物：
 *
 *     dist/arale_onnx_v1-macos-arm64.zip
 *     dist/arale_onnx_v1-windows-x64.zip
 *     dist/catalog-entry-<target>.json      # 单个平台条目（含 sha256/bytes）
 *     ../repositories/default.jsonl         # 仓库：一行一个引擎，平台差异在 assets 里
 *
 * 归档根的布局（应用解到 `<userData>/extensions/ocr-arale_onnx_v1/` 后直接跑）：
 *
 *     extension.json                 # 自描述 + runner
 *     python/bin/python3 | python/python.exe  # 自带解释器（应用只给它 chmod）
 *     ocr/ocr_run.py                # ONNX 推理 + Mokuro 几何适配器
 *     engine/                       # Python 依赖，无 PyTorch
 *     models/{detector,manga-ocr-encoder,manga-ocr-decoder-init,manga-ocr-decoder-step}.onnx + vocab.txt
 *     LICENSE
 *
 * ONNX Runtime 的 Python wheel 已包含在 engine/ 内；构建时不联网。
 *
 * ## 为什么只有一个可执行文件
 *
 * 应用解压走 `native/arale-native` 的 zip 解包，**不还原 unix 权限位**；唯一补权限的
 * 地方是 `src/main/extensions/service.ts` 的 `chmodExecutable()`，而它只给
 * `extension.json → runner.program` 那一个路径 chmod 0755。所以 runner 直接指向
 * `python/bin/python3`，不能指向需要再启动其它未补权限文件的 shell 脚本。
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
import { Transform, Writable, pipeline } from 'node:stream';
import { fileURLToPath } from 'node:url';
import { promisify } from 'node:util';

const pipe = promisify(pipeline);
const here = path.dirname(fileURLToPath(import.meta.url));
const root = path.resolve(here, '..');

const ENGINE_ID = 'arale_onnx_v1';
const EXTENSION_ID = `ocr-${ENGINE_ID}`;
const ENGINE_NAME = 'arale_onnx_v1（ONNX 版 OCR）';
const ENGINE_SUMMARY =
  'Mokuro 几何流程 + ONNX Runtime 推理，自带 Python 解释器、依赖和模型，不需要用户安装 Python 或 PyTorch。';
const REPO = 'heyanLE/arale-book-ocr-manga';
const LICENSE = 'GPL-3.0';
const ENGINE_VERSION = '0.2.0';
const HOMEPAGE = `https://github.com/${REPO}`;

/** 检测器、编码器、缓存解码器首步/续步图 + 词表；旧无缓存图仅用于开发对照。 */
const MODEL_FILES = [
  'detector.onnx',
  'manga-ocr-encoder.onnx',
  'manga-ocr-decoder-init.onnx',
  'manga-ocr-decoder-step.onnx',
  'vocab.txt',
];
const MODEL_HASHES = JSON.parse(fs.readFileSync(path.join(here, 'model-manifest.json'), 'utf8')).files;

async function sha256File(file) {
  const hash = crypto.createHash('sha256');
  for await (const chunk of fs.createReadStream(file)) hash.update(chunk);
  return hash.digest('hex');
}

const TARGETS = {
  'darwin-arm64': { platform: 'darwin', arch: 'arm64', asset: 'arale_onnx_v1-macos-arm64.zip', python: 'python/bin/python3' },
  'darwin-x64': { platform: 'darwin', arch: 'x64', asset: 'arale_onnx_v1-macos-x64.zip', python: 'python/bin/python3' },
  'win32-x64': { platform: 'win32', arch: 'x64', asset: 'arale_onnx_v1-windows-x64.zip', python: 'python/python.exe' },
};

function parseArgs(argv) {
  const args = {
    target: null,
    models: path.join(here, 'models'),
    runtime: null,
    version: null,
    out: path.join(here, 'dist'),
    level: 6,
    keepStaging: false,
    debug: false,
    indexCrossBuild: false,
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
    else if (flag === '--runtime') args.runtime = next();
    else if (flag === '--version') args.version = next();
    else if (flag === '--out') args.out = path.resolve(next());
    else if (flag === '--level') args.level = Number(next());
    else if (flag === '--keep-staging') args.keepStaging = true;
    else if (flag === '--debug') args.debug = true;
    else if (flag === '--index-cross-build') args.indexCrossBuild = true;
    else if (flag === '--help' || flag === '-h') args.help = true;
    else throw new Error(`未知参数：${flag}`);
  }
  return args;
}

const USAGE = `用法：
  node build.mjs --target <darwin-arm64|darwin-x64|win32-x64|all> [选项]

选项：
  --models <dir>    模型目录（缺省 <引擎目录>/models，被 gitignore 排除）
  --runtime <dir>  包内 Python + 依赖目录（缺省 runtime/<target>/，被 gitignore 排除）
  --version <v>   扩展版本（缺省 ${ENGINE_VERSION}）
  --out <dir>       产物目录（缺省 <引擎目录>/dist）
  --level <0-9>     deflate 级别（缺省 6）
  --keep-staging    保留中间目录（排障）
  --debug           只生成 build/dev-<target>/，供开发版直接加载，不制作发布归档
  --index-cross-build  允许把非本机平台归档写进 JSONL（仅在目标机器验收后使用）`;

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
    minMacOS: 14,
    license: LICENSE,
    homepage: HOMEPAGE,
    description: ENGINE_SUMMARY,
    runner: {
      program: spec.python,
      args: ['-s', '-u', 'ocr/ocr_run.py', '--pages-file', '{pagesFile}'],
      env: { PYTHONPATH: 'engine' },
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
  const modelDir = path.resolve(args.models);
  const runtimeDir = path.resolve(args.runtime ?? path.join(here, 'runtime', target));
  for (const name of MODEL_FILES) {
    if (!fs.existsSync(path.join(modelDir, name))) throw new Error(`模型目录里缺 ${name}：${modelDir}`);
    const actual = await sha256File(path.join(modelDir, name));
    if (actual !== MODEL_HASHES[name]) throw new Error(`${name} sha256 不匹配 model-manifest.json：${actual}`);
  }
  if (!fs.existsSync(path.join(runtimeDir, spec.python))) {
    throw new Error(`缺少目标平台包内 Python：${path.join(runtimeDir, spec.python)}`);
  }
  if (!fs.existsSync(path.join(runtimeDir, 'engine', 'onnxruntime'))) {
    throw new Error(`缺少 ONNX Runtime Python 包：${path.join(runtimeDir, 'engine', 'onnxruntime')}`);
  }
  if (fs.existsSync(path.join(runtimeDir, 'engine', 'torch'))) {
    throw new Error('runtime 中仍含 PyTorch；拒绝打包');
  }
  const staged = path.join(here, 'build', `${args.debug ? 'dev' : 'staging'}-${target}`);
  await fsp.rm(staged, { recursive: true, force: true });
  await fsp.cp(path.join(runtimeDir, 'python'), path.join(staged, 'python'), { recursive: true });
  await fsp.cp(path.join(runtimeDir, 'engine'), path.join(staged, 'engine'), { recursive: true });
  await fsp.cp(path.join(here, 'python'), path.join(staged, 'ocr'), { recursive: true });
  await fsp.mkdir(path.join(staged, 'models'), { recursive: true });
  if (target.startsWith('darwin')) await fsp.chmod(path.join(staged, spec.python), 0o755);

  // ② 模型（平台无关，两个归档各一份）
  for (const name of MODEL_FILES) {
    const from = path.join(modelDir, name);
    await fsp.copyFile(from, path.join(staged, 'models', name));
  }

  // 自描述 + 许可
  writeManifest(staged, target, version);
  await fsp.copyFile(path.join(root, 'LICENSE'), path.join(staged, 'LICENSE'));
  await fsp.copyFile(path.join(here, 'THIRD_PARTY.md'), path.join(staged, 'THIRD_PARTY.md'));

  // 开发包保留完整目录，应用直接按 extension.json 加载，不依赖用户安装记录。
  if (args.debug) {
    console.log(`[debug] ${staged}`);
    return null;
  }

  // ⑤ 打包
  await fsp.mkdir(args.out, { recursive: true });
  const outFile = path.join(args.out, spec.asset);
  const files = listFiles(staged);
  const total = files.reduce((sum, file) => sum + file.bytes, 0);
  console.log(`[zip] ${spec.asset}：${files.length} 个文件，未压缩 ${mb(total)}`);
  const bytes = await writeZip(outFile, files, args.level);
  const sha256 = await sha256File(outFile);

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
    minMacOS: 14,
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

/** 一个仓库一个 JSONL 文件；只更新本引擎那一行，保留其它引擎。 */
async function writeRepository(version, built) {
  const repoPath = path.join(root, 'repositories', 'default.jsonl');
  let entries = [];
  if (fs.existsSync(repoPath)) {
    entries = fs.readFileSync(repoPath, 'utf8').split(/\r?\n/).filter(Boolean).map((line) => JSON.parse(line));
  }
  const assets = {};
  for (const { target, entry } of built) assets[`${TARGETS[target].platform}-${TARGETS[target].arch}`] = entry;
  const existing = entries.find((item) => item.id === EXTENSION_ID);
  const merged = {
    id: EXTENSION_ID,
    name: ENGINE_NAME,
    summary: ENGINE_SUMMARY,
    version,
    kind: 'ocr-engine',
    provides: ENGINE_ID,
    minMacOS: 14,
    release: { repo: REPO, tag: `v${version}`, assets: { ...(existing?.version === version ? existing.release?.assets ?? {} : {}), ...assets } },
    license: LICENSE,
    homepage: HOMEPAGE,
    requires: [],
  };
  entries = [...entries.filter((item) => item.id !== EXTENSION_ID), merged];
  await fsp.mkdir(path.dirname(repoPath), { recursive: true });
  await fsp.writeFile(repoPath, `${entries.map((item) => JSON.stringify(item)).join('\n')}\n`);
  console.log(`[ok ] ${repoPath}`);
}

function hostTarget() {
  return `${process.platform}-${process.arch}`;
}

async function main() {
  const args = parseArgs(process.argv.slice(2));
  if (args.help || args.target === null) {
    console.log(USAGE);
    if (args.help) return;
    process.exitCode = 2;
    return;
  }
  const version = args.version ?? ENGINE_VERSION;
  const targets = args.target === 'all' ? ['darwin-arm64', 'win32-x64'] : args.target.split(',');
  for (const target of targets) {
    if (TARGETS[target] === undefined) throw new Error(`未知 --target：${target}`);
  }
  const built = [];
  for (const target of targets) built.push({ target, entry: await buildTarget(target, args, version) });
  const indexed = built.map(({ target, entry }) => ({
    target,
    entry: args.indexCrossBuild || target === hostTarget() ? entry : { ...entry, sha256: '' },
  }));
  if (!args.debug && indexed.length > 0) await writeRepository(version, indexed);
  for (const { target } of built) {
    if (!args.debug && target !== hostTarget() && !args.indexCrossBuild) {
      console.log(`[warn] ${target} 是交叉打包，JSONL 中 sha256 留空，安装被拒绝；在目标平台验收后使用 --index-cross-build`);
    }
  }
}

main().catch((error) => {
  console.error(`[fail] ${error.message}`);
  process.exitCode = 1;
});
