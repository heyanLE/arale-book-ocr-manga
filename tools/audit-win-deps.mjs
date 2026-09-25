/**
 * Windows 归档的**依赖体检**：扫包里所有 PE 文件（.pyd / .dll / .exe）的导入表，
 * 指出「既不在包里、也不是已知系统 DLL」的依赖。
 *
 * ## 为什么需要它
 *
 * Windows 归档是**在 macOS 上交叉安装出来的**（`pip install --platform win_amd64`），
 * 本机没有 Windows 可以真跑。静态核对只会数「.pyd 有几个」——那证明不了**加载得起来**。
 * 而这类问题的真实症状是：用户机器上 `import torch` 直接失败，整个引擎变成「不可用」，
 * 报错还只有一句 DLL load failed。
 *
 * 实测（2026-09）它就是抓到了两条：
 * - `vcruntime140_threads.dll`：`torch_cpu.dll` 要它，而 embedding CPython 只带
 *   `vcruntime140.dll` / `vcruntime140_1.dll`，**包里没有**——没装新版 VC++ 运行库的
 *   机器上 torch 加载不起来；
 * - `mfplat.dll` / `mf.dll` / `mfreadwrite.dll`：`cv2.pyd` 要它们（Media Foundation），
 *   Windows **N / KN 版**（欧盟常见的无媒体功能版）默认没有，cv2 就加载不起来。
 *   我们只用图像处理，用不到视频——但这个依赖是 `cv2.pyd` 的静态导入表，躲不开。
 *
 * 用法：
 *   node tools/audit-win-deps.mjs build/win32-x64 [--strict]
 * `--strict` 把「条件性系统 DLL」也当成失败（默认只警告）。
 */

import fs from 'node:fs';
import path from 'node:path';

/** Windows 10+ 上一定有的系统 DLL（缺失才是问题）。 */
const BASE_OS = new Set(
  [
    'kernel32.dll', 'kernelbase.dll', 'ntdll.dll', 'user32.dll', 'gdi32.dll', 'advapi32.dll',
    'shell32.dll', 'shlwapi.dll', 'ole32.dll', 'oleaut32.dll', 'combase.dll', 'rpcrt4.dll',
    'ucrtbase.dll', 'msvcrt.dll', 'vcruntime140.dll', 'vcruntime140_1.dll',
    'ws2_32.dll', 'mswsock.dll', 'dnsapi.dll', 'iphlpapi.dll', 'secur32.dll', 'sechost.dll',
    'bcrypt.dll', 'bcryptprimitives.dll', 'ncrypt.dll', 'crypt32.dll', 'cryptbase.dll', 'wintrust.dll',
    'userenv.dll', 'version.dll', 'psapi.dll', 'dbghelp.dll', 'imagehlp.dll', 'rpcns4.dll',
    'comctl32.dll', 'comdlg32.dll', 'imm32.dll', 'winmm.dll', 'setupapi.dll', 'cfgmgr32.dll',
    'powrprof.dll', 'mpr.dll', 'netapi32.dll', 'wtsapi32.dll', 'avrt.dll', 'gdiplus.dll',
    'uxtheme.dll', 'dwmapi.dll', 'propsys.dll', 'hid.dll', 'pdh.dll', 'msi.dll', 'normaliz.dll',
    'd3d11.dll', 'dxgi.dll', 'opengl32.dll', 'usp10.dll', 'dwrite.dll', 'wldap32.dll', 'winhttp.dll',
    'wininet.dll', 'cabinet.dll', 'msvcp140.dll', 'msvcp140_1.dll', 'msvcp140_2.dll',
    'concrt140.dll', 'vcruntime140_threads.dll'.replace('vcruntime140_threads.dll', '__skip__'),
  ].filter((name) => name !== '__skip__'),
);

/**
 * 「装了才有」的系统 DLL：不属于基础系统，缺了会让对应组件加载失败。
 * 单列出来是因为它们**不是我们的 bug**，但必须让用户/安装器知道。
 */
const CONDITIONAL = new Map([
  ['vcruntime140_threads.dll', 'VC++ 2015–2022 运行库（VS2022 17.8+）。缺了 torch_cpu.dll 加载不起来'],
  ['msvcp140.dll', 'VC++ 运行库（多数包自带私有副本，见 *.libs/）'],
  ['msvcp140_atomic_wait.dll', 'VC++ 运行库（VS2019 16.8+）。缺了 torch_python.dll 加载不起来'],
  ['mfplat.dll', 'Media Foundation（Windows N/KN 版需装 Media Feature Pack）。cv2.pyd 需要'],
  ['mf.dll', 'Media Foundation（同上）'],
  ['mfreadwrite.dll', 'Media Foundation（同上）'],
  ['cudart64_12.dll', 'CUDA 运行库（CPU 版不该出现，出现说明装错了 torch）'],
]);

const SYSTEM_PREFIX = ['api-ms-win', 'ext-ms-win'];

// ---------------------------------------------------------------------------
// 最小 PE 导入表解析（不引依赖；只读我们需要的那几个目录）
// ---------------------------------------------------------------------------

function readImportDlls(file) {
  const buf = fs.readFileSync(file);
  if (buf.length < 0x40 || buf.readUInt16LE(0) !== 0x5a4d) return null; // 'MZ'
  const peOffset = buf.readUInt32LE(0x3c);
  if (buf.readUInt32LE(peOffset) !== 0x00004550) return null; // 'PE\0\0'
  const coff = peOffset + 4;
  const sectionCount = buf.readUInt16LE(coff + 2);
  const optionalSize = buf.readUInt16LE(coff + 16);
  const optional = coff + 20;
  const magic = buf.readUInt16LE(optional);
  const isPe32Plus = magic === 0x20b;
  // 数据目录从 optional header 偏移 112（PE32+）或 96（PE32）开始，第 1 项是导入表。
  const dataDir = optional + (isPe32Plus ? 112 : 96);
  const importRva = buf.readUInt32LE(dataDir + 8);
  if (importRva === 0) return [];

  const sections = [];
  const sectionStart = optional + optionalSize;
  for (let i = 0; i < sectionCount; i += 1) {
    const base = sectionStart + i * 40;
    sections.push({
      virtualSize: buf.readUInt32LE(base + 8),
      virtualAddress: buf.readUInt32LE(base + 12),
      rawSize: buf.readUInt32LE(base + 16),
      rawOffset: buf.readUInt32LE(base + 20),
    });
  }
  const toOffset = (rva) => {
    for (const section of sections) {
      const span = Math.max(section.virtualSize, section.rawSize);
      if (rva >= section.virtualAddress && rva < section.virtualAddress + span) {
        return section.rawOffset + (rva - section.virtualAddress);
      }
    }
    return null;
  };

  const names = [];
  let cursor = toOffset(importRva);
  if (cursor === null) return [];
  // IMAGE_IMPORT_DESCRIPTOR 20 字节一项，全 0 结束。
  for (let guard = 0; guard < 512; guard += 1) {
    const descriptor = cursor + guard * 20;
    if (descriptor + 20 > buf.length) break;
    const nameRva = buf.readUInt32LE(descriptor + 12);
    const firstThunk = buf.readUInt32LE(descriptor + 16);
    if (nameRva === 0 && firstThunk === 0) break;
    if (nameRva === 0) continue;
    const nameOffset = toOffset(nameRva);
    if (nameOffset === null) continue;
    let end = nameOffset;
    while (end < buf.length && buf[end] !== 0) end += 1;
    names.push(buf.subarray(nameOffset, end).toString('ascii').toLowerCase());
  }
  return names;
}

// ---------------------------------------------------------------------------

function walk(dir, out = []) {
  for (const entry of fs.readdirSync(dir, { withFileTypes: true })) {
    const abs = path.join(dir, entry.name);
    if (entry.isDirectory()) walk(abs, out);
    else if (/\.(pyd|dll|exe)$/i.test(entry.name)) out.push(abs);
  }
  return out;
}

const root = process.argv[2];
const strict = process.argv.includes('--strict');
if (root === undefined || !fs.existsSync(root)) {
  console.error('用法：node tools/audit-win-deps.mjs <win32-x64 目录> [--strict]');
  process.exit(2);
}

const binaries = walk(root);
const provided = new Set(
  binaries.filter((file) => file.toLowerCase().endsWith('.dll')).map((file) => path.basename(file).toLowerCase()),
);
const missing = new Map(); // name → Set(users)
let scanned = 0;
for (const file of binaries) {
  const names = readImportDlls(file);
  if (names === null) continue;
  scanned += 1;
  for (const name of names) {
    if (provided.has(name) || SYSTEM_PREFIX.some((prefix) => name.startsWith(prefix))) continue;
    const conditional = CONDITIONAL.has(name);
    if (!conditional && BASE_OS.has(name)) continue;
    if (missing.has(name)) missing.get(name).add(file);
    else missing.set(name, new Set([file]));
  }
}

const hard = [];
const conditionals = [];
for (const [name, users] of missing) {
  const list = [...users].map((file) => path.relative(root, file));
  (CONDITIONAL.has(name) ? conditionals : hard).push({ name, list, why: CONDITIONAL.get(name) ?? '' });
}
hard.sort((a, b) => b.list.length - a.list.length);

console.log(`[audit] 扫描 ${scanned} 个 PE 文件；包内自带 DLL ${provided.size} 个`);
if (hard.length === 0) {
  console.log('[audit] ✔ 没有「既不在包里、也不是系统 DLL」的依赖');
} else {
  console.log(`[audit] ✘ ${hard.length} 种依赖既不在包里、也不在已知系统 DLL 里：`);
  for (const item of hard) {
    console.log(`         ${item.name}  ← ${item.list.length} 个文件，例如 ${item.list[0]}`);
  }
}
if (conditionals.length > 0) {
  console.log(`[audit] ⚠ ${conditionals.length} 种「装了才有」的依赖（不是 bug，但要知道）：`);
  for (const item of conditionals) {
    console.log(`         ${item.name}  ← ${item.list.length} 个文件（${item.why}）`);
  }
}
process.exit(hard.length > 0 || (strict && conditionals.length > 0) ? 1 : 0);
