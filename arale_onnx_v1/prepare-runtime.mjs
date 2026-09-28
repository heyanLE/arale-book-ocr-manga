#!/usr/bin/env node
/** Assemble a self-contained Python/ORT runtime from platform-matched inputs.
 * No network, no PyTorch in the output. All output stays under ignored runtime/.
 */
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const here = path.dirname(fileURLToPath(import.meta.url));
const argv = process.argv.slice(2);
const arg = (flag) => {
  const index = argv.indexOf(flag);
  if (index < 0 || !argv[index + 1]) throw new Error(`缺少 ${flag}`);
  return argv[index + 1];
};
const optionalArg = (flag) => argv.includes(flag) ? path.resolve(arg(flag)) : null;
const name = arg('--target');
if (!['darwin-arm64', 'darwin-x64', 'win32-x64'].includes(name)) throw new Error(`不支持的目标：${name}`);
const pythonDir = path.resolve(arg('--python-dir'));
const sitePackages = path.resolve(arg('--site-packages'));
const ortPackages = path.resolve(arg('--ort-packages'));
const opencvPackages = optionalArg('--opencv-packages');
const output = path.join(here, 'runtime', name);
const skip = /^(?:torch|torchgen|torchsummary|torchvision|functorch|transformers|tokenizers|unidic_lite|fugashi|comic_text_detector|manga_ocr|mokuro|onnxruntime)(?:$|[-_.])/i;

for (const [label, directory] of [['Python', pythonDir], ['依赖', sitePackages], ['ONNX Runtime', ortPackages]]) {
  if (!fs.statSync(directory, { throwIfNoEntry: false })?.isDirectory()) throw new Error(`${label} 目录不存在：${directory}`);
}
fs.rmSync(output, { recursive: true, force: true });
fs.mkdirSync(output, { recursive: true });
fs.cpSync(pythonDir, path.join(output, 'python'), { recursive: true });
if (name === 'win32-x64') {
  const pythonOutput = path.join(output, 'python');
  const pthFiles = fs.readdirSync(pythonOutput).filter((entry) => /^python\d+\._pth$/i.test(entry));
  if (pthFiles.length !== 1) {
    throw new Error(`Windows 嵌入式 Python 应有且仅有一个 _pth 文件，实际找到 ${pthFiles.length} 个`);
  }
  const pthFile = path.join(pythonOutput, pthFiles[0]);
  const lines = fs.readFileSync(pthFile, 'utf8').split(/\r?\n/);
  if (!lines.some((line) => line.trim() === '..\\ocr')) {
    let insertAt = lines.findIndex((line) => line.trim() === 'import site');
    if (insertAt < 0) {
      insertAt = lines.length;
      while (insertAt > 0 && lines[insertAt - 1] === '') insertAt -= 1;
    }
    lines.splice(insertAt, 0, '..\\ocr');
    fs.writeFileSync(pthFile, lines.join('\r\n'));
  }
}
fs.mkdirSync(path.join(output, 'engine'), { recursive: true });
for (const entry of fs.readdirSync(sitePackages, { withFileTypes: true })) {
  if (skip.test(entry.name)) continue;
  fs.cpSync(path.join(sitePackages, entry.name), path.join(output, 'engine', entry.name), {
    recursive: true,
    filter: (source) => !source.split(path.sep).some((part) => part === '__pycache__' || part.endsWith('.pyc')),
  });
}
for (const entry of fs.readdirSync(ortPackages)) {
  if (!entry.startsWith('onnxruntime')) continue;
  fs.cpSync(path.join(ortPackages, entry), path.join(output, 'engine', entry), { recursive: true });
}
if (opencvPackages) {
  if (!fs.existsSync(path.join(opencvPackages, 'cv2'))) throw new Error('headless OpenCV wheel 缺 cv2/');
  const engine = path.join(output, 'engine');
  fs.rmSync(path.join(engine, 'cv2'), { recursive: true, force: true });
  for (const name of fs.readdirSync(engine)) {
    if (name.startsWith('opencv_python') && name.endsWith('.dist-info')) {
      fs.rmSync(path.join(engine, name), { recursive: true, force: true });
    }
  }
  for (const name of fs.readdirSync(opencvPackages)) {
    if (name === 'cv2' || (name.startsWith('opencv_python_headless') && name.endsWith('.dist-info'))) {
      fs.cpSync(path.join(opencvPackages, name), path.join(engine, name), { recursive: true });
    }
  }
}
if (!fs.existsSync(path.join(output, 'engine', 'onnxruntime'))) throw new Error('ONNX Runtime Python 包不在 --ort-packages 中');
console.log(`准备完成：${output}`);
