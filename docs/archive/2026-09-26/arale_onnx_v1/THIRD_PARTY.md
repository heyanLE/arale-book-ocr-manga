# Third-party components

- Mokuro 0.2.5 geometry and OCR flow: GPL-3.0, <https://github.com/kha-white/mokuro>.
- comic-text-detector utility modules in `python/mokuro_compat/`: GPL-3.0, <https://github.com/kha-white/comic-text-detector>. The copied files were adjusted to remove PyTorch-only branches and use local imports.
- Manga OCR code and `kha-white/manga-ocr-base` weights: Apache-2.0, <https://github.com/kha-white/manga-ocr> and <https://huggingface.co/kha-white/manga-ocr-base>.
- `comictextdetector.pt` source: <https://github.com/zyddnys/manga-image-translator/releases/tag/beta-0.2.1>; it is converted to ONNX at build time.
- ONNX Runtime: MIT, <https://github.com/microsoft/onnxruntime>. Its package includes its own `LICENSE` and `ThirdPartyNotices.txt`.
- Embedded CPython and other wheels retain their license metadata in the runtime. Review the platform-specific runtime contents before publishing.
