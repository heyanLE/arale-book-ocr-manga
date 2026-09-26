#!/usr/bin/env python3
"""Build-time export of the Mokuro detector and manga-ocr models to fp32 ONNX.

Requires PyTorch, transformers, comic-text-detector and their build dependencies.
None of those packages is needed by the distributed runner.
"""

import argparse
import hashlib
import shutil
from pathlib import Path

import cv2
import torch
from transformers import GenerationMixin, VisionEncoderDecoderModel
from comic_text_detector.basemodel import TextDetBase
from comic_text_detector.inference import preprocess_img


class MangaOcrModel(VisionEncoderDecoderModel, GenerationMixin):
    pass


class Encoder(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, pixel_values):
        return self.model(pixel_values).last_hidden_state


class Decoder(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, encoder_hidden_states, input_ids):
        return self.model(input_ids=input_ids, encoder_hidden_states=encoder_hidden_states, use_cache=False).logits


class Detector(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, image):
        return self.model(image)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--recognizer-model', required=True, help='Local manga-ocr-base directory')
    parser.add_argument('--detector-model', required=True, help='Local comictextdetector.pt')
    parser.add_argument('--detector-example', required=True, help='Representative manga page for tracing')
    parser.add_argument('--output', default=str(Path(__file__).resolve().parents[1] / 'models'))
    args = parser.parse_args()
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)

    recognizer = MangaOcrModel.from_pretrained(args.recognizer_model).eval()
    pixels = torch.randn(1, 3, 224, 224)
    encoder = Encoder(recognizer.encoder).eval()
    torch.onnx.export(encoder, (pixels,), str(output / 'manga-ocr-encoder.onnx'),
                      input_names=['pixel_values'], output_names=['last_hidden_state'],
                      dynamic_axes={'pixel_values': {0: 'batch'}, 'last_hidden_state': {0: 'batch'}},
                      opset_version=17, dynamo=False)
    hidden = encoder(pixels)
    torch.onnx.export(Decoder(recognizer.decoder).eval(), (hidden, torch.tensor([[2, 100, 200]])),
                      str(output / 'manga-ocr-decoder.onnx'),
                      input_names=['encoder_hidden_states', 'input_ids'], output_names=['logits'],
                      dynamic_axes={'encoder_hidden_states': {0: 'batch'},
                                    'input_ids': {0: 'batch', 1: 'seq'},
                                    'logits': {0: 'batch', 1: 'seq'}},
                      opset_version=17, dynamo=False)
    shutil.copyfile(Path(args.recognizer_model) / 'vocab.txt', output / 'vocab.txt')

    detector = TextDetBase(args.detector_model, device='cpu', act='leaky').eval()
    image = cv2.imread(args.detector_example)
    if image is None:
        raise ValueError(f'Cannot read {args.detector_example}')
    tensor, _, _, _ = preprocess_img(image, input_size=(1024, 1024), device='cpu', bgr2rgb=True, to_tensor=True)
    torch.onnx.export(Detector(detector).eval(), (tensor,), str(output / 'detector.onnx'),
                      input_names=['image'], output_names=['blks', 'mask', 'lines_map'],
                      opset_version=17, dynamo=False)
    for file in sorted(output.iterdir()):
        if file.is_file():
            with file.open('rb') as handle:
                digest = hashlib.file_digest(handle, 'sha256').hexdigest()
            print(file.name, file.stat().st_size, digest)
    print('Validate the models against Mokuro, then update model-manifest.json before packaging.')


if __name__ == '__main__':
    main()
