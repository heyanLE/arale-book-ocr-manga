#!/usr/bin/env python3
"""Mokuro-compatible page OCR using ONNX Runtime, without importing PyTorch.

The four small compat modules retain mokuro/comic-text-detector's OpenCV
geometry. The neural nets are the original weights exported to ONNX offline.
stdout is reserved for the application's page NDJSON protocol.
"""

import argparse
import json
import os
import re
import sys
from pathlib import Path

import cv2
import jaconv
import numpy as np
import onnxruntime as ort
from PIL import Image
from scipy.signal.windows import gaussian

from mokuro_compat.db_utils import SegDetectorRepresenter
from mokuro_compat.imgproc_utils import letterbox
from mokuro_compat.textblock import group_output
from mokuro_compat.textmask import refine_mask, refine_undetected_mask


ROOT = Path(__file__).resolve().parent.parent
MODELS = ROOT / "models"
ENGINE = "arale_onnx_v1"


def open_session(name):
    options = ort.SessionOptions()
    options.intra_op_num_threads = max(1, int(os.environ.get("ARALE_OCR_THREADS", "4")))
    return ort.InferenceSession(str(MODELS / name), sess_options=options, providers=["CPUExecutionProvider"])


def emit(value):
    print(json.dumps(value, ensure_ascii=False), flush=True)


def top_k(scores, k):
    # HF 5.x tie-break: higher flattened index first.
    indices = np.arange(len(scores))
    order = np.lexsort((-indices, -scores))[:k]
    return order.tolist()


def banned_ngrams(tokens, n=3):
    if len(tokens) < n:
        return []
    prefix = tokens[-(n - 1):]
    return [tokens[i + n - 1] for i in range(len(tokens) - n + 1)
            if tokens[i:i + n - 1] == prefix]


class Recognizer:
    def __init__(self):
        self.encoder = open_session("manga-ocr-encoder.onnx")
        self.use_cache = (os.environ.get("ARALE_OCR_KV_CACHE", "1") != "0"
                          and (MODELS / "manga-ocr-decoder-init.onnx").is_file()
                          and (MODELS / "manga-ocr-decoder-step.onnx").is_file())
        if self.use_cache:
            self.decoder_init = open_session("manga-ocr-decoder-init.onnx")
            self.decoder_step = open_session("manga-ocr-decoder-step.onnx")
        else:
            if not (MODELS / "manga-ocr-decoder.onnx").is_file():
                raise RuntimeError("无缓存解码图仅供开发对照；请取消 ARALE_OCR_KV_CACHE=0")
            self.decoder = open_session("manga-ocr-decoder.onnx")
        self.vocab = (MODELS / "vocab.txt").read_text(encoding="utf-8").splitlines()
        self.cls = self.vocab.index("[CLS]")
        self.sep = self.vocab.index("[SEP]")
        self.pad = self.vocab.index("[PAD]")

    def __call__(self, crop):
        # MangaOcr.__call__: PIL treats the OpenCV byte array as RGB, then L→RGB.
        img = Image.fromarray(crop).convert("L").convert("RGB")
        img = img.resize((224, 224), Image.Resampling.BILINEAR)
        pixels = (np.asarray(img, dtype=np.float32) / 255.0 - 0.5) / 0.5
        pixels = np.transpose(pixels, (2, 0, 1))[None].copy()
        hidden = self.encoder.run(None, {"pixel_values": pixels})[0]
        ids = self._beam_search(hidden)
        text = "".join(self.vocab[i] for i in ids if 0 <= i < len(self.vocab) and not self.vocab[i].startswith("["))
        text = "".join(text.split()).replace("…", "...")
        text = re.sub(r"[・.]{2,}", lambda m: "." * len(m.group()), text)
        return jaconv.h2z(text, ascii=True, digit=True)

    def _beam_search(self, hidden):
        # Port of transformers 5.x beam search used by manga-ocr 0.1.16.
        beams, max_len, neg = 4, 300, np.float32(-1e9)
        running = [[self.cls] for _ in range(beams)]
        running_scores = np.array([0, neg, neg, neg], dtype=np.float32)
        finished = [[self.cls] for _ in range(beams)]
        finished_scores = np.full(beams, neg, dtype=np.float32)
        finished_flags = [False] * beams
        unsat = True
        self_cache = None
        cross_cache = None
        hidden_batch = np.repeat(hidden, beams, axis=0) if self.use_cache else None
        while len(running[0]) + 1 < max_len:
            length = len(running[0])
            if self.use_cache:
                if self_cache is None:
                    outputs = self.decoder_init.run(None, {
                        "encoder_hidden_states": hidden_batch,
                        "input_ids": np.full((beams, 1), self.cls, dtype=np.int64),
                    })
                    cross_cache = outputs[5:9]
                else:
                    inputs = np.array([[seq[-1]] for seq in running], dtype=np.int64)
                    names = ("past_self_key_0", "past_self_value_0", "past_self_key_1", "past_self_value_1",
                             "past_cross_key_0", "past_cross_value_0", "past_cross_key_1", "past_cross_value_1")
                    feed = {"input_ids": inputs, **dict(zip(names, self_cache + cross_cache))}
                    outputs = self.decoder_step.run(None, feed)
                logits = outputs[0][:, -1, :]
                present_self = outputs[1:5]
            else:
                batch_hidden = np.repeat(hidden, beams, axis=0)
                inputs = np.array(running, dtype=np.int64)
                logits = self.decoder.run(None, {
                    "encoder_hidden_states": batch_hidden,
                    "input_ids": inputs,
                })[0][:, -1, :]
            vocab_size = logits.shape[1]
            maxima = logits.max(axis=1, keepdims=True)
            log_probs = logits - maxima - np.log(np.exp(logits - maxima).sum(axis=1, keepdims=True))
            for i, seq in enumerate(running):
                log_probs[i, banned_ngrams(seq)] = -np.inf
            scores = (log_probs + running_scores[:, None]).reshape(-1)
            chosen = top_k(scores, 2 * beams)
            top_scores = scores[chosen]
            parents = [int(i // vocab_size) for i in chosen]
            tokens = [int(i % vocab_size) for i in chosen]
            candidates = [running[p] + [token] for p, token in zip(parents, tokens)]
            hits = [token == self.sep for token in tokens]
            masked = top_scores + np.array([neg if hit else 0 for hit in hits], dtype=np.float32)
            keep_running = top_k(masked, beams)
            running = [candidates[i] for i in keep_running]
            running_scores = masked[keep_running]
            if self.use_cache:
                # Each candidate inherits the past of its parent beam. Duplicate or
                # reorder both self- and cross-attention caches exactly like HF generate.
                selected_parents = np.array([parents[i] for i in keep_running], dtype=np.intp)
                self_cache = [np.ascontiguousarray(value[selected_parents]) for value in present_self]
                cross_cache = [np.ascontiguousarray(value[selected_parents]) for value in cross_cache]

            denom = np.float32(length ** 2)
            final_scores = top_scores / denom
            if all(finished_flags) or not unsat:
                final_scores = final_scores + neg
            final_scores = final_scores + np.array(
                [0 if hits[i] and i < beams else neg for i in range(2 * beams)], dtype=np.float32)
            merged_scores = np.concatenate([finished_scores, final_scores])
            merged_sequences = finished + candidates
            merged_flags = finished_flags + [hits[i] and i < beams for i in range(2 * beams)]
            keep_finished = top_k(merged_scores, beams)
            finished = [merged_sequences[i] for i in keep_finished]
            finished_scores = merged_scores[keep_finished]
            finished_flags = [merged_flags[i] for i in keep_finished]

            current_len = length + 1
            hypothetical = running_scores[0] / np.float32((current_len - 1) ** 2)
            worst = min(finished_scores)
            unsat = unsat and any(hypothetical > (worst if done else neg) for done in finished_flags)
            if not (unsat and not all(finished_flags) and not all(hits)):
                break
        return finished[0]


def split_crop(crop, mask, block, line_index, textheight=64, max_ratio=16, anchor_window=2):
    region = block.get_transformed_region(crop, line_index, textheight)
    ratio = region.shape[1] / region.shape[0]
    if ratio <= max_ratio:
        return [region]
    line_mask = block.get_transformed_region(mask, line_index, textheight)
    count = int(np.ceil(ratio / max_ratio))
    anchors = np.linspace(0, region.shape[1], count + 1)[1:-1]
    kernel = gaussian(textheight * 2, textheight / 8)
    density = np.convolve(line_mask.sum(axis=0), kernel, "same")
    if density.max() > 0:
        density /= density.max()
    cuts = []
    for anchor in anchors:
        start = max(0, int(anchor) - anchor_window * textheight // 2)
        end = min(region.shape[1], int(anchor) + anchor_window * textheight // 2)
        cuts.append(start + int(density[start:end].argmin()))
    return np.split(region, cuts, axis=1)


def yolo_blocks(prediction, resize_ratio):
    """NumPy equivalent of comic-text-detector's torch/torchvision NMS path."""
    rows = prediction[0]
    rows = rows[rows[:, 4] > 0.4].copy()
    if not len(rows):
        return np.empty((0, 4), dtype=np.int32), np.empty(0, dtype=np.int32), np.empty(0)
    rows[:, 5:] *= rows[:, 4:5]
    labels = rows[:, 5:].argmax(axis=1)
    confidence = rows[np.arange(len(rows)), labels + 5]
    keep = confidence > 0.4
    rows, labels, confidence = rows[keep], labels[keep], confidence[keep]
    boxes = np.column_stack((rows[:, 0] - rows[:, 2] / 2, rows[:, 1] - rows[:, 3] / 2,
                             rows[:, 0] + rows[:, 2] / 2, rows[:, 1] + rows[:, 3] / 2))
    ordered = np.argsort(-confidence)[:30000]
    selected = []
    while len(ordered) and len(selected) < 300:
        current = ordered[0]
        selected.append(current)
        rest = ordered[1:]
        if not len(rest):
            break
        left = np.maximum(boxes[current, :2], boxes[rest, :2])
        right = np.minimum(boxes[current, 2:], boxes[rest, 2:])
        wh = np.maximum(right - left, 0)
        intersection = wh[:, 0] * wh[:, 1]
        area_current = np.prod(boxes[current, 2:] - boxes[current, :2])
        area_rest = np.prod(boxes[rest, 2:] - boxes[rest, :2], axis=1)
        overlap = intersection / np.maximum(area_current + area_rest - intersection, 1)
        ordered = rest[(labels[rest] != labels[current]) | (overlap <= 0.35)]
    boxes = boxes[selected]
    boxes[:, [0, 2]] *= resize_ratio[0]
    boxes[:, [1, 3]] *= resize_ratio[1]
    return boxes.astype(np.int32), labels[selected].astype(np.int32), np.round(confidence[selected], 3)


class PageOcr:
    def __init__(self):
        self.detector = open_session("detector.onnx")
        self.representer = SegDetectorRepresenter(thresh=0.3)
        self.recognizer = Recognizer()

    def __call__(self, image_path):
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f"无法读取图片：{image_path}")
        height, width = image.shape[:2]
        scaled, _, (dw, dh) = letterbox(image, new_shape=(1024, 1024), auto=False, stride=64)
        pixels = scaled.transpose((2, 0, 1))[::-1]
        pixels = np.array([np.ascontiguousarray(pixels)], dtype=np.float32) / 255.0
        blks, mask, lines_map = self.detector.run(None, {"image": pixels})
        ratio = (width / (1024 - int(dw)), height / (1024 - int(dh)))
        mask = (mask.squeeze() * 255).astype(np.uint8)
        lines, scores = self.representer((1024, 1024), lines_map)
        indices = np.where(scores[0] > 0.6)[0]
        lines = lines[0][indices].astype(np.float64)
        lines[..., 0] *= ratio[0]
        lines[..., 1] *= ratio[1]
        lines = lines.astype(np.int32)
        mask = cv2.resize(mask[:1024 - int(dh), :1024 - int(dw)], (width, height), interpolation=cv2.INTER_LINEAR)
        blocks = group_output(yolo_blocks(blks, ratio), lines, width, height, mask)
        refined = refine_mask(image, mask, blocks, refine_mode=1)
        refined = refine_undetected_mask(image, mask, refined, blocks, refine_mode=1)
        result = []
        for block in blocks:
            for index, polygon in enumerate(block.lines_array()):
                pieces = split_crop(image, refined, block, index, max_ratio=16 if block.vertical else 8)
                text = "".join(self.recognizer(cv2.rotate(part, cv2.ROTATE_90_CLOCKWISE) if block.vertical else part)
                               for part in pieces)
                if not text:
                    continue
                x1, y1 = polygon.min(axis=0)
                x2, y2 = polygon.max(axis=0)
                result.append({"text": text, "confidence": 1.0,
                               "box": [float(x1), float(y1), float(x2), float(y2)],
                               "vertical": bool(block.vertical)})
        return width, height, result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pages-file")
    parser.add_argument("--probe", action="store_true")
    args = parser.parse_args()
    if args.probe:
        try:
            PageOcr()
            emit({"kind": "probe", "ok": True})
        except Exception as error:
            emit({"kind": "probe", "ok": False, "error": str(error)})
        return
    if not args.pages_file:
        parser.error("缺少 --pages-file")
    raw = json.loads(Path(args.pages_file).read_text(encoding="utf-8"))
    pages = raw["pages"] if isinstance(raw, dict) else raw
    emit({"kind": "meta", "engine": ENGINE, "languages": ["ja-JP"], "requested": ["ja-JP"]})
    try:
        engine = PageOcr()
    except Exception as error:
        emit({"kind": "fatal", "error": str(error)})
        return
    for page in pages:
        file = page["rel"]
        try:
            width, height, lines = engine(page["absPath"])
            emit({"kind": "page", "file": file, "ok": True,
                  "width": width, "height": height, "lines": lines})
        except Exception as error:
            print(f"{file}: {error}", file=sys.stderr, flush=True)
            emit({"kind": "page", "file": file, "ok": False,
                  "width": 0, "height": 0, "lines": [], "error": str(error)})


if __name__ == "__main__":
    main()
