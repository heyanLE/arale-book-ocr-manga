#!/usr/bin/env python3
"""Compare the ONNX runner against Mokuro 0.2.5 on the same page images.

Build-time tool only. Use a Python environment with torch, mokuro and
onnxruntime installed; it is never copied into the user-facing engine ZIP.
"""

import argparse
import json
import os
import sys
from pathlib import Path


def iou(a, b):
    intersection = max(0, min(a[2], b[2]) - max(a[0], b[0])) * max(0, min(a[3], b[3]) - max(a[1], b[1]))
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - intersection
    return intersection / union if union else 0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--recognizer-model', required=True, help='Local manga-ocr-base directory')
    parser.add_argument('--cache-root', required=True, help='Directory containing manga-ocr/comictextdetector.pt')
    parser.add_argument('images', nargs='+')
    args = parser.parse_args()
    os.environ['XDG_CACHE_HOME'] = args.cache_root
    os.environ['HF_HUB_OFFLINE'] = '1'
    os.environ['TRANSFORMERS_OFFLINE'] = '1'
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'python'))
    import torch
    from mokuro.manga_page_ocr import MangaPageOcr
    from ocr_run import PageOcr

    torch.set_num_threads(4)
    reference = MangaPageOcr(pretrained_model_name_or_path=args.recognizer_model, force_cpu=True)
    candidate = PageOcr()
    totals = {'pages': 0, 'reference': 0, 'ours': 0, 'matched': 0, 'exact': 0}
    for filename in args.images:
        output = reference(Path(filename))
        ref_lines = []
        for block in output['blocks']:
            for text, polygon in zip(block['lines'], block['lines_coords']):
                xs = [point[0] for point in polygon]
                ys = [point[1] for point in polygon]
                ref_lines.append((text, [min(xs), min(ys), max(xs), max(ys)]))
        _, _, ours = candidate(filename)
        used = set()
        paired = []
        for line in ours:
            ranked = sorted(((iou(line['box'], box), index) for index, (_, box) in enumerate(ref_lines)
                             if index not in used), reverse=True)
            if ranked and ranked[0][0] >= 0.5:
                score, index = ranked[0]
                used.add(index)
                paired.append((score, line['text'], ref_lines[index][0]))
        row = {'page': Path(filename).name, 'reference': len(ref_lines), 'ours': len(ours),
               'matched': len(paired), 'exact': sum(a == b for _, a, b in paired),
               'diffs': [(a, b) for _, a, b in paired if a != b]}
        print(json.dumps(row, ensure_ascii=False), flush=True)
        for key in totals:
            totals[key] += 1 if key == 'pages' else row[key]
    print(json.dumps({'total': totals, 'exact_rate': totals['exact'] / totals['matched'] if totals['matched'] else 0},
                     ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
