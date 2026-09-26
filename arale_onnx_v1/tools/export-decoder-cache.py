#!/usr/bin/env python3
"""Export Manga OCR's BERT decoder with reusable self/cross-attention KV cache.

Build-time only: requires PyTorch and transformers 5.x. The distributed runner
uses only ONNX Runtime and never imports either package.
"""

import argparse
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch
from transformers import GenerationMixin, VisionEncoderDecoderModel
from transformers.cache_utils import DynamicCache, EncoderDecoderCache


class Model(VisionEncoderDecoderModel, GenerationMixin):
    pass


class Init(torch.nn.Module):
    def __init__(self, decoder):
        super().__init__()
        self.dec = decoder

    def forward(self, encoder_hidden_states, input_ids):
        output = self.dec(input_ids=input_ids, encoder_hidden_states=encoder_hidden_states,
                              use_cache=True, return_dict=True)
        cache = output.past_key_values
        values = [output.logits]
        for layer in cache.self_attention_cache.layers:
            values.extend((layer.keys, layer.values))
        for layer in cache.cross_attention_cache.layers:
            values.extend((layer.keys, layer.values))
        return tuple(values)


class Step(torch.nn.Module):
    def __init__(self, decoder):
        super().__init__()
        self.dec = decoder

    def forward(self, encoder_hidden_states, input_ids,
                s0k, s0v, s1k, s1v, c0k, c0v, c1k, c1v):
        cache = EncoderDecoderCache(
            DynamicCache([(s0k, s0v), (s1k, s1v)]),
            DynamicCache([(c0k, c0v), (c1k, c1v)]),
        )
        output = self.dec(input_ids=input_ids, encoder_hidden_states=encoder_hidden_states,
                              past_key_values=cache, use_cache=True, return_dict=True)
        values = [output.logits]
        for layer in output.past_key_values.self_attention_cache.layers:
            values.extend((layer.keys, layer.values))
        return tuple(values)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--recognizer-model', required=True, help='Local manga-ocr-base directory')
    parser.add_argument('--output', default=str(Path(__file__).resolve().parents[1] / 'models'))
    args = parser.parse_args()
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    model = Model.from_pretrained(args.recognizer_model).eval()
    with torch.no_grad():
        hidden = model.encoder(torch.randn(1, 3, 224, 224)).last_hidden_state.repeat(4, 1, 1)
        ids = torch.full((4, 1), 2, dtype=torch.long)
        init = Init(model.decoder).eval()
        cache_names = ['self_key_0', 'self_value_0', 'self_key_1', 'self_value_1',
                       'cross_key_0', 'cross_value_0', 'cross_key_1', 'cross_value_1']
        outputs = ['logits', *cache_names]
        axes = {'encoder_hidden_states': {0: 'batch'}, 'input_ids': {0: 'batch'}}
        axes.update({name: {0: 'batch'} for name in outputs})
        torch.onnx.export(init, (hidden, ids), str(output_dir / 'manga-ocr-decoder-init.onnx'),
                          input_names=['encoder_hidden_states', 'input_ids'], output_names=outputs,
                          dynamic_axes=axes, opset_version=17, dynamo=False)

        init_values = init(hidden, ids)
        step_ids = torch.tensor([[100], [200], [300], [400]], dtype=torch.long)
        input_names = ['encoder_hidden_states', 'input_ids',
                       'past_self_key_0', 'past_self_value_0', 'past_self_key_1', 'past_self_value_1',
                       'past_cross_key_0', 'past_cross_value_0', 'past_cross_key_1', 'past_cross_value_1']
        output_names = ['logits', 'present_self_key_0', 'present_self_value_0',
                        'present_self_key_1', 'present_self_value_1']
        axes = {'encoder_hidden_states': {0: 'batch'}, 'input_ids': {0: 'batch'}}
        for name in input_names[2:]:
            axes[name] = {0: 'batch', 2: 'past' if name.startswith('past_self') else 'encoder_seq'}
        for name in output_names:
            axes[name] = {0: 'batch'}
        for name in output_names[1:]:
            axes[name][2] = 'present'
        step = Step(model.decoder).eval()
        torch.onnx.export(step,
                          (hidden, step_ids, *init_values[1:]),
                          str(output_dir / 'manga-ocr-decoder-step.onnx'),
                          input_names=input_names, output_names=output_names,
                          dynamic_axes=axes, opset_version=17, dynamo=False)

        # The traced cache branches must agree numerically across multiple
        # steps; a successful export alone does not prove the past is reused.
        options = ort.SessionOptions()
        options.intra_op_num_threads = 4
        init_ort = ort.InferenceSession(str(output_dir / 'manga-ocr-decoder-init.onnx'),
                                        sess_options=options, providers=['CPUExecutionProvider'])
        step_ort = ort.InferenceSession(str(output_dir / 'manga-ocr-decoder-step.onnx'),
                                        sess_options=options, providers=['CPUExecutionProvider'])
        initial = init_ort.run(None, {'encoder_hidden_states': hidden.numpy(), 'input_ids': ids.numpy()})

        def close(actual, expected, label):
            differences = [float(np.max(np.abs(a - b.detach().numpy()))) for a, b in zip(actual, expected)]
            maximum = max(differences)
            print(label, 'max|Δ|', maximum)
            if maximum > 1e-3:
                raise RuntimeError(f'{label} ONNX / PyTorch parity failed: {maximum}')

        close(initial, init_values, 'initial')
        needed = {item.name for item in step_ort.get_inputs()}
        past_ort = initial
        past_ref = init_values
        for iteration in range(2):
            tokens = torch.tensor([[100 + iteration], [200 + iteration],
                                   [300 + iteration], [400 + iteration]], dtype=torch.long)
            reference = step(hidden, tokens, *past_ref[1:5], *init_values[5:])
            arguments = (hidden.numpy(), tokens.numpy(), *past_ort[1:5], *initial[5:])
            feed = {name: value for name, value in zip(input_names, arguments) if name in needed}
            actual = step_ort.run(None, feed)
            close(actual, reference, f'step {iteration + 1}')
            past_ort, past_ref = actual, reference

    for name in ('manga-ocr-decoder-init.onnx', 'manga-ocr-decoder-step.onnx'):
        file = output_dir / name
        print(name, file.stat().st_size)
    print('Validate tensor parity and OCR text parity before updating model-manifest.json.')


if __name__ == '__main__':
    main()
