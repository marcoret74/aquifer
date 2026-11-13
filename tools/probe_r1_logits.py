#!/usr/bin/env python3
"""Sonda: logits del primer token con lm_head REAL vs embedding atado.

Motivo: `oracle_qwen.Qwen.forward` asume `tie_word_embeddings=True` y calcula
los logits con `embed_tokens`. DeepSeek-R1-Distill-Qwen-1.5B declara
`tie_word_embeddings=False` y trae un `lm_head.weight` propio. Esta sonda mide
si esa suposicion explica una salida rota, comparando ambos caminos.

No cuantiza (quant=False): aisla la MATEMATICA del ruido int4.
"""
import argparse
import json
import os
import sys

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from oracle_qwen import Qwen, Tokenizer  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', required=True)
    ap.add_argument('--prompt', default='La capital de Francia es')
    ap.add_argument('--top', type=int, default=8)
    a = ap.parse_args()

    cfg = json.load(open(os.path.join(a.model, 'config.json')))
    cfg['rope_theta'] = float(cfg['rope_theta'])
    tok = Tokenizer(os.path.join(a.model, 'tokenizer.json'))
    ids = tok.encode(a.prompt, allow_special=False)
    print('prompt ids:', ids)

    m = Qwen(a.model, cfg)
    xn, _, _ = m.forward(ids)

    tied = F.linear(xn, m.wq('embed_tokens.weight'))
    head = m.st.get('lm_head.weight').float()
    real = F.linear(xn, head)

    for etiqueta, lg in (('ATADO (embed)', tied), ('lm_head REAL', real)):
        p = torch.softmax(lg[-1].float(), -1)
        v, i = p.topk(a.top)
        print(f'\n-- {etiqueta}')
        for pi, ti in zip(v.tolist(), i.tolist()):
            print(f'   {ti:>7d}  p={pi:8.5f}  {tok.decode([ti])!r}')

    # Cuanto difieren ambos caminos
    d = (real[-1] - tied[-1]).abs()
    print(f'\n|lm_head - atado|: mean={d.mean():.4f} max={d.max():.3f}')
    print('coseno entre filas de lm_head y embed:',
          F.cosine_similarity(head[:1000], m.wq('embed_tokens.weight')[:1000]).mean().item())


if __name__ == '__main__':
    main()
