#!/usr/bin/env python3
"""Sonda de logits contra el oraculo PyTorch (fp32, sin cuantizar).

Sirve para separar dos causas antes de tocar C++:
  * el pack/kernel lee mal los pesos  -> C++ y oraculo NO coinciden
  * el modelo realmente produce eso   -> coinciden, y el problema es otro

Maneja los dos casos de lm_head: atado (usa embed_tokens) y propio.

  python3 tools/probe_logits.py --model DIR --prompt "texto"
  python3 tools/probe_logits.py --model DIR --ids 1,2,3
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
    ap.add_argument('--prompt', default=None)
    ap.add_argument('--ids', default=None, help='ids separados por coma')
    ap.add_argument('--top', type=int, default=8)
    ap.add_argument('--norms', action='store_true', help='norma del estado por capa')
    ap.add_argument('--quant', action='store_true', help='int4 g<qgroup> al vuelo')
    ap.add_argument('--qgroup', type=int, default=32)
    ap.add_argument('--xq', action='store_true',
                    help='int8 simetrico en cada linear (lo que hace el C++)')
    ap.add_argument('--xq-group', type=int, default=0,
                    help='0 = una escala por token; G = una escala por grupo de G')
    ap.add_argument('--layers', type=int, default=-1,
                    help='override de num_hidden_layers (bisectar profundidad)')
    a = ap.parse_args()

    cfg = json.load(open(os.path.join(a.model, 'config.json')))
    cfg['rope_theta'] = float(cfg['rope_theta'])
    if a.layers >= 0:
        cfg['num_hidden_layers'] = a.layers
    tok = Tokenizer(os.path.join(a.model, 'tokenizer.json'))

    if a.ids:
        ids = [int(x) for x in a.ids.split(',')]
    else:
        ids = tok.encode(a.prompt, allow_special=False)
    print('ids:', ids)
    print('texto:', repr(tok.decode(ids)))

    if a.xq:
        # El motor cuantiza x a int8 (amax/127) antes de cada GEMV. Se replica
        # aqui para saber si esa perdida, y no el motor, es la que rompe.
        _orig = F.linear

        G = a.xq_group

        def _q8(t):
            T, K = t.shape
            if G and K % G == 0:
                g = t.view(T, K // G, G)
                amax = g.abs().amax(-1, keepdim=True)
                esc = torch.clamp(amax / 127.0, min=1e-8)
                return ((g / esc).round().clamp(-127, 127) * esc).view(T, K)
            amax = t.abs().amax(-1, keepdim=True)
            esc = torch.clamp(amax / 127.0, min=1e-8)
            return (t / esc).round().clamp(-127, 127) * esc

        def _lin(inp, w, b=None):
            return _orig(_q8(inp), w, b)
        F.linear = _lin

    m = Qwen(a.model, cfg)
    if a.quant:
        m.quant = True
        m.qgroup = a.qgroup
    xn, _, hs = m.forward(ids)
    if a.norms:
        for k, h in enumerate(hs):
            print(f'NORM L={k:2d} |x|={float(h[-1].norm()):.5f} x0={float(h[-1][0]):.5f} x1={float(h[-1][1]):.5f}')

    tied = F.linear(xn, m.wq('embed_tokens.weight'))
    caminos = [('ATADO (embed)', tied)]
    if 'lm_head.weight' in m.st.header:
        caminos.append(('lm_head propio',
                        F.linear(xn, m.st.get('lm_head.weight').float())))

    for etiqueta, lg in caminos:
        p = torch.softmax(lg[-1].float(), -1)
        v, i = p.topk(a.top)
        print(f'\n-- {etiqueta}')
        for pi, ti in zip(v.tolist(), i.tolist()):
            print(f'   {ti:>7d}  p={pi:8.5f}  {tok.decode([ti])!r}')


if __name__ == '__main__':
    main()
