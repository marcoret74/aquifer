#!/usr/bin/env python3
"""Recomputa la capa 0 en numpy leyendo el ARTEFACTO (int4 ya empacado).

Compara contra lo que imprime `generate_qwen --dump`. Sirve para decidir si el
motor (gemv) o el artefacto (pack) es el que se aparta.

  python3 tools/probe_layer0.py --model DIR --artifact DIR --tok 1
"""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from check_pack import dequant  # noqa: E402
from oracle_qwen import SafeTensors  # noqa: E402


def cargar(art):
    idx = {}
    for line in open(os.path.join(art, 'index.txt')):
        f = line.split()
        if f and f[0] == 'LINEAR' and len(f) == 8:      # split
            idx[f[1]] = (int(f[2]), int(f[3]), int(f[4]), int(f[5]),
                         int(f[6]), int(f[7]))
    W = open(os.path.join(art, 'weights.bin'), 'rb').read()
    S = open(os.path.join(art, 'scales.bin'), 'rb').read()
    return idx, W, S


def mat(idx, W, S, name, filas=None):
    N, K, off, pb, soff, sb = idx[name]
    G = sb // (2 * N)
    n = N if filas is None else min(filas, N)
    pk = np.frombuffer(W, np.uint8, n * (K // 2), off).reshape(n, K // 2)
    sp = np.frombuffer(S, np.uint8, n * G * 2, soff).reshape(n, G * 2)
    return dequant(pk, sp, n, K, G), N, K, G


def norm(x):
    return float(np.sqrt((x.astype(np.float64) ** 2).sum()))


def rmsnorm(x, w, eps):
    r = 1.0 / np.sqrt((x.astype(np.float32) ** 2).mean() + eps)
    return x * r * w


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', required=True)
    ap.add_argument('--artifact', required=True)
    ap.add_argument('--tok', type=int, default=1)
    ap.add_argument('--layer', type=int, default=0)
    ap.add_argument('--head-dim', type=int, default=0)
    a = ap.parse_args()

    cfg = json.load(open(os.path.join(a.model, 'config.json')))
    eps = float(cfg['rms_norm_eps'])
    H = cfg['hidden_size']
    d = a.head_dim or H // cfg['num_attention_heads']
    nh, nkv = cfg['num_attention_heads'], cfg['num_key_value_heads']

    st = SafeTensors(os.path.join(a.model, 'model.safetensors'))
    idx, W, S = cargar(a.artifact)
    P = f'layers.{a.layer}.'

    x = st.get('model.embed_tokens.weight')[a.tok].float().numpy()
    print(f'embed  |x|={norm(x):.5f}  x0={x[0]:.5f} x1={x[1]:.5f}')

    h = rmsnorm(x, st.get('model.' + P + 'input_layernorm.weight').float().numpy(), eps)
    print(f'L0 |h_in|={norm(h):.5f}')

    def lin(nombre):
        m, N, K, G = mat(idx, W, S, 'model.' + nombre)
        return m

    q = lin(P + 'self_attn.q_proj.weight') @ h
    k = lin(P + 'self_attn.k_proj.weight') @ h
    v = lin(P + 'self_attn.v_proj.weight') @ h
    print(f'L0 |q|={norm(q):.5f} |k|={norm(k):.5f} |v|={norm(v):.5f}')

    # 1 token: atencion sobre si mismo = v
    o = np.tile(v.reshape(nkv, d), (nh // nkv, 1)).reshape(-1)
    print(f'L0 |o_attn|={norm(o):.5f}')
    oproj = lin(P + 'self_attn.o_proj.weight') @ o
    print(f'L0 |o_proj|={norm(oproj):.5f}')

    x1 = x + oproj
    h2 = rmsnorm(x1, st.get('model.' + P + 'post_attention_layernorm.weight').float().numpy(), eps)
    g = lin(P + 'mlp.gate_proj.weight') @ h2
    u = lin(P + 'mlp.up_proj.weight') @ h2
    print(f'L0 |gate|={norm(g):.5f} |up|={norm(u):.5f}')
    mid = (g / (1.0 + np.exp(-g))) * u
    out = lin(P + 'mlp.down_proj.weight') @ mid
    print(f'L0 |mlp_out|={norm(out):.5f}')
    x2 = x1 + out
    print(f'L0 |x|={norm(x2):.5f}  x0={x2[0]:.5f} x1={x2[1]:.5f}')


if __name__ == '__main__':
    main()
