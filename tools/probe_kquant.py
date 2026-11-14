#!/usr/bin/env python3
"""Simula el esquema tipo Q4_K (super-bloque 256, 16 niveles con minimo,
escalas y minimos a 6 bits) y mide su perplejidad SIN tocar el kernel.

Sirve para decidir con numeros si vale la pena implementarlo en C++.

  python3 tools/probe_kquant.py --model DIR --text F --window 512

Esquema simulado, por super-bloque de 256 (8 sub-bloques de 32):
  w = d*sc_i*q - dmin*m_i          q in [0,15]
  d    = max_i(scale_i)/63         sc_i = round(scale_i/d)      (6 bits)
  dmin = max_i(min_i)/63           m_i  = round(min_i/dmin)     (6 bits)
  scale_i = (max_i - min_i)/15
Bits reales: 4 + (8*6*2 + 32)/256 = 4,5 bpw  (igual que el g32 actual)
"""
import argparse
import json
import math
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from oracle_qwen import Qwen, Tokenizer  # noqa: E402

SB = 256      # valores por super-bloque
SUB = 32      # valores por sub-bloque (grupo de escala)


def kquant(w):
    """w [N,K] float32 -> dequantizado float32 con el esquema simulado."""
    N, K = w.shape
    if K % SB:
        return w
    G = K // SB
    x = w.reshape(N, G, SB)
    sub = x.reshape(N, G, SB // SUB, SUB)
    mn = sub.min(axis=3)
    mx = sub.max(axis=3)
    sc = (mx - mn) / 15.0
    d = np.maximum(sc.max(axis=2, keepdims=True) / 63.0, 1e-8)
    sc_q = np.rint(sc / d).clip(0, 63)
    dm = np.maximum(mn.max(axis=2, keepdims=True) / 63.0, 1e-8)
    # el minimo se maneja como magnitud positiva: w = d*sc*q - dmin*m
    mn_mag = np.maximum(-mn, 0.0)
    dmin = np.maximum(mn_mag.max(axis=2, keepdims=True) / 63.0, 1e-8)
    m_q = np.rint(mn_mag / dmin).clip(0, 63)
    # cuantizar q con la escala efectiva (d*sc) y el minimo real (-dmin*m)
    s_eff = d * sc_q
    m_eff = -dmin * m_q
    q = np.rint((sub - m_eff[:, :, :, None]) / np.maximum(s_eff[:, :, :, None], 1e-8))
    q = q.clip(0, 15)
    wq = s_eff[:, :, :, None] * q + m_eff[:, :, :, None]
    return wq.reshape(N, K).astype(np.float32)


def bpc(nll, ids, txt, chars=None):
    chars = chars if chars else len(txt)
    mitad = len(ids) // 2
    seg = float(nll[mitad:].sum())
    n_seg = len(ids) - 1 - mitad
    ch = chars * n_seg / (len(ids) - 1)
    return math.exp(seg / max(1, n_seg)), seg / (math.log(2) * ch)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', required=True)
    ap.add_argument('--text', required=True)
    ap.add_argument('--window', type=int, default=512)
    a = ap.parse_args()

    txt = open(a.text, encoding='utf-8').read()
    cfg = json.load(open(os.path.join(a.model, 'config.json')))
    cfg['rope_theta'] = float(cfg['rope_theta'])
    tok = Tokenizer(os.path.join(a.model, 'tokenizer.json'))
    ids = tok.encode(txt, allow_special=False)[:a.window]
    medio = {}          # cache del modo actual

    m = Qwen(a.model, cfg)
    m.lowmem = True

    def con(modo):
        def get(nombre):
            t = m.st.get('model.' + nombre).float()
            if t.dim() != 2:
                return t
            if modo == 'bf16':
                return t
            if modo == 'kquant':
                return torch.from_numpy(kquant(t.numpy()))
            return t
        medio['get'] = get
        m.w = get
        m.wb = lambda nombre: (m.st.get('model.' + nombre).float()
                               if ('model.' + nombre) in m.st.header else None)

    for modo in ('bf16', 'kquant'):
        con(modo)
        with torch.no_grad():
            _, lg, _ = m.forward(ids)
        lp = torch.log_softmax(lg.float(), -1)
        nll = torch.tensor([-lp[t, ids[t + 1]].item() for t in range(len(ids) - 1)])
        ppl, b = bpc(nll, ids, txt, len(tok.decode(ids)))
        print(f'{modo:8s} tokens={len(ids):4d}  PPL(2a mitad)={ppl:8.4f}  BPC={b:.4f}')


if __name__ == '__main__':
    main()
