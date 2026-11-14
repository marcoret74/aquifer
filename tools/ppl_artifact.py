#!/usr/bin/env python3
"""Perplejidad del ARTEFACTO EMPAQUETADO (no del oraculo bf16).

Necesario para medir cambios del packer (grilla de clipping, escalas): el
oraculo cuantiza al vuelo con su propia convencion, asi que no ve el efecto.

  python3 tools/ppl_artifact.py --model DIR --artifact DIR --text F --window 512

Puntua la SEGUNDA MITAD de la ventana, que es lo que hace llama-perplexity.
"""
import argparse
import json
import math
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from check_pack import dequant, dequant_kmin, dequant_affine   # noqa: E402
from oracle_qwen import Qwen, Tokenizer  # noqa: E402


class Artefacto:
    def __init__(self, art):
        self.lin, self.aux_off, self.aux_cnt = {}, {}, {}
        for line in open(os.path.join(art, 'index.txt')):
            f = line.split()
            if not f:
                continue
            if f[0] == 'LINEARA':                      # afin (escala fp16 + minimo fp16)
                self.lin[f[1]] = dict(N=int(f[2]), K=int(f[3]), off=int(f[4]),
                                      pb=int(f[5]), soff=int(f[6]), moff=int(f[7]),
                                      affine=True)
            elif f[0] == 'LINEARK' and len(f) == 8:      # kmin
                self.lin[f[1]] = dict(N=int(f[2]), K=int(f[3]), off=int(f[4]),
                                      pb=int(f[5]), soff=int(f[6]), sb=int(f[7]),
                                      kmin=True)
            elif f[0] == 'LINEAR' and len(f) == 8:     # split
                self.lin[f[1]] = dict(N=int(f[2]), K=int(f[3]), off=int(f[4]),
                                      pb=int(f[5]), soff=int(f[6]), sb=int(f[7]))
            elif f[0] == 'LINEAR' and len(f) == 7:     # intercalado
                self.lin[f[1]] = dict(N=int(f[2]), K=int(f[3]), off=int(f[4]),
                                      stride=int(f[5]), groups=int(f[6]))
            elif f[0] == 'AUX':
                self.aux_off[f[1]] = int(f[3])
                self.aux_cnt[f[1]] = int(f[2])
            elif f[0] == 'EMBED':
                pass
        self.W = open(os.path.join(art, 'weights.bin'), 'rb').read() \
            if os.path.exists(os.path.join(art, 'weights.bin')) else None
        self.S = open(os.path.join(art, 'scales.bin'), 'rb').read() \
            if os.path.exists(os.path.join(art, 'scales.bin')) else None
        if self.W is None:                              # intercalado
            self.W = open(os.path.join(art, 'w.bin'), 'rb').read()
        self.A = open(os.path.join(art, 'aux.bin'), 'rb').read()

    def tiene(self, nombre):
        return nombre in self.lin or nombre in self.aux_off

    def get(self, nombre):
        if nombre in self.aux_off:
            o, n = self.aux_off[nombre], self.aux_cnt[nombre]
            a = np.frombuffer(self.A, np.float32, n, o * 4).copy()
            return torch.from_numpy(a)
        m = self.lin[nombre]
        N, K = m['N'], m['K']
        if m.get('affine'):
            G = K // 32
            pk = np.frombuffer(self.W, np.uint8, N * (K // 2), m['off']).reshape(N, K // 2)
            sp = np.frombuffer(self.S, np.uint8, N * G * 2, m['soff']).reshape(N, G * 2)
            mn = np.frombuffer(self.S, np.uint8, N * G * 2, m['moff']).reshape(N, G * 2)
            return torch.from_numpy(dequant_affine(pk, sp, mn, N, K, G))
        if m.get('kmin'):
            pk = np.frombuffer(self.W, np.uint8, N * (K // 2), m['off']).reshape(N, K // 2)
            hb = N * (K // 256) * 16
            hd = np.frombuffer(self.S, np.uint8, hb, m['soff']).reshape(N, K // 256 * 16)
            return torch.from_numpy(dequant_kmin(pk, hd, N, K))
        if 'stride' in m:                               # intercalado
            G = m['groups']
            rows = np.frombuffer(self.W, np.uint8, N * m['stride'],
                                 m['off']).reshape(N, m['stride'])
            pk = np.ascontiguousarray(rows[:, :K // 2])
            sp = np.ascontiguousarray(rows[:, K // 2:K // 2 + G * 2])
        else:
            G = m['sb'] // (2 * N)
            pk = np.frombuffer(self.W, np.uint8, N * (K // 2), m['off']).reshape(N, K // 2)
            sp = np.frombuffer(self.S, np.uint8, N * G * 2, m['soff']).reshape(N, G * 2)
        return torch.from_numpy(dequant(pk, sp, N, K, G))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', required=True)
    ap.add_argument('--artifact', required=True)
    ap.add_argument('--text', required=True)
    ap.add_argument('--window', type=int, default=512)
    a = ap.parse_args()

    txt = open(a.text, encoding='utf-8').read()
    chars_tot = len(txt)
    cfg = json.load(open(os.path.join(a.model, 'config.json')))
    cfg['rope_theta'] = float(cfg['rope_theta'])
    tok = Tokenizer(os.path.join(a.model, 'tokenizer.json'))
    ids = tok.encode(txt, allow_special=False)[:a.window]
    chars_tot = len(tok.decode(ids))     # caracteres REALES de la ventana

    art = Artefacto(a.artifact)
    m = Qwen(a.model, cfg)
    m.lowmem = True
    m.quant = False
    m.w = lambda nombre: art.get('model.' + nombre)
    m.wb = lambda nombre: (art.get('model.' + nombre)
                           if art.tiene('model.' + nombre) else None)

    with torch.no_grad():
        _, lg, _ = m.forward(ids)
    lp = torch.log_softmax(lg.float(), -1)
    nll = torch.tensor([-lp[t, ids[t + 1]].item() for t in range(len(ids) - 1)])
    mitad = len(ids) // 2
    seg = float(nll[mitad:].sum())
    n_seg = len(ids) - 1 - mitad
    ch_seg = chars_tot * n_seg / (len(ids) - 1)
    ppl = math.exp(seg / max(1, n_seg))
    bpc = seg / (math.log(2) * ch_seg)
    print(f'{os.path.basename(a.artifact):16s} tokens={len(ids):4d}  '
          f'PPL(2a mitad)={ppl:8.4f}  BPC={bpc:.4f}')


if __name__ == '__main__':
    main()
