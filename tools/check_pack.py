#!/usr/bin/env python3
"""Verifica el artefacto: dequantiza lo empacado y lo compara con el original.

Separa dos causas antes de tocar C++:
  * si el error es grande  -> el PACKER (o el layout) esta mal
  * si el error es ~cuantizacion (5-12 %) -> los datos estan bien y el bug
    esta en el consumo (el motor)

  python3 tools/check_pack.py --model DIR --artifact DIR [--max 5]
"""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from oracle_qwen import SafeTensors  # noqa: E402


def dequant(packed_u8, scales_u8, N, K, G):
    """packed_u8: (N, K/2) uint8 con dos nibbles por byte, ya con offset +8.

    El kernel usa `w = code + 8` (nibble 0..15 YA es unsigned, sin LUT):
    ver include/caudal/gemv_int4.hpp. El nibble bajo es j par, el alto j impar.
    """
    q = np.empty((N, K), dtype=np.float32)
    q[:, 0::2] = (packed_u8 & 0x0F).astype(np.float32) - 8.0
    q[:, 1::2] = ((packed_u8 >> 4) & 0x0F).astype(np.float32) - 8.0
    sw = scales_u8.view(np.float16).astype(np.float32).reshape(N, G, 1)
    return (q.reshape(N, G, K // G) * sw).reshape(N, K)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', required=True)
    ap.add_argument('--artifact', required=True)
    ap.add_argument('--max', type=int, default=5, help='cuantas matrices revisar')
    ap.add_argument('--rows', type=int, default=64, help='filas a comparar')
    a = ap.parse_args()

    idx = {}
    for line in open(os.path.join(a.artifact, 'index.txt')):
        f = line.split()
        if f and f[0] == 'LINEAR':
            name = f[1]
            if len(f) == 8:                       # split: base packed scale_off scale_bytes
                idx[name] = dict(N=int(f[2]), K=int(f[3]), off=int(f[4]),
                                 pb=int(f[5]), soff=int(f[6]), sb=int(f[7]))
            else:                                 # intercalado: base stride groups
                idx[name] = dict(N=int(f[2]), K=int(f[3]), off=int(f[4]),
                                 stride=int(f[5]), groups=int(f[6]))

    W = open(os.path.join(a.artifact, 'weights.bin'), 'rb').read()
    S = open(os.path.join(a.artifact, 'scales.bin'), 'rb').read()
    st = SafeTensors(os.path.join(a.model, 'model.safetensors'))

    revisados = 0
    for name, m in idx.items():
        if revisados >= a.max:
            break
        N, K = m['N'], m['K']
        nrows = min(a.rows, N)
        if 'stride' in m:                          # intercalado
            G = m['groups']
            rows = np.frombuffer(W, dtype=np.uint8, count=nrows * m['stride'],
                                 offset=m['off']).reshape(nrows, m['stride'])
            pk, sp = rows[:, :K // 2], rows[:, K // 2:K // 2 + G * 2]
            G = G
        else:                                      # split
            G = m['sb'] // (N * 2)
            pk = np.frombuffer(W, dtype=np.uint8, count=nrows * (K // 2),
                               offset=m['off']).reshape(nrows, K // 2)
            sp = np.frombuffer(S, dtype=np.uint8, count=nrows * G * 2,
                               offset=m['soff']).reshape(nrows, G * 2)
        got = dequant(pk, sp, nrows, K, G)
        ref = st.get(name).float().numpy()[:nrows]
        err = np.abs(got - ref).mean() / (np.abs(ref).mean() + 1e-9)
        print(f'{name:<46} N={N:>6} K={K:>5} G={G:>3}  err_rel={100*err:6.3f} %')
        revisados += 1

    # de paso: tamano de embed.bf16 vs lo declarado
    e = st.header['model.embed_tokens.weight']
    print(f'\nembed.bf16: {os.path.getsize(os.path.join(a.artifact, "embed.bf16"))} bytes, '
          f'esperado {e["shape"][0]*e["shape"][1]*2}  shape={e["shape"]}')


if __name__ == '__main__':
    main()
