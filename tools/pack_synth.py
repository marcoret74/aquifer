#!/usr/bin/env python3
# =============================================================================
#  caudal · tools/pack_synth.py
#  Packer SINTÉTICO: genera el artefacto exacto del layout A0 + su referencia.
# =============================================================================
#  Sólo stdlib. Sin red, sin numpy, sin safetensors.
#
#  Su trabajo no es ser rápido: es ser la definición ejecutable del layout.
#  Si el kernel C++ discrepa de `ref.bin`, el bug está en el layout o en el
#  kernel, y el packer dice cuál.
#
#  ── Formato (todas las filas contiguas, sin padding) ────────────────────────
#    w_packed.bin   N * (K/2)      int4 empaquetado, 2 por byte.
#                                  byte i = (u[2i] & 0xF) | (u[2i+1] << 4)
#                                  u = code + 8  ->  nibble UNSIGNED 0..15
#    w_scale.bin    N * (K/64) * 2 fp16, escala por grupo de 64 (simétrica)
#    x_perm.bin     M * K          int8 CON SIGNO (zero-point 0), PRE-PERMUTADO
#    x_scale.bin    M * 4          fp32
#    ref.bin        M * N * 4      fp32, calculado en float64 desde los
#                                  mismos operandos cuantizados
#
#  ── Por qué los nibbles son unsigned ───────────────────────────────────────
#  vpmaddubsw exige primer operando unsigned y segundo signed. Poniendo w como
#  unsigned (code+8) y x como int8 con signo:
#
#      Σ u·x = Σ (code+8)·x = Σ code·x + 8·Σx
#
#  El término 8·Σx depende sólo del token, así que NO viaja en el artefacto:
#  el kernel lo calcula una vez por token. Eso elimina `w_codesum` (+0.5
#  bits/param) y, sobre todo, elimina la LUT vpshufb: el nibble ya es el valor.
#
#      4.75 -> 4.25 bits/param      (−10.5 % de tráfico de DRAM)
#      4 -> 2 ops de shuffle / 32 B (−50 % en el recurso escaso de AVX2)
#
#  ── El pre-permutado de x ───────────────────────────────────────────────────
#  vpunpcklbw/hbw entrega las mitades como ([v0..15, v32..47], [v16..31, v48..63]).
#  Guardando x en ese orden, el kernel empareja directo y no gasta vperm2i128.
#  Cualquier permutación CONSISTENTE da el mismo producto punto.
# =============================================================================

import argparse
import json
import os
import random
import struct

GROUP = 64  # valores por escala

PERM = (list(range(0, 16)) + list(range(32, 48))
        + list(range(16, 32)) + list(range(48, 64)))


def f16(v: float) -> float:
    """Redondea a fp16 y devuelve el valor como float: la referencia usa
    exactamente lo que el C++ va a leer."""
    return struct.unpack('<e', struct.pack('<e', v))[0]


def pack_nibbles(codes):
    """codes en [-8, 7] -> bytes con nibbles unsigned (code+8) en 0..15."""
    out = bytearray(len(codes) // 2)
    for i in range(len(codes) // 2):
        lo = (codes[2 * i] + 8) & 0xF
        hi = (codes[2 * i + 1] + 8) & 0xF
        out[i] = lo | (hi << 4)
    return bytes(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--N', type=int, default=256, help='filas de salida')
    ap.add_argument('--K', type=int, default=512, help='profundidad (multiplo de 64)')
    ap.add_argument('--M', type=int, default=4, help='tokens x')
    ap.add_argument('--out', default='build/artefacto')
    ap.add_argument('--seed', type=int, default=12345)
    args = ap.parse_args()

    N, K, M = args.N, args.K, args.M
    if K % GROUP or K % 2:
        raise SystemExit('K debe ser multiplo de 64')
    G = K // GROUP
    rnd = random.Random(args.seed)

    W = [[rnd.gauss(0.0, 1.0) for _ in range(K)] for _ in range(N)]

    codes = [[0] * K for _ in range(N)]
    scales = [[0.0] * G for _ in range(N)]

    for n in range(N):
        for g in range(G):
            blk = W[n][g * GROUP:(g + 1) * GROUP]
            amax = max(abs(v) for v in blk)
            sw = f16(amax / 7.0) if amax > 0 else 1.0
            if sw == 0.0:
                sw = 1.0
            scales[n][g] = sw
            for j, v in enumerate(blk):
                c = int(round(v / sw))
                codes[n][g * GROUP + j] = max(-8, min(7, c))

    # --- Activaciones: int8 CON SIGNO, zero-point 0 --------------------------
    X = [[rnd.gauss(0.0, 1.0) for _ in range(K)] for _ in range(M)]
    x_scale = []
    xq = []
    for m in range(M):
        amax = max(abs(v) for v in X[m]) or 1.0
        sx = amax / 127.0
        x_scale.append(sx)
        xq.append([max(-127, min(127, int(round(X[m][k] / sx)))) for k in range(K)])

    x_perm = []
    for m in range(M):
        out = bytearray(K)
        for g in range(G):
            for i, p in enumerate(PERM):
                out[g * GROUP + i] = xq[m][g * GROUP + p] & 0xFF   # two's complement
        x_perm.append(bytes(out))

    # --- Referencia (float64 desde los operandos cuantizados) ----------------
    ref = [[0.0] * N for _ in range(M)]
    for m in range(M):
        for n in range(N):
            acc = 0.0
            for g in range(G):
                s = 0
                for j in range(GROUP):
                    s += codes[n][g * GROUP + j] * xq[m][g * GROUP + j]
                acc += s * (scales[n][g] * x_scale[m])
            ref[m][n] = acc

    # --- Escritura -----------------------------------------------------------
    os.makedirs(args.out, exist_ok=True)

    def path(name):
        return os.path.join(args.out, name)

    with open(path('w_packed.bin'), 'wb') as f:
        for n in range(N):
            f.write(pack_nibbles(codes[n]))

    with open(path('w_scale.bin'), 'wb') as f:
        for n in range(N):
            for g in range(G):
                f.write(struct.pack('<e', scales[n][g]))

    with open(path('x_perm.bin'), 'wb') as f:
        for m in range(M):
            f.write(x_perm[m])

    with open(path('x_scale.bin'), 'wb') as f:
        for m in range(M):
            f.write(struct.pack('<f', x_scale[m]))

    with open(path('ref.bin'), 'wb') as f:
        for m in range(M):
            for n in range(N):
                f.write(struct.pack('<f', ref[m][n]))

    row_bytes = K // 2
    assert row_bytes % 16 == 0, 'el stride de fila debe ser multiplo de 16 B'

    total_bits = 4.0 + 16.0 / GROUP
    manifest = {
        'N': N, 'K': K, 'M': M, 'group': GROUP,
        'row_bytes': row_bytes,
        'scale_bytes_per_row': G * 2,
        'bits_per_param': total_bits,
        'nibble': 'unsigned (code+8), 0..15',
        'x': 'int8 signed, zero_point 0, pre-permutado',
        'perm': PERM,
        'no_codesum': 'el sesgo 8*sum(x) se calcula por token en el kernel',
        'files': ['w_packed.bin', 'w_scale.bin', 'x_perm.bin',
                  'x_scale.bin', 'ref.bin'],
    }
    with open(path('manifest.json'), 'w') as f:
        json.dump(manifest, f, indent=2)

    print(f'artefacto en {args.out}/')
    print(f'  N={N} K={K} M={M} grupos={G}  row_bytes={row_bytes}')
    print(f'  {total_bits:.2f} bits/param  (sin w_codesum)')
    print(f'  w_packed   {N*row_bytes:>10} B')
    print(f'  w_scale    {N*G*2:>10} B')
    print(f'  x_perm     {M*K:>10} B')
    print(f'  ref        {M*N*4:>10} B')


if __name__ == '__main__':
    main()
