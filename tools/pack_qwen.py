#!/usr/bin/env python3
# =============================================================================
#  caudal · tools/pack_qwen.py
#  Safetensors reales -> artefacto A0 en el layout CONGELADO (4.25 bits/param).
# =============================================================================
#  Sin `safetensors`, sin `transformers`. El formato safetensors son 8 bytes
#  (uint64 LE = tamaño del header) + un JSON + el bloque de datos contiguo.
#
#  Produce:
#    <out>/weights.bin   int4 empaquetado de TODOS los tensores, 64 B alineado
#    <out>/scales.bin    fp16 por grupo de 64, 64 B alineado
#    <out>/manifest.json índice: nombre -> (N, K, grupos, offsets)
#
#  Y opcionalmente (--layout-tensor) el set de 5 archivos que consume
#  `test_pack_roundtrip`, para validar el kernel C++ con PESOS REALES.
#
#  Formato congelado (ver include/caudal/gemv_int4.hpp):
#    nibble  = code + 8      (unsigned 0..15, sin LUT)
#    byte i  = lo | (hi << 4)   con lo = v[2i], hi = v[2i+1]
#    x       = int8 con signo, zero-point 0   (el sesgo 8*sum(x) es por token)
#
#  Uso:
#    python3 tools/pack_qwen.py --model /tmp/opencode/qwen --out build/qwen05
#    python3 tools/pack_qwen.py --model ... --out ... --layout-tensor \
#            model.layers.0.mlp.down_proj.weight --layout-out build/real_layer
# =============================================================================

import argparse
import json
import os
import struct
import sys

import numpy as np

GROUP = 64
PERM = (list(range(0, 16)) + list(range(32, 48))
        + list(range(16, 32)) + list(range(48, 64)))

# tensores que SON lineales (W [out, in]) y por lo tanto van al artefacto
LINEAR_SUFFIX = ('q_proj.weight', 'k_proj.weight', 'v_proj.weight', 'o_proj.weight',
                 'gate_proj.weight', 'up_proj.weight', 'down_proj.weight',
                 'lm_head.weight')


# -----------------------------------------------------------------------------
#  Lectura de safetensors sin dependencias
# -----------------------------------------------------------------------------
class SafeTensors:
    def __init__(self, path):
        self.path = path
        self.f = open(path, 'rb')
        raw = self.f.read(8)
        if len(raw) != 8:
            raise SystemExit('archivo safetensors truncado')
        (hlen,) = struct.unpack('<Q', raw)
        self.header = json.loads(self.f.read(hlen).decode('utf-8'))
        self.data_start = 8 + hlen
        self.header.pop('__metadata__', None)

    def names(self):
        return list(self.header.keys())

    def info(self, name):
        e = self.header[name]
        return e['dtype'], e['shape'], e['data_offsets']

    def tensor_f32(self, name):
        """Devuelve float32 [..] desde BF16 o F16 o F32."""
        dtype, shape, (a, b) = self.info(name)
        self.f.seek(self.data_start + a)
        buf = self.f.read(b - a)
        if dtype == 'BF16':
            u = np.frombuffer(buf, dtype=np.uint16)
            f = (u.astype(np.uint32) << 16).view(np.float32)
        elif dtype == 'F16':
            f = np.frombuffer(buf, dtype=np.float16).astype(np.float32)
        elif dtype == 'F32':
            f = np.frombuffer(buf, dtype=np.float32)
        else:
            raise SystemExit(f'dtype no soportado: {dtype}')
        return f.reshape(shape)


# -----------------------------------------------------------------------------
#  Cuantización int4 por grupo de 64 (vectorizada)
# -----------------------------------------------------------------------------
def quantize(w):
    """w: float32 [N, K] -> (packed uint8 [N, K/2], scales fp16 [N, G], err)"""
    N, K = w.shape
    if K % GROUP:
        raise SystemExit(f'K={K} no es multiplo de {GROUP}')
    G = K // GROUP

    g = w.reshape(N, G, GROUP)
    amax = np.abs(g).max(axis=2)
    sw = (amax / 7.0).astype(np.float32)
    sw[sw < 1e-8] = 1e-8                      # nunca dividir por cero
    sw16 = sw.astype(np.float16)
    sw_back = sw16.astype(np.float32)          # el valor REAL que leerá el kernel

    codes = np.rint(g / sw_back[:, :, None])
    np.clip(codes, -8, 7, out=codes)
    deq = codes * sw_back[:, :, None]

    num = float(np.linalg.norm(g - deq))
    den = float(np.linalg.norm(g)) or 1.0
    err = num / den

    u = (codes.astype(np.int16) + 8).astype(np.uint8)   # 0..15
    u = u.reshape(N, K)
    lo = u[:, 0::2]
    hi = u[:, 1::2]
    packed = (lo | (hi << 4)).astype(np.uint8)          # [N, K/2]
    return np.ascontiguousarray(packed), np.ascontiguousarray(sw16), err


def quantize_x(x):
    """x: float32 [K] -> (x_int8 [K], scale) y pre-permuta por bloques de 64."""
    amax = float(np.abs(x).max()) or 1.0
    sx = np.float32(amax / 127.0)
    q = np.clip(np.rint(x / sx), -127, 127).astype(np.int8)
    K = q.size
    out = np.empty(K, dtype=np.int8)
    for gi in range(K // GROUP):
        base = gi * GROUP
        for i, p in enumerate(PERM):
            out[base + i] = q[base + p]
    return out, float(sx)


def align(n, a=64):
    return (n + a - 1) & ~(a - 1)


# -----------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', required=True, help='directorio con model.safetensors')
    ap.add_argument('--out', required=True)
    ap.add_argument('--layout-tensor', default=None,
                    help='tensor a emitir tambien en formato de 5 archivos')
    ap.add_argument('--layout-out', default=None)
    ap.add_argument('--layout-tokens', type=int, default=2,
                    help='cuantos x sinteticos para el set de 5 archivos')
    args = ap.parse_args()

    st_path = os.path.join(args.model, 'model.safetensors')
    if not os.path.exists(st_path):
        raise SystemExit(f'no existe {st_path}')
    st = SafeTensors(st_path)
    cfg_path = os.path.join(args.model, 'config.json')
    cfg = json.load(open(cfg_path)) if os.path.exists(cfg_path) else {}

    # --- Elegir tensores lineales 2D -----------------------------------------
    todo = []
    for name in sorted(st.names()):
        if not name.endswith(LINEAR_SUFFIX):
            continue
        dtype, shape, _ = st.info(name)
        if len(shape) != 2:
            print(f'  (salto {name}: shape {shape})')
            continue
        todo.append((name, shape))

    if not todo:
        raise SystemExit('no se encontro ningun tensor lineal 2D')

    print(f'modelo : {args.model}')
    print(f'tensores lineales: {len(todo)}')
    print(f'{"nombre":<44}{"N":>6}{"K":>6}{"bits":>7}{"err%":>9}')
    print('-' * 74)

    os.makedirs(args.out, exist_ok=True)
    blobs_w, blobs_s, index = [], [], {}
    off_w = off_s = 0
    total_params = 0

    for name, (N, K) in todo:
        w = st.tensor_f32(name)
        packed, scales, err = quantize(w)
        total_params += N * K

        o_w = align(off_w)
        o_s = align(off_s)
        blobs_w.append((o_w, packed.tobytes()))
        blobs_s.append((o_s, scales.tobytes()))
        index[name] = {
            'N': int(N), 'K': int(K), 'groups': int(K // GROUP),
            'packed_off': o_w, 'packed_bytes': int(packed.size),
            'scale_off': o_s, 'scale_bytes': int(scales.size * 2),
            'quant_err': round(err, 6),
        }
        off_w = o_w + packed.size
        off_s = o_s + scales.size * 2
        print(f'{name:<44}{N:>6}{K:>6}{4.25:>7.2f}{100*err:>9.3f}')

    # --- Escritura -----------------------------------------------------------
    def write_blob(path, parts, total):
        with open(path, 'wb') as f:
            for off, data in parts:
                f.write(b'\0' * (off - f.tell()))
                f.write(data)
            f.write(b'\0' * (align(total) - f.tell()))

    w_path = os.path.join(args.out, 'weights.bin')
    s_path = os.path.join(args.out, 'scales.bin')
    write_blob(w_path, blobs_w, off_w)
    write_blob(s_path, blobs_s, off_s)

    errs = [v['quant_err'] for v in index.values()]
    manifest = {
        'model': os.path.basename(args.model.rstrip('/')),
        'config': cfg,
        'group': GROUP,
        'format': {'nibble': 'unsigned code+8', 'x': 'int8 signed, zero_point 0',
                   'perm': PERM},
        'bits_per_param': 4.25,
        'total_params': total_params,
        'weights_bytes': os.path.getsize(w_path),
        'scales_bytes': os.path.getsize(s_path),
        'effective_bits': round(8.0 * (os.path.getsize(w_path) +
                                       os.path.getsize(s_path)) / total_params, 3),
        'quant_err_mean': round(float(np.mean(errs)), 6),
        'quant_err_max': round(float(np.max(errs)), 6),
        'tensors': index,
    }
    with open(os.path.join(args.out, 'manifest.json'), 'w') as f:
        json.dump(manifest, f, indent=1)

    print('-' * 74)
    print(f'params            : {total_params/1e6:.1f} M')
    print(f'weights.bin       : {manifest["weights_bytes"]/1e6:.1f} MB')
    print(f'scales.bin        : {manifest["scales_bytes"]/1e6:.1f} MB')
    print(f'bits/param real   : {manifest["effective_bits"]:.3f}')
    print(f'error cuant. medio: {100*manifest["quant_err_mean"]:.3f} %   '
          f'max: {100*manifest["quant_err_max"]:.3f} %')

    # --- Set de 5 archivos para validar el kernel C++ con pesos REALES -------
    if args.layout_tensor:
        if args.layout_tensor not in index:
            raise SystemExit(f'no existe el tensor {args.layout_tensor}')
        meta = index[args.layout_tensor]
        N, K, G = meta['N'], meta['K'], meta['groups']
        print(f'\nemitiendo layout de 5 archivos para {args.layout_tensor}')

        with open(w_path, 'rb') as f:
            f.seek(meta['packed_off'])
            packed = np.frombuffer(f.read(meta['packed_bytes']), dtype=np.uint8)
        with open(s_path, 'rb') as f:
            f.seek(meta['scale_off'])
            scales = np.frombuffer(f.read(meta['scale_bytes']), dtype=np.float16)

        outd = args.layout_out or os.path.join(args.out, 'layout')
        os.makedirs(outd, exist_ok=True)

        codes = np.empty((N, K), dtype=np.int16)
        pb = packed.reshape(N, K // 2)
        codes[:, 0::2] = (pb & 0x0F).astype(np.int16) - 8
        codes[:, 1::2] = ((pb >> 4) & 0x0F).astype(np.int16) - 8
        sw = scales.reshape(N, G).astype(np.float32)

        # x sinteticos reproducibles (el kernel solo necesita el layout)
        rng = np.random.default_rng(7)
        xs, xsc = [], []
        for _ in range(args.layout_tokens):
            x = rng.standard_normal(K).astype(np.float32)
            xi, sx = quantize_x(x)
            xs.append(xi)
            xsc.append(sx)

        ref = np.zeros((len(xs), N), dtype=np.float32)
        for m, (xi, sx) in enumerate(zip(xs, xsc)):
            xsign = xi.astype(np.int32)
            for gi in range(G):
                sl = slice(gi * GROUP, (gi + 1) * GROUP)
                # deshacer la pre-permutacion para la referencia escalar
                xnat = np.empty(GROUP, dtype=np.int32)
                for i, p in enumerate(PERM):
                    xnat[p] = xsign[sl][i]
                dot = codes[:, sl].astype(np.int64) @ xnat.astype(np.int64)
                ref[m] += (dot * sw[:, gi].astype(np.float64) * sx).astype(np.float32)

        open(os.path.join(outd, 'w_packed.bin'), 'wb').write(packed.tobytes())
        open(os.path.join(outd, 'w_scale.bin'), 'wb').write(scales.tobytes())
        open(os.path.join(outd, 'x_perm.bin'), 'wb').write(
            np.concatenate(xs).tobytes())
        open(os.path.join(outd, 'x_scale.bin'), 'wb').write(
            np.array(xsc, dtype=np.float32).tobytes())
        open(os.path.join(outd, 'ref.bin'), 'wb').write(ref.tobytes())
        json.dump({'N': N, 'K': K, 'M': len(xs), 'group': GROUP,
                   'bits_per_param': 4.25,
                   'note': 'pesos REALES de Qwen2.5; ref en float64'},
                  open(os.path.join(outd, 'manifest.json'), 'w'), indent=1)
        print(f'  -> {outd}  (N={N} K={K} M={len(xs)})')


if __name__ == '__main__':
    sys.exit(main())
