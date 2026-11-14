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

GROUP = 64   # se puede sobreescribir con --group
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
def quantize(w, group=GROUP):
    """w: float32 [N, K] -> (packed uint8 [N, K/2], scales fp16 [N, G], err)"""
    N, K = w.shape
    GROUP = group
    if K % GROUP:
        raise SystemExit(f'K={K} no es multiplo de {GROUP}')
    G = K // GROUP

    g = w.reshape(N, G, GROUP)
    amax = np.abs(g).max(axis=2)
    # OJO: clampear en fp32 antes del cast NO alcanza. fp16 no representa 1e-8
    # (subnormal mínimo ~6e-8): 1e-8 -> 0 -> división por cero -> nan.
    # Hay grupos con amax==0 (canales muertos), así que el clamp va en dominio
    # fp16 y en el mínimo NORMAL, que sobrevive seguro.
    MIN_F16 = np.float16(6.103515625e-05)      # 2^-14, menor normal de fp16
    sw16 = np.maximum((amax / 7.0).astype(np.float16), MIN_F16)
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
    ap.add_argument('--stride-align', type=int, default=64,
                    help='alineacion del stride en layout inter (0 = exacto, sin '
                         'huecos: era el agujero que hundio el intercalado)')
    ap.add_argument('--layout', default='split', choices=['split', 'inter'],
                    help='split = weights.bin+scales.bin (MEDIDO mas rapido: 22.2 '
                         'tok/s); inter = w.bin con filas [pesos][escalas] (18.0)')
    ap.add_argument('--group', type=int, default=64, choices=[16, 32, 64],
                    help='valores por escala (64 = 4.25 bits, 32 = 4.75 bits)')
    args = ap.parse_args()
    global GROUP
    GROUP = args.group

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

    # lm_head atado: si no hay lm_head.weight, el GEMV final usa embed_tokens
    if 'lm_head.weight' not in st.header and 'model.embed_tokens.weight' in st.header:
        e = st.header['model.embed_tokens.weight']
        if len(e['shape']) == 2:
            todo.append(('model.embed_tokens.weight', tuple(e['shape'])))
            todo.sort()
            print('  (lm_head atado -> se empaqueta embed_tokens como lineal)')

    total_bits = 4.0 + 16.0 / GROUP
    print(f'modelo : {args.model}   (grupo {GROUP} -> {total_bits:.2f} bits/param)')
    print(f'tensores lineales: {len(todo)}')
    print(f'{"nombre":<44}{"N":>6}{"K":>6}{"bits":>7}{"err%":>9}')
    print('-' * 74)

    os.makedirs(args.out, exist_ok=True)
    index = {}
    total_params = 0

    # ── Layout INTERCALADO: cada fila = [pesos K/2][escalas G*2] ────────────
    # Medido en probe3_ceiling: 98.7 % del techo continuo contra 90.4 % del
    # layout de dos arrays separados. El prefetcher sigue UN stream por hilo.
    splitted = (args.layout == 'split')
    w_path = os.path.join(args.out, 'weights.bin' if splitted else 'w.bin')
    s_path = os.path.join(args.out, 'scales.bin')
    fw = open(w_path, 'wb')
    fs = open(s_path, 'wb') if splitted else None
    try:
        for name, (N, K) in todo:
            w = st.tensor_f32(name)
            packed, scales, err = quantize(w, GROUP)
            total_params += N * K

            rb, sbb = K // 2, (K // GROUP) * 2
            if splitted:
                ow = align(fw.tell()); fw.write(b'\0' * (ow - fw.tell()))
                fw.write(packed.tobytes())
                os_ = align(fs.tell()); fs.write(b'\0' * (os_ - fs.tell()))
                fs.write(scales.tobytes())
                index[name] = {'N': int(N), 'K': int(K), 'groups': int(K // GROUP),
                               'base_off': ow, 'stride': 0, 'scale_off': os_,
                               'packed_bytes': int(packed.size),
                               'scale_bytes': int(scales.size * 2),
                               'quant_err': round(err, 6)}
            else:
                if args.stride_align > 0:
                    a_ = args.stride_align
                    stride = (rb + sbb + a_ - 1) & ~(a_ - 1)
                else:
                    stride = rb + sbb
                rows = np.empty((N, stride), dtype=np.uint8)
                rows[:, :rb] = packed
                rows[:, rb:rb + sbb] = scales.view(np.uint8).reshape(N, sbb)
                o = align(fw.tell()); fw.write(b'\0' * (o - fw.tell()))
                fw.write(rows.tobytes())
                index[name] = {'N': int(N), 'K': int(K), 'groups': int(K // GROUP),
                               'base_off': o, 'stride': stride,
                               'packed_bytes': int(packed.size),
                               'scale_bytes': int(scales.size * 2),
                               'quant_err': round(err, 6)}
            print(f'{name:<44}{N:>6}{K:>6}{total_bits:>7.2f}{100*err:>9.3f}')
    finally:
        fw.close()
        if fs: fs.close()

    errs = [v['quant_err'] for v in index.values()]
    wsize = os.path.getsize(w_path) + (os.path.getsize(s_path) if splitted else 0)
    manifest = {
        'model': os.path.basename(args.model.rstrip('/')),
        'config': cfg,
        'group': GROUP,
        'layout': 'interleaved: fila = [pesos K/2][escalas G*2]',
        'format': {'nibble': 'unsigned code+8', 'x': 'int8 signed, zero_point 0',
                   'perm': PERM},
        'bits_per_param': round(total_bits, 3),
        'total_params': total_params,
        'w_bytes': wsize,
        'effective_bits': round(8.0 * wsize / total_params, 3),
        'quant_err_mean': round(float(np.mean(errs)), 6),
        'quant_err_max': round(float(np.max(errs)), 6),
        'layout': ('split: weights.bin + scales.bin' if splitted else
                   'interleaved: fila = [pesos K/2][escalas G*2]'),
        'tensors': index,
    }
    with open(os.path.join(args.out, 'manifest.json'), 'w') as f:
        json.dump(manifest, f, indent=1)

    print('-' * 74)
    print(f'params            : {total_params/1e6:.1f} M')
    print(f'artefacto         : {manifest["w_bytes"]/1e6:.1f} MB  layout={manifest["layout"]}')
    print(f'bits/param real   : {manifest["effective_bits"]:.3f}')
    print(f'error cuant. medio: {100*manifest["quant_err_mean"]:.3f} %   '
          f'max: {100*manifest["quant_err_max"]:.3f} %')

    # --- Material para el forward en C++ (sin JSON ni safetensors) -----------
    # index.txt: LINEAL / AUX / EMBED, parseable con >>
    # embed.bf16: tabla de lookup (bf16 crudo, sin tocar)
    # aux.bin   : todos los tensores 1-D (normas y biases) en float32
    lines = []
    for name, m in index.items():
        if splitted:
            lines.append(f'LINEAR {name} {m["N"]} {m["K"]} {m["base_off"]} '
                         f'{m["packed_bytes"]} {m["scale_off"]} {m["scale_bytes"]}')
        else:
            lines.append(f'LINEAR {name} {m["N"]} {m["K"]} {m["base_off"]} '
                         f'{m["stride"]} {m["groups"]}')

    aux_names = [n for n in sorted(st.names()) if len(st.info(n)[1]) == 1]
    aux_off = 0
    with open(os.path.join(args.out, 'aux.bin'), 'wb') as fa:
        for n in aux_names:
            a = st.tensor_f32(n).astype(np.float32)
            fa.write(a.tobytes())
            lines.append(f'AUX {n} {a.size} {aux_off}')
            aux_off += a.size

    # embed_tokens crudo en bf16 (lookup exacto, no cuantizado)
    if 'model.embed_tokens.weight' in st.header:
        e = st.header['model.embed_tokens.weight']
        a, b = e['data_offsets']
        st.f.seek(st.data_start + a)
        with open(os.path.join(args.out, 'embed.bf16'), 'wb') as fe:
            fe.write(st.f.read(b - a))
        lines.append(f'EMBED model.embed_tokens.weight {e["shape"][0]} {e["shape"][1]}')

    # model.txt: hiperparametros en una linea, para que el C++ no parsee JSON
    if cfg:
        with open(os.path.join(args.out, 'model.txt'), 'w') as fm:
            for k, v in [('H', cfg.get('hidden_size')), ('NLY', cfg.get('num_hidden_layers')),
                         ('NH', cfg.get('num_attention_heads')), ('NKV', cfg.get('num_key_value_heads')),
                         ('INTER', cfg.get('intermediate_size')), ('V', cfg.get('vocab_size')),
                         ('EPS', cfg.get('rms_norm_eps')), ('THETA', cfg.get('rope_theta'))]:
                fm.write(f'{k} {v}\n')
            fm.write(f'DH {cfg["hidden_size"] // cfg["num_attention_heads"]}\n')

    with open(os.path.join(args.out, 'index.txt'), 'w') as fi:
        fi.write('\n'.join(lines) + '\n')
    print(f'index.txt         : {len(index)} lineales, {len(aux_names)} aux')

    # --- Set de 5 archivos para validar el kernel C++ con pesos REALES -------
    if args.layout_tensor:
        if args.layout_tensor not in index:
            raise SystemExit(f'no existe el tensor {args.layout_tensor}')
        meta = index[args.layout_tensor]
        N, K, G = meta['N'], meta['K'], meta['groups']
        print(f'\nemitiendo layout de 5 archivos para {args.layout_tensor}')

        stride = meta['stride']
        with open(w_path, 'rb') as f:
            f.seek(meta['base_off'])
            raw = np.frombuffer(f.read(meta['packed_bytes'] + meta['scale_bytes']),
                                dtype=np.uint8).reshape(N, stride)
        packed = raw[:, :meta['packed_bytes']].reshape(-1).copy()
        scales = raw[:, meta['packed_bytes']:].copy().view(np.float16).reshape(-1)

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
                   'bits_per_param': round(total_bits, 3),
                   'note': 'pesos REALES de Qwen2.5; ref en float64'},
                  open(os.path.join(outd, 'manifest.json'), 'w'), indent=1)
        print(f'  -> {outd}  (N={N} K={K} M={len(xs)})')


if __name__ == '__main__':
    sys.exit(main())
