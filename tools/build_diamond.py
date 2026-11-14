#!/usr/bin/env python3
"""
build_diamond.py — Generador del Cristal Cúbico 3D para Modelos Caudal / Aquifer.
Descompone espectralmente las matrices MLP (gate_proj) de cada capa en 3 ejes ópticos
y proyecta todas las neuronas dentro del hipercubo [-1, 1]³.
Genera un artefacto ultraliviano 'diamond.bin' (~1.7 MB a ~5.9 MB).
"""

import argparse
import os
import struct
import numpy as np

def load_matrix_affine(model_dir, tensor_name):
    index_path = os.path.join(model_dir, "index.txt")
    weights_path = os.path.join(model_dir, "weights.bin")
    scales_path = os.path.join(model_dir, "scales.bin")

    found = None
    with open(index_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 8 and parts[1] == tensor_name:
                found = parts
                break
    if not found:
        return None

    N, K = int(found[2]), int(found[3])
    w_off, w_bytes = int(found[4]), int(found[5])
    s_off, m_off = int(found[6]), int(found[7])

    with open(weights_path, "rb") as f:
        f.seek(w_off)
        raw_w = f.read(w_bytes)

    # Determinar grupo desde manifest o K/G
    group = 64
    G = K // group

    with open(scales_path, "rb") as f:
        f.seek(s_off)
        raw_s = f.read(N * G * 2)
        f.seek(m_off)
        raw_m = f.read(N * G * 2)

    w_u8 = np.frombuffer(raw_w, dtype=np.uint8).reshape(N, K // 2)
    s_f16 = np.frombuffer(raw_s, dtype=np.float16).reshape(N, G).astype(np.float32)
    m_f16 = np.frombuffer(raw_m, dtype=np.float16).reshape(N, G).astype(np.float32)

    lo = (w_u8 & 0x0F).astype(np.float32)
    hi = ((w_u8 >> 4) & 0x0F).astype(np.float32)
    q = np.empty((N, K), dtype=np.float32)
    q[:, 0::2] = lo
    q[:, 1::2] = hi

    W = np.empty((N, K), dtype=np.float32)
    for g in range(G):
        c0, c1 = g * group, (g + 1) * group
        W[:, c0:c1] = q[:, c0:c1] * s_f16[:, g:g+1] + m_f16[:, g:g+1]

    return W

def main():
    ap = argparse.ArgumentParser(description="Construye el índice óptico 3D (diamond.bin) para un modelo cuantizado")
    ap.add_argument("--model", required=True, help="Ruta al artefacto del modelo (ej. build/qwen05_affine)")
    ap.add_argument("--out", default=None, help="Ruta de salida para diamond.bin (por defecto dentro del modelo)")
    args = ap.parse_args()

    model_dir = args.model
    out_path = args.out if args.out else os.path.join(model_dir, "diamond.bin")

    print(f"Construyendo Cristal Cúbico 3D para {model_dir}...")
    
    # Leer hiperparámetros desde model.txt
    model_txt = os.path.join(model_dir, "model.txt")
    cfg = {}
    with open(model_txt, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 2:
                cfg[parts[0]] = int(parts[1]) if parts[1].isdigit() else parts[1]

    NLY = int(cfg["NLY"])
    H = int(cfg["H"])
    INTER = int(cfg["INTER"])
    print(f"Modelo: Capas={NLY}, H={H}, INTER={INTER}")

    out_file = open(out_path, "wb")
    # Cabecera mágica: "DIAMOND3", NLY (u32), H (u32), INTER (u32)
    magic = b"DIAMOND3"
    out_file.write(magic)
    out_file.write(struct.pack("<III", NLY, H, INTER))

    total_bytes = 8 + 12

    for L in range(NLY):
        name = f"model.layers.{L}.mlp.gate_proj.weight"
        print(f"  Proyectando Capa {L:2d}/{NLY-1}: {name}...", end="", flush=True)
        W = load_matrix_affine(model_dir, name)
        if W is None:
            raise RuntimeError(f"No se pudo cargar {name}")
        
        N, K = W.shape
        W_mean = np.mean(W, axis=0, keepdims=True)
        W_centered = W - W_mean

        # Covarianza K x K
        cov = (W_centered.T @ W_centered) / N
        eigenvals, eigenvecs = np.linalg.eigh(cov)
        top3_idx = np.argsort(eigenvals)[::-1][:3]
        V3 = eigenvecs[:, top3_idx].astype(np.float32)  # Shape: (K, 3)

        # Proyección de las N neuronas en el Cubo 3D
        P3 = (W_centered @ V3).astype(np.float32)       # Shape: (N, 3)
        # Normalizar al cubo unitario [-1, 1]³
        p_max = np.max(np.abs(P3), axis=0, keepdims=True)
        P3_norm = P3 / (p_max + 1e-9)

        # Escribir en diamond.bin:
        # V3: 3 * K floats (3 ejes directores de luz)
        # W_mean: K floats
        # P3_norm: N * 3 floats (coordenadas de neuronas en [-1, 1]³)
        b_v3 = V3.T.tobytes() # (3, K)
        b_mean = W_mean.astype(np.float32).tobytes() # (K,)
        b_p3 = P3_norm.tobytes() # (N, 3)

        out_file.write(b_v3)
        out_file.write(b_mean)
        out_file.write(b_p3)

        layer_bytes = len(b_v3) + len(b_mean) + len(b_p3)
        total_bytes += layer_bytes
        print(f" ✓ ({layer_bytes / 1024:.1f} KB)")

    out_file.close()
    print(f"\n¡Cristal 3D generado con éxito en: {out_path}!")
    print(f"Tamaño total de diamond.bin: {total_bytes / (1024 * 1024):.2f} MB")

if __name__ == "__main__":
    main()
