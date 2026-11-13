#!/usr/bin/env python3
# =============================================================================
#  caudal · tools/probe_projection.py
#  TEST 1: ¿la proyección 1536D -> 3D es señal o estética geométrica?
# =============================================================================
#  Contrato declarado ANTES de medir:
#
#    Y = (x, y, z) del concepto en semantic.db, en [-1,1]^3.
#    X = embedding de Qwen2.5-1.5B para la MISMA palabra.
#    Modelo  = ridge 1536 -> 3 sobre X estandarizada, split 80/20 por palabra.
#    Métrica = R² por eje en el HOLD-OUT (nunca en train).
#    Control = las mismas X con Y barajada. Si el control también sube, el
#              pipeline filtra información y el número no vale.
#
#  Dos contratos de "vector de la palabra", porque la elección mueve el R² más
#  que el algoritmo:
#    estricto : la palabra es 1 token  -> sin decisión de pooling
#    pooled   : media de los embeddings de sus tokens -> más cobertura
#
#  Umbrales fijados de antemano (del consejo):
#    R² >= 0.5  -> señal utilizable      R² < 0.2 -> ruido con estética
#
#  Uso: python3 tools/probe_projection.py [--n 12000] [--pooled 30000]
# =============================================================================

import argparse
import json
import os
import sqlite3
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from oracle_qwen import Tokenizer  # noqa: E402

DB = '/home/marco/source/repos/sg/semantic.db'
MODEL = '/home/marco/.cache/caudal/qwen15'
ART = '/home/marco/.cache/caudal/qwen15_g32'

HIDDEN = 0


def embed_matrix():
    V = HIDDEN
    raw = os.path.getsize(os.path.join(ART, 'embed.bf16')) // (2 * V)
    return np.memmap(os.path.join(ART, 'embed.bf16'), dtype=np.uint16,
                     mode='r', shape=(raw, V))


def rows_of(emb, ids):
    out = (np.asarray(emb[np.asarray(ids, dtype=np.int64)]).astype(np.uint32) << 16)
    return out.view(np.float32).astype(np.float64)


def load_candidates(limit=200000):
    con = sqlite3.connect(f'file:{DB}?mode=ro', uri=True)
    cur = con.execute(
        "SELECT word, x, y, z FROM concepts "
        "WHERE lang='es' AND x IS NOT NULL AND y IS NOT NULL AND z IS NOT NULL "
        "  AND word NOT LIKE '% %' AND length(word) >= 3 AND length(word) <= 18 "
        "ORDER BY RANDOM() LIMIT ?", (limit,))
    words, coords, seen = [], [], set()
    for w, x, y, z in cur:
        lw = w.lower()
        if lw in seen or not lw.isalpha():
            continue
        seen.add(lw)
        words.append(lw)
        coords.append((x, y, z))
    con.close()
    return words, np.array(coords, dtype=np.float64)


def r2_per_axis(Yt, Yp):
    ss_res = ((Yt - Yp) ** 2).sum(axis=0)
    ss_tot = ((Yt - Yt.mean(axis=0)) ** 2).sum(axis=0)
    return 1.0 - ss_res / ss_tot


def fit_eval(X, Y, lam, seed=2026):
    N = len(X)
    perm = np.random.default_rng(seed).permutation(N)
    ntr = int(0.8 * N)
    tr, te = perm[:ntr], perm[ntr:]

    mu = X[tr].mean(axis=0)
    sd = X[tr].std(axis=0) + 1e-9
    Xtr = (X[tr] - mu) / sd
    Xte = (X[te] - mu) / sd

    def ridge(Ytr):
        A = Xtr.T @ Xtr + lam * np.eye(Xtr.shape[1])
        return np.linalg.solve(A, Xtr.T @ Ytr)

    W = ridge(Y[tr])
    r2_tr = r2_per_axis(Y[tr], Xtr @ W)      # in-sample: detecta sobreajuste
    r2 = r2_per_axis(Y[te], Xte @ W)

    Ys = Y[np.random.default_rng(7).permutation(N)]
    Ws = ridge(Ys[tr])
    r2s = r2_per_axis(Ys[te], Xte @ Ws)
    return r2, r2s, (len(tr), len(te)), r2_tr


def report(tag, r2, r2s, ntr, nte, pca=None, r2_tr=None):
    print(f'\n--- {tag}  (train {ntr} / test {nte}) ---')
    if r2_tr is not None:
        print(f'  in-sample    R2 = {r2_tr[0]:+.4f}  {r2_tr[1]:+.4f}  {r2_tr[2]:+.4f}')
    print(f'  ridge        R2 = {r2[0]:+.4f}  {r2[1]:+.4f}  {r2[2]:+.4f}')
    print(f'  control      R2 = {r2s[0]:+.4f}  {r2s[1]:+.4f}  {r2s[2]:+.4f}   (Y barajada)')
    if pca is not None:
        print(f'  varianza de X en 3 PCs = {100*pca:.2f} %')
    return r2.max()


def main():
    global HIDDEN
    ap = argparse.ArgumentParser()
    ap.add_argument('--n', type=int, default=12000, help='max anclas estrictas')
    ap.add_argument('--pooled', type=int, default=30000, help='max anclas pooled')
    ap.add_argument('--lam', type=float, default=1.0)
    ap.add_argument('--cand', type=int, default=200000)
    args = ap.parse_args()

    cfg = json.load(open(os.path.join(MODEL, 'config.json')))
    HIDDEN = cfg['hidden_size']
    tok = Tokenizer(os.path.join(MODEL, 'tokenizer.json'))
    print(f'modelo : Qwen2.5-1.5B (d={HIDDEN})')
    print(f'anclas : semantic.db, conceptos es\n')

    words, Y = load_candidates(args.cand)
    print(f'candidatas aleatorias: {len(words)}')

    emb = embed_matrix()

    s_ids, s_y = [], []
    p_flat, p_off, p_y = [], [0], []
    for w, c in zip(words, Y):
        ids = tok.encode(' ' + w, allow_special=False)
        if len(ids) == 1 and len(s_ids) < args.n:
            s_ids.append(ids[0]); s_y.append(c)
        if len(ids) <= 8 and len(p_y) < args.pooled:
            p_flat.extend(ids); p_off.append(len(p_flat)); p_y.append(c)

    print(f'  estrictas (1 token)  : {len(s_ids)}  ({100*len(s_ids)/len(words):.2f} % de las candidatas)')
    print(f'  pooled   (<=8 tokens): {len(p_y)}')
    print('\nSESGO DECLARADO: las palabras 1-token son las FRECUENTES (BPE fusiona')
    print('secuencias comunes). El set estricto esta sesgado hacia lo comun; el')
    print('pooled cubre mas pero impone una decision de pooling.')

    best = -9.0

    if len(s_ids) >= 500:
        Xs = rows_of(emb, s_ids)
        Ys = np.array(s_y, dtype=np.float64)
        for i, ax in enumerate('xyz'):
            print(f'  {ax}: media {Ys[:,i].mean():+.3f} std {Ys[:,i].std():.3f}')
        Xc = Xs - Xs.mean(axis=0)
        sv = np.linalg.svd(Xc, compute_uv=False)
        pca = (sv[:3] ** 2).sum() / (sv ** 2).sum()
        del Xc
        r2, r2s, (ntr, nte), r2tr = fit_eval(Xs, Ys, args.lam)
        best = max(best, report('ESTRICTO (1 token)', r2, r2s, ntr, nte, pca, r2tr))
        del Xs, Ys

    if len(p_y) >= 500:
        sumv = np.zeros((len(p_y), HIDDEN))
        cnt = np.zeros(len(p_y))
        flat = rows_of(emb, p_flat)
        for i in range(len(p_y)):
            a, b = p_off[i], p_off[i + 1]
            sumv[i] = flat[a:b].sum(axis=0)
            cnt[i] = b - a
        Xp = sumv / cnt[:, None]
        del sumv, flat
        Yp = np.array(p_y, dtype=np.float64)
        r2, r2s, (ntr, nte), r2tr = fit_eval(Xp, Yp, args.lam)
        best = max(best, report('POOLED (media de tokens)', r2, r2s, ntr, nte, None, r2tr))
        del Xp, Yp

    # --- CONTROL POSITIVO ----------------------------------------------------
    # Si el pipeline estuviera roto, todo daria ~0. Se predice dim 0 del
    # embedding (escalar) desde las otras 1535 con el MISMO ridge. Tiene que
    # dar R2 alto: los embeddings son redundantes.
    if len(p_y) >= 500:
        Cp = rows_of(emb, p_flat)
        sumv = np.zeros((len(p_y), HIDDEN)); cnt = np.zeros(len(p_y))
        for i in range(len(p_y)):
            a, b = p_off[i], p_off[i + 1]
            sumv[i] = Cp[a:b].sum(axis=0); cnt[i] = b - a
        Xp = sumv / cnt[:, None]
        tgt = Xp[:, :1].copy()
        Xo = np.delete(Xp, 0, axis=1)
        r2p, _, nn, r2ptr = fit_eval(Xo, tgt, args.lam)
        print(f'\n--- CONTROL POSITIVO (predecir dim 0 del embedding desde el resto) ---')
        print(f'  in-sample R2 = {r2ptr[0]:+.4f}   hold-out R2 = {r2p[0]:+.4f}')
        print('  (si esto no da alto, el pipeline esta roto y el resto no vale)')

    print(f'\n{"-"*60}')
    print(f'R2 maximo observado = {best:+.4f}')
    if best >= 0.5:
        print('VEREDICTO: SENAL utilizable en al menos un eje')
    elif best >= 0.2:
        print('VEREDICTO: senal DEBIL (entre 0.2 y 0.5)')
    else:
        print('VEREDICTO: RUIDO con estetica geometrica')
    return 0


if __name__ == '__main__':
    sys.exit(main())
