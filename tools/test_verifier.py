#!/usr/bin/env python3
# =============================================================================
#  caudal · tools/test_verifier.py
#  TEST 2: ¿sirve el Juez sobre texto REALMENTE generado por caudal?
# =============================================================================
#  Pipeline completo:
#    1. caudal/generate_qwen produce texto con Qwen2.5-1.5B int4 a ~21 tok/s.
#    2. Se extraen claims (s, r, o) con LOS MISMOS relatores que SG reconoce.
#       No se inventa un extractor semantico: se mide cuantas oraciones del
#       texto generado son siquiera AFIRMACIONES VERIFICABLES.
#    3. Cada claim va a tools/sg.py verify (daemon, sub-ms).
#
#  Metricas que importan:
#    cobertura   = % de oraciones con claim extraible
#    definitivo  = % de claims con CONFIRMED/CONFIRMED_BY_PATH/REFUTED
#                  (el grafo respondio algo)
#    silencio    = % NO_EVIDENCE + NODE_ABSENT (el grafo calla)
#
#  Control anti-alucinacion: claims DELIBERADAMENTE FALSOS. Si el Juez no los
#  marca, no hay arbitro. Umbral del repo: >=99.5% en negativos.
#
#  Uso: python3 tools/test_verifier.py [--gen 60]
# =============================================================================

import argparse
import json
import os
import re
import subprocess
import sys

sys.path.insert(0, '/home/marco/source/repos/sg/tools')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from oracle_qwen import Tokenizer      # noqa: E402
from sg import SG                      # noqa: E402

CAUDAL = '/home/marco/source/repos/caudal/build/generate_qwen'
ART = '/home/marco/.cache/caudal/qwen15_g32'
MODEL = '/home/marco/.cache/caudal/qwen15'

# Relatores EXACTOS que reconoce parse_claim() de SG.
REL = ('es parte de', 'son parte de', 'is part of', 'es', 'son', 'are', 'is')
REL_MULTI = [r for r in REL if ' ' in r]
REL_SINGLE = [r for r in REL if ' ' not in r]

ARTICLES = ('el ', 'la ', 'los ', 'las ', 'un ', 'una ', 'unos ', 'unas ',
            'the ', 'a ', 'an ', 'lo ')
# Palabras que NO pueden cerrar un claim: si el objeto termina en una de estas,
# la ventana de 4 palabras corto la frase y el claim queda truncado. Recortarlas
# es obligatorio para no confirmar hiperonimos por accidente.
STOP = {'que', 'de', 'del', 'la', 'el', 'los', 'las', 'en', 'se', 'y', 'o',
        'con', 'por', 'para', 'un', 'una', 'su', 'al', 'es', 'son',
        'the', 'of', 'a', 'an', 'and', 'or', 'to', 'in', 'that', 'which',
        'it', 'is', 'are', 'as', 'from', 'by', 'with', 'their'}


def trim_stop(s):
    w = s.split()
    while w and w[-1].lower().strip('.,;:') in STOP:
        w.pop()
    return ' '.join(w)


def strip_art(s):
    s = s.strip().strip('.,;:¡!¿?"\'()[]')
    low = s.lower()
    for a in ARTICLES:
        if low.startswith(a):
            return s[len(a):].strip()
    return s


def split_sentences(text):
    parts = re.split(r'[.!?;:\n]+', text)
    return [p.strip() for p in parts if len(p.strip().split()) >= 3]


def extract_one(s):
    """Busca el primer relator DENTRO de la oracion y toma una ventana a cada
    lado. Exigir que la oracion entera sea 'sujeto + relator + objeto' descarta
    casi todo el texto real, que es lo que hacia la version anterior."""
    w = s.split()
    low = [x.lower().strip('.,;:¡!¿?"\'()[]') for x in w]
    for i in range(len(w)):
        for r in REL_MULTI:
            parts = r.split()
            if low[i:i + len(parts)] == parts:
                subj = trim_stop(strip_art(' '.join(w[max(0, i - 4):i])))
                obj = trim_stop(strip_art(' '.join(w[i + len(parts):i + len(parts) + 4])))
                if subj and obj:
                    return f'{subj} {r} {obj}'
        if low[i] in REL_SINGLE:
            subj = trim_stop(strip_art(' '.join(w[max(0, i - 4):i])))
            obj = trim_stop(strip_art(' '.join(w[i + 1:i + 5])))
            if subj and obj:
                return f'{subj} {low[i]} {obj}'
    return None


def extract(sents):
    out = []
    for s in sents:
        c = extract_one(s)
        if c:
            out.append(c)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--gen', type=int, default=60)
    args = ap.parse_args()

    cfg = json.load(open(os.path.join(MODEL, 'config.json')))
    tok = Tokenizer(os.path.join(MODEL, 'tokenizer.json'))

    prompts = [
        '¿Qué es el agua?',
        'Explícame qué es un mamífero.',
        '¿Qué es el derecho del trabajo?',
        'What is a cell?',
        'Háblame de la fotosíntesis.',
        '¿Qué es la electricidad?',
    ]

    sg = SG()
    all_sents, all_claims, all_text = [], [], []

    for p in prompts:
        text = f'<|im_start|>user\n{p}<|im_end|>\n<|im_start|>assistant\n'
        ids = tok.encode(text)
        r = subprocess.run([CAUDAL, ART, *map(str, ids), '--n', str(args.gen),
                            '--threads', '4'],
                           capture_output=True, text=True, timeout=1800)
        line = [l for l in r.stdout.splitlines() if l.startswith('IDS ')]
        if not line:
            print('fallo generate_qwen:', r.stderr[-300:]); continue
        gen = [int(x) for x in line[0][4:].split()]
        out = tok.decode(gen)
        all_text.append((p, out))
        s = split_sentences(out)
        all_sents.extend(s)
        all_claims.extend(extract(s))

    print(f'\n{"="*64}\nTEXTO GENERADO\n{"="*64}')
    for p, o in all_text:
        print(f'\n[{p}]\n{o[:260]}')

    print(f'\n{"="*64}\nCOBERTURA\n{"="*64}')
    print(f'oraciones            : {len(all_sents)}')
    print(f'con claim extraible  : {len(all_claims)}  '
          f'({100*len(all_claims)/max(len(all_sents),1):.1f} %)')

    # --- verificación --------------------------------------------------------
    from collections import Counter
    verd = Counter()
    detail = []
    for c in all_claims:
        try:
            r = sg.verify(c)
            v = r.get('verdict', 'ERROR')
        except Exception as e:
            v = 'ERROR'
        verd[v] += 1
        detail.append((c, v))

    print(f'\n{"="*64}\nVEREDICTOS ({len(all_claims)} claims generados por Qwen)\n{"="*64}')
    for v, n in verd.most_common():
        print(f'  {v:<22} {n:4d}  ({100*n/max(len(all_claims),1):5.1f} %)')

    definitivo = verd['CONFIRMED'] + verd['CONFIRMED_BY_PATH'] + verd['REFUTED']
    silencio = verd['NO_EVIDENCE'] + verd['NODE_ABSENT']
    print(f'\n  definitivo (el grafo respondio): {definitivo}  '
          f'({100*definitivo/max(len(all_claims),1):.1f} %)')
    print(f'  silencio   (NO_EVIDENCE/ABSENT): {silencio}  '
          f'({100*silencio/max(len(all_claims),1):.1f} %)')

    print(f'\n{"="*64}\nCONTROL ANTI-ALUCINACION (claims falsos por construccion)\n{"="*64}')
    falsos = ['el gato es un planeta', 'el agua es un metal',
              'el sol es un mamifero', 'la luna es un pez',
              'el perro es una estrella', 'el hierro es un gas noble']
    mal = 0
    for f in falsos:
        try:
            r = sg.verify(f); v = r.get('verdict', 'ERROR')
        except Exception:
            v = 'ERROR'
        ok = v in ('NO_EVIDENCE', 'NODE_ABSENT', 'REFUTED')
        mal += 0 if ok else 1
        print(f'  [{v:<18}] {f}   {"OK" if ok else "<-- CONFIRMO UN FALSO"}')
    print(f'\n  falsos confirmados: {mal}/{len(falsos)}  '
          f'(exigido: 0)')

    print(f'\n{"="*64}')
    print('MUESTRA DE CLAIMS')
    print(f'{"="*64}')
    for c, v in detail[:14]:
        print(f'  [{v:<18}] {c}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
