#!/usr/bin/env python3
# =============================================================================
#  caudal · tools/ppl_qwen.py
#  PERPLEJIDAD con teacher forcing: la métrica que decide si la cuantización
#  sirve. Dos párrafos de texto NO son una medición.
# =============================================================================
#  Compara bf16 contra int4 a distintos tamaños de grupo usando el MISMO
#  artefacto lógico (cuantiza-decuantiza en memoria), así el único eje que
#  cambia es el grupo.
#
#  Uso:
#    python3 tools/ppl_qwen.py --model DIR [--groups 64,32,16] [--text "..."]
# =============================================================================

import argparse
import json
import math
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from oracle_qwen import Tokenizer, Qwen  # noqa: E402

TEXTO_POR_DEFECTO = (
    "La inteligencia artificial es una disciplina que estudia cómo construir "
    "sistemas capaces de realizar tareas que normalmente requieren inteligencia "
    "humana, como el reconocimiento de imágenes, la traducción automática y la "
    "toma de decisiones. Su desarrollo ha avanzado rápidamente en las últimas "
    "décadas gracias al aumento de la capacidad de cómputo y a la disponibilidad "
    "de grandes cantidades de datos.")


def ppl(m, ids):
    with torch.no_grad():
        _, logits, _ = m.forward(ids)
    nll, n = 0.0, 0
    for t in range(len(ids) - 1):
        lp = torch.log_softmax(logits[t].float(), -1)
        nll += -lp[ids[t + 1]].item()
        n += 1
    return math.exp(nll / n)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', required=True)
    ap.add_argument('--groups', default='64,32')
    ap.add_argument('--text', default=TEXTO_POR_DEFECTO)
    args = ap.parse_args()

    cfg = json.load(open(os.path.join(args.model, 'config.json')))
    tok = Tokenizer(os.path.join(args.model, 'tokenizer.json'))
    ids = tok.encode(args.text, allow_special=False)
    print(f'modelo : {args.model}')
    print(f'texto  : {len(ids)} tokens\n')

    m = Qwen(args.model, cfg)
    m.lowmem = True          # streaming: un tensor a la vez, sin cache

    base = ppl(m, ids)
    print(f'{"bf16":<12} {"16.00":>7} bits   ppl {base:8.3f}   (referencia)')

    for g in [int(x) for x in args.groups.split(',')]:
        m.quant = True
        m.qgroup = g
        p = ppl(m, ids)
        bits = 4.0 + 16.0 / g
        print(f'{"int4 g" + str(g):<12} {bits:7.2f} bits   ppl {p:8.3f}   '
              f'({100.0 * (p / base - 1.0):+.1f} %)')


if __name__ == '__main__':
    main()
