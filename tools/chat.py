#!/usr/bin/env python3
"""chat.py — Shell interactiva para conversar con Qwen cuantizado en Caudal.

Uso:
    python3 tools/chat.py
    python3 tools/chat.py --threads 4 --n 256
"""

import argparse
import os
import readline  # habilita flechas arriba/abajo e historial en input()
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from oracle_qwen import Tokenizer


# Colores ANSI
CYAN = "\033[1;36m"
GREEN = "\033[1;32m"
YELLOW = "\033[1;33m"
BOLD = "\033[1m"
RESET = "\033[0m"


def format_chatml(history, system_prompt=None):
    """Genera la secuencia ChatML completa para Qwen."""
    parts = []
    if system_prompt:
        parts.append(f"<|im_start|>system\n{system_prompt}<|im_end|>\n")
    for role, text in history:
        parts.append(f"<|im_start|>{role}\n{text}<|im_end|>\n")
    parts.append("<|im_start|>assistant\n")
    return "".join(parts)


def main():
    repo_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    default_bin = os.path.join(repo_dir, "build", "generate_qwen")
    default_art = os.path.join(repo_dir, "build", "qwen15_affine")
    default_tok = "/home/marco/.cache/caudal/qwen15/tokenizer.json"

    ap = argparse.ArgumentParser(description="Chat interactivo con Qwen en Caudal")
    ap.add_argument("--bin", default=default_bin, help="Ruta al binario generate_qwen")
    ap.add_argument("--artifact", default=default_art, help="Ruta al artefacto cuantizado")
    ap.add_argument("--tokenizer", default=default_tok, help="Ruta a tokenizer.json")
    ap.add_argument("--threads", type=int, default=4, help="Número de hilos AVX2")
    ap.add_argument("--n", type=int, default=256, help="Tokens máximos generados por turno")
    ap.add_argument("--ctx", type=int, default=1024, help="Ventana de contexto")
    ap.add_argument("--system", default="Eres Qwen, un asistente inteligente y servicial. Responde siempre en español.",
                    help="Prompt de sistema")
    args = ap.parse_args()

    if not os.path.exists(args.bin):
        print(f"Error: Binario no encontrado en {args.bin}. Compílalo primero con make.")
        return 1
    if not os.path.exists(args.artifact):
        print(f"Error: Artefacto no encontrado en {args.artifact}.")
        return 1
    if not os.path.exists(args.tokenizer):
        print(f"Error: Tokenizer no encontrado en {args.tokenizer}.")
        return 1

    print(f"{BOLD}Iniciando Caudal Engine (Qwen 1.5B Afín 5.0 bpw)...{RESET}")
    tok = Tokenizer(args.tokenizer)

    cmd = [
        args.bin,
        args.artifact,
        "--interactive",
        "--threads", str(args.threads),
        "--n", str(args.n),
        "--ctx", str(args.ctx)
    ]

    proc = subprocess.Popen(
        cmd,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        text=True,
        bufsize=1
    )

    # Esperar confirmación READY de Caudal
    for line in proc.stdout:
        line_clean = line.strip()
        if line_clean == "READY":
            break
        elif "THP" in line_clean:
            print(f"  {line_clean}")

    print(f"\n{BOLD}================================================================={RESET}")
    print(f"{BOLD}      Caudal Shell Interactiva · Qwen 1.5B (17.4 tok/s)         {RESET}")
    print(f"  Comandos especiales:                                           ")
    print(f"    {YELLOW}/reset{RESET}  -> Reiniciar memoria de la conversación           ")
    print(f"    {YELLOW}/salir{RESET}  -> Salir de la shell                             ")
    print(f"{BOLD}================================================================={RESET}\n")

    history = []

    try:
        while True:
            try:
                user_input = input(f"{GREEN}{BOLD}Tú > {RESET}").strip()
            except (EOFError, KeyboardInterrupt):
                print("\nSaliendo...")
                break

            if not user_input:
                continue

            if user_input.lower() in ("/salir", "/exit", "exit", "quit"):
                break
            elif user_input.lower() in ("/reset", "/limpiar", "clear"):
                history.clear()
                print(f"{YELLOW}[Memoria de conversación reiniciada]{RESET}\n")
                continue

            history.append(("user", user_input))
            full_prompt = format_chatml(history, args.system)
            tokens = tok.encode(full_prompt)

            if len(tokens) >= args.ctx - args.n:
                print(f"{YELLOW}[Aviso: Contexto largo ({len(tokens)} tokens), recortando historia]{RESET}")
                if len(history) > 2:
                    history = history[-2:]
                    full_prompt = format_chatml(history, args.system)
                    tokens = tok.encode(full_prompt)

            tokens_str = ",".join(map(str, tokens))
            proc.stdin.write(f"PROMPT {tokens_str}\n")
            proc.stdin.flush()

            sys.stdout.write(f"{CYAN}{BOLD}Qwen > {RESET}")
            sys.stdout.flush()

            resp_tokens = []
            for line in proc.stdout:
                line_str = line.strip()
                if line_str == "DONE":
                    break
                if line_str.startswith("TOKEN "):
                    t_id = int(line_str.split()[1])
                    resp_tokens.append(t_id)
                    token_text = tok.decode([t_id])
                    # No imprimir etiquetas especiales
                    if "<|im_end|>" in token_text or "<|endoftext|>" in token_text:
                        continue
                    sys.stdout.write(token_text)
                    sys.stdout.flush()

            sys.stdout.write("\n\n")
            sys.stdout.flush()

            assistant_text = tok.decode(resp_tokens).replace("<|im_end|>", "").strip()
            history.append(("assistant", assistant_text))

    finally:
        try:
            proc.stdin.write("QUIT\n")
            proc.stdin.flush()
            proc.wait(timeout=2)
        except Exception:
            proc.kill()


if __name__ == "__main__":
    sys.exit(main())
