#!/usr/bin/env python3
"""chat.py — Shell interactiva para conversar con modelos cuantizados en Caudal.

Soporta:
    - Qwen 2.5-1.5B Instruct          (--model qwen)
    - Qwen 2.5-0.5B Instruct (~52 t/s) (--model qwen05)
    - Qwen 2.5-Coder-1.5B (Código)    (--model coder)
    - DeepSeek-R1-Distill-1.5B        (--model r1)
    - TinyLlama-1.1B Chat             (--model tiny)
    - SmolLM2-1.7B Instruct           (--model smol2)

Uso:
    python3 tools/chat.py --model qwen05
    python3 tools/chat.py --model coder
    python3 tools/chat.py --model qwen
    python3 tools/chat.py --model r1
    python3 tools/chat.py --model tiny
    python3 tools/chat.py --model smol2
"""

import argparse
import os
import readline  # habilita flechas arriba/abajo e historial en input()
import subprocess
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from oracle_qwen import Tokenizer


# Colores ANSI
CYAN = "\033[1;36m"
GREEN = "\033[1;32m"
YELLOW = "\033[1;33m"
MAGENTA = "\033[1;35m"
BLUE = "\033[1;34m"
DIM = "\033[90m\033[3m"
BOLD = "\033[1m"
RESET = "\033[0m"


def format_prompt(history, model_type="qwen", system_prompt=None):
    """Genera la secuencia de prompt adecuada para el modelo."""
    if model_type == "r1":
        parts = []
        for role, text in history:
            if role == "user":
                parts.append(f"<｜User｜>{text}")
            elif role == "assistant":
                parts.append(f"<｜Assistant｜>{text}<｜end▁of▁sentence｜>")
        parts.append("<｜Assistant｜><think>\n")
        return "".join(parts)
    elif model_type == "tiny":
        parts = []
        if system_prompt:
            parts.append(f"<|system|>\n{system_prompt}</s>\n")
        for role, text in history:
            if role == "user":
                parts.append(f"<|user|>\n{text}</s>\n")
            elif role == "assistant":
                parts.append(f"<|assistant|>\n{text}</s>\n")
        parts.append("<|assistant|>\n")
        return "".join(parts)
    else:  # qwen, qwen05, coder y smol2 usan ChatML
        parts = []
        if system_prompt:
            parts.append(f"<|im_start|>system\n{system_prompt}<|im_end|>\n")
        for role, text in history:
            parts.append(f"<|im_start|>{role}\n{text}<|im_end|>\n")
        parts.append("<|im_start|>assistant\n")
        return "".join(parts)


def main():
    repo_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    default_bin = os.path.join(repo_dir, "build", "generate_aquifer")
    if not os.path.exists(default_bin):
        default_bin = os.path.join(repo_dir, "build", "generate_qwen")

    def resolve_artifact(model_name):
        local_art = os.path.join(repo_dir, "build", model_name)
        if os.path.exists(local_art):
            return local_art
        caudal_art = os.path.join("/home/marco/source/repos/caudal/build", model_name)
        if os.path.exists(caudal_art):
            return caudal_art
        return local_art

    ap = argparse.ArgumentParser(description="Chat interactivo en Aquifer Engine (Zero-Copy Paging, 5.00 bpw)")
    ap.add_argument("--model", choices=["qwen", "qwen05", "qwen3", "coder", "r1", "tiny", "smol2"], default="qwen",
                    help="Modelo: 'qwen' (1.5B), 'qwen05' (0.5B rápido), 'qwen3' (3B Instruct), 'coder' (1.5B Coder), 'r1' (R1 Distill), 'tiny' (1.1B), 'smol2' (1.7B)")
    ap.add_argument("--bin", default=default_bin, help="Ruta al binario generate_aquifer")
    ap.add_argument("--artifact", default=None, help="Ruta al artefacto cuantizado")
    ap.add_argument("--tokenizer", default=None, help="Ruta a tokenizer.json")
    ap.add_argument("--threads", type=int, default=4, help="Número de hilos AVX2")
    ap.add_argument("--n", type=int, default=None, help="Tokens máximos generados por turno")
    ap.add_argument("--ctx", type=int, default=1024, help="Ventana de contexto")
    ap.add_argument("--system", default=None, help="Prompt de sistema")
    ap.add_argument("--diamond", type=float, default=None, help="Porcentaje de volumen del haz óptico en el Diamante 3D (ej: 85)")
    args = ap.parse_args()

    if args.diamond is not None and args.bin == default_bin:
        diamond_bin = os.path.join(repo_dir, "build", "generate_aquifer_diamond")
        if os.path.exists(diamond_bin):
            args.bin = diamond_bin

    # Configuraciones específicas por modelo
    if args.model == "r1":
        if args.artifact is None:
            args.artifact = resolve_artifact("r1_affine")
        if args.tokenizer is None:
            args.tokenizer = "/home/marco/.cache/caudal/r1/tokenizer.json"
        if args.system is None:
            args.system = ""
        if args.n is None:
            args.n = 512
        model_title = "DeepSeek-R1-Distill-1.5B (Razonamiento)"
        bot_color = YELLOW
        bot_label = "DeepSeek-R1"
        eos_str = "151643"
    elif args.model == "qwen05":
        if args.artifact is None:
            args.artifact = resolve_artifact("qwen05_affine")
        if args.tokenizer is None:
            args.tokenizer = "/home/marco/.cache/caudal/qwen05/tokenizer.json"
        if args.system is None:
            args.system = "Eres un asistente servicial que responde siempre en español de forma concisa."
        if args.n is None:
            args.n = 256
        model_title = "Qwen 2.5-0.5B Instruct (Ultra-Rápido ~52 tok/s)"
        bot_color = BLUE
        bot_label = "Qwen-0.5B"
        eos_str = "151643,151645"
    elif args.model == "qwen3":
        if args.artifact is None:
            args.artifact = resolve_artifact("qwen3b_affine")
        if args.tokenizer is None:
            args.tokenizer = "/home/marco/.cache/caudal/qwen3b/tokenizer.json"
        if args.system is None:
            args.system = "Eres un asistente servicial que responde siempre en español de forma concisa."
        if args.n is None:
            args.n = 512
        model_title = "Qwen 2.5-3B Instruct (Aquifer Zero-Copy Paging)"
        bot_color = CYAN
        bot_label = "Qwen-3B"
        eos_str = "151643,151645"
    elif args.model == "coder":
        if args.artifact is None:
            args.artifact = resolve_artifact("coder15_affine")
        if args.tokenizer is None:
            args.tokenizer = "/home/marco/.cache/caudal/coder15/tokenizer.json"
        if args.system is None:
            args.system = "Eres un asistente experto en programación de software, arquitectura de sistemas y algoritmos. Genera código limpio, moderno, tipado y bien explicado."
        if args.n is None:
            args.n = 512
        model_title = "Qwen 2.5-Coder-1.5B Instruct (Código y Sistemas)"
        bot_color = GREEN
        bot_label = "Qwen-Coder"
        eos_str = "151643,151645"
    elif args.model == "tiny":
        if args.artifact is None:
            args.artifact = resolve_artifact("tiny_affine")
        if args.tokenizer is None:
            args.tokenizer = "/home/marco/.cache/caudal/tiny/tokenizer.json"
        if args.system is None:
            args.system = "Eres un asistente servicial que responde siempre en español."
        if args.n is None:
            args.n = 256
        model_title = "TinyLlama-1.1B Chat"
        bot_color = MAGENTA
        bot_label = "TinyLlama"
        eos_str = "2"
    elif args.model == "smol2":
        if args.artifact is None:
            args.artifact = resolve_artifact("smol2_affine")
        if args.tokenizer is None:
            args.tokenizer = "/home/marco/.cache/caudal/smol2/tokenizer.json"
        if args.system is None:
            args.system = "Eres un asistente servicial que responde siempre en español."
        if args.n is None:
            args.n = 256
        model_title = "SmolLM2-1.7B Instruct"
        bot_color = BLUE
        bot_label = "SmolLM2"
        eos_str = "0,2"
    else:  # qwen
        if args.artifact is None:
            args.artifact = resolve_artifact("qwen15_affine")
        if args.tokenizer is None:
            args.tokenizer = "/home/marco/.cache/caudal/qwen15/tokenizer.json"
        if args.system is None:
            args.system = "Eres un asistente servicial que responde siempre en español."
        if args.n is None:
            args.n = 256
        model_title = "Qwen 2.5-1.5B Instruct"
        bot_color = CYAN
        bot_label = "Qwen"
        eos_str = "151643,151645"

    if not os.path.exists(args.bin):
        print(f"Error: Binario no encontrado en {args.bin}. Compílalo primero.")
        return 1
    if not os.path.exists(args.artifact):
        print(f"Error: Artefacto no encontrado en {args.artifact}.")
        return 1
    if not os.path.exists(args.tokenizer):
        print(f"Error: Tokenizer no encontrado en {args.tokenizer}.")
        return 1

    print(f"{BOLD}Iniciando Aquifer Paging Engine ({model_title} Afín 5.0 bpw · Zero-Copy HiperRAM)...{RESET}")
    tok = Tokenizer(args.tokenizer)

    cmd = [
        args.bin,
        args.artifact,
        "--interactive",
        "--threads", str(args.threads),
        "--n", str(args.n),
        "--ctx", str(args.ctx),
        "--eos", eos_str
    ]
    if args.diamond is not None:
        cmd.extend(["--diamond", str(args.diamond)])

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

    # Pre-calentar system prompt en el KV-cache para que el turno 1 responda de inmediato
    sys_tokens_str = ""
    if args.system and args.model in ("qwen", "qwen05", "coder", "tiny", "smol2"):
        if args.model == "tiny":
            sys_str = f"<|system|>\n{args.system}</s>\n"
        else:
            sys_str = f"<|im_start|>system\n{args.system}<|im_end|>\n"
        sys_tokens = tok.encode(sys_str)
        sys_tokens_str = ",".join(map(str, sys_tokens))
        proc.stdin.write(f"PREWARM {sys_tokens_str}\n")
        proc.stdin.flush()
        for line in proc.stdout:
            if line.strip() == "PREWARMED":
                break

    print(f"\n{BOLD}================================================================={RESET}")
    print(f"{BOLD}      Caudal Shell Interactiva · {model_title}                  {RESET}")
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
                proc.stdin.write("RESET\n")
                proc.stdin.flush()
                for line in proc.stdout:
                    if line.strip() == "RESET_OK":
                        break
                if sys_tokens_str:
                    proc.stdin.write(f"PREWARM {sys_tokens_str}\n")
                    proc.stdin.flush()
                    for line in proc.stdout:
                        if line.strip() == "PREWARMED":
                            break
                print(f"{YELLOW}[Memoria y KV-cache reiniciados]{RESET}\n")
                continue

            history.append(("user", user_input))
            full_prompt = format_prompt(history, args.model, args.system)
            tokens = tok.encode(full_prompt)

            if len(tokens) >= args.ctx - args.n:
                print(f"{YELLOW}[Aviso: Contexto largo ({len(tokens)} tokens), recortando historia]{RESET}")
                if len(history) > 2:
                    history = history[-2:]
                    full_prompt = format_prompt(history, args.model, args.system)
                    tokens = tok.encode(full_prompt)
                    proc.stdin.write("RESET\n")
                    proc.stdin.flush()
                    for line in proc.stdout:
                        if line.strip() == "RESET_OK":
                            break

            tokens_str = ",".join(map(str, tokens))
            proc.stdin.write(f"PROMPT {tokens_str}\n")
            proc.stdin.flush()

            in_thinking = (args.model == "r1")
            if in_thinking:
                sys.stdout.write(f"{YELLOW}[Pensando...]{RESET}\n{DIM}")
            else:
                sys.stdout.write(f"{bot_color}{BOLD}{bot_label} > {RESET}")
            sys.stdout.flush()

            t_start = time.time()
            ttft = None
            resp_tokens = []
            for line in proc.stdout:
                line_str = line.strip()
                if line_str == "DONE":
                    break
                if line_str.startswith("TOKEN "):
                    if ttft is None:
                        ttft = time.time() - t_start
                    t_id = int(line_str.split()[1])
                    resp_tokens.append(t_id)

                    if t_id == 151649:  # </think> de DeepSeek-R1
                        in_thinking = False
                        sys.stdout.write(f"{RESET}\n\n{bot_color}{BOLD}Respuesta > {RESET}")
                        sys.stdout.flush()
                        continue

                    token_text = tok.decode([t_id])
                    # No imprimir etiquetas especiales
                    if any(sp in token_text for sp in ["<|im_end|>", "<|endoftext|>", "<｜end", "<think>", "</s>", "<|user|>", "<|assistant|>"]):
                        continue
                    sys.stdout.write(token_text)
                    sys.stdout.flush()

            t_end = time.time()
            ttft_val = ttft if ttft is not None else 0.0
            gen_time = (t_end - (t_start + ttft_val)) if len(resp_tokens) > 1 else max(0.001, t_end - t_start)
            tok_s = (len(resp_tokens) - 1) / max(0.001, gen_time) if len(resp_tokens) > 1 else 0.0

            sys.stdout.write(f"\n{BOLD}\033[90m[inicio: {ttft_val*1000:.0f}ms | {len(resp_tokens)} toks | {tok_s:.1f} tok/s]{RESET}\n\n")
            sys.stdout.flush()

            assistant_text = tok.decode(resp_tokens).replace("<|im_end|>", "").replace("</s>", "").strip()
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
