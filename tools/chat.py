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


MODELS_CATALOG = [
    {
        "key": "qwen05",
        "name": "Qwen 2.5-0.5B Instruct",
        "params": "490M",
        "ram": "~37 MB",
        "speed": "35-50 tok/s",
        "artifact": "qwen05_affine",
        "desc": "Ultra-fast response, lightweight direct answers"
    },
    {
        "key": "qwen",
        "name": "Qwen 2.5-1.5B Instruct",
        "params": "1.54B",
        "ram": "~50 MB",
        "speed": "15-20 tok/s",
        "artifact": "qwen15_affine",
        "desc": "Golden balance between intelligence, Spanish & speed"
    },
    {
        "key": "qwen3",
        "name": "Qwen 2.5-3B Instruct",
        "params": "3.09B",
        "ram": "~75 MB",
        "speed": "6-7 tok/s",
        "artifact": "qwen3b_affine",
        "desc": "Deep reasoning, eloquent writing & rich vocabulary"
    },
    {
        "key": "coder",
        "name": "Qwen 2.5-Coder-1.5B",
        "params": "1.54B",
        "ram": "~50 MB",
        "speed": "15-20 tok/s",
        "artifact": "coder15_affine",
        "desc": "Code generation, C++, Python, algorithms & systems"
    },
    {
        "key": "r1",
        "name": "DeepSeek-R1-Distill-1.5B",
        "params": "1.54B",
        "ram": "~50 MB",
        "speed": "15-20 tok/s",
        "artifact": "r1_affine",
        "desc": "Chain-of-thought reasoning with <think> tags"
    },
    {
        "key": "smol2",
        "name": "SmolLM2-1.7B Instruct",
        "params": "1.71B",
        "ram": "~50 MB",
        "speed": "15-18 tok/s",
        "artifact": "smol2_affine",
        "desc": "HuggingFace curated knowledge dataset"
    },
    {
        "key": "tiny",
        "name": "TinyLlama-1.1B Chat",
        "params": "1.10B",
        "ram": "~45 MB",
        "speed": "20-25 tok/s",
        "artifact": "tiny_affine",
        "desc": "Compact Llama-2 architecture"
    },
]


def print_model_catalog(resolve_artifact):
    print(f"\n{BOLD}{CYAN}Available Models in Aquifer Engine (Zero-Copy HiperRAM Paging):{RESET}")
    print(f"  {CYAN}⚡ Powered by Semantic Geometry Chile{RESET}")
    print("=" * 95)
    print(f"{'Key':<8} {'Model Name':<26} {'Params':<8} {'RAM (VmRSS)':<12} {'Speed':<13} {'Status'}")
    print("-" * 95)
    for m in MODELS_CATALOG:
        art_path = resolve_artifact(m["artifact"])
        exists = os.path.exists(art_path)
        status_str = f"{GREEN}● Ready{RESET}" if exists else f"{DIM}○ Missing{RESET}"
        print(f"{YELLOW}{m['key']:<8}{RESET} {m['name']:<26} {m['params']:<8} {m['ram']:<12} {m['speed']:<13} {status_str}")
        print(f"         {DIM}{m['desc']}{RESET}")
    print("=" * 95)
    print(f"Run model  : {BOLD}python3 tools/chat.py --model <key>{RESET}")
    print(f"Diamond 3D : {BOLD}python3 tools/chat.py --model <key> --diamond 90{RESET}\n")


def get_model_config(model_name, resolve_artifact):
    cfg = {}
    if model_name == "r1":
        cfg["artifact"] = resolve_artifact("r1_affine")
        cfg["tokenizer"] = "/home/marco/.cache/caudal/r1/tokenizer.json"
        cfg["system"] = ""
        cfg["n"] = 512
        cfg["title"] = "DeepSeek-R1-Distill-1.5B (Razonamiento)"
        cfg["color"] = YELLOW
        cfg["label"] = "DeepSeek-R1"
        cfg["eos"] = "151643"
    elif model_name == "qwen05":
        cfg["artifact"] = resolve_artifact("qwen05_affine")
        cfg["tokenizer"] = "/home/marco/.cache/caudal/qwen05/tokenizer.json"
        cfg["system"] = "You are a helpful assistant."
        cfg["n"] = 256
        cfg["title"] = "Qwen 2.5-0.5B Instruct (Ultra-Fast ~52 tok/s)"
        cfg["color"] = BLUE
        cfg["label"] = "Qwen-0.5B"
        cfg["eos"] = "151643,151645"
    elif model_name == "qwen3":
        cfg["artifact"] = resolve_artifact("qwen3b_affine")
        cfg["tokenizer"] = "/home/marco/.cache/caudal/qwen3b/tokenizer.json"
        cfg["system"] = "You are a helpful assistant."
        cfg["n"] = 512
        cfg["title"] = "Qwen 2.5-3B Instruct (Aquifer Zero-Copy Paging)"
        cfg["color"] = CYAN
        cfg["label"] = "Qwen-3B"
        cfg["eos"] = "151643,151645"
    elif model_name == "coder":
        cfg["artifact"] = resolve_artifact("coder15_affine")
        cfg["tokenizer"] = "/home/marco/.cache/caudal/coder15/tokenizer.json"
        cfg["system"] = "You are an expert assistant in software engineering, systems architecture, and algorithms. Write clean, modern, typed, and well-explained code."
        cfg["n"] = 512
        cfg["title"] = "Qwen 2.5-Coder-1.5B Instruct (Code & Systems)"
        cfg["color"] = GREEN
        cfg["label"] = "Qwen-Coder"
        cfg["eos"] = "151643,151645"
    elif model_name == "tiny":
        cfg["artifact"] = resolve_artifact("tiny_affine")
        cfg["tokenizer"] = "/home/marco/.cache/caudal/tiny/tokenizer.json"
        cfg["system"] = "You are a helpful assistant."
        cfg["n"] = 256
        cfg["title"] = "TinyLlama-1.1B Chat"
        cfg["color"] = MAGENTA
        cfg["label"] = "TinyLlama"
        cfg["eos"] = "2"
    elif model_name == "smol2":
        cfg["artifact"] = resolve_artifact("smol2_affine")
        cfg["tokenizer"] = "/home/marco/.cache/caudal/smol2/tokenizer.json"
        cfg["system"] = "You are a helpful assistant."
        cfg["n"] = 256
        cfg["title"] = "SmolLM2-1.7B Instruct"
        cfg["color"] = BLUE
        cfg["label"] = "SmolLM2"
        cfg["eos"] = "0,2"
    else:  # qwen
        cfg["artifact"] = resolve_artifact("qwen15_affine")
        cfg["tokenizer"] = "/home/marco/.cache/caudal/qwen15/tokenizer.json"
        cfg["system"] = "You are a helpful assistant."
        cfg["n"] = 256
        cfg["title"] = "Qwen 2.5-1.5B Instruct"
        cfg["color"] = CYAN
        cfg["label"] = "Qwen"
        cfg["eos"] = "151643,151645"
    return cfg


def start_engine(mcfg, args, repo_dir, default_bin, resolve_artifact):
    bin_path = args.bin
    if args.diamond is not None and bin_path == default_bin:
        diamond_bin = os.path.join(repo_dir, "build", "generate_aquifer_diamond")
        if os.path.exists(diamond_bin):
            bin_path = diamond_bin

    if not os.path.exists(bin_path):
        print(f"Error: Binario no encontrado en {bin_path}. Compílalo primero.")
        return None, None, ""

    artifact_path = mcfg["artifact"]
    if not os.path.exists(artifact_path):
        print(f"\n{YELLOW}El modelo '{args.model}' no se encuentra en el disco local ({artifact_path}).{RESET}")
        try:
            resp = input(f"{BOLD}¿Deseas descargarlo ahora automáticamente? [S/n]: {RESET}").strip().lower()
        except (EOFError, KeyboardInterrupt):
            resp = "n"
        if resp in ("", "s", "si", "y", "yes"):
            try:
                from pull import pull_model
                ok = pull_model(args.model)
                if not ok:
                    return None, None, ""
                mcfg = get_model_config(args.model, resolve_artifact)
                artifact_path = mcfg["artifact"]
            except ImportError:
                print("Error: No se pudo cargar tools/pull.py.")
                return None, None, ""
        else:
            print("Operación cancelada.")
            return None, None, ""

    tok_path = mcfg["tokenizer"]
    if not os.path.exists(tok_path):
        local_tok = os.path.join(artifact_path, "tokenizer.json")
        if os.path.exists(local_tok):
            tok_path = local_tok
        else:
            print(f"Error: Tokenizer no encontrado en {tok_path}.")
            return None, None, ""

    print(f"{BOLD}Iniciando Aquifer Paging Engine ({mcfg['title']} Afín 5.0 bpw · Zero-Copy HiperRAM)...{RESET}")
    tok = Tokenizer(tok_path)

    gen_n = args.n if args.n is not None else mcfg["n"]
    cmd = [
        bin_path,
        artifact_path,
        "--interactive",
        "--threads", str(args.threads),
        "--n", str(gen_n),
        "--ctx", str(args.ctx),
        "--eos", mcfg["eos"]
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

    # Esperar confirmación READY de Caudal/Aquifer
    for line in proc.stdout:
        line_clean = line.strip()
        if line_clean == "READY":
            break
        elif "THP" in line_clean:
            print(f"  {line_clean}")

    # Pre-calentar system prompt en el KV-cache para que el turno 1 responda de inmediato
    sys_tokens_str = ""
    system_text = args.system if args.system is not None else mcfg["system"]
    if system_text and args.model in ("qwen", "qwen05", "qwen3", "coder", "tiny", "smol2"):
        if args.model == "tiny":
            sys_str = f"<|system|>\n{system_text}</s>\n"
        else:
            sys_str = f"<|im_start|>system\n{system_text}<|im_end|>\n"
        sys_tokens = tok.encode(sys_str)
        sys_tokens_str = ",".join(map(str, sys_tokens))
        proc.stdin.write(f"PREWARM {sys_tokens_str}\n")
        proc.stdin.flush()
        for line in proc.stdout:
            if line.strip() == "PREWARMED":
                break

    return proc, tok, sys_tokens_str


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
    ap.add_argument("--list", "-l", action="store_true", help="Listar todos los modelos disponibles y salir")
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

    if args.list:
        print_model_catalog(resolve_artifact)
        return 0

    mcfg = get_model_config(args.model, resolve_artifact)
    if args.artifact:
        mcfg["artifact"] = args.artifact
    if args.tokenizer:
        mcfg["tokenizer"] = args.tokenizer

    proc, tok, sys_tokens_str = start_engine(mcfg, args, repo_dir, default_bin, resolve_artifact)
    if proc is None:
        return 1

    print(f"\n{BOLD}================================================================={RESET}")
    print(f"{BOLD}      Aquifer Interactive Shell · {mcfg['title']}                  {RESET}")
    print(f"      {CYAN}⚡ Powered by Semantic Geometry Chile{RESET}")
    print(f"  Commands:                                                      ")
    print(f"    {YELLOW}/models{RESET} (or {YELLOW}/list{RESET})          -> List all available models              ")
    print(f"    {YELLOW}/switch <model>{RESET} (or {YELLOW}/model{RESET}) -> Switch active model dynamically      ")
    print(f"    {YELLOW}/reset{RESET}  (or {YELLOW}/clear{RESET})         -> Reset conversation history & KV cache  ")
    print(f"    {YELLOW}/exit{RESET}   (or {YELLOW}/quit{RESET})          -> Exit interactive shell                 ")
    print(f"{BOLD}================================================================={RESET}\n")

    history = []

    try:
        while True:
            try:
                user_input = input(f"{GREEN}{BOLD}You > {RESET}").strip()
            except (EOFError, KeyboardInterrupt):
                print("\nExiting...")
                break

            if not user_input:
                continue

            if user_input.lower() in ("/exit", "/quit", "exit", "quit", "/salir", "salir"):
                break
            elif user_input.lower() in ("/models", "/list", "models", "list", "/modelos", "modelos"):
                print_model_catalog(resolve_artifact)
                continue
            elif user_input.lower().startswith(("/switch", "/model", "/use", "/select")):
                parts = user_input.split()
                target_key = parts[1] if len(parts) > 1 else None
                if not target_key:
                    print_model_catalog(resolve_artifact)
                    try:
                        target_key = input(f"{BOLD}Selecciona modelo [ej: qwen05, qwen, qwen3, coder, r1, smol2, tiny]: {RESET}").strip()
                    except (EOFError, KeyboardInterrupt):
                        continue
                if target_key in [str(i + 1) for i in range(len(MODELS_CATALOG))]:
                    target_key = MODELS_CATALOG[int(target_key) - 1]["key"]
                if target_key not in [m["key"] for m in MODELS_CATALOG]:
                    print(f"{YELLOW}Modelo '{target_key}' no reconocido. Usa /models para ver las opciones.{RESET}\n")
                    continue
                if target_key == args.model:
                    print(f"{YELLOW}Ya estás usando '{target_key}'.{RESET}\n")
                    continue

                print(f"\n{BOLD}Cambiando a modelo '{target_key}'...{RESET}")
                try:
                    proc.stdin.write("RESET\n")
                    proc.stdin.flush()
                except Exception:
                    pass
                proc.terminate()
                try:
                    proc.wait(timeout=3)
                except subprocess.TimeoutExpired:
                    proc.kill()

                args.model = target_key
                mcfg = get_model_config(args.model, resolve_artifact)
                new_proc, new_tok, new_sys_tokens_str = start_engine(mcfg, args, repo_dir, default_bin, resolve_artifact)
                if new_proc is None:
                    print(f"Error al iniciar '{target_key}'. Restaurando qwen05...")
                    args.model = "qwen05"
                    mcfg = get_model_config(args.model, resolve_artifact)
                    proc, tok, sys_tokens_str = start_engine(mcfg, args, repo_dir, default_bin, resolve_artifact)
                else:
                    proc = new_proc
                    tok = new_tok
                    sys_tokens_str = new_sys_tokens_str
                history.clear()
                print(f"\n{GREEN}{BOLD}✓ Modelo activo cambiado a: {mcfg['title']}{RESET}")
                print(f"  {CYAN}⚡ Powered by Semantic Geometry Chile{RESET}\n")
                continue
            elif user_input.lower() in ("/reset", "/clear", "clear", "/limpiar", "reset"):
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
                print(f"{YELLOW}[Memory and KV-cache reset]{RESET}\n")
                continue

            history.append(("user", user_input))
            full_prompt = format_prompt(history, args.model, args.system if args.system is not None else mcfg["system"])
            tokens = tok.encode(full_prompt)

            bot_color = mcfg["color"]
            bot_label = mcfg["label"]
            gen_n = args.n if args.n is not None else mcfg["n"]

            if len(tokens) >= args.ctx - gen_n:
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
