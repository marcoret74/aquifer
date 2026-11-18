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
import json
import os
import re
import readline  # habilita flechas arriba/abajo e historial en input()
import subprocess
import sys
import tempfile
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



class StreamDecoder:
    """Decodificador streaming con soporte incremental de multi-byte UTF-8 y SentencePiece."""
    def __init__(self, tok):
        self.tok = tok
        self.first_token = True
        if tok.mode == 'sp':
            self.pend = bytearray()
        else:
            import codecs
            self.utf8 = codecs.getincrementaldecoder('utf-8')(errors='replace')

    def decode(self, t_id):
        s = self.tok.id2s.get(int(t_id))
        if s is None:
            return ''
        if self.tok.mode == 'sp':
            if len(s) == 6 and s.startswith('<0x') and s.endswith('>'):
                try:
                    self.pend.append(int(s[3:5], 16))
                    return ''
                except ValueError:
                    pass
            prefix = ''
            if self.pend:
                prefix = self.pend.decode('utf-8', 'replace')
                self.pend = bytearray()
            txt = (prefix + s).replace('▁', ' ')
            if self.first_token and txt.startswith(' '):
                txt = txt.lstrip(' ')
            if txt:
                self.first_token = False
            return txt
        else:
            if s in self.tok.added:
                b = s.encode('utf-8')
            else:
                b = bytes(self.tok.u2b.get(c, 63) for c in s)
            chunk = self.utf8.decode(b, final=False)
            if self.first_token and chunk.startswith(' '):
                chunk = chunk.lstrip(' ')
            if chunk:
                self.first_token = False
            return chunk

def system_prompt_text(model_type, system_prompt):
    """Prefijo exacto del system prompt para el chat template del modelo."""
    if not system_prompt or model_type == "r1":
        return ""
    if model_type == "tiny":
        return f"<|system|>\n{system_prompt}</s>\n"
    return f"<|im_start|>system\n{system_prompt}<|im_end|>\n"


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
        "desc": "El más liviano; respuestas directas y rápidas"
    },
    {
        "key": "qwen",
        "name": "Qwen 2.5-1.5B Instruct",
        "params": "1.54B",
        "ram": "~50 MB",
        "speed": "15-20 tok/s",
        "artifact": "qwen15_affine",
        "desc": "Equilibrio entre calidad, español y velocidad"
    },
    {
        "key": "qwen3",
        "name": "Qwen 2.5-3B Instruct",
        "params": "3.09B",
        "ram": "~75 MB",
        "speed": "6-7 tok/s",
        "artifact": "qwen3b_affine",
        "desc": "Razonamiento y redacción más elaborada; el más lento"
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
    print(f"\n{BOLD}{CYAN}Modelos disponibles en Aquifer (pesos paginados desde disco):{RESET}")
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
        cfg["title"] = "Qwen 2.5-0.5B Instruct (~52 tok/s medidos)"
        cfg["color"] = BLUE
        cfg["label"] = "Qwen-0.5B"
        cfg["eos"] = "151643,151645"
    elif model_name == "qwen3":
        cfg["artifact"] = resolve_artifact("qwen3b_affine")
        cfg["tokenizer"] = "/home/marco/.cache/caudal/qwen3b/tokenizer.json"
        cfg["system"] = "You are a helpful assistant."
        cfg["n"] = 512
        cfg["title"] = "Qwen 2.5-3B Instruct"
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


SUMMARY_SYSTEM = ("Resume conversaciones con hechos concretos: nombres, "
                  "números, preferencias y decisiones. No saludes, no preguntes "
                  "y no inventes información.")

SESSION_VERSION = 1


def session_paths(base):
    """Un archivo pareja: <base>.kv (binario del motor) y <base>.json (historial)."""
    return base + ".kv", base + ".json"


def load_session(base, model):
    """Devuelve (history, motivo_de_omision). None en history = no restaurar."""
    kv_path, json_path = session_paths(base)
    if not os.path.exists(json_path):
        return None, None
    try:
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, ValueError):
        return None, "archivo de sesión ilegible"
    if data.get("version") != SESSION_VERSION:
        return None, "versión de sesión distinta"
    if data.get("model") != model:
        return None, "pertenece al modelo '%s'" % data.get("model")
    if not os.path.exists(kv_path):
        return None, "falta el KV de la sesión"
    hist = [(str(r), str(t)) for r, t in data.get("history", [])]
    return hist, None


def read_session_meta(base):
    _, json_path = session_paths(base)
    try:
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, ValueError):
        return None
    return data if isinstance(data, dict) else None


def write_session_meta(base, model, system_text, history, parent=None):
    """Escribe el historial y metadatos. Conserva created/parent si existían."""
    _, json_path = session_paths(base)
    os.makedirs(os.path.dirname(os.path.abspath(json_path)), exist_ok=True)
    meta = read_session_meta(base) or {}
    now = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    payload = {
        "version": SESSION_VERSION,
        "model": model,
        "system": system_text or "",
        "history": [[r, t] for r, t in history],
        "created": meta.get("created") or now,
        "updated": now,
        "parent": parent if parent is not None else meta.get("parent"),
        "turns": sum(1 for r, _ in history if r == "user"),
    }
    tmp = json_path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=1)
    os.replace(tmp, json_path)


def clean_summary(text):
    for mark in ("<|im_end|>", "<|endoftext|>", "</s>", "<|assistant|>",
                 "<|user|>", "<think>", "</think>"):
        text = text.replace(mark, "")
    text = " ".join(text.split())
    if text.lower().startswith("resumen:"):
        text = text[len("resumen:"):].strip()
    # Un resumen no debería ser un saludo ni una pregunta: esos turnos
    # confunden al modelo en la siguiente respuesta.
    parts = []
    for sentence in re.split(r"(?<=[.!?])\s+", text):
        low = sentence.strip().lower()
        if not low or low.endswith("?"):
            continue
        if low.startswith(("hola", "claro", "por supuesto", "aquí tienes")):
            continue
        parts.append(sentence.strip())
    return " ".join(parts).strip()


def default_session_dir():
    return os.path.join(os.path.expanduser("~"), ".cache", "aquifer", "sessions")


def session_dir_for(args):
    if getattr(args, "session_dir", None):
        return os.path.abspath(os.path.expanduser(args.session_dir))
    if getattr(args, "session", None):
        return os.path.dirname(os.path.abspath(os.path.expanduser(args.session)))
    return default_session_dir()


SESSION_NAME_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,63}")


def resolve_session_base(name, directory):
    """Convierte un nombre simple en ruta base dentro del directorio de sesiones."""
    name = (name or "").strip()
    if not SESSION_NAME_RE.fullmatch(name):
        raise ValueError(
            "nombre inválido: usa letras, números, punto, guion o guion bajo "
            "(máximo 64, sin rutas)")
    return os.path.join(directory, name)


def list_sessions(directory):
    rows = []
    try:
        entries = list(os.scandir(directory))
    except (FileNotFoundError, NotADirectoryError):
        return rows
    for entry in entries:
        if not entry.is_file() or not entry.name.endswith(".json"):
            continue
        base = entry.path[:-5]
        meta = read_session_meta(base)
        if not meta:
            continue
        kv_path = base + ".kv"
        updated = meta.get("updated")
        if not updated:
            try:
                updated = time.strftime("%Y-%m-%dT%H:%M:%SZ",
                                        time.gmtime(entry.stat().st_mtime))
            except OSError:
                updated = "?"
        rows.append({
            "name": os.path.basename(base),
            "base": base,
            "model": meta.get("model", "?"),
            "turns": meta.get("turns", len(meta.get("history", [])) // 2),
            "updated": updated,
            "parent": meta.get("parent"),
            "size": os.path.getsize(kv_path) if os.path.exists(kv_path) else 0,
            "has_kv": os.path.exists(kv_path),
        })
    rows.sort(key=lambda r: r["updated"], reverse=True)
    return rows


def print_sessions(directory, active_base=None):
    rows = list_sessions(directory)
    print(f"{BOLD}Sesiones en {directory}{RESET}")
    if not rows:
        print(f"  {DIM}(vacío){RESET}")
        return
    active = os.path.basename(active_base) if active_base else None
    print(f"  {'':2}{'nombre':<20} {'modelo':<8} {'turnos':>6}  {'tamaño':>9}  actualizado")
    for row in rows:
        mark = "*" if row["name"] == active else " "
        size = "%.0f KB" % (row["size"] / 1024.0) if row["size"] else "sin KV"
        parent = f" <- {row['parent']}" if row.get("parent") else ""
        print(f"  {mark} {row['name']:<20} {row['model']:<8} {row['turns']:>6}  "
              f"{size:>9}  {row['updated']}{parent}")
    if active:
        print(f"  {DIM}* sesión activa para autoguardado{RESET}")


def engine_save_kv_path(proc, kv_path):
    os.makedirs(os.path.dirname(os.path.abspath(kv_path)), exist_ok=True)
    try:
        proc.stdin.write("SAVEKV %s\n" % kv_path)
        proc.stdin.flush()
    except Exception:
        return None
    for line in proc.stdout:
        line = line.strip()
        if line.startswith(("SAVED", "SAVE_ERR")):
            return line
    return None


def engine_load_kv_path(proc, kv_path):
    try:
        proc.stdin.write("LOADKV %s\n" % kv_path)
        proc.stdin.flush()
    except Exception:
        return None
    for line in proc.stdout:
        line = line.strip()
        if line.startswith(("LOADED", "LOAD_ERR")):
            return line
    return None


def engine_save_kv(proc, base):
    """Pide al motor que persista su KV. Devuelve la línea de respuesta."""
    kv_path, _ = session_paths(base)
    return engine_save_kv_path(proc, kv_path)


def engine_config(proc, n, temperature, top_k, top_p, seed,
                  repeat_penalty, repeat_last_n):
    """Ajusta n y sampling en caliente. Devuelve True si el motor aceptó."""
    line = "CONFIG %d %.6g %d %.6g %d %.6g %d" % (
        n, temperature, top_k, top_p, seed, repeat_penalty, repeat_last_n)
    try:
        proc.stdin.write(line + "\n")
        proc.stdin.flush()
    except Exception:
        return False
    for raw in proc.stdout:
        s = raw.strip()
        if s == "CONFIG_OK":
            return True
        if s == "CONFIG_ERR":
            return False
    return False


def start_engine(mcfg, args, repo_dir, default_bin, resolve_artifact, restore=False):
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

    print(f"{BOLD}Iniciando Aquifer ({mcfg['title']} · afín 5.0 bpw · pesos paginados desde disco)...{RESET}")
    tok = Tokenizer(tok_path)

    gen_n = args.n if args.n is not None else mcfg["n"]
    cmd = [
        bin_path,
        artifact_path,
        "--interactive",
        "--threads", str(args.threads),
        "--n", str(gen_n),
        "--ctx", str(args.ctx),
        "--eos", mcfg["eos"],
        "--prefill-batch", str(args.prefill_batch),
        "--mem-policy", args.mem_policy,
        "--kv-dtype", args.kv_dtype,
        "--spec", str(args.spec),
        "--spec-min", str(args.spec_min),
        "--spec-max", str(args.spec_max),
        "--temperature", str(args.temperature),
        "--top-k", str(args.top_k),
        "--top-p", str(args.top_p),
        "--seed", str(args.seed),
        "--repeat-penalty", str(args.repeat_penalty),
        "--repeat-last-n", str(args.repeat_last_n)
    ]
    if args.diamond is not None:
        print(f"{YELLOW}[Experimental] El modo Diamante actual selecciona neuronas "
              f"dispersas sin reordenar filas: midió más lento que denso.{RESET}")
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
    sys_str = system_prompt_text(args.model, system_text)
    if sys_str:
        sys_tokens = tok.encode(sys_str)
        sys_tokens_str = ",".join(map(str, sys_tokens))

    restored = 0
    if restore and args.session:
        kv_path, _ = session_paths(args.session)
        if os.path.exists(kv_path):
            proc.stdin.write("LOADKV %s\n" % kv_path)
            proc.stdin.flush()
            for line in proc.stdout:
                ls = line.strip()
                if ls.startswith("LOADED"):
                    try:
                        restored = int(ls.split()[1])
                    except (IndexError, ValueError):
                        restored = 0
                    break
                if ls.startswith("LOAD_ERR"):
                    print(f"{YELLOW}[Sesión: {ls} — se ignora el KV guardado]{RESET}")
                    break
    if restored:
        print(f"{DIM}[Sesión restaurada: {restored} tokens de KV reutilizados]{RESET}")
    elif sys_tokens_str:
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

    ap = argparse.ArgumentParser(description="Chat interactivo en Aquifer (artefacto afín 5.00 bpw, pesos paginados)")
    ap.add_argument("--list", "-l", action="store_true", help="Listar todos los modelos disponibles y salir")
    ap.add_argument("--model", choices=["qwen", "qwen05", "qwen3", "coder", "r1", "tiny", "smol2"], default="qwen",
                    help="Modelo: 'qwen' (1.5B), 'qwen05' (0.5B rápido), 'qwen3' (3B Instruct), 'coder' (1.5B Coder), 'r1' (R1 Distill), 'tiny' (1.1B), 'smol2' (1.7B)")
    ap.add_argument("--bin", default=default_bin, help="Ruta al binario generate_aquifer")
    ap.add_argument("--artifact", default=None, help="Ruta al artefacto cuantizado")
    ap.add_argument("--tokenizer", default=None, help="Ruta a tokenizer.json")
    ap.add_argument("--threads", type=int, default=0, help="Hilos AVX2 (0 = núcleos físicos)")
    ap.add_argument("--n", type=int, default=None, help="Tokens máximos generados por turno")
    ap.add_argument("--ctx", type=int, default=2048, help="Ventana de contexto (default: 2048)")
    ap.add_argument("--system", default=None, help="Prompt de sistema")
    ap.add_argument("--prefill-batch", type=int, default=8, help="Tokens prellenados por pasada (1-8; default: 8)")
    ap.add_argument("--mem-policy", choices=["auto", "paged", "resident"], default="auto",
                    help="auto usa RAM libre como caché y pagina bajo presión; paged fuerza VmRSS mínimo; resident nunca desaloja")
    ap.add_argument("--kv-dtype", choices=["fp16", "fp32"], default="fp16",
                    help="Precisión del KV cache (fp16 = mitad de memoria y tráfico)")
    ap.add_argument("--spec", type=int, default=4,
                    help="Tokens especulativos por pasada con prompt-lookup (0 = desactivado)")
    ap.add_argument("--spec-min", type=int, default=2,
                    help="Longitud mínima del n-grama para especular")
    ap.add_argument("--spec-max", type=int, default=8,
                    help="Longitud máxima del n-grama a buscar")
    ap.add_argument("--temperature", type=float, default=0.0,
                    help="Temperatura de muestreo (0 = greedy, default)")
    ap.add_argument("--top-k", type=int, default=40,
                    help="Top-k de muestreo (solo si temperature > 0)")
    ap.add_argument("--top-p", type=float, default=0.9,
                    help="Nucleus sampling (solo si temperature > 0)")
    ap.add_argument("--seed", type=int, default=0,
                    help="Semilla de muestreo (0 = aleatoria)")
    ap.add_argument("--repeat-penalty", type=float, default=1.0,
                    help="Penalización de repetición (1.0 = off; ej. 1.15)")
    ap.add_argument("--repeat-last-n", type=int, default=64,
                    help="Ventana de tokens para repetition penalty")
    ap.add_argument("--no-context-summary", dest="context_summary",
                    action="store_false", default=True,
                    help="No resumir el contexto descartado antes de compactar")
    ap.add_argument("--summary-tokens", type=int, default=160,
                    help="Tokens máximos del resumen de contexto (default: 160)")
    ap.add_argument("--no-kv-compact", dest="kv_compact", action="store_false",
                    default=True,
                    help="Al llenarse el contexto, resetear el KV en vez de "
                         "conservar el system prompt y la cola reciente")
    ap.add_argument("--session", default=None, metavar="RUTA",
                    help="Sesión persistente: guarda/carga KV cache e historial en "
                         "RUTA.kv y RUTA.json entre ejecuciones")
    ap.add_argument("--session-dir", default=None, metavar="DIR",
                    help="Directorio de sesiones para /sessions, /save, /load y "
                         "/branch (default: el de --session o ~/.cache/aquifer/sessions)")
    ap.add_argument("--sessions", action="store_true",
                    help="Listar las sesiones guardadas y salir")
    ap.add_argument("--stop", action="append", default=None, metavar="TEXTO",
                    help="Secuencia de parada por texto (repetible). Se detecta en streaming, "
                         "se suprime junto con lo que siga y se corta la generación")
    ap.add_argument("--diamond", type=float, default=None, help="Porcentaje de volumen del haz óptico en el Diamante 3D (ej: 85)")
    args = ap.parse_args()

    if args.list:
        print_model_catalog(resolve_artifact)
        return 0
    if args.sessions:
        print_sessions(session_dir_for(args), args.session)
        return 0

    mcfg = get_model_config(args.model, resolve_artifact)
    if args.artifact:
        mcfg["artifact"] = args.artifact
    if args.tokenizer:
        mcfg["tokenizer"] = args.tokenizer

    session_history = None
    if args.session:
        session_history, session_skip = load_session(args.session, args.model)
        if session_skip:
            print(f"{YELLOW}[Sesión no restaurada: {session_skip}]{RESET}")

    proc, tok, sys_tokens_str = start_engine(
        mcfg, args, repo_dir, default_bin, resolve_artifact,
        restore=session_history is not None)
    if proc is None:
        return 1

    print(f"\n{BOLD}================================================================={RESET}")
    print(f"{BOLD}      Aquifer Interactive Shell · {mcfg['title']}                  {RESET}")
    print(f"      {CYAN}⚡ Powered by Semantic Geometry Chile{RESET}")
    print(f"  Commands:                                                      ")
    print(f"    {YELLOW}/models{RESET} (or {YELLOW}/list{RESET})          -> List all available models              ")
    print(f"    {YELLOW}/switch <model>{RESET} (or {YELLOW}/model{RESET}) -> Switch active model dynamically      ")
    print(f"    {YELLOW}/reset{RESET}  (or {YELLOW}/clear{RESET})         -> Reset conversation history & KV cache  ")
    print(f"    {YELLOW}/sessions{RESET}                     -> List saved sessions                   ")
    print(f"    {YELLOW}/save <n>{RESET} (or {YELLOW}/branch <n>{RESET})   -> Snapshot / branch to a new session  ")
    print(f"    {YELLOW}/load <n>{RESET}                     -> Load a saved session                  ")
    print(f"    {YELLOW}/exit{RESET}   (or {YELLOW}/quit{RESET})          -> Exit interactive shell                 ")
    print(f"{BOLD}================================================================={RESET}\n")

    history = session_history if session_history is not None else []
    session_dirty = False
    session_dir = session_dir_for(args)

    def current_system():
        return args.system if args.system is not None else mcfg["system"]

    def snapshot(base, parent=None):
        """Persiste KV + historial en `base`. Devuelve (ok, mensaje)."""
        msg = engine_save_kv(proc, base)
        if msg and msg.startswith("SAVED"):
            write_session_meta(base, args.model, current_system(), history,
                               parent=parent)
            return True, msg
        return False, msg or "el motor no respondió"

    def summarize(messages, budget, gen_n):
        """Resume `messages` con el mismo motor, sin perder el KV actual.

        Guarda el KV a un temporal, genera el resumen, lo restaura y devuelve
        el texto. Si algo falla, devuelve None y el KV se restaura igual.
        """
        if not messages or budget <= 0:
            return None
        transcript = "\n".join(
            ("Usuario: " if r == "user" else "Asistente: ") + t
            for r, t in messages)
        overhead = len(tok.encode(format_prompt(
            [("user", "")], args.model, SUMMARY_SYSTEM)))
        max_ctx = args.ctx - budget - 16
        if max_ctx <= overhead + 32:
            return None
        ids = tok.encode(transcript)
        if len(ids) + overhead > max_ctx:
            keep = max_ctx - overhead
            if keep < 32:
                return None
            transcript = ("[...inicio del contexto recortado...]\n" +
                          tok.decode(ids[-keep:]))
        instruction = ("Resume la conversación en 2-4 frases factuales. Incluye "
                       "nombres, números y decisiones. No saludes ni hagas "
                       "preguntas. Devuelve solo el resumen.\n\n" + transcript)
        prompt = format_prompt([("user", instruction)], args.model, SUMMARY_SYSTEM)
        ptokens = tok.encode(prompt)
        if len(ptokens) + budget >= args.ctx:
            return None

        fd, tmp = tempfile.mkstemp(prefix="aquifer-ctx-", suffix=".kv")
        os.close(fd)
        saved = engine_save_kv_path(proc, tmp)
        if not (saved and saved.startswith("SAVED")):
            try:
                os.remove(tmp)
            except OSError:
                pass
            return None
        text = ""
        try:
            if not engine_config(proc, budget, 0.0, 40, 0.9, 0, 1.0, 64):
                return None
            proc.stdin.write("PROMPT " + ",".join(map(str, ptokens)) + "\n")
            proc.stdin.flush()
            out_ids = []
            for line in proc.stdout:
                line = line.strip()
                if line == "DONE":
                    break
                if line.startswith("TOKEN "):
                    try:
                        out_ids.append(int(line.split()[1]))
                    except (IndexError, ValueError):
                        pass
            text = tok.decode(out_ids)
        finally:
            loaded = engine_load_kv_path(proc, tmp)
            try:
                os.remove(tmp)
            except OSError:
                pass
            engine_config(proc, gen_n, args.temperature, args.top_k,
                          args.top_p, args.seed, args.repeat_penalty,
                          args.repeat_last_n)
            if not (loaded and loaded.startswith("LOADED")):
                text = ""
        return clean_summary(text) or None

    try:
        while True:
            try:
                user_input = input(f"\001{GREEN}{BOLD}\002You > \001{RESET}\002").strip()
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
            elif user_input.lower() in ("/sessions", "/sesiones"):
                print_sessions(session_dir, args.session)
                continue
            elif user_input.lower() in ("/session", "/sesion"):
                if args.session:
                    meta = read_session_meta(args.session) or {}
                    kv_path, json_path = session_paths(args.session)
                    size = os.path.getsize(kv_path) if os.path.exists(kv_path) else 0
                    print(f"{BOLD}Sesión activa:{RESET} {os.path.basename(args.session)}")
                    print(f"  directorio : {session_dir}")
                    print(f"  modelo     : {args.model}")
                    print(f"  turnos     : {meta.get('turns', '?')}")
                    print(f"  KV         : {('%.0f KB' % (size / 1024.0)) if size else 'sin archivo'}")
                    if meta.get("parent"):
                        print(f"  rama de    : {meta['parent']}")
                    print(f"  archivos   : {kv_path} · {json_path}")
                else:
                    print(f"{YELLOW}Sin sesión activa.{RESET} Usa {BOLD}/save <nombre>{RESET} "
                          f"o {BOLD}/branch <nombre>{RESET}.")
                    print(f"{DIM}Directorio: {session_dir}{RESET}")
                continue
            elif user_input.lower().startswith(("/save ", "/guardar ")):
                name = user_input.split(None, 1)[1].strip()
                try:
                    base = resolve_session_base(name, session_dir)
                except ValueError as err:
                    print(f"{YELLOW}{err}{RESET}")
                    continue
                parent = os.path.basename(args.session) if args.session else None
                ok, msg = snapshot(base, parent=parent)
                if ok:
                    print(f"{GREEN}[Guardada '{name}' · {msg} · {base}]{RESET}")
                else:
                    print(f"{YELLOW}[No se pudo guardar: {msg}]{RESET}")
                continue
            elif user_input.lower().startswith(("/branch ", "/rama ")):
                name = user_input.split(None, 1)[1].strip()
                try:
                    base = resolve_session_base(name, session_dir)
                except ValueError as err:
                    print(f"{YELLOW}{err}{RESET}")
                    continue
                if any(os.path.exists(_p) for _p in session_paths(base)):
                    print(f"{YELLOW}[Ya existe la sesión '{name}'; elige otro "
                          f"nombre o bórrala con /delete]{RESET}")
                    continue
                parent = os.path.basename(args.session) if args.session else None
                ok, msg = snapshot(base, parent=parent)
                if ok:
                    args.session = base
                    session_dirty = False
                    print(f"{GREEN}[Rama '{name}' guardada y activa · {msg}]{RESET}")
                else:
                    print(f"{YELLOW}[No se pudo crear la rama: {msg}]{RESET}")
                continue
            elif user_input.lower().startswith(("/load ", "/cargar ")):
                name = user_input.split(None, 1)[1].strip()
                try:
                    base = resolve_session_base(name, session_dir)
                except ValueError as err:
                    print(f"{YELLOW}{err}{RESET}")
                    continue
                meta = read_session_meta(base)
                kv_path, _ = session_paths(base)
                if not meta:
                    print(f"{YELLOW}[No existe la sesión '{name}' en {session_dir}]{RESET}")
                    continue
                if meta.get("model") != args.model:
                    print(f"{YELLOW}[La sesión '{name}' es del modelo "
                          f"'{meta.get('model')}'; cambia con /switch]{RESET}")
                    continue
                if not os.path.exists(kv_path):
                    print(f"{YELLOW}[Falta el KV de la sesión '{name}']{RESET}")
                    continue
                try:
                    proc.stdin.write("LOADKV %s\n" % kv_path)
                    proc.stdin.flush()
                except Exception:
                    print(f"{YELLOW}[El motor no responde]{RESET}")
                    continue
                loaded = None
                for rline in proc.stdout:
                    ls = rline.strip()
                    if ls.startswith("LOADED"):
                        try:
                            loaded = int(ls.split()[1])
                        except (IndexError, ValueError):
                            loaded = 0
                        break
                    if ls.startswith("LOAD_ERR"):
                        loaded = ls
                        break
                if isinstance(loaded, int):
                    history = [(str(r), str(t)) for r, t in meta.get("history", [])]
                    args.session = base
                    session_dirty = False
                    print(f"{GREEN}[Sesión '{name}' cargada · {loaded} tokens de KV]{RESET}")
                else:
                    print(f"{YELLOW}[No se pudo cargar '{name}': {loaded}]{RESET}")
                    if sys_tokens_str:
                        proc.stdin.write(f"PREWARM {sys_tokens_str}\n")
                        proc.stdin.flush()
                        for rline in proc.stdout:
                            if rline.strip() == "PREWARMED":
                                break
                continue
            elif user_input.lower().startswith(("/delete ", "/borrar ")):
                name = user_input.split(None, 1)[1].strip()
                try:
                    base = resolve_session_base(name, session_dir)
                except ValueError as err:
                    print(f"{YELLOW}{err}{RESET}")
                    continue
                kv_path, json_path = session_paths(base)
                if not os.path.exists(kv_path) and not os.path.exists(json_path):
                    print(f"{YELLOW}[No existe la sesión '{name}']{RESET}")
                    continue
                try:
                    ans = input(f"{BOLD}¿Borrar la sesión '{name}'? [s/N]: {RESET}").strip().lower()
                except (EOFError, KeyboardInterrupt):
                    continue
                if ans not in ("s", "si", "sí", "y", "yes"):
                    continue
                for _p in (kv_path, json_path):
                    try:
                        os.remove(_p)
                    except OSError:
                        pass
                if args.session == base:
                    args.session = None
                    session_dirty = False
                print(f"{YELLOW}[Sesión '{name}' borrada]{RESET}")
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
                new_proc, new_tok, new_sys_tokens_str = start_engine(
                    mcfg, args, repo_dir, default_bin, resolve_artifact, restore=False)
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
                session_dirty = False
                if args.session:
                    print(f"{DIM}[Sesión desvinculada; los archivos se conservan]{RESET}")
                    args.session = None
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
                session_dirty = False
                if args.session:
                    args.session = None
                    print(f"{DIM}[Sesión desvinculada; los archivos se conservan]{RESET}")
                print(f"{YELLOW}[Memory and KV-cache reset]{RESET}\n")
                continue

            history.append(("user", user_input))
            full_prompt = format_prompt(history, args.model, args.system if args.system is not None else mcfg["system"])
            tokens = tok.encode(full_prompt)

            bot_color = mcfg["color"]
            bot_label = mcfg["label"]
            gen_n = args.n if args.n is not None else mcfg["n"]

            sys_text = args.system if args.system is not None else mcfg["system"]
            if len(tokens) >= args.ctx - gen_n:
                print(f"{YELLOW}[Aviso: Contexto largo ({len(tokens)} tokens), compactando historia]{RESET}")
                prev_len = len(history)
                dropped = []
                while len(tokens) >= args.ctx - gen_n and len(history) > 1:
                    dropped.append(history.pop(0))
                    while len(history) > 1 and history[0][0] != "user":
                        dropped.append(history.pop(0))
                    full_prompt = format_prompt(history, args.model, sys_text)
                    tokens = tok.encode(full_prompt)

                summary = None
                if args.context_summary and dropped and args.summary_tokens > 0:
                    budget = min(args.summary_tokens, max(48, args.ctx // 4))
                    print(f"{DIM}[Resumiendo {len(dropped)} mensajes descartados "
                          f"(≤{budget} tokens)...]{RESET}")
                    summary = summarize(dropped, budget, gen_n)
                if summary:
                    user_text = history[-1][1]
                    history[-1] = ("user", "[Contexto previo resumido]\n" +
                                   summary + "\n\n" + user_text)
                    tokens = tok.encode(format_prompt(history, args.model, sys_text))
                    while len(tokens) >= args.ctx - gen_n and len(history) > 1:
                        history.pop(0)
                        while len(history) > 1 and history[0][0] != "user":
                            history.pop(0)
                        tokens = tok.encode(format_prompt(history, args.model, sys_text))
                    if len(tokens) >= args.ctx - gen_n:
                        history[-1] = ("user", user_text)
                        tokens = tok.encode(format_prompt(history, args.model, sys_text))
                        summary = None
                        print(f"{YELLOW}[El resumen no cupo; se descarta]{RESET}")
                    else:
                        print(f"{DIM}[Contexto resumido: {len(summary)} caracteres]{RESET}")
                elif args.context_summary and dropped:
                    print(f"{DIM}[No se pudo generar resumen; se compacta sin él]{RESET}")

                compacted = False
                # Conserva el system prompt en el KV y la cola que coincide con
                # el prompt recortado; el hueco descartado no se vuelve a prellenar.
                sink = len(sys_tokens_str.split(",")) if sys_tokens_str else 0
                if (args.kv_compact and len(history) < prev_len
                        and sink < len(tokens)):
                    try:
                        proc.stdin.write("COMPACT %d %s\n" % (
                            sink, ",".join(map(str, tokens[sink:]))))
                        proc.stdin.flush()
                        for rline in proc.stdout:
                            ls = rline.strip()
                            if ls.startswith("COMPACTED"):
                                kept = ls.split()[1] if len(ls.split()) > 1 else "?"
                                print(f"{DIM}[KV compactado: system + {kept} tokens conservados]{RESET}")
                                compacted = True
                                break
                            if ls.startswith(("COMPACT_NOMATCH", "COMPACT_ERR")):
                                break
                    except Exception:
                        compacted = False

                if not compacted:
                    proc.stdin.write("RESET\n")
                    proc.stdin.flush()
                    for rline in proc.stdout:
                        if rline.strip() == "RESET_OK":
                            break
                    if sys_tokens_str:
                        proc.stdin.write(f"PREWARM {sys_tokens_str}\n")
                        proc.stdin.flush()
                        for rline in proc.stdout:
                            if rline.strip() == "PREWARMED":
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
            streamer = StreamDecoder(tok)
            stop_seqs = [x for x in (args.stop or []) if x]
            stop_buf = ""
            answer_parts = []
            stopped_by_seq = False
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
                        streamer.first_token = True
                        continue

                    # Detección temprana de bucle degenerativo de repetición (ej: TinyLlama en español)
                    has_loop = False
                    for L in range(3, 20):
                        if len(resp_tokens) >= L * 3:
                            if resp_tokens[-L:] == resp_tokens[-2*L:-L] == resp_tokens[-3*L:-2*L]:
                                has_loop = True
                                break
                    if has_loop:
                        # Cortar el generador y drenar hasta DONE. Si no, los
                        # TOKEN que el hijo siga emitiendo se leerían como la
                        # respuesta del turno siguiente.
                        try:
                            proc.stdin.write("STOP\n")
                            proc.stdin.flush()
                        except Exception:
                            pass
                        for extra in proc.stdout:
                            if extra.strip() == "DONE":
                                break
                        break

                    token_text = streamer.decode(t_id)
                    # No imprimir etiquetas especiales
                    if any(sp in token_text for sp in ["<|im_end|>", "<|endoftext|>", "<｜end", "<think>", "</s>", "<|user|>", "<|assistant|>"]):
                        continue

                    if stop_seqs:
                        # Buffer rodante: detecta la parada aunque cruce tokens.
                        stop_buf += token_text
                        hit = None
                        for seq in stop_seqs:
                            idx = stop_buf.find(seq)
                            if idx != -1 and (hit is None or idx < hit[0]):
                                hit = (idx, seq)
                        if hit is not None:
                            prefix = stop_buf[:hit[0]]
                            if prefix:
                                answer_parts.append(prefix)
                                sys.stdout.write(prefix)
                                sys.stdout.flush()
                            stopped_by_seq = True
                            try:
                                proc.stdin.write("STOP\n")
                                proc.stdin.flush()
                            except Exception:
                                pass
                            for extra in proc.stdout:
                                if extra.strip() == "DONE":
                                    break
                            break
                        # Retiene solo el sufijo que podría ser prefijo de una parada.
                        keep = 0
                        for seq in stop_seqs:
                            for k in range(min(len(seq) - 1, len(stop_buf)), 0, -1):
                                if stop_buf.endswith(seq[:k]):
                                    keep = max(keep, k)
                                    break
                        emit = len(stop_buf) - keep
                        if emit > 0:
                            out = stop_buf[:emit]
                            stop_buf = stop_buf[emit:]
                            answer_parts.append(out)
                            sys.stdout.write(out)
                            sys.stdout.flush()
                    else:
                        answer_parts.append(token_text)
                        sys.stdout.write(token_text)
                        sys.stdout.flush()
            if stop_seqs and stop_buf and not stopped_by_seq:
                answer_parts.append(stop_buf)
                sys.stdout.write(stop_buf)
                sys.stdout.flush()
                stop_buf = ""
            t_end = time.time()
            ttft_val = ttft if ttft is not None else 0.0
            gen_time = (t_end - (t_start + ttft_val)) if len(resp_tokens) > 1 else max(0.001, t_end - t_start)
            tok_s = (len(resp_tokens) - 1) / max(0.001, gen_time) if len(resp_tokens) > 1 else 0.0

            sys.stdout.write(f"\n{BOLD}\033[90m[inicio: {ttft_val*1000:.0f}ms | {len(resp_tokens)} toks | {tok_s:.1f} tok/s]{RESET}\n\n")
            sys.stdout.flush()

            if stop_seqs:
                assistant_text = "".join(answer_parts).strip()
            else:
                assistant_text = tok.decode(resp_tokens).replace("<|im_end|>", "").replace("</s>", "").strip()
            history.append(("assistant", assistant_text))
            session_dirty = True
            if args.session:
                msg = engine_save_kv(proc, args.session)
                if msg and msg.startswith("SAVED"):
                    write_session_meta(args.session, args.model, sys_text, history)
                    session_dirty = False
                elif msg:
                    print(f"{YELLOW}[Sesión: {msg}]{RESET}")

    finally:
        if args.session and session_dirty:
            try:
                msg = engine_save_kv(proc, args.session)
                if msg and msg.startswith("SAVED"):
                    _sys = args.system if args.system is not None else mcfg["system"]
                    write_session_meta(args.session, args.model, _sys, history)
            except Exception:
                pass
        try:
            proc.stdin.write("QUIT\n")
            proc.stdin.flush()
            proc.wait(timeout=2)
        except Exception:
            proc.kill()


if __name__ == "__main__":
    sys.exit(main())
