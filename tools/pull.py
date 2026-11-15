#!/usr/bin/env python3
"""
pull.py — Gestor de descarga y distribución de modelos para Aquifer / Caudal Engine.
Descarga modelos empaquetados y pre-cuantizados directamente en silicio local
sin requerir PyTorch, Transformers, ni Docker/Ollama.
"""

import argparse
import json
import os
import sys
import time
import urllib.request
import urllib.error

# Colores ANSI
CYAN = "\033[1;36m"
GREEN = "\033[1;32m"
YELLOW = "\033[1;33m"
DIM = "\033[90m\033[3m"
BOLD = "\033[1m"
RESET = "\033[0m"

DEFAULT_HUB = os.environ.get("AQUIFER_HUB", "https://huggingface.co/marcoret74/aquifer-models/resolve/main")

REGISTRY = {
    "qwen05": {
        "name": "Qwen 2.5-0.5B Instruct",
        "params": "490M",
        "size_mb": 295,
        "artifact": "qwen05_affine",
        "desc": "Ultra-rápido (~52 tok/s), 37 MB RAM",
        "files": ["weights.bin", "scales.bin", "embed.bf16", "aux.bin", "index.txt", "model.txt", "manifest.json", "tokenizer.json", "diamond.bin"]
    },
    "qwen": {
        "name": "Qwen 2.5-1.5B Instruct",
        "params": "1.54B",
        "size_mb": 980,
        "artifact": "qwen15_affine",
        "desc": "Balance óptimo inteligencia/velocidad, 50 MB RAM",
        "files": ["weights.bin", "scales.bin", "embed.bf16", "aux.bin", "index.txt", "model.txt", "manifest.json", "tokenizer.json"]
    },
    "qwen3": {
        "name": "Qwen 2.5-3B Instruct",
        "params": "3.09B",
        "size_mb": 1940,
        "artifact": "qwen3b_affine",
        "desc": "Elocuencia profunda y razonamiento, 75 MB RAM",
        "files": ["weights.bin", "scales.bin", "embed.bf16", "aux.bin", "index.txt", "model.txt", "manifest.json", "tokenizer.json", "diamond.bin"]
    },
    "coder": {
        "name": "Qwen 2.5-Coder-1.5B",
        "params": "1.54B",
        "size_mb": 980,
        "artifact": "coder15_affine",
        "desc": "Especialista en código C++, Python, algoritmos",
        "files": ["weights.bin", "scales.bin", "embed.bf16", "aux.bin", "index.txt", "model.txt", "manifest.json", "tokenizer.json"]
    },
    "r1": {
        "name": "DeepSeek-R1-Distill-1.5B",
        "params": "1.54B",
        "size_mb": 1050,
        "artifact": "r1_affine",
        "desc": "Razonamiento con bloques <think>",
        "files": ["weights.bin", "scales.bin", "embed.bf16", "aux.bin", "index.txt", "model.txt", "manifest.json", "tokenizer.json"]
    },
    "smol2": {
        "name": "SmolLM2-1.7B Instruct",
        "params": "1.71B",
        "size_mb": 1080,
        "artifact": "smol2_affine",
        "desc": "Hugging Face dataset ultra-curado",
        "files": ["weights.bin", "scales.bin", "embed.bf16", "aux.bin", "index.txt", "model.txt", "manifest.json", "tokenizer.json"]
    },
    "tiny": {
        "name": "TinyLlama-1.1B Chat",
        "params": "1.10B",
        "size_mb": 670,
        "artifact": "tiny_affine",
        "desc": "Arquitectura compacta Llama 2",
        "files": ["weights.bin", "scales.bin", "embed.bf16", "aux.bin", "index.txt", "model.txt", "manifest.json", "tokenizer.json"]
    }
}


def download_file_with_progress(url, dest_path, desc=""):
    os.makedirs(os.path.dirname(os.path.abspath(dest_path)), exist_ok=True)
    temp_path = dest_path + ".tmp"
    
    req = urllib.request.Request(
        url,
        headers={"User-Agent": "Aquifer-Engine/1.0 (Linux; x86_64)"}
    )
    
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            total_size = int(resp.headers.get("content-length", 0))
            downloaded = 0
            start_time = time.time()
            chunk_size = 1024 * 1024  # 1 MB chunks
            
            with open(temp_path, "wb") as f:
                while True:
                    chunk = resp.read(chunk_size)
                    if not chunk:
                        break
                    f.write(chunk)
                    downloaded += len(chunk)
                    
                    elapsed = time.time() - start_time
                    speed_mb = (downloaded / (1024 * 1024)) / (elapsed if elapsed > 0 else 1e-6)
                    
                    if total_size > 0:
                        pct = (downloaded / total_size) * 100
                        mb_done = downloaded / (1024 * 1024)
                        mb_tot = total_size / (1024 * 1024)
                        bar_len = 25
                        filled = int(bar_len * downloaded // total_size)
                        bar = "█" * filled + "░" * (bar_len - filled)
                        sys.stdout.write(f"\r  {CYAN}{desc:<16}{RESET} [{bar}] {pct:5.1f}% ({mb_done:6.1f}/{mb_tot:6.1f} MB) {speed_mb:5.1f} MB/s")
                    else:
                        mb_done = downloaded / (1024 * 1024)
                        sys.stdout.write(f"\r  {CYAN}{desc:<16}{RESET} {mb_done:6.1f} MB descargados ({speed_mb:5.1f} MB/s)")
                    sys.stdout.flush()
        
        sys.stdout.write("\n")
        os.replace(temp_path, dest_path)
        return True
    except Exception as e:
        sys.stdout.write(f"\n  {YELLOW}Aviso: No se pudo descargar {desc}: {e}{RESET}\n")
        if os.path.exists(temp_path):
            os.remove(temp_path)
        return False


def pull_model(model_key, target_dir=None, hub_base=DEFAULT_HUB):
    if model_key not in REGISTRY:
        print(f"Error: Modelo '{model_key}' no reconocido en el registro de Aquifer.")
        print(f"Modelos válidos: {', '.join(REGISTRY.keys())}")
        return False
        
    info = REGISTRY[model_key]
    repo_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if target_dir is None:
        target_dir = os.path.join(repo_dir, "build", info["artifact"])
        
    os.makedirs(target_dir, exist_ok=True)
    print(f"\n{BOLD}================================================================={RESET}")
    print(f"{BOLD}  Aquifer Model Pull · {info['name']} ({info['params']}){RESET}")
    print(f"  {CYAN}⚡ Powered by Semantic Geometry Chile{RESET}")
    print(f"  Destino : {target_dir}")
    print(f"  Tamaño  : ~{info['size_mb']} MB (Cuantizado int4 afín / Zero-Copy)")
    print(f"{BOLD}================================================================={RESET}\n")
    
    # URL base del modelo en el Hub
    base_url = f"{hub_base}/{info['artifact']}"
    
    success_count = 0
    for filename in info["files"]:
        file_url = f"{base_url}/{filename}"
        dest_file = os.path.join(target_dir, filename)
        
        if os.path.exists(dest_file) and os.path.getsize(dest_file) > 0:
            print(f"  {GREEN}✓{RESET} {filename:<16} (ya existe en disco)")
            success_count += 1
            continue
            
        ok = download_file_with_progress(file_url, dest_file, desc=filename)
        if ok:
            success_count += 1
            
    if success_count > 0:
        print(f"\n{GREEN}{BOLD}✓ Modelo '{model_key}' listo para ejecución en Aquifer.{RESET}")
        print(f"Ejecuta: {BOLD}python3 tools/chat.py --model {model_key}{RESET}\n")
        return True
    else:
        print(f"\n{YELLOW}No se completó la descarga del modelo. Verifica la conexión a la URL del Hub.{RESET}\n")
        return False


def main():
    ap = argparse.ArgumentParser(description="Aquifer Model Pull — Descarga modelos pre-cuantizados para inferencia local")
    ap.add_argument("model", nargs="?", default=None, help=f"Clave del modelo: {', '.join(REGISTRY.keys())}")
    ap.add_argument("--list", "-l", action="store_true", help="Listar modelos disponibles en el Hub")
    ap.add_argument("--hub", default=DEFAULT_HUB, help="URL base del repositorio de modelos")
    ap.add_argument("--dir", default=None, help="Directorio destino personalizado")
    args = ap.parse_args()
    
    if args.list or args.model is None:
        print(f"\n{BOLD}{CYAN}Catálogo de Modelos para Aquifer / Caudal Engine:{RESET}")
        print(f"  {CYAN}⚡ Powered by Semantic Geometry Chile{RESET}")
        print("=" * 85)
        print(f"{'Clave':<8} {'Nombre':<26} {'Params':<8} {'Descarga':<11} {'RAM (VmRSS)':<12} {'Velocidad'}")
        print("-" * 85)
        for k, v in REGISTRY.items():
            print(f"{YELLOW}{k:<8}{RESET} {v['name']:<26} {v['params']:<8} ~{v['size_mb']} MB    {v['desc']}")
        print("=" * 85)
        print(f"Uso: {BOLD}python3 tools/pull.py <clave>{RESET}\n")
        return 0
        
    pull_model(args.model, target_dir=args.dir, hub_base=args.hub)
    return 0


if __name__ == "__main__":
    sys.exit(main())
