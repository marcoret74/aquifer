#!/usr/bin/env python3
"""prefix_cache.py — Mide la caché de prefijo de Aquifer y Ollama.

Mismos turnos, mismo system prompt. En Aquifer usa `--prefix-report` para ver
cuántos tokens reutiliza; en Ollama usa prompt_eval_count/duration del API.

Uso:
  python3 bench/prefix_cache.py \
    --artifact /ruta/qwen3b_affine \
    --tokenizer /home/marco/.cache/caudal/qwen3b/tokenizer.json \
    --ollama-model qwen2.5:3b
"""
import argparse
import json
import os
import subprocess
import sys
import time
import urllib.request

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "tools"))
from oracle_qwen import Tokenizer  # noqa: E402

TURNS = [
    "Hola, responde en una frase.",
    "¿Y qué más puedes decirme?",
    "Dame un ejemplo breve.",
]


def chatml(history, system):
    parts = []
    if system:
        parts.append(f"<|im_start|>system\n{system}<|im_end|>\n")
    for role, text in history:
        parts.append(f"<|im_start|>{role}\n{text}<|im_end|>\n")
    parts.append("<|im_start|>assistant\n")
    return "".join(parts)


def run_aquifer(a, tok):
    cmd = [a.engine, a.artifact, "--interactive", "--n", str(a.n),
           "--ctx", str(a.ctx), "--eos", "-1", "--threads", str(a.threads),
           "--mem-policy", "auto", "--prefill-batch", str(a.prefill_batch),
           "--prefix-report"]
    p = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                         stderr=subprocess.STDOUT, text=True, bufsize=1)
    for line in p.stdout:
        if line.strip() == "READY":
            break

    sys_tokens = tok.encode(chatml([], a.system))
    p.stdin.write("PREWARM " + ",".join(map(str, sys_tokens)) + "\n")
    p.stdin.flush()
    for line in p.stdout:
        if line.strip() == "PREWARMED":
            break

    history = []
    rows = []
    for turn, user in enumerate(TURNS[:a.turns], 1):
        history.append(("user", user))
        ids = tok.encode(chatml(history, a.system))
        prefix = None
        first = None
        resp = []
        t0 = time.perf_counter()
        p.stdin.write("PROMPT " + ",".join(map(str, ids)) + "\n")
        p.stdin.flush()
        for line in p.stdout:
            s = line.strip()
            if s.startswith("PREFIX "):
                _, common, total = s.split()
                prefix = (int(common), int(total))
            elif s.startswith("TOKEN "):
                if first is None:
                    first = time.perf_counter() - t0
                resp.append(int(s.split()[1]))
            elif s == "DONE":
                break
        total_t = time.perf_counter() - t0
        text = tok.decode(resp).replace("<|im_end|>", "").replace("</s>", "").strip()
        history.append(("assistant", text))
        dec = (len(resp) - 1) / max(1e-9, total_t - (first or 0)) if len(resp) > 1 else 0.0
        rows.append({"turn": turn, "prefix": prefix, "ttft_ms": (first or 0) * 1000,
                     "tokens": len(resp), "decode_tok_s": dec, "wall_s": total_t})
    p.stdin.write("QUIT\n")
    p.stdin.flush()
    try:
        p.wait(timeout=5)
    except Exception:
        p.kill()
    return rows


def run_ollama(a):
    url = a.ollama_url.rstrip("/") + "/api/chat"
    msgs = []
    if a.system:
        msgs.append({"role": "system", "content": a.system})
    rows = []
    for turn, user in enumerate(TURNS[:a.turns], 1):
        msgs.append({"role": "user", "content": user})
        body = json.dumps({
            "model": a.ollama_model, "messages": msgs, "stream": False,
            "keep_alive": "10m",
            "options": {"num_predict": a.n, "num_ctx": a.ctx,
                        "num_thread": a.threads, "num_gpu": 0},
        }).encode()
        req = urllib.request.Request(url, data=body,
                                     headers={"Content-Type": "application/json"})
        t0 = time.perf_counter()
        with urllib.request.urlopen(req, timeout=300) as r:
            d = json.load(r)
        wall = time.perf_counter() - t0
        pe_n = d.get("prompt_eval_count", 0)
        pe = d.get("prompt_eval_duration", 1) / 1e9
        ev_n = d.get("eval_count", 0)
        ev = d.get("eval_duration", 1) / 1e9
        rows.append({"turn": turn, "prompt_tokens": pe_n, "prefill_s": pe,
                     "prefill_tok_s": pe_n / pe if pe else 0,
                     "decode_tok_s": ev_n / ev if ev else 0,
                     "ttft_s": pe + (ev / ev_n if ev_n else 0), "wall_s": wall})
        msgs.append({"role": "assistant", "content": d["message"]["content"]})
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--engine", default=os.path.join(
        os.path.dirname(__file__), "..", "build", "generate_aquifer"))
    ap.add_argument("--artifact", required=True)
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--ollama-model", default="qwen2.5:3b")
    ap.add_argument("--ollama-url", default="http://localhost:11434")
    ap.add_argument("--turns", type=int, default=3)
    ap.add_argument("--n", type=int, default=12)
    ap.add_argument("--ctx", type=int, default=1024)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--prefill-batch", type=int, default=8)
    ap.add_argument("--system", default="You are a helpful assistant.")
    ap.add_argument("--nonce", default=None,
                    help="Sufijo único del system para forzar prompt frío (default: timestamp)")
    ap.add_argument("--no-ollama", action="store_true")
    a = ap.parse_args()

    nonce = a.nonce if a.nonce else str(time.time_ns())
    a.system = a.system + f" [run {nonce}]"
    tok = Tokenizer(a.tokenizer)
    print(f"prefix cache bench · turns={a.turns} n={a.n} ctx={a.ctx} threads={a.threads} nonce={nonce}")
    print(f"artifact: {a.artifact}\n")

    rows = run_aquifer(a, tok)
    print("AQUIFER")
    print(f"{'turn':<5} {'prefix':<12} {'TTFT ms':<10} {'decode tok/s':<14}")
    for r in rows:
        pref = f"{r['prefix'][0]}/{r['prefix'][1]}" if r["prefix"] else "?"
        print(f"{r['turn']:<5} {pref:<12} {r['ttft_ms']:<10.0f} {r['decode_tok_s']:<14.2f}")

    if a.no_ollama:
        return
    try:
        o = run_ollama(a)
    except Exception as e:
        print(f"\nOLLAMA: no disponible ({e})")
        return
    print("\nOLLAMA")
    print(f"{'turn':<5} {'prompt tok':<11} {'prefill s':<10} {'prefill tok/s':<14} "
          f"{'decode tok/s':<14} {'TTFT s':<8}")
    for r in o:
        print(f"{r['turn']:<5} {r['prompt_tokens']:<11} {r['prefill_s']:<10.3f} "
              f"{r['prefill_tok_s']:<14.1f} {r['decode_tok_s']:<14.2f} {r['ttft_s']:<8.3f}")


if __name__ == "__main__":
    main()
