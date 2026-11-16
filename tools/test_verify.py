#!/usr/bin/env python3
"""test_verify.py — pruebas del CLI de verificación end-to-end.

Uso: python3 tools/test_verify.py
"""

import hashlib
import hmac
import json
import os
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
VERIFY = os.path.join(HERE, "verify.py")
sys.path.insert(0, HERE)
import bundle as B  # noqa: E402
import receipts  # noqa: E402

FAILED = []
TOTAL = 0


def check(name, cond, detail=""):
    global TOTAL
    TOTAL += 1
    if not cond:
        FAILED.append(name + ((" :: " + str(detail)) if detail else ""))


def run(*args):
    return subprocess.run([sys.executable, VERIFY] + list(args),
                          capture_output=True, text=True)


def make_session(root):
    base = os.path.join(root, "ses")
    session = {
        "version": 1, "model": "qwen05", "system": "Eres breve.",
        "history": [["user", "Hola"], ["assistant", "Hola!"],
                    ["user", "¿Qué tal?"], ["assistant", "Bien."]],
        "turns": 2, "created": "2026-01-01T00:00:00Z",
        "updated": "2026-01-01T00:01:00Z", "parent": None,
    }
    with open(base + ".json", "w", encoding="utf-8") as f:
        json.dump(session, f, ensure_ascii=False, indent=1)
    with open(base + ".kv", "wb") as f:
        f.write(b"kv" * 500)
    return base


def make_trace(path, key):
    body = {
        "request_id": "chatcmpl-x", "ts": "2026-01-01T00:00:00Z",
        "status": 200, "model": "qwen05", "prompt_tokens": 10,
        "completion_tokens": 5, "finish_reason": "stop",
        "prompt_sha256": "a" * 64, "conversation_sha256": "b" * 64,
        "response_sha256": "c" * 64, "latency_ms": 100.0,
    }
    body["receipt"] = receipts.sign(body, key)
    prev = "0" * 64
    entry = B.canonical_hash({"prev": prev, "body": body})
    sig = hmac.new(key, entry.encode("ascii"), hashlib.sha256).hexdigest()
    body["chain"] = {"seq": 1, "prev": prev, "sha256": entry, "sig": sig}
    with open(path, "w", encoding="utf-8") as f:
        f.write(json.dumps(body, ensure_ascii=False) + "\n")


def main():
    with tempfile.TemporaryDirectory(prefix="aquifer-verify-") as root:
        key = os.urandom(32)
        key_path = os.path.join(root, "key")
        with open(key_path, "wb") as f:
            f.write(key)
        base = make_session(root)
        trace = os.path.join(root, "traces.jsonl")
        make_trace(trace, key)
        out = os.path.join(root, "bundle")
        B.export_bundle(base, out, include_kv=True, traces=trace,
                        hmac_key=key_path, traces_hmac_key=key_path)

        r = run("--bundle", out, "--hmac-key", key_path,
                "--traces-hmac-key", key_path)
        check("verify ok", r.returncode == 0, r.stderr or r.stdout)
        check("verificado", "VERIFICADO" in r.stdout, r.stdout)
        check("recibos", "recibos" in r.stdout.lower(), r.stdout)
        check("cadena", "cadena de trazas" in r.stdout.lower(), r.stdout)

        r = run("--bundle", out, "--hmac-key", key_path,
                "--traces-hmac-key", key_path, "--json")
        data = json.loads(r.stdout) if r.returncode == 0 else {}
        check("json ok", data.get("ok") is True, data)
        check("json recibos", (data.get("traces", {}).get("receipts", {})
                               .get("entries")) == 1, data)
        check("json checks", data.get("checks", {}).get("signature") is True,
              data)

        r = run("--bundle", out)
        check("sin claves falla", r.returncode == 1, r.stdout)

        session_path = os.path.join(out, "session.json")
        with open(session_path, "ab") as f:
            f.write(b" ")
        r = run("--bundle", out, "--hmac-key", key_path,
                "--traces-hmac-key", key_path)
        check("tamper falla", r.returncode == 1, r.stdout)
        check("tamper mensaje", "FALLÓ" in r.stdout, r.stdout)

    if FAILED:
        print("FALLARON %d de %d pruebas:" % (len(FAILED), TOTAL))
        for name in FAILED:
            print("  - " + name)
        return 1
    print("verify: %d pruebas OK" % TOTAL)
    return 0


if __name__ == "__main__":
    sys.exit(main())
