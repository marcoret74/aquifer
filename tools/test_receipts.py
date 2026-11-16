#!/usr/bin/env python3
"""test_receipts.py — pruebas de recibos HMAC por request (sin modelo).

Uso: python3 tools/test_receipts.py
"""

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import receipts  # noqa: E402

FAILED = []
TOTAL = 0


def check(name, cond, detail=""):
    global TOTAL
    TOTAL += 1
    if not cond:
        FAILED.append(name + ((" :: " + str(detail)) if detail else ""))


def sample():
    return {
        "request_id": "chatcmpl-x",
        "ts": "2026-01-01T00:00:00Z",
        "status": 200,
        "model": "qwen05",
        "stream": False,
        "prompt_tokens": 10,
        "completion_tokens": 5,
        "finish_reason": "stop",
        "prompt_sha256": "a" * 64,
        "conversation_sha256": "b" * 64,
        "response_sha256": "c" * 64,
        "latency_ms": 120.0,
    }


def main():
    key = os.urandom(32)
    trace = sample()
    trace["receipt"] = receipts.sign(trace, key)

    ok, why = receipts.verify(trace, key)
    check("roundtrip", ok, why)
    check("key_id", trace["receipt"]["key_sha256"] == receipts.key_id(key))
    check("alg", trace["receipt"]["alg"] == "hmac-sha256")

    tampered = dict(trace)
    tampered["completion_tokens"] = 6
    ok, why = receipts.verify(tampered, key)
    check("detecta tamper", not ok, why)

    ok, why = receipts.verify(trace, os.urandom(32))
    check("detecta clave distinta", not ok, why)

    ok, why = receipts.verify(sample(), key)
    check("detecta sin recibo", not ok, why)

    # Payload no incluye el recibo: cambiarlo no invalida si no toca campos.
    other = dict(trace)
    other["receipt"] = dict(trace["receipt"], extra="x")
    ok, why = receipts.verify(other, key)
    check("recibo extra no afecta", ok, why)

    lines = [json.dumps(trace).encode()]
    result = receipts.verify_lines(lines, key)
    check("verify_lines", result["ok"] is True and result["entries"] == 1,
          result)

    result = receipts.verify_lines(lines, None)
    check("verify_lines sin clave", result["ok"] is None and
          result["entries"] == 1 and not result["verified"], result)

    bad = [json.dumps(tampered).encode()]
    result = receipts.verify_lines(bad, key)
    check("verify_lines tamper", result["ok"] is False, result)

    if FAILED:
        print("FALLARON %d de %d pruebas:" % (len(FAILED), TOTAL))
        for name in FAILED:
            print("  - " + name)
        return 1
    print("receipts: %d pruebas OK" % TOTAL)
    return 0


if __name__ == "__main__":
    sys.exit(main())
