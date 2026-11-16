#!/usr/bin/env python3
"""receipts.py — recibos HMAC-SHA256 por request.

Un recibo firma un subconjunto canónico de la traza: id, timestamp, modelo,
estado, hashes de prompt/conversación/esquema/respuesta, tokens, latencia,
herramientas y error. No guarda contenido: solo hashes y metadatos.

La misma clave HMAC que firma los bundles (`--hmac-key`) puede firmar recibos
(`--receipt-key` en el servidor); `bundle.py` los verifica con
`--traces-hmac-key`.
"""

import hashlib
import hmac
import json

FIELDS = (
    "request_id", "ts", "model", "stream", "prompt_tokens",
    "completion_tokens", "finish_reason", "prompt_sha256",
    "conversation_sha256", "schema_sha256", "schema_attempts",
    "schema_invalid", "response_sha256", "latency_ms", "ttft_ms",
    "tool_execution", "tool_rounds", "tool_local_calls", "error_type",
)


def canonical_bytes(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True,
                      separators=(",", ":")).encode("utf-8")


def sha256_hex(data):
    if isinstance(data, str):
        data = data.encode("utf-8")
    return hashlib.sha256(data).hexdigest()


def key_id(key):
    return sha256_hex(key)


def payload(data, status=None):
    out = {}
    effective = status if status is not None else data.get("status")
    if effective is not None:
        out["status"] = effective
    for field in FIELDS:
        value = data.get(field)
        if value is not None:
            out[field] = value
    return out


def sign(data, key, status=None):
    raw = canonical_bytes(payload(data, status))
    return {
        "alg": "hmac-sha256",
        "key_sha256": key_id(key),
        "payload_sha256": sha256_hex(raw),
        "sig": hmac.new(key, raw, hashlib.sha256).hexdigest(),
    }


def verify(data, key):
    receipt = data.get("receipt")
    if not isinstance(receipt, dict):
        return False, "sin recibo"
    if receipt.get("alg") != "hmac-sha256":
        return False, "algoritmo no soportado"
    if receipt.get("key_sha256") != key_id(key):
        return False, "clave distinta"
    raw = canonical_bytes(payload(data))
    expected = hmac.new(key, raw, hashlib.sha256).hexdigest()
    if not hmac.compare_digest(expected, str(receipt.get("sig", ""))):
        return False, "firma no coincide"
    if receipt.get("payload_sha256") != sha256_hex(raw):
        return False, "payload_sha256 no coincide"
    return True, ""


def verify_lines(lines, key=None):
    """Verifica los recibos de un trace. Sin clave, solo los cuenta."""
    entries = 0
    problems = []
    verified = key is not None
    for lineno, line in enumerate(lines, 1):
        if isinstance(line, bytes):
            try:
                line = line.decode("utf-8")
            except UnicodeDecodeError:
                problems.append("línea %d: UTF-8 inválido" % lineno)
                continue
        line = line.strip()
        if not line:
            continue
        try:
            data = json.loads(line)
        except ValueError:
            problems.append("línea %d: JSON inválido" % lineno)
            continue
        if not isinstance(data.get("receipt"), dict):
            continue
        entries += 1
        if not verified:
            continue
        ok, why = verify(data, key)
        if not ok:
            problems.append("línea %d: %s" % (lineno, why))
    return {
        "ok": (None if not verified else not problems),
        "verified": verified,
        "entries": entries,
        "problems": problems,
    }
