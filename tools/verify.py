#!/usr/bin/env python3
"""verify.py — verificación end-to-end de un bundle de Aquifer.

Comprueba en un solo comando:
  - hashes de archivos y del manifiesto,
  - hash semántico de la conversación,
  - firma del manifiesto (HMAC u OpenSSL),
  - cadena de trazas append-only,
  - recibos HMAC por request,
  - cadena de bundles padres.

Uso:
    python3 tools/verify.py --bundle RUTA [--hmac-key CLAVE]
        [--pubkey PEM] [--traces-hmac-key CLAVE] [--parent-bundle RUTA] [--json]
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bundle  # noqa: E402


def _check(checks, name):
    for check in checks:
        if check[0] == name:
            return check
    return None


def _files(checks):
    return [c for c in checks if c[0].startswith("archivo ")]


def build_summary(bundle_path, result):
    manifest = result.get("manifest") or {}
    checks = result.get("checks") or []
    session = manifest.get("session") or {}
    files = _files(checks)
    manifest_check = _check(checks, "manifest.sha256")
    signature_check = _check(checks, "firma")
    chain_check = _check(checks, "cadena de trazas")
    receipt_check = _check(checks, "recibos")
    parent_check = _check(checks, "cadena de padres")
    traces = manifest.get("traces") or {}

    def ok_of(check):
        return bool(check[1]) if check else None

    return {
        "ok": bool(result.get("ok")),
        "bundle": os.path.abspath(bundle_path),
        "session": {
            "name": session.get("name"),
            "model": session.get("model"),
            "turns": session.get("turns"),
            "messages": session.get("messages"),
            "conversation_sha256": session.get("conversation_sha256"),
        },
        "manifest_sha256": manifest_check[2] if manifest_check else None,
        "kv": manifest.get("kv"),
        "traces": {
            "included": bool(traces.get("included")),
            "chain": traces.get("chain"),
            "receipts": traces.get("receipts"),
        },
        "signature": signature_check[2] if signature_check else None,
        "files": [{"name": c[0], "ok": bool(c[1]), "detail": c[2]}
                  for c in files],
        "checks": {
            "manifest": ok_of(manifest_check),
            "session": ok_of(_check(checks, "session.json")),
            "conversation": ok_of(_check(checks, "conversation_sha256")),
            "signature": ok_of(signature_check),
            "trace_chain": ok_of(chain_check),
            "receipts": ok_of(receipt_check),
            "parent": ok_of(parent_check),
        },
        "problems": result.get("problems") or [],
    }


def print_summary(summary):
    session = summary["session"]
    kv = summary.get("kv") or {}
    traces = summary.get("traces") or {}
    chain = traces.get("chain") or {}
    receipts = traces.get("receipts") or {}

    def mark(value):
        if value is True:
            return "OK   "
        if value is False:
            return "FALLA"
        return "  -  "

    print("AQUIFER · VERIFICACIÓN END-TO-END")
    print("=" * 58)
    print("Bundle      : %s" % summary["bundle"])
    print("Sesión      : %s (%s, %s turnos)" % (
        session.get("name") or "?", session.get("model") or "?",
        session.get("turns") if session.get("turns") is not None else "?"))
    print("Conversación: %s" % (session.get("conversation_sha256") or "?"))
    print("Manifiesto  : %s" % (summary.get("manifest_sha256") or "?"))
    if kv:
        print("KV          : %s (%s)" % (
            "incluido" if kv.get("included") else "solo hash",
            kv.get("sha256", "")[:16]))
    else:
        print("KV          : no registrado")
    if traces.get("included"):
        print("Trazas      : %s entradas, cadena %s, recibos %s" % (
            chain.get("entries", "?"),
            "OK" if chain.get("ok") else "FALLA",
            receipts.get("entries", 0)))
    print("-" * 58)
    print("[%s] manifiesto y archivos (%d)" %
          (mark(summary["checks"]["manifest"]), len(summary["files"])))
    print("[%s] historial de sesión" % mark(summary["checks"]["session"]))
    print("[%s] hash de conversación" % mark(summary["checks"]["conversation"]))
    print("[%s] firma (%s)" %
          (mark(summary["checks"]["signature"]),
           summary.get("signature") or "sin firma"))
    if traces.get("included"):
        print("[%s] cadena de trazas" % mark(summary["checks"]["trace_chain"]))
        if receipts.get("entries"):
            print("[%s] recibos por request (%d)" %
                  (mark(summary["checks"]["receipts"]),
                   receipts.get("entries")))
    if summary["checks"]["parent"] is not None:
        print("[%s] cadena de bundles padres" %
              mark(summary["checks"]["parent"]))
    print("-" * 58)
    if summary["ok"]:
        print("RESULTADO: VERIFICADO")
    else:
        print("RESULTADO: FALLÓ")
        for problem in summary["problems"]:
            print("  - %s" % problem)


def main():
    ap = argparse.ArgumentParser(
        description="Verificación end-to-end de un bundle de Aquifer")
    ap.add_argument("--bundle", required=True, metavar="RUTA")
    ap.add_argument("--hmac-key", default=None, metavar="CLAVE")
    ap.add_argument("--pubkey", default=None, metavar="PEM")
    ap.add_argument("--traces-hmac-key", default=None, metavar="CLAVE")
    ap.add_argument("--parent-bundle", default=None, metavar="RUTA")
    ap.add_argument("--json", action="store_true",
                    help="Salida JSON para integrar en un orquestador")
    args = ap.parse_args()

    try:
        result = bundle.verify_bundle(
            args.bundle, hmac_key=args.hmac_key, pubkey=args.pubkey,
            parent_bundle=args.parent_bundle,
            traces_hmac_key=args.traces_hmac_key)
    except bundle.BundleError as err:
        print("Error: %s" % err, file=sys.stderr)
        return 2

    summary = build_summary(args.bundle, result)
    if args.json:
        print(json.dumps(summary, ensure_ascii=False, indent=2,
                         sort_keys=True))
    else:
        print_summary(summary)
    return 0 if summary["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
