#!/usr/bin/env python3
"""bundle.py — empaqueta una sesión de Aquifer en un bundle auditable.

Un bundle contiene:

    session.json      historial y metadatos de la sesión
    session.kv        estado del motor (opcional con --include-kv)
    traces.jsonl      trazas del servidor (opcional con --traces)
    manifest.json     hashes SHA-256, metadatos y cadena de padres
    manifest.sha256   hash del manifiesto (formato sha256sum)
    manifest.sig      firma del manifiesto (opcional)
    signature.json    algoritmo y huella de la clave (opcional)

La firma cubre los bytes exactos de manifest.json. El manifiesto cubre el
contenido de los archivos. No interpreta el contenido: sirve para que un
orquestador o un tercero pueda verificar que un paquete no fue alterado.

Uso:
    python3 tools/bundle.py export --session RUTA --out DESTINO \\
        [--include-kv] [--traces TRACE.jsonl] [--note TEXTO] \\
        [--hmac-key CLAVE] [--sign-key CLAVE.pem] [--parent-bundle B]

    python3 tools/bundle.py verify --bundle DESTINO \\
        [--hmac-key CLAVE] [--pubkey PUB.pem] [--parent-bundle B]

    python3 tools/bundle.py inspect --bundle DESTINO

DESTINO puede ser un directorio o un archivo .zip.
"""

import argparse
import hashlib
import hmac
import json
import os
import subprocess
import sys
import tempfile
import time
import zipfile

BUNDLE_VERSION = 1
GENERATOR = "aquifer-tools/bundle"


class BundleError(Exception):
    pass


def canonical_bytes(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True,
                      separators=(",", ":")).encode("utf-8")


def sha256_bytes(data):
    return hashlib.sha256(data).hexdigest()


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def canonical_hash(value):
    return sha256_bytes(canonical_bytes(value))


def verify_trace_chain(lines, hmac_key=None):
    """Verifica una cadena de trazas: prev, seq, sha256 y HMAC opcional."""
    zeros = "0" * 64
    prev = zeros
    expected_seq = 1
    entries = 0
    head = zeros
    problems = []
    warnings = []
    for lineno, raw in enumerate(lines, 1):
        if isinstance(raw, bytes):
            try:
                raw = raw.decode("utf-8")
            except UnicodeDecodeError:
                problems.append("línea %d: UTF-8 inválido" % lineno)
                continue
        line = raw.strip()
        if not line:
            continue
        try:
            data = json.loads(line)
        except ValueError:
            problems.append("línea %d: JSON inválido" % lineno)
            continue
        chain = data.get("chain")
        if not isinstance(chain, dict):
            # Prefijo legado sin cadena: se ignora para la verificación.
            continue
        body = {k: v for k, v in data.items() if k != "chain"}
        entry = canonical_hash({"prev": chain.get("prev"), "body": body})
        if chain.get("prev") != prev:
            problems.append("línea %d: prev no continúa la cadena" % lineno)
        if chain.get("sha256") != entry:
            problems.append("línea %d: hash de entrada no coincide" % lineno)
        if chain.get("seq") != expected_seq:
            problems.append("línea %d: seq %r != %d" %
                            (lineno, chain.get("seq"), expected_seq))
        sig = chain.get("sig")
        if sig:
            if not hmac_key:
                warnings.append("línea %d: firma HMAC no verificada" % lineno)
            else:
                expected = hmac.new(hmac_key, entry.encode("ascii"),
                                    hashlib.sha256).hexdigest()
                if not hmac.compare_digest(expected, sig):
                    problems.append("línea %d: HMAC no coincide" % lineno)
        prev = str(chain.get("sha256") or "")
        head = prev
        expected_seq += 1
        entries += 1
    return {
        "ok": not problems,
        "entries": entries,
        "head": head,
        "problems": problems,
        "warnings": warnings,
    }


def conversation_hash(session):
    messages = [[str(r), str(t)] for r, t in session.get("history", [])]
    return sha256_bytes(canonical_bytes({
        "system": session.get("system", ""),
        "messages": messages,
    }))


def _read_bundle(path):
    """Devuelve {nombre: bytes} para un directorio o zip de bundle."""
    if os.path.isdir(path):
        out = {}
        for name in sorted(os.listdir(path)):
            full = os.path.join(path, name)
            if os.path.isfile(full):
                with open(full, "rb") as f:
                    out[name] = f.read()
        return out
    if zipfile.is_zipfile(path):
        out = {}
        with zipfile.ZipFile(path) as z:
            for name in z.namelist():
                if name.endswith("/"):
                    continue
                out[os.path.basename(name)] = z.read(name)
        return out
    raise BundleError("no es un directorio ni un zip: %s" % path)


def _parse_manifest(raw):
    try:
        data = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, ValueError) as err:
        raise BundleError("manifest.json inválido: %s" % err)
    if not isinstance(data, dict):
        raise BundleError("manifest.json no es un objeto")
    if data.get("bundle_version") != BUNDLE_VERSION:
        raise BundleError("versión de bundle no soportada")
    return data


def _key_is_ed(key_path):
    try:
        out = subprocess.run(
            ["openssl", "pkey", "-in", key_path, "-noout", "-text"],
            capture_output=True, text=True)
    except OSError:
        raise BundleError("openssl no está disponible")
    text = (out.stdout or "").upper()
    return "ED25519" in text or "ED448" in text


def _pubkey_fingerprint(key_path):
    try:
        out = subprocess.run(
            ["openssl", "pkey", "-in", key_path, "-pubout", "-outform", "DER"],
            capture_output=True)
    except OSError:
        raise BundleError("openssl no está disponible")
    if out.returncode != 0 or not out.stdout:
        raise BundleError("no se pudo derivar la clave pública de %s" % key_path)
    return sha256_bytes(out.stdout)


def _sign_manifest(manifest_path, sig_path, key_path):
    ed = _key_is_ed(key_path)
    if ed:
        cmd = ["openssl", "pkeyutl", "-sign", "-inkey", key_path, "-rawin",
               "-in", manifest_path, "-out", sig_path]
    else:
        cmd = ["openssl", "dgst", "-sha256", "-sign", key_path,
               "-out", sig_path, manifest_path]
    try:
        out = subprocess.run(cmd, capture_output=True, text=True)
    except OSError:
        raise BundleError("openssl no está disponible")
    if out.returncode != 0:
        raise BundleError("falló la firma: %s" % (out.stderr.strip() or "openssl"))
    return "ed25519" if ed else "openssl-sha256"


def _verify_manifest(manifest_path, sig_path, pubkey_path, alg):
    if alg == "ed25519":
        cmd = ["openssl", "pkeyutl", "-verify", "-pubin", "-inkey", pubkey_path,
               "-rawin", "-in", manifest_path, "-sigfile", sig_path]
    else:
        cmd = ["openssl", "dgst", "-sha256", "-verify", pubkey_path,
               "-signature", sig_path, manifest_path]
    try:
        out = subprocess.run(cmd, capture_output=True, text=True)
    except OSError:
        raise BundleError("openssl no está disponible")
    return out.returncode == 0, (out.stdout + out.stderr).strip()


def _write_bundle(stage, out):
    files = sorted(os.listdir(stage))
    if out.lower().endswith(".zip"):
        if os.path.exists(out):
            os.remove(out)
        os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
        with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
            for name in files:
                z.write(os.path.join(stage, name), name)
        return out
    if os.path.exists(out) and os.listdir(out):
        raise BundleError("el destino existe y no está vacío: %s" % out)
    os.makedirs(out, exist_ok=True)
    for name in files:
        with open(os.path.join(stage, name), "rb") as src:
            with open(os.path.join(out, name), "wb") as dst:
                dst.write(src.read())
    return out


def export_bundle(session_base, out, include_kv=False, traces=None, note=None,
                  hmac_key=None, sign_key=None, parent_bundle=None,
                  traces_hmac_key=None):
    kv_path = session_base + ".kv"
    json_path = session_base + ".json"
    if not os.path.exists(json_path):
        raise BundleError("no existe la sesión: %s" % json_path)
    if include_kv and not os.path.exists(kv_path):
        raise BundleError("no existe el KV: %s" % kv_path)
    if hmac_key and sign_key:
        raise BundleError("elegí una firma: --hmac-key o --sign-key, no ambas")

    with open(json_path, "rb") as f:
        session_raw = f.read()
    try:
        session = json.loads(session_raw.decode("utf-8"))
    except (UnicodeDecodeError, ValueError) as err:
        raise BundleError("session.json inválido: %s" % err)

    parent = None
    if parent_bundle:
        parent_files = _read_bundle(parent_bundle)
        if "manifest.json" not in parent_files:
            raise BundleError("el bundle padre no tiene manifest.json")
        parent_manifest = parent_files["manifest.json"]
        _parse_manifest(parent_manifest)
        parent = {
            "manifest_sha256": sha256_bytes(parent_manifest),
            "path": os.path.abspath(parent_bundle),
        }

    entries = []
    stage = tempfile.mkdtemp(prefix="aquifer-bundle-")
    try:
        with open(os.path.join(stage, "session.json"), "wb") as f:
            f.write(session_raw)
        entries.append({
            "path": "session.json",
            "sha256": sha256_bytes(session_raw),
            "bytes": len(session_raw),
            "included": True,
        })

        kv_meta = None
        if os.path.exists(kv_path):
            kv_hash = sha256_file(kv_path)
            kv_bytes = os.path.getsize(kv_path)
            kv_meta = {"sha256": kv_hash, "bytes": kv_bytes,
                       "included": bool(include_kv)}
            entries.append({"path": "session.kv", "sha256": kv_hash,
                            "bytes": kv_bytes, "included": bool(include_kv)})
            if include_kv:
                with open(kv_path, "rb") as src, \
                        open(os.path.join(stage, "session.kv"), "wb") as dst:
                    for chunk in iter(lambda: src.read(1024 * 1024), b""):
                        dst.write(chunk)

        traces_meta = None
        if traces:
            if not os.path.exists(traces):
                raise BundleError("no existe el trace log: %s" % traces)
            with open(traces, "rb") as f:
                traces_raw = f.read()
            chain_key = None
            if traces_hmac_key:
                with open(traces_hmac_key, "rb") as f:
                    chain_key = f.read()
            chain = verify_trace_chain(traces_raw.splitlines(), chain_key)
            if not chain["ok"]:
                raise BundleError("la cadena de trazas está rota: %s" %
                                  "; ".join(chain["problems"][:3]))
            with open(os.path.join(stage, "traces.jsonl"), "wb") as f:
                f.write(traces_raw)
            traces_meta = {
                "sha256": sha256_bytes(traces_raw),
                "bytes": len(traces_raw),
                "included": True,
                "lines": traces_raw.count(b"\n"),
                "source": os.path.basename(traces),
                "chain": {
                    "ok": chain["ok"],
                    "entries": chain["entries"],
                    "head": chain["head"],
                },
            }
            entries.append({"path": "traces.jsonl",
                            "sha256": traces_meta["sha256"],
                            "bytes": traces_meta["bytes"], "included": True})

        manifest = {
            "bundle_version": BUNDLE_VERSION,
            "generator": GENERATOR,
            "created": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "session": {
                "name": os.path.basename(session_base),
                "model": session.get("model"),
                "created": session.get("created"),
                "updated": session.get("updated"),
                "turns": session.get("turns"),
                "parent": session.get("parent"),
                "system_sha256": sha256_bytes(
                    str(session.get("system", "")).encode("utf-8")),
                "conversation_sha256": conversation_hash(session),
                "messages": len(session.get("history", [])),
            },
            "files": entries,
            "kv": kv_meta,
            "traces": traces_meta,
            "parent": parent,
            "note": note,
        }

        manifest_path = os.path.join(stage, "manifest.json")
        manifest_raw = (json.dumps(manifest, ensure_ascii=False, indent=2,
                                   sort_keys=True) + "\n").encode("utf-8")
        with open(manifest_path, "wb") as f:
            f.write(manifest_raw)
        with open(os.path.join(stage, "manifest.sha256"), "w",
                  encoding="ascii") as f:
            f.write("%s  manifest.json\n" % sha256_bytes(manifest_raw))

        signature = None
        if hmac_key:
            with open(hmac_key, "rb") as f:
                key = f.read()
            if len(key) < 16:
                raise BundleError("la clave HMAC debe tener al menos 16 bytes")
            sig = hmac.new(key, manifest_raw, hashlib.sha256).digest()
            with open(os.path.join(stage, "manifest.sig"), "wb") as f:
                f.write(sig)
            signature = {"alg": "hmac-sha256",
                         "key_sha256": sha256_bytes(key)}
        elif sign_key:
            sig_path = os.path.join(stage, "manifest.sig")
            alg = _sign_manifest(manifest_path, sig_path, sign_key)
            signature = {"alg": alg,
                         "key_sha256": _pubkey_fingerprint(sign_key)}
        if signature:
            with open(os.path.join(stage, "signature.json"), "w",
                      encoding="utf-8") as f:
                json.dump(signature, f, ensure_ascii=False, indent=2,
                          sort_keys=True)
                f.write("\n")

        result = _write_bundle(stage, out)
    finally:
        for name in os.listdir(stage):
            try:
                os.remove(os.path.join(stage, name))
            except OSError:
                pass
        try:
            os.rmdir(stage)
        except OSError:
            pass

    return {
        "bundle": result,
        "manifest_sha256": sha256_bytes(manifest_raw),
        "conversation_sha256": manifest["session"]["conversation_sha256"],
        "files": [e["path"] for e in entries],
        "signed": signature["alg"] if signature else None,
        "kv_included": bool(include_kv),
    }


def verify_bundle(bundle, hmac_key=None, pubkey=None, parent_bundle=None,
                  traces_hmac_key=None):
    files = _read_bundle(bundle)
    problems = []
    checks = []

    def check(name, ok, detail=""):
        checks.append((name, bool(ok), detail))
        if not ok:
            problems.append(name)

    if "manifest.json" not in files:
        raise BundleError("falta manifest.json")
    manifest_raw = files["manifest.json"]
    manifest = _parse_manifest(manifest_raw)

    expected_hash = sha256_bytes(manifest_raw)
    if "manifest.sha256" in files:
        line = files["manifest.sha256"].decode("ascii", "replace").split()
        check("manifest.sha256", bool(line) and line[0] == expected_hash,
              "%s" % expected_hash)
    else:
        check("manifest.sha256", False, "falta manifest.sha256")

    session = None
    if "session.json" in files:
        try:
            session = json.loads(files["session.json"].decode("utf-8"))
        except (UnicodeDecodeError, ValueError):
            session = None
    check("session.json", session is not None)
    if session is not None:
        check("conversation_sha256",
              conversation_hash(session) ==
              manifest.get("session", {}).get("conversation_sha256"),
              manifest.get("session", {}).get("conversation_sha256", ""))

    for entry in manifest.get("files", []):
        path = entry.get("path")
        if not path or path == "manifest.json":
            continue
        if not entry.get("included", True):
            check("archivo %s" % path, True,
                  "no incluido (sha256 registrado: %s)" % entry.get("sha256"))
            continue
        data = files.get(path)
        if data is None:
            check("archivo %s" % path, False, "falta en el bundle")
            continue
        ok = sha256_bytes(data) == entry.get("sha256")
        check("archivo %s" % path, ok, "%d bytes" % len(data))

    signature = None
    if "signature.json" in files:
        try:
            signature = json.loads(files["signature.json"].decode("utf-8"))
        except (UnicodeDecodeError, ValueError):
            signature = None
    if "manifest.sig" in files:
        if signature is None:
            check("firma", False, "falta signature.json")
        elif signature.get("alg") == "hmac-sha256":
            if not hmac_key:
                check("firma", False, "se requiere --hmac-key")
            else:
                with open(hmac_key, "rb") as f:
                    key = f.read()
                expected = hmac.new(key, manifest_raw, hashlib.sha256).digest()
                check("firma", hmac.compare_digest(expected, files["manifest.sig"]),
                      "hmac-sha256")
        else:
            if not pubkey:
                check("firma", False, "se requiere --pubkey")
            else:
                with tempfile.TemporaryDirectory(prefix="aquifer-verify-") as tmp:
                    mpath = os.path.join(tmp, "manifest.json")
                    spath = os.path.join(tmp, "manifest.sig")
                    with open(mpath, "wb") as f:
                        f.write(manifest_raw)
                    with open(spath, "wb") as f:
                        f.write(files["manifest.sig"])
                    ok, detail = _verify_manifest(
                        mpath, spath, pubkey, signature.get("alg"))
                    check("firma", ok, signature.get("alg", "?"))
    else:
        check("firma", True, "sin firma (solo hashes)")

    if "traces.jsonl" in files and manifest.get("traces"):
        chain_key = None
        if traces_hmac_key:
            with open(traces_hmac_key, "rb") as f:
                chain_key = f.read()
        chain = verify_trace_chain(files["traces.jsonl"].splitlines(), chain_key)
        check("cadena de trazas", chain["ok"],
              "%d entradas, head %s" % (chain["entries"], chain["head"][:16]))
        for warn in chain["warnings"][:3]:
            checks.append(("traza (aviso)", True, warn))

    parent = manifest.get("parent") or {}
    if parent.get("manifest_sha256"):
        if parent_bundle:
            try:
                pfiles = _read_bundle(parent_bundle)
                phash = sha256_bytes(pfiles["manifest.json"])
            except (BundleError, KeyError, OSError) as err:
                check("cadena de padres", False, str(err))
            else:
                check("cadena de padres",
                      phash == parent["manifest_sha256"], phash)
        else:
            check("cadena de padres", True,
                  "registrada; no se verificó (falta --parent-bundle)")

    return {
        "ok": not problems,
        "checks": checks,
        "manifest": manifest,
        "problems": problems,
    }


def _print_checks(result):
    for name, ok, detail in result["checks"]:
        mark = "OK  " if ok else "FALLA"
        print("  [%s] %-24s %s" % (mark, name, detail))
    print("")
    if result["ok"]:
        print("BUNDLE OK")
    else:
        print("BUNDLE INVÁLIDO: %s" % ", ".join(result["problems"]))


def main():
    ap = argparse.ArgumentParser(
        description="Bundle auditable de una sesión de Aquifer")
    sub = ap.add_subparsers(dest="cmd", required=True)

    ex = sub.add_parser("export", help="Crear un bundle")
    ex.add_argument("--session", required=True, metavar="RUTA",
                    help="Ruta base de la sesión (sin .json/.kv)")
    ex.add_argument("--out", required=True, metavar="DESTINO",
                    help="Directorio o archivo .zip")
    ex.add_argument("--include-kv", action="store_true",
                    help="Incluir session.kv (si no, solo se registra su hash)")
    ex.add_argument("--traces", default=None, metavar="JSONL",
                    help="Incluir un trace log del servidor")
    ex.add_argument("--note", default=None, help="Nota libre en el manifiesto")
    ex.add_argument("--traces-hmac-key", default=None, metavar="CLAVE",
                    help="Clave HMAC para verificar la cadena de trazas")
    ex.add_argument("--hmac-key", default=None, metavar="CLAVE",
                    help="Firmar con HMAC-SHA256 usando esta clave")
    ex.add_argument("--sign-key", default=None, metavar="PEM",
                    help="Firmar con OpenSSL (RSA/EC/Ed25519)")
    ex.add_argument("--parent-bundle", default=None, metavar="BUNDLE",
                    help="Bundle padre para registrar la cadena")

    ve = sub.add_parser("verify", help="Verificar un bundle")
    ve.add_argument("--bundle", required=True, metavar="DESTINO")
    ve.add_argument("--hmac-key", default=None, metavar="CLAVE")
    ve.add_argument("--pubkey", default=None, metavar="PEM")
    ve.add_argument("--parent-bundle", default=None, metavar="BUNDLE")
    ve.add_argument("--traces-hmac-key", default=None, metavar="CLAVE")

    ins = sub.add_parser("inspect", help="Mostrar el manifiesto")
    ins.add_argument("--bundle", required=True, metavar="DESTINO")

    vt = sub.add_parser("verify-traces",
                        help="Verificar una cadena de trazas JSONL")
    vt.add_argument("--file", required=True, metavar="JSONL")
    vt.add_argument("--hmac-key", default=None, metavar="CLAVE")

    args = ap.parse_args()
    try:
        if args.cmd == "export":
            result = export_bundle(
                args.session, args.out, include_kv=args.include_kv,
                traces=args.traces, note=args.note,
                hmac_key=args.hmac_key, sign_key=args.sign_key,
                parent_bundle=args.parent_bundle,
                traces_hmac_key=args.traces_hmac_key)
            print("Bundle   : %s" % result["bundle"])
            print("Manifest : %s" % result["manifest_sha256"])
            print("Conversa : %s" % result["conversation_sha256"])
            print("Archivos : %s" % ", ".join(result["files"]))
            print("KV       : %s" % ("incluido" if result["kv_included"]
                                     else "solo hash"))
            print("Firma    : %s" % (result["signed"] or "sin firma"))
            return 0
        if args.cmd == "verify":
            result = verify_bundle(args.bundle, hmac_key=args.hmac_key,
                                   pubkey=args.pubkey,
                                   parent_bundle=args.parent_bundle,
                                   traces_hmac_key=args.traces_hmac_key)
            _print_checks(result)
            return 0 if result["ok"] else 1
        if args.cmd == "inspect":
            files = _read_bundle(args.bundle)
            manifest = _parse_manifest(files["manifest.json"])
            print(json.dumps(manifest, ensure_ascii=False, indent=2,
                             sort_keys=True))
            return 0
        if args.cmd == "verify-traces":
            with open(args.file, "rb") as f:
                lines = f.read().splitlines()
            key = None
            if args.hmac_key:
                with open(args.hmac_key, "rb") as f:
                    key = f.read()
            result = verify_trace_chain(lines, key)
            print("Entradas : %d" % result["entries"])
            print("Head     : %s" % result["head"])
            for warn in result["warnings"][:10]:
                print("  [aviso] %s" % warn)
            if result["problems"]:
                print("PROBLEMAS:")
                for problem in result["problems"][:20]:
                    print("  - %s" % problem)
                return 1
            print("CADENA OK")
            return 0
    except BundleError as err:
        print("Error: %s" % err, file=sys.stderr)
        return 2
    return 2


if __name__ == "__main__":
    sys.exit(main())
