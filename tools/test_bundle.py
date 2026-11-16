#!/usr/bin/env python3
"""test_bundle.py — pruebas del bundle auditable (sin modelo).

Uso: python3 tools/test_bundle.py
"""

import json
import os
import shutil
import subprocess
import sys
import tempfile
import time

HERE = os.path.dirname(os.path.abspath(__file__))
BUNDLE = os.path.join(HERE, "bundle.py")
FAILED = []
TOTAL = 0


def check(name, cond, detail=""):
    global TOTAL
    TOTAL += 1
    if not cond:
        FAILED.append(name + ((" :: " + str(detail)) if detail else ""))


def run(*args):
    return subprocess.run([sys.executable, BUNDLE] + list(args),
                          capture_output=True, text=True)


def make_session(root, name="ses"):
    base = os.path.join(root, name)
    session = {
        "version": 1,
        "model": "qwen05",
        "system": "Eres breve.",
        "history": [["user", "Hola"], ["assistant", "Hola!"],
                    ["user", "¿Cómo estás?"], ["assistant", "Bien."]],
        "turns": 2,
        "created": "2026-01-01T00:00:00Z",
        "updated": "2026-01-01T00:01:00Z",
        "parent": None,
    }
    with open(base + ".json", "w", encoding="utf-8") as f:
        json.dump(session, f, ensure_ascii=False, indent=1)
    with open(base + ".kv", "wb") as f:
        f.write(os.urandom(2048))
    return base


def tamper(path):
    with open(path, "ab") as f:
        f.write(b" ")


def test_hmac(root):
    base = make_session(root, "ses_hmac")
    key = os.path.join(root, "hmac.key")
    with open(key, "wb") as f:
        f.write(os.urandom(32))
    out = os.path.join(root, "b_hmac")

    r = run("export", "--session", base, "--out", out, "--hmac-key", key)
    check("hmac export", r.returncode == 0, r.stderr or r.stdout)
    check("hmac manifest", os.path.exists(os.path.join(out, "manifest.json")))
    check("hmac sig", os.path.exists(os.path.join(out, "manifest.sig")))

    r = run("verify", "--bundle", out, "--hmac-key", key)
    check("hmac verify", r.returncode == 0, r.stdout)

    r = run("verify", "--bundle", out)
    check("hmac sin clave falla", r.returncode == 1, r.stdout)

    session = os.path.join(out, "session.json")
    tamper(session)
    r = run("verify", "--bundle", out, "--hmac-key", key)
    check("hmac detecta tamper", r.returncode == 1, r.stdout)
    # restaurar
    r = run("export", "--session", base, "--out", os.path.join(root, "b_hmac2"),
            "--hmac-key", key)
    check("hmac re-export", r.returncode == 0, r.stderr)
    out2 = os.path.join(root, "b_hmac2")
    sig = os.path.join(out2, "manifest.sig")
    with open(sig, "r+b") as f:
        f.seek(0)
        f.write(b"\x00")
    r = run("verify", "--bundle", out2, "--hmac-key", key)
    check("hmac detecta firma alterada", r.returncode == 1, r.stdout)


def test_kv_and_zip(root):
    base = make_session(root, "ses_kv")
    out = os.path.join(root, "b_kv")
    r = run("export", "--session", base, "--out", out, "--include-kv")
    check("kv export", r.returncode == 0, r.stderr)
    check("kv incluido", os.path.exists(os.path.join(out, "session.kv")))
    r = run("verify", "--bundle", out)
    check("kv verify", r.returncode == 0, r.stdout)

    tamper(os.path.join(out, "session.kv"))
    r = run("verify", "--bundle", out)
    check("kv detecta tamper", r.returncode == 1, r.stdout)

    zpath = os.path.join(root, "b.zip")
    r = run("export", "--session", base, "--out", zpath, "--include-kv")
    check("zip export", r.returncode == 0, r.stderr)
    check("zip existe", os.path.exists(zpath))
    r = run("verify", "--bundle", zpath)
    check("zip verify", r.returncode == 0, r.stdout)
    r = run("inspect", "--bundle", zpath)
    check("zip inspect", r.returncode == 0 and "conversation_sha256" in r.stdout,
          r.stderr)


def test_parent(root):
    base = make_session(root, "ses_parent")
    parent = os.path.join(root, "b_parent")
    child = os.path.join(root, "b_child")
    r = run("export", "--session", base, "--out", parent)
    check("parent export", r.returncode == 0, r.stderr)
    r = run("export", "--session", base, "--out", child,
            "--parent-bundle", parent)
    check("child export", r.returncode == 0, r.stderr)
    r = run("verify", "--bundle", child, "--parent-bundle", parent)
    check("cadena ok", r.returncode == 0, r.stdout)
    r = run("verify", "--bundle", child, "--parent-bundle",
            os.path.join(root, "b_kv"))
    check("cadena rota", r.returncode == 1, r.stdout)


def test_openssl(root):
    if not shutil.which("openssl"):
        print("openssl no disponible: se salta la firma asimétrica")
        return
    base = make_session(root, "ses_ossl")

    priv = os.path.join(root, "rsa.pem")
    pub = os.path.join(root, "rsa.pub")
    r = subprocess.run(["openssl", "genpkey", "-algorithm", "RSA",
                        "-pkeyopt", "rsa_keygen_bits:2048", "-out", priv],
                       capture_output=True, text=True)
    check("generar RSA", r.returncode == 0, r.stderr)
    if r.returncode != 0:
        return
    r = subprocess.run(["openssl", "pkey", "-in", priv, "-pubout", "-out", pub],
                       capture_output=True, text=True)
    check("derivar pública RSA", r.returncode == 0, r.stderr)

    out = os.path.join(root, "b_rsa")
    r = run("export", "--session", base, "--out", out, "--sign-key", priv)
    check("firma RSA", r.returncode == 0, r.stderr)
    r = run("verify", "--bundle", out, "--pubkey", pub)
    check("verify RSA", r.returncode == 0, r.stdout)
    r = run("verify", "--bundle", out)
    check("RSA sin pubkey falla", r.returncode == 1, r.stdout)

    ed = os.path.join(root, "ed.pem")
    edpub = os.path.join(root, "ed.pub")
    r = subprocess.run(["openssl", "genpkey", "-algorithm", "ED25519",
                        "-out", ed], capture_output=True, text=True)
    check("generar Ed25519", r.returncode == 0, r.stderr)
    if r.returncode == 0:
        r = subprocess.run(["openssl", "pkey", "-in", ed, "-pubout",
                            "-out", edpub], capture_output=True, text=True)
        check("derivar pública Ed25519", r.returncode == 0, r.stderr)
        edout = os.path.join(root, "b_ed")
        r = run("export", "--session", base, "--out", edout, "--sign-key", ed)
        check("firma Ed25519", r.returncode == 0, r.stderr)
        r = run("verify", "--bundle", edout, "--pubkey", edpub)
        check("verify Ed25519", r.returncode == 0, r.stdout)


def main():
    with tempfile.TemporaryDirectory(prefix="aquifer-bundle-test-") as root:
        test_hmac(root)
        test_kv_and_zip(root)
        test_parent(root)
        test_openssl(root)
    if FAILED:
        print("FALLARON %d de %d pruebas:" % (len(FAILED), TOTAL))
        for name in FAILED:
            print("  - " + name)
        return 1
    print("bundle: %d pruebas OK" % TOTAL)
    return 0


if __name__ == "__main__":
    sys.exit(main())
