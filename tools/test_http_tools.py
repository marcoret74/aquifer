#!/usr/bin/env python3
"""test_http_tools.py — pruebas del registro HTTP con allowlist (sin modelo).

Uso: python3 tools/test_http_tools.py
"""

import json
import os
import sys
import tempfile
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlsplit

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import http_tools as ht  # noqa: E402

FAILED = []
TOTAL = 0


def check(name, cond, detail=""):
    global TOTAL
    TOTAL += 1
    if not cond:
        FAILED.append(name + ((" :: " + str(detail)) if detail else ""))


def expect_error(name, fn, fragment=""):
    try:
        fn()
    except ht.ToolError as err:
        check(name, fragment in str(err), str(err))
        return
    check(name, False, "no lanzó ToolError")


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def _send(self, code, payload, ctype="application/json"):
        body = payload if isinstance(payload, bytes) else json.dumps(
            payload).encode()
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self):
        length = int(self.headers.get("Content-Length") or 0)
        raw = self.rfile.read(length)
        try:
            data = json.loads(raw or b"{}")
        except ValueError:
            data = {}
        self._send(200, {"echo": data, "path": self.path})

    def do_GET(self):
        parsed = urlsplit(self.path)
        if parsed.path == "/fail":
            self._send(500, {"error": "boom"})
            return
        if parsed.path == "/slow":
            time.sleep(1.0)
            self._send(200, {"ok": True})
            return
        if parsed.path == "/redirect":
            self.send_response(302)
            self.send_header("Location", "/query")
            self.end_headers()
            return
        if parsed.path == "/big":
            self._send(200, "x" * 5000, "text/plain")
            return
        self._send(200, {"query": parse_qs(parsed.query)})


def params(values=None, required=()):
    return {
        "type": "object",
        "properties": values or {},
        "required": list(required),
        "additionalProperties": False,
    }


def main():
    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    port = server.server_address[1]
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base = "http://127.0.0.1:%d" % port

    config = {"tools": {
        "echo": {"url": base + "/echo", "method": "POST",
                 "description": "Echo",
                 "parameters": params({"texto": {"type": "string"}}, ["texto"])},
        "query": {"url": base + "/query", "method": "GET",
                  "description": "Query",
                  "parameters": params({"texto": {"type": "string"}}, ["texto"])},
        "fail": {"url": base + "/fail", "method": "GET",
                 "description": "Falla", "parameters": params()},
        "slow": {"url": base + "/slow", "method": "GET", "timeout": 0.2,
                 "description": "Lento", "parameters": params()},
        "redirect": {"url": base + "/redirect", "method": "GET",
                     "description": "Redirect", "parameters": params()},
        "big": {"url": base + "/big", "method": "GET",
                "description": "Grande", "parameters": params()},
    }}

    with tempfile.TemporaryDirectory(prefix="aquifer-http-") as tmp:
        path = os.path.join(tmp, "tools.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(config, f)

        reg = ht.HttpToolRegistry(path, default_timeout=2, max_bytes=1024)
        check("nombres", "echo" in reg.names() and "query" in reg.names(),
              reg.names())
        catalog = reg.catalog()
        check("catalog", len(catalog) == 6)
        check("catalog sin url",
              all("url" not in t["function"] for t in catalog))

        result = reg.execute("echo", {"texto": "hola"})
        check("POST", result["result"]["echo"] == {"texto": "hola"}, result)
        check("POST status", result["status"] == 200)

        result = reg.execute("query", {"texto": "hola"})
        check("GET query", result["result"]["query"] == {"texto": ["hola"]},
              result)

        expect_error("HTTP 500", lambda: reg.execute("fail", {}), "HTTP 500")
        expect_error("timeout", lambda: reg.execute("slow", {}), "fallo HTTP")
        expect_error("redirect", lambda: reg.execute("redirect", {}), "HTTP 302")

        result = reg.execute("big", {})
        check("truncado", result["truncated"] is True, result)
        check("truncado tamaño",
              len(result["result"]["result"]) <= 1024, result)

        expect_error("tipo inválido",
                     lambda: reg.execute("echo", {"texto": 1}), "texto")
        expect_error("desconocida",
                     lambda: reg.execute("rm", {}), "desconocida")

        def load(data):
            return ht.HttpToolRegistry(_write(tmp, data))

        def bad(spec):
            return load({"tools": {"x": spec}})

        expect_error("scheme", lambda: bad({"url": "ftp://127.0.0.1/x"}),
                     "http/https")
        expect_error("credenciales",
                     lambda: bad({"url": "http://u:p@127.0.0.1/x"}),
                     "credenciales")
        expect_error("host no permitido",
                     lambda: bad({"url": "http://example.com/x"}),
                     "allowlist")
        expect_error("método",
                     lambda: bad({"url": "http://127.0.0.1/x",
                                  "method": "PUT"}), "método")
        expect_error("header prohibido",
                     lambda: bad({"url": "http://127.0.0.1/x",
                                  "headers": {"Host": "x"}}), "header")
        expect_error("nombre inválido",
                     lambda: load({"tools": {"a b": {"url": base + "/x"}}}),
                     "nombre")

        private = ht.HttpToolRegistry(
            _write(tmp, {"tools": {"p": {"url": "http://10.0.0.1/x"}}}),
            allow_hosts=["10.0.0.1"])
        expect_error("IP privada", lambda: private.execute("p", {}), "privada")

        wild = ht.HttpToolRegistry(None, allow_hosts=["*.example.com"])
        check("wildcard sí", wild._host_allowed("api.example.com"))
        check("wildcard no raíz", not wild._host_allowed("example.com"))

    server.shutdown()
    server.server_close()

    if FAILED:
        print("FALLARON %d de %d pruebas:" % (len(FAILED), TOTAL))
        for name in FAILED:
            print("  - " + name)
        return 1
    print("http_tools: %d pruebas OK" % TOTAL)
    return 0


def _write(tmp, data):
    path = os.path.join(tmp, "bad.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f)
    return path


if __name__ == "__main__":
    sys.exit(main())
