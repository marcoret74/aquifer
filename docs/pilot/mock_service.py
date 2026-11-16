#!/usr/bin/env python3
"""mock_service.py — servicio HTTP local de clima para la demo del piloto.

Escucha solo en 127.0.0.1. No es parte del motor ni del Juez: es un mock
genérico para mostrar tool calling + allowlist + recibos.
"""

import hashlib
import json
import os
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlsplit

CITIES = {
    "rancagua": 24,
    "santiago": 27,
    "valparaiso": 19,
    "concepcion": 16,
    "la serena": 21,
}


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def _json(self, code, payload):
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _clima(self, ciudad):
        key = ciudad.strip().lower()
        if key not in CITIES:
            return {"ciudad": ciudad or "?", "error": "ciudad no encontrada",
                    "disponibles": sorted(CITIES)}
        digest = hashlib.sha256(key.encode()).hexdigest()
        humedad = 40 + int(digest[:2], 16) % 50
        condicion = "despejado" if int(digest[2:4], 16) % 2 else "nublado"
        return {"ciudad": ciudad.title(), "temp_c": CITIES[key],
                "humedad": humedad, "condicion": condicion,
                "fuente": "mock local"}

    def do_GET(self):
        parsed = urlsplit(self.path)
        if parsed.path == "/salud":
            self._json(200, {"ok": True})
            return
        if parsed.path == "/clima":
            query = parse_qs(parsed.query)
            ciudad = (query.get("ciudad") or [""])[0]
            self._json(200, self._clima(ciudad))
            return
        self._json(404, {"error": "ruta no encontrada"})

    def do_POST(self):
        length = int(self.headers.get("Content-Length") or 0)
        try:
            data = json.loads(self.rfile.read(length) or b"{}")
        except ValueError:
            data = {}
        self._json(200, self._clima(str(data.get("ciudad", ""))))


def main():
    port = int(os.environ.get("MOCK_PORT", "9100"))
    server = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    print("mock de clima en http://127.0.0.1:%d" % port, flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
