#!/usr/bin/env python3
"""http_tools.py — herramientas HTTP con allowlist para el piloto.

La configuración la define el operador (nunca el cliente): URL, método, headers
y esquema. El cliente solo manda argumentos que se validan contra el esquema.

Defensas:
  - solo http/https, sin credenciales embebidas;
  - host en allowlist (default loopback);
  - hostnames que resuelven a IPs privadas/loopback/link-local se rechazan
    (salvo loopback explícito);
  - sin redirecciones;
  - timeout por herramienta y límite de bytes de respuesta;
  - headers del cliente prohibidos (Host, Content-Length, etc.);
  - no se exponen URLs, headers ni tokens en el catálogo.
"""

import ipaddress
import json
import os
import re
import socket
import stat
import sys
import time
from urllib import error, parse, request

from structured import validate

DEFAULT_HOSTS = ("127.0.0.1", "localhost", "::1")
FORBIDDEN_HEADERS = {"host", "content-length", "transfer-encoding",
                     "connection", "proxy-authorization"}
NAME_RE = re.compile(r"[A-Za-z0-9_-]{1,64}")


class ToolError(Exception):
    pass


class _NoRedirect(request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


class HttpToolRegistry:
    def __init__(self, path=None, allow_hosts=None, default_timeout=15.0,
                 max_bytes=65536):
        self.allow_hosts = [h.strip().lower() for h in
                            (allow_hosts or DEFAULT_HOSTS) if h.strip()]
        if not self.allow_hosts:
            self.allow_hosts = list(DEFAULT_HOSTS)
        self.default_timeout = self._clamp_timeout(default_timeout)
        self.max_bytes = max(1024, int(max_bytes))
        self.tools = {}
        self.opener = request.build_opener(_NoRedirect())
        if path:
            self._load(path)

    @staticmethod
    def _clamp_timeout(value):
        try:
            value = float(value)
        except (TypeError, ValueError):
            value = 15.0
        return min(60.0, max(0.1, value))

    def _host_allowed(self, host):
        host = host.lower()
        for pattern in self.allow_hosts:
            if pattern.startswith("*."):
                if host.endswith(pattern[1:]) and host != pattern[2:]:
                    return True
            elif host == pattern:
                return True
        return False

    def _check_addresses(self, host):
        if host.lower() in ("127.0.0.1", "localhost", "::1"):
            return
        try:
            infos = socket.getaddrinfo(host, None)
        except OSError:
            raise ToolError("no se pudo resolver el host de la herramienta")
        for info in infos:
            try:
                ip = ipaddress.ip_address(info[4][0])
            except ValueError:
                raise ToolError("dirección inválida para el host de la herramienta")
            if (ip.is_private or ip.is_loopback or ip.is_link_local
                    or ip.is_reserved or ip.is_multicast or ip.is_unspecified):
                raise ToolError(
                    "el host de la herramienta resuelve a una IP privada o local")

    def _load(self, path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except (OSError, ValueError) as err:
            raise ToolError("no se pudo leer la configuración: %s" % err)
        tools = data.get("tools") if isinstance(data, dict) else None
        if not isinstance(tools, dict) or not tools:
            raise ToolError("la configuración debe tener un objeto 'tools' no vacío")
        for name, spec in tools.items():
            self._add(name, spec)
        if any(spec["headers"] for spec in self.tools.values()):
            try:
                mode = stat.S_IMODE(os.stat(path).st_mode)
                if mode & 0o077:
                    sys.stderr.write(
                        "[http-tools] aviso: %s es legible por otros (modo %o); "
                        "puede contener tokens\n" % (path, mode))
            except OSError:
                pass

    def _add(self, name, spec):
        if not isinstance(name, str) or not NAME_RE.fullmatch(name):
            raise ToolError("nombre de herramienta HTTP inválido: %r" % (name,))
        if not isinstance(spec, dict):
            raise ToolError("la herramienta '%s' debe ser un objeto" % name)
        url = str(spec.get("url", ""))
        parsed = parse.urlsplit(url)
        if parsed.scheme not in ("http", "https"):
            raise ToolError("'%s': solo se permite http/https" % name)
        if parsed.username or parsed.password:
            raise ToolError("'%s': la URL no puede incluir credenciales" % name)
        host = (parsed.hostname or "").lower()
        if not host:
            raise ToolError("'%s': la URL no tiene host" % name)
        if not self._host_allowed(host):
            raise ToolError(
                "'%s': host '%s' no está en la allowlist (%s)" %
                (name, host, ", ".join(self.allow_hosts)))
        method = str(spec.get("method", "POST")).upper()
        if method not in ("GET", "POST"):
            raise ToolError("'%s': método no soportado (GET|POST)" % name)
        headers = spec.get("headers") or {}
        if not isinstance(headers, dict):
            raise ToolError("'%s': headers debe ser un objeto" % name)
        clean_headers = {}
        for key, value in headers.items():
            if not isinstance(key, str) or not isinstance(value, str):
                raise ToolError("'%s': headers debe ser texto" % name)
            if key.lower() in FORBIDDEN_HEADERS:
                raise ToolError("'%s': header no permitido: %s" % (name, key))
            if "\r" in key or "\n" in key or "\r" in value or "\n" in value:
                raise ToolError("'%s': header con salto de línea" % name)
            clean_headers[key] = value
        params = spec.get("parameters")
        if params is None:
            params = {"type": "object", "properties": {}}
        if not isinstance(params, dict):
            raise ToolError("'%s': parameters debe ser un objeto" % name)
        timeout = self._clamp_timeout(spec.get("timeout", self.default_timeout))
        self.tools[name] = {
            "name": name,
            "url": url,
            "host": host,
            "method": method,
            "headers": clean_headers,
            "parameters": params,
            "description": str(spec.get("description") or ""),
            "timeout": timeout,
        }

    def has(self, name):
        return name in self.tools

    def names(self):
        return sorted(self.tools)

    def catalog(self):
        return [{
            "type": "function",
            "function": {
                "name": name,
                "description": spec["description"],
                "parameters": spec["parameters"],
            },
        } for name, spec in sorted(self.tools.items())]

    def execute(self, name, args, timeout_cap=None):
        spec = self.tools.get(name)
        if spec is None:
            raise ToolError("herramienta HTTP desconocida: %s" % name)
        if not isinstance(args, dict):
            raise ToolError("arguments debe ser un objeto")
        errors = validate(args, spec["parameters"])
        if errors:
            raise ToolError("; ".join(errors[:5]))
        self._check_addresses(spec["host"])

        url = spec["url"]
        body = None
        if spec["method"] == "GET":
            if args:
                url += ("&" if "?" in url else "?") + parse.urlencode(
                    args, doseq=True)
        else:
            body = json.dumps(args, ensure_ascii=False).encode("utf-8")

        req = request.Request(url, data=body, method=spec["method"])
        req.add_header("User-Agent", "aquifer-http-tools/1")
        req.add_header("Accept", "application/json, text/plain;q=0.9, */*;q=0.5")
        if body is not None:
            req.add_header("Content-Type", "application/json")
        for key, value in spec["headers"].items():
            req.add_header(key, value)

        timeout = spec["timeout"]
        if timeout_cap is not None:
            timeout = max(0.1, min(timeout, float(timeout_cap)))
        started = time.monotonic()
        try:
            with self.opener.open(req, timeout=timeout) as resp:
                raw = resp.read(self.max_bytes + 1)
                status = getattr(resp, "status", 200)
                ctype = resp.headers.get("Content-Type", "")
        except error.HTTPError as err:
            raise ToolError("HTTP %s en la herramienta '%s'" %
                            (err.code, name))
        except (error.URLError, TimeoutError, socket.timeout, OSError):
            raise ToolError("fallo HTTP en la herramienta '%s'" % name)
        duration = round((time.monotonic() - started) * 1000.0, 1)

        truncated = len(raw) > self.max_bytes
        raw = raw[:self.max_bytes]
        text = raw.decode("utf-8", "replace")
        if "json" in ctype.lower():
            try:
                result = json.loads(text)
            except ValueError:
                result = text
        else:
            result = text
        if truncated:
            result = {"truncated": True, "bytes": self.max_bytes,
                      "result": result}
        return {
            "result": result,
            "status": status,
            "duration_ms": duration,
            "truncated": truncated,
        }
