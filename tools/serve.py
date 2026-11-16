#!/usr/bin/env python3
"""serve.py — servidor HTTP local con API de chat compatible con OpenAI.

Expone el motor Aquifer por HTTP en 127.0.0.1, sin dependencias externas:

    GET  /health
    GET  /v1/models
    POST /v1/chat/completions      (stream: true | false)

El servidor es orquestador: no toca el kernel. Habla con `generate_aquifer`
por stdin/stdout, aplica sampling y stop sequences por request, y reutiliza el
KV cache del motor (prefijo) entre turnos de la misma conversación.

Uso:
    python3 tools/serve.py --model qwen05 --port 8080
    curl -s localhost:8080/v1/chat/completions -H 'Content-Type: application/json' \\
      -d '{"messages":[{"role":"user","content":"Hola"}]}'
"""

import argparse
import hashlib
import hmac
import json
import os
import re
import signal
import sys
import threading
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from types import SimpleNamespace
from urllib.parse import parse_qs, urlsplit

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from chat import (MODELS_CATALOG, StreamDecoder, format_prompt,  # noqa: E402
                  get_model_config, start_engine, system_prompt_text)
from structured import SchemaError, extract_json, validate  # noqa: E402

SPECIAL_MARKERS = ("<|im_end|>", "<|endoftext|>", "<｜end", "<think>",
                   "</s>", "<|user|>", "<|assistant|>")


class ApiError(Exception):
    def __init__(self, status, message, kind="invalid_request_error", details=None):
        super().__init__(message)
        self.status = status
        self.message = message
        self.kind = kind
        self.details = details


class Generation:
    """Estado de una generación: tokens, tiempos y compactación."""

    __slots__ = ("tokens", "finish", "done", "first_token_at", "queue_ms",
                 "compact")

    def __init__(self):
        self.tokens = 0
        self.finish = "stop"
        self.done = False
        self.first_token_at = None
        self.queue_ms = 0.0
        self.compact = None


class Metrics:
    """Contadores agregados, histogramas y percentiles simples, con lock."""

    MAX_SAMPLES = 2000
    BUCKETS_MS = (50, 100, 200, 400, 800, 1600, 3200, 6400, 12800)

    def __init__(self):
        self.lock = threading.Lock()
        self.total = 0
        self.ok = 0
        self.errors = 0
        self.streamed = 0
        self.structured = 0
        self.schema_retried = 0
        self.schema_failed = 0
        self.schema_streamed = 0
        self.compactions = 0
        self.tool_requests = 0
        self.tool_calls = 0
        self.tokens_in = 0
        self.tokens_out = 0
        self.by_status = {}
        self.latencies = []
        self.ttfts = []
        self.latency_buckets = [0] * len(self.BUCKETS_MS)
        self.ttft_buckets = [0] * len(self.BUCKETS_MS)
        self.latency_sum = 0.0
        self.ttft_sum = 0.0
        self.latency_count = 0
        self.ttft_count = 0
        self.in_flight = 0

    def begin(self):
        with self.lock:
            self.in_flight += 1

    def record(self, trace):
        with self.lock:
            self.total += 1
            status = trace.get("status", 0)
            key = str(status)
            self.by_status[key] = self.by_status.get(key, 0) + 1
            if 200 <= status < 300:
                self.ok += 1
            else:
                self.errors += 1
            if trace.get("stream"):
                self.streamed += 1
            if trace.get("response_format"):
                self.structured += 1
                if trace.get("stream"):
                    self.schema_streamed += 1
            if (trace.get("schema_attempts") or 0) > 1:
                self.schema_retried += 1
            if (trace.get("error_type") == "invalid_schema_output"
                    or trace.get("schema_invalid")):
                self.schema_failed += 1
            compact = trace.get("compact") or {}
            if compact.get("status") == "ok":
                self.compactions += 1
            if trace.get("tools"):
                self.tool_requests += 1
            if trace.get("tool_calls"):
                self.tool_calls += 1
            self.tokens_in += int(trace.get("prompt_tokens") or 0)
            self.tokens_out += int(trace.get("completion_tokens") or 0)
            lat = trace.get("latency_ms")
            ttft = trace.get("ttft_ms")
            self._push(self.latencies, lat)
            self._push(self.ttfts, ttft)
            if isinstance(lat, (int, float)):
                self.latency_sum += float(lat)
                self.latency_count += 1
                self._bucket(self.latency_buckets, float(lat))
            if isinstance(ttft, (int, float)):
                self.ttft_sum += float(ttft)
                self.ttft_count += 1
                self._bucket(self.ttft_buckets, float(ttft))
            self.in_flight = max(0, self.in_flight - 1)

    @classmethod
    def _bucket(cls, counts, value):
        for i, limit in enumerate(cls.BUCKETS_MS):
            if value <= limit:
                counts[i] += 1

    @classmethod
    def _push(cls, values, v):
        if not isinstance(v, (int, float)):
            return
        values.append(float(v))
        if len(values) > cls.MAX_SAMPLES:
            del values[:len(values) - cls.MAX_SAMPLES]

    @staticmethod
    def _pct(values, p):
        if not values:
            return None
        data = sorted(values)
        idx = min(len(data) - 1, max(0, int(round((p / 100.0) * (len(data) - 1)))))
        return round(data[idx], 1)

    def snapshot(self):
        with self.lock:
            lat = list(self.latencies)
            ttft = list(self.ttfts)
            return {
                "requests": {
                    "total": self.total, "ok": self.ok, "errors": self.errors,
                    "stream": self.streamed, "structured": self.structured,
                    "in_flight": self.in_flight,
                },
                "by_status": dict(sorted(self.by_status.items())),
                "tokens": {"prompt": self.tokens_in, "completion": self.tokens_out,
                           "total": self.tokens_in + self.tokens_out},
                "latency_ms": {
                    "avg": round(sum(lat) / len(lat), 1) if lat else None,
                    "p50": self._pct(lat, 50), "p95": self._pct(lat, 95),
                    "max": round(max(lat), 1) if lat else None,
                    "samples": len(lat),
                },
                "ttft_ms": {
                    "avg": round(sum(ttft) / len(ttft), 1) if ttft else None,
                    "p50": self._pct(ttft, 50), "p95": self._pct(ttft, 95),
                    "samples": len(ttft),
                },
                "schema": {"retried": self.schema_retried,
                           "failed": self.schema_failed,
                           "streamed": self.schema_streamed},
                "tools": {"requests": self.tool_requests,
                          "calls": self.tool_calls},
                "compactions": self.compactions,
            }

    @staticmethod
    def _fmt(value):
        if isinstance(value, float):
            if value == int(value) and abs(value) < 1e15:
                return str(int(value))
            return ("%.3f" % value).rstrip("0").rstrip(".")
        return str(value)

    @staticmethod
    def _label(value):
        return (str(value).replace("\\", "\\\\").replace("\"", "\\\"")
                .replace("\n", "\\n"))

    def prometheus(self, model, ctx, uptime_s, started_epoch):
        """Exposición en formato Prometheus (text/plain 0.0.4)."""
        with self.lock:
            lat_b = list(self.latency_buckets)
            ttft_b = list(self.ttft_buckets)
            data = {
                "total": self.total, "ok": self.ok, "errors": self.errors,
                "stream": self.streamed, "structured": self.structured,
                "retried": self.schema_retried, "failed": self.schema_failed,
                "schema_streamed": self.schema_streamed,
                "compactions": self.compactions,
                "tool_requests": self.tool_requests,
                "tool_calls": self.tool_calls,
                "tokens_in": self.tokens_in, "tokens_out": self.tokens_out,
                "in_flight": self.in_flight,
                "lat_sum": self.latency_sum, "lat_count": self.latency_count,
                "ttft_sum": self.ttft_sum, "ttft_count": self.ttft_count,
                "by_status": dict(self.by_status),
            }
        out = []
        add = out.append

        add("# HELP aquifer_info Servidor y modelo")
        add("# TYPE aquifer_info gauge")
        add('aquifer_info{model="%s",ctx="%d"} 1' %
            (self._label(model), int(ctx)))
        add("# HELP aquifer_uptime_seconds Segundos desde el arranque")
        add("# TYPE aquifer_uptime_seconds gauge")
        add("aquifer_uptime_seconds %s" % self._fmt(float(uptime_s)))
        add("# HELP aquifer_start_time_seconds Arranque (epoch)")
        add("# TYPE aquifer_start_time_seconds gauge")
        add("aquifer_start_time_seconds %s" % self._fmt(float(started_epoch)))

        add("# HELP aquifer_requests_total Peticiones por estado HTTP")
        add("# TYPE aquifer_requests_total counter")
        for status, count in sorted(data["by_status"].items()):
            add('aquifer_requests_total{status="%s"} %d' %
                (self._label(status), count))
        for name, help_text in (
                ("aquifer_requests_ok_total", "Peticiones 2xx"),
                ("aquifer_requests_errors_total", "Peticiones no 2xx"),
                ("aquifer_requests_in_flight", "Peticiones en curso"),
                ("aquifer_stream_requests_total", "Peticiones en streaming"),
                ("aquifer_structured_requests_total", "Peticiones estructuradas"),
                ("aquifer_schema_retried_total", "Peticiones con reintento de esquema"),
                ("aquifer_schema_failed_total", "Salidas estructuradas inválidas"),
                ("aquifer_schema_streamed_total", "Salidas estructuradas en streaming"),
                ("aquifer_tool_requests_total", "Peticiones con herramientas"),
                ("aquifer_tool_calls_total", "Llamadas de herramienta emitidas"),
                ("aquifer_compactions_total", "Compactaciones de KV")):
            kind = "gauge" if name.endswith("in_flight") else "counter"
            add("# HELP %s %s" % (name, help_text))
            add("# TYPE %s %s" % (name, kind))
            key = {"aquifer_requests_ok_total": "ok",
                   "aquifer_requests_errors_total": "errors",
                   "aquifer_requests_in_flight": "in_flight",
                   "aquifer_stream_requests_total": "stream",
                   "aquifer_structured_requests_total": "structured",
                   "aquifer_schema_retried_total": "retried",
                   "aquifer_schema_failed_total": "failed",
                   "aquifer_schema_streamed_total": "schema_streamed",
                   "aquifer_tool_requests_total": "tool_requests",
                   "aquifer_tool_calls_total": "tool_calls",
                   "aquifer_compactions_total": "compactions"}[name]
            add("%s %d" % (name, data[key]))

        add("# HELP aquifer_tokens_total Tokens procesados o generados")
        add("# TYPE aquifer_tokens_total counter")
        add('aquifer_tokens_total{direction="prompt"} %d' % data["tokens_in"])
        add('aquifer_tokens_total{direction="completion"} %d' % data["tokens_out"])

        for name, buckets, total_sum, count, help_text in (
                ("aquifer_request_latency_ms", lat_b, data["lat_sum"],
                 data["lat_count"],
                 "Latencia total del request en milisegundos"),
                ("aquifer_ttft_ms", ttft_b, data["ttft_sum"],
                 data["ttft_count"],
                 "Tiempo hasta el primer token en milisegundos")):
            add("# HELP %s %s" % (name, help_text))
            add("# TYPE %s histogram" % name)
            for limit, count_b in zip(self.BUCKETS_MS, buckets):
                add('%s_bucket{le="%d"} %d' % (name, limit, count_b))
            add('%s_bucket{le="+Inf"} %d' % (name, count))
            add("%s_sum %s" % (name, self._fmt(float(total_sum))))
            add("%s_count %d" % (name, count))

        return "\n".join(out) + "\n"


def stop_hit(buf, stops):
    hit = None
    for seq in stops:
        idx = buf.find(seq)
        if idx != -1 and (hit is None or idx < hit[0]):
            hit = (idx, seq)
    return hit


def stop_keep(buf, stops):
    """Sufijo que todavía podría ser el comienzo de una parada."""
    keep = 0
    for seq in stops:
        for k in range(min(len(seq) - 1, len(buf)), 0, -1):
            if buf.endswith(seq[:k]):
                keep = max(keep, k)
                break
    return keep


def canonical_hash(value):
    data = json.dumps(value, ensure_ascii=False, sort_keys=True,
                      separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(data).hexdigest()


class JsonStreamFilter:
    """Emite el JSON superior desde el primer '{'/'[' y corta al cerrarse.

    Descarta cercas markdown y preámbulo. Si no aparece un objeto/arreglo en
    `max_prefix` caracteres, emite best-effort y deja que la validación final
    decida.
    """

    def __init__(self, max_prefix=512):
        self.max_prefix = max_prefix
        self.buf = ""
        self.started = False
        self.depth = 0
        self.in_str = False
        self.esc = False
        self.done = False
        self.text = ""

    def feed(self, chunk):
        if self.done or not chunk:
            return ""
        self.buf += chunk
        if not self.started:
            idx = next((i for i, ch in enumerate(self.buf) if ch in "{["), None)
            if idx is None:
                if len(self.buf) < self.max_prefix:
                    return ""
                out = self.buf
                self.buf = ""
                self.text += out
                return out
            self.buf = self.buf[idx:]
            self.started = True

        out = []
        for ch in self.buf:
            out.append(ch)
            if self.in_str:
                if self.esc:
                    self.esc = False
                elif ch == "\\":
                    self.esc = True
                elif ch == '"':
                    self.in_str = False
                continue
            if ch == '"':
                self.in_str = True
            elif ch in "{[":
                self.depth += 1
            elif ch in "}]":
                self.depth -= 1
                if self.depth <= 0:
                    self.done = True
                    break
        self.buf = ""
        emitted = "".join(out)
        self.text += emitted
        return emitted


def read_last_chain(path):
    """Devuelve (seq, hash) del último eslabón, o (0, ceros) si no hay."""
    zeros = "0" * 64
    if not path or not os.path.exists(path):
        return 0, zeros
    last = None
    try:
        with open(path, "rb") as f:
            for line in f:
                line = line.strip()
                if line:
                    last = line
    except OSError as err:
        raise SystemExit("no se pudo leer el trace log: %s" % err)
    if not last:
        return 0, zeros
    try:
        data = json.loads(last.decode("utf-8"))
    except (UnicodeDecodeError, ValueError):
        raise SystemExit(
            "trace log corrupto en la última línea; usa otro --trace-log")
    chain = data.get("chain") or {}
    if chain.get("sha256"):
        return int(chain.get("seq", 0)), str(chain["sha256"])
    return 0, zeros


def relaxed_json(text):
    """Repara JSON laxo típico de modelos chicos (claves sin comillas, etc.)."""
    s = text.strip()
    if not s:
        raise SchemaError("salida vacía")
    s = re.sub(r"([{,{]\s*)([A-Za-z_][A-Za-z0-9_]*)\s*:", r'\1"\2":', s)
    s = s.replace("'", '"')
    s = re.sub(r",\s*([}\]])", r"\1", s)
    s = re.sub(r"\bTrue\b", "true", s)
    s = re.sub(r"\bFalse\b", "false", s)
    s = re.sub(r"\bNone\b", "null", s)
    return extract_json(s)


def content_text(content):
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        out = []
        for part in content:
            if isinstance(part, dict) and part.get("type") in ("text", "input_text"):
                out.append(str(part.get("text", "")))
        return "".join(out)
    return ""


class Engine:
    """Un proceso del motor, serializado por lock (una generación a la vez)."""

    def __init__(self, mcfg, args, repo_dir, default_bin, resolve_artifact):
        self.mcfg = mcfg
        self.args = args
        self.lock = threading.Lock()
        self.proc, self.tok, _ = start_engine(
            mcfg, args, repo_dir, default_bin, resolve_artifact)
        if self.proc is None:
            raise SystemExit("No se pudo iniciar el motor")

    def alive(self):
        return self.proc is not None and self.proc.poll() is None

    def _send(self, line):
        try:
            self.proc.stdin.write(line + "\n")
            self.proc.stdin.flush()
        except (BrokenPipeError, ValueError, OSError):
            raise ApiError(503, "el motor no está disponible", "server_error")

    def _expect(self, ok, err):
        for raw in self.proc.stdout:
            line = raw.strip()
            if line == ok:
                return True
            if line == err:
                return False
        raise ApiError(503, "el motor terminó la conexión", "server_error")

    def _read_until(self, prefixes):
        for raw in self.proc.stdout:
            line = raw.strip()
            for p in prefixes:
                if line.startswith(p):
                    return line
        raise ApiError(503, "el motor terminó la conexión", "server_error")

    def configure(self, opts):
        line = "CONFIG %d %.6g %d %.6g %d %.6g %d" % (
            opts["max_tokens"], opts["temperature"], opts["top_k"],
            opts["top_p"], opts["seed"], opts["repeat_penalty"],
            opts["repeat_last_n"])
        self._send(line)
        if not self._expect("CONFIG_OK", "CONFIG_ERR"):
            raise ApiError(500, "el motor rechazó los parámetros", "server_error")

    def stream(self, tokens, opts, gen):
        """Genera texto. Debe consumirse entero o cerrarse (envía STOP)."""
        stops = opts.get("stop") or []
        hit_stop = False
        t_wait = time.monotonic()
        with self.lock:
            gen.queue_ms = (time.monotonic() - t_wait) * 1000.0
            try:
                if not self.alive():
                    raise ApiError(503, "el motor no está disponible", "server_error")
                self.configure(opts)
                compact = opts.get("compact")
                if compact and compact.get("tokens"):
                    try:
                        self._send("COMPACT %d %s" % (
                            compact["sink"],
                            ",".join(map(str, compact["tokens"]))))
                        line = self._read_until(
                            ("COMPACTED", "COMPACT_NOMATCH", "COMPACT_ERR"))
                        if line.startswith("COMPACTED"):
                            parts = line.split()
                            try:
                                gen.compact = {
                                    "status": "ok",
                                    "kept": int(parts[1]),
                                    "start": int(parts[2]),
                                    "pos_base": int(parts[3]),
                                }
                            except (IndexError, ValueError):
                                gen.compact = {"status": "ok"}
                        elif line.startswith("COMPACT_NOMATCH"):
                            gen.compact = {"status": "nomatch"}
                        else:
                            gen.compact = {"status": "error"}
                        sys.stderr.write("[compact] %s\n" % line)
                    except ApiError:
                        pass
                self._send("PROMPT " + ",".join(map(str, tokens)))
                streamer = StreamDecoder(self.tok)
                buf = ""
                for raw in self.proc.stdout:
                    line = raw.strip()
                    if line == "DONE":
                        gen.done = True
                        break
                    if not line.startswith("TOKEN "):
                        continue
                    try:
                        tid = int(line.split()[1])
                    except (IndexError, ValueError):
                        continue
                    gen.tokens += 1
                    if gen.first_token_at is None:
                        gen.first_token_at = time.monotonic()
                    text = streamer.decode(tid)
                    if any(mark in text for mark in SPECIAL_MARKERS):
                        continue
                    if not stops:
                        if text:
                            yield text
                        continue
                    buf += text
                    hit = stop_hit(buf, stops)
                    if hit is not None:
                        prefix = buf[:hit[0]]
                        if prefix:
                            yield prefix
                        hit_stop = True
                        try:
                            self._send("STOP")
                        except ApiError:
                            pass
                        for extra in self.proc.stdout:
                            if extra.strip() == "DONE":
                                gen.done = True
                                break
                        break
                    keep = stop_keep(buf, stops)
                    emit = len(buf) - keep
                    if emit > 0:
                        out = buf[:emit]
                        buf = buf[emit:]
                        yield out
                if not stops and not gen.done:
                    gen.done = True
                if stops and buf and not hit_stop:
                    # El motor terminó sin tocar la parada: se emite el resto.
                    yield buf
                    buf = ""
            finally:
                if not gen.done:
                    # Cliente desconectado o excepción: cortar y drenar.
                    try:
                        self._send("STOP")
                        for extra in self.proc.stdout:
                            if extra.strip() == "DONE":
                                break
                    except Exception:
                        pass
                    gen.done = True
                gen.finish = ("stop" if hit_stop or gen.tokens < opts["max_tokens"]
                              else "length")

    def close(self):
        try:
            self.proc.stdin.write("QUIT\n")
            self.proc.stdin.flush()
            self.proc.wait(timeout=2)
        except Exception:
            try:
                self.proc.kill()
            except Exception:
                pass


class Server(ThreadingHTTPServer):
    daemon_threads = True

    def handle_error(self, request, client_address):
        exc = sys.exc_info()[1]
        if isinstance(exc, (ConnectionResetError, BrokenPipeError)):
            return
        super().handle_error(request, client_address)


class State:
    def __init__(self, engine, args, mcfg, ctx, system_text):
        self.engine = engine
        self.args = args
        self.mcfg = mcfg
        self.model = args.model
        self.ctx = ctx
        self.system = system_text
        self.api_key = args.api_key
        self.cors = args.cors
        self.created = int(time.time())
        self.started = time.time()
        self.metrics = Metrics()
        self.trace_path = args.trace_log
        self.trace_content = args.trace_content
        self.trace_lock = threading.Lock()
        self.trace_hmac_key = None
        if getattr(args, "trace_hmac_key", None):
            with open(args.trace_hmac_key, "rb") as f:
                self.trace_hmac_key = f.read()
            if len(self.trace_hmac_key) < 16:
                raise SystemExit("la clave de trazas debe tener al menos 16 bytes")
        self.chain_seq, self.chain_last = read_last_chain(self.trace_path)


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    server_version = "AquiferServe/0.1"

    # --- infraestructura ---------------------------------------------------
    @property
    def state(self):
        return self.server.state

    def log_message(self, fmt, *args):
        sys.stderr.write("[%s] %s\n" % (time.strftime("%H:%M:%S"), fmt % args))

    def _cors(self):
        if self.state.cors:
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("Access-Control-Allow-Headers", "Authorization, Content-Type")
            self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")

    def _send_json(self, status, payload, headers=None):
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        for key, value in (headers or {}).items():
            self.send_header(key, str(value))
        self._cors()
        self.end_headers()
        self.wfile.write(body)

    def _send_text(self, status, text, content_type="text/plain; charset=utf-8"):
        body = text.encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self._cors()
        self.end_headers()
        self.wfile.write(body)

    def _error(self, err):
        # Si no se consumió el cuerpo, cerrar evita desincronizar keep-alive.
        self.close_connection = True
        payload = {"message": err.message, "type": err.kind, "code": err.status}
        if err.details is not None:
            payload["details"] = err.details
        self._send_json(err.status, {"error": payload})

    def _auth(self):
        key = self.state.api_key
        if not key:
            return True
        header = self.headers.get("Authorization", "")
        token = header[7:] if header.startswith("Bearer ") else ""
        return hmac.compare_digest(token, key)

    def _body(self):
        try:
            length = int(self.headers.get("Content-Length") or 0)
        except ValueError:
            raise ApiError(400, "Content-Length inválido")
        if length <= 0:
            raise ApiError(400, "cuerpo JSON vacío")
        if length > 8 * 1024 * 1024:
            raise ApiError(413, "cuerpo demasiado grande")
        raw = self.rfile.read(length)
        try:
            return json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, ValueError):
            raise ApiError(400, "JSON inválido")

    def _sse(self, payload):
        data = "data: " + json.dumps(payload, ensure_ascii=False) + "\n\n"
        self.wfile.write(data.encode("utf-8"))
        self.wfile.flush()

    # --- trazas ------------------------------------------------------------
    def _new_trace(self):
        return {"_t0": time.monotonic(),
                "request_id": "chatcmpl-" + uuid.uuid4().hex[:24]}

    def _record(self, trace, status, err=None):
        trace["status"] = status
        if err is not None:
            trace["error_type"] = err.kind
            trace["error_message"] = err.message
            if isinstance(err.details, dict):
                tool_err = err.kind == "invalid_tool_call"
                if "attempts" in err.details:
                    trace["tool_attempts" if tool_err else "schema_attempts"] = \
                        err.details["attempts"]
                if "errors" in err.details:
                    trace["tool_errors" if tool_err else "schema_errors"] = \
                        err.details["errors"][:10]
        if "latency_ms" not in trace:
            trace["latency_ms"] = round(
                (time.monotonic() - trace["_t0"]) * 1000.0, 1)
        self.state.metrics.record(trace)
        path = self.state.trace_path
        if not path:
            return
        row = {k: v for k, v in trace.items() if not k.startswith("_")}
        row["ts"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        try:
            with self.state.trace_lock:
                seq = self.state.chain_seq + 1
                prev = self.state.chain_last
                entry = canonical_hash({"prev": prev, "body": row})
                chain = {"seq": seq, "prev": prev, "sha256": entry}
                if self.state.trace_hmac_key:
                    chain["sig"] = hmac.new(
                        self.state.trace_hmac_key, entry.encode("ascii"),
                        hashlib.sha256).hexdigest()
                row["chain"] = chain
                self.state.chain_seq = seq
                self.state.chain_last = entry
                with open(path, "a", encoding="utf-8") as f:
                    f.write(json.dumps(row, ensure_ascii=False) + "\n")
        except OSError as exc:
            sys.stderr.write("[trace] no se pudo escribir: %s\n" % exc)

    @staticmethod
    def _public_trace(trace):
        keys = ("request_id", "model", "stream", "latency_ms", "ttft_ms",
                "queue_ms", "prompt_tokens", "completion_tokens",
                "tokens_per_s", "finish_reason", "prompt_sha256",
                "conversation_sha256", "response_format", "schema_name",
                "schema_sha256", "schema_attempts", "compact", "tools",
                "tool_choice", "tool_calls", "tool_name", "tool_attempts")
        return {k: trace[k] for k in keys if trace.get(k) is not None}

    # --- rutas -------------------------------------------------------------
    def do_OPTIONS(self):
        self.send_response(204)
        self._cors()
        self.send_header("Content-Length", "0")
        self.end_headers()

    def do_GET(self):
        try:
            parsed = urlsplit(self.path)
            path = parsed.path
            if path in ("/health", "/healthz"):
                st = self.state
                self._send_json(200, {
                    "status": "ok" if st.engine.alive() else "engine_down",
                    "model": st.model,
                    "ctx": st.ctx,
                    "uptime_s": round(time.time() - st.started, 1),
                })
            elif path in ("/metrics", "/metrics/prometheus"):
                want_prom = (path == "/metrics/prometheus" or
                             "prometheus" in parse_qs(parsed.query).get("format", []))
                st = self.state
                if want_prom:
                    text = st.metrics.prometheus(
                        st.model, st.ctx, time.time() - st.started, st.started)
                    self._send_text(
                        200, text,
                        "text/plain; version=0.0.4; charset=utf-8")
                else:
                    snap = st.metrics.snapshot()
                    snap["uptime_s"] = round(time.time() - st.started, 1)
                    snap["model"] = st.model
                    snap["ctx"] = st.ctx
                    self._send_json(200, snap)
            elif path.startswith("/v1/") and not self._auth():
                raise ApiError(401, "API key inválida", "authentication_error")
            elif path == "/v1/models":
                self._send_json(200, {"object": "list", "data": [{
                    "id": self.state.model,
                    "object": "model",
                    "created": self.state.created,
                    "owned_by": "semantic-geometry",
                }]})
            elif path == "/v1/models/catalog":
                data = []
                for m in MODELS_CATALOG:
                    data.append({
                        "id": m["key"], "object": "model",
                        "created": self.state.created,
                        "owned_by": "semantic-geometry",
                        "name": m["name"], "params": m["params"],
                    })
                self._send_json(200, {"object": "list", "data": data})
            else:
                self._send_json(404, {"error": {
                    "message": "ruta no encontrada", "type": "not_found"}})
        except ApiError as err:
            try:
                self._error(err)
            except BrokenPipeError:
                pass
        except BrokenPipeError:
            pass

    def do_POST(self):
        if self.path != "/v1/chat/completions":
            try:
                self._error(ApiError(404, "ruta no encontrada", "not_found"))
            except BrokenPipeError:
                pass
            return
        trace = self._new_trace()
        self.state.metrics.begin()
        try:
            if not self._auth():
                raise ApiError(401, "API key inválida", "authentication_error")
            body = self._body()
            self._chat(body, trace)
            self._record(trace, 200)
        except ApiError as err:
            self._record(trace, err.status, err)
            try:
                self._error(err)
            except BrokenPipeError:
                pass
        except BrokenPipeError:
            self._record(trace, 499)
        except Exception as err:  # noqa: BLE001
            wrapped = ApiError(500, "error interno: %s" % err, "server_error")
            self._record(trace, 500, wrapped)
            try:
                self._error(wrapped)
            except BrokenPipeError:
                pass

    # --- chat --------------------------------------------------------------
    def _opts(self, body):
        st = self.state
        args = st.args

        def num(key, default, cast, lo=None, hi=None, alt=None):
            val = body.get(key)
            if val is None and alt:
                val = body.get(alt)
            if val is None:
                return default
            try:
                out = cast(val)
            except (TypeError, ValueError):
                raise ApiError(400, "%s inválido" % key)
            if lo is not None:
                out = max(lo, out)
            if hi is not None:
                out = min(hi, out)
            return out

        max_tokens = num("max_tokens", args.n, int, 1,
                         max(1, st.ctx - 1), alt="max_completion_tokens")
        temperature = num("temperature", args.temperature, float, 0.0, 4.0)
        top_p = num("top_p", args.top_p, float, 0.0, 1.0)
        top_k = num("top_k", args.top_k, int, 1, 100000)
        seed = num("seed", args.seed, int, 0, 2**63 - 1)
        repeat_penalty = num("repeat_penalty", args.repeat_penalty, float, 1.0, 4.0)
        repeat_last_n = num("repeat_last_n", args.repeat_last_n, int, 0, st.ctx)

        stop = body.get("stop")
        if stop is None:
            stops = []
        elif isinstance(stop, str):
            stops = [stop]
        elif isinstance(stop, list):
            stops = [str(s) for s in stop if str(s)]
        else:
            raise ApiError(400, "stop debe ser string o lista")
        stops = [s for s in stops if s][:32]

        return {
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_k": top_k,
            "top_p": top_p,
            "seed": seed,
            "repeat_penalty": repeat_penalty,
            "repeat_last_n": repeat_last_n,
            "stop": stops,
        }

    def _prompt(self, body, opts):
        messages = body.get("messages")
        if not isinstance(messages, list) or not messages:
            raise ApiError(400, "messages debe ser una lista no vacía")
        system_parts = []
        history = []
        for msg in messages:
            if not isinstance(msg, dict):
                raise ApiError(400, "cada mensaje debe ser un objeto")
            role = str(msg.get("role", "")).strip()
            text = content_text(msg.get("content", ""))
            if role == "system":
                system_parts.append(text)
            elif role == "user":
                history.append(("user", text))
            elif role == "assistant":
                calls = msg.get("tool_calls")
                if isinstance(calls, list) and calls:
                    rendered = []
                    for call in calls:
                        if not isinstance(call, dict):
                            continue
                        fn = call.get("function") or {}
                        name = str(fn.get("name", ""))
                        args = fn.get("arguments", {})
                        if isinstance(args, str):
                            try:
                                args = json.loads(args)
                            except ValueError:
                                args = {}
                        if not isinstance(args, dict):
                            args = {}
                        rendered.append("<tool_call>" + json.dumps(
                            {"name": name, "arguments": args},
                            ensure_ascii=False, separators=(",", ":")) +
                            "</tool_call>")
                    if rendered:
                        text = ((text + "\n") if text else "") + "".join(rendered)
                history.append(("assistant", text))
            elif role == "tool":
                call_id = str(msg.get("tool_call_id", "") or "")
                label = (" " + call_id) if call_id else ""
                history.append(("user",
                                "[Resultado de herramienta%s]\n%s" % (label, text)))
            else:
                raise ApiError(400, "rol no soportado: %r" % role)
        if not history:
            raise ApiError(400, "se requiere al menos un mensaje user/assistant")
        system_text = "\n\n".join(p for p in system_parts if p)
        if not system_parts:
            system_text = self.state.system or ""
        return system_text, history

    def _sink_tokens(self, system_text):
        if not system_text:
            return 0
        try:
            prefix = system_prompt_text(self.state.model, system_text)
            return len(self.state.engine.tok.encode(prefix)) if prefix else 0
        except Exception:
            return 0

    def _tokens(self, system_text, history, opts):
        model = self.state.model
        ctx = self.state.ctx
        prompt = format_prompt(history, model, system_text or None)
        tokens = self.state.engine.tok.encode(prompt)
        trimmed = False
        while len(history) > 1 and len(tokens) + opts["max_tokens"] >= ctx:
            history.pop(0)
            while len(history) > 1 and history[0][0] != "user":
                history.pop(0)
            prompt = format_prompt(history, model, system_text or None)
            tokens = self.state.engine.tok.encode(prompt)
            trimmed = True
        if len(tokens) + opts["max_tokens"] >= ctx:
            raise ApiError(
                400,
                "la conversación no cabe en ctx=%d; reduce max_tokens o el "
                "historial" % ctx)
        if len(tokens) >= ctx:
            raise ApiError(400, "prompt demasiado largo para ctx=%d" % ctx)
        compact = None
        if trimmed:
            sink = self._sink_tokens(system_text)
            if sink < len(tokens):
                compact = {"sink": sink, "tokens": tokens[sink:]}
        return tokens, compact

    # --- salida estructurada ----------------------------------------------
    def _response_format(self, body):
        rf = body.get("response_format")
        if rf is None:
            return None
        if not isinstance(rf, dict):
            raise ApiError(400, "response_format debe ser un objeto")
        kind = rf.get("type")
        if kind in (None, "text"):
            return None
        if kind == "json_object":
            return {"type": "json_object", "schema": None, "schema_name": None}
        if kind == "json_schema":
            js = rf.get("json_schema")
            if not isinstance(js, dict):
                raise ApiError(400, "json_schema debe ser un objeto")
            schema = js.get("schema")
            if not isinstance(schema, dict):
                raise ApiError(400, "json_schema.schema debe ser un objeto")
            return {"type": "json_schema", "schema": schema,
                    "schema_name": js.get("name") or "response"}
        raise ApiError(400, "response_format.type no soportado: %r" % (kind,))

    def _json_system(self, system_text, fmt):
        if fmt.get("schema") is not None:
            instr = ("Devuelve exclusivamente un objeto JSON válido que cumpla "
                     "este esquema. No uses markdown ni ``` y no agregues texto "
                     "fuera del JSON.\nEsquema:\n" +
                     json.dumps(fmt["schema"], ensure_ascii=False,
                                separators=(",", ":")))
        else:
            instr = ("Devuelve exclusivamente un objeto JSON válido. No uses "
                     "markdown ni ``` y no agregues texto fuera del JSON.")
        return (system_text + "\n\n" + instr).strip() if system_text else instr

    # --- herramientas (function calling) ----------------------------------
    def _tools(self, body):
        raw = body.get("tools")
        if raw is None:
            return None, "auto"
        if not isinstance(raw, list):
            raise ApiError(400, "tools debe ser una lista")
        specs = []
        for entry in raw:
            if not isinstance(entry, dict) or entry.get("type") != "function":
                raise ApiError(400, "cada tool debe ser "
                                    "{type:'function', function:{...}}")
            fn = entry.get("function")
            if not isinstance(fn, dict):
                raise ApiError(400, "tool.function debe ser un objeto")
            name = fn.get("name")
            if not isinstance(name, str) or not re.fullmatch(r"[A-Za-z0-9_-]{1,64}", name):
                raise ApiError(400, "nombre de herramienta inválido")
            if any(spec["name"] == name for spec in specs):
                raise ApiError(400, "herramienta duplicada: %s" % name)
            params = fn.get("parameters")
            if params is None:
                params = {"type": "object", "properties": {}}
            if not isinstance(params, dict):
                raise ApiError(400, "parameters debe ser un objeto JSON Schema")
            specs.append({
                "name": name,
                "description": str(fn.get("description") or ""),
                "parameters": params,
            })
        choice = body.get("tool_choice", "auto")
        if isinstance(choice, dict):
            fn = choice.get("function") or {}
            name = fn.get("name") if isinstance(fn, dict) else None
            if choice.get("type") != "function" or not name:
                raise ApiError(400, "tool_choice inválido")
            if not any(spec["name"] == name for spec in specs):
                raise ApiError(400, "tool_choice apunta a una herramienta inexistente")
            choice = name
        elif choice not in ("auto", "none"):
            raise ApiError(400, "tool_choice inválido")
        if not specs:
            return None, "auto"
        return specs, choice

    @staticmethod
    def _tool_signature(spec):
        params = spec.get("parameters") or {}
        props = params.get("properties") or {}
        required = params.get("required") or list(props.keys())
        bits = []
        for key in required:
            typ = (props.get(key) or {}).get("type", "any")
            if isinstance(typ, list):
                typ = "/".join(str(t) for t in typ)
            bits.append("%s: %s" % (key, typ))
        return "%s(%s)" % (spec["name"], ", ".join(bits))

    @staticmethod
    def _example_args(params):
        props = params.get("properties") or {}
        out = {}
        for key in params.get("required", []) or []:
            typ = (props.get(key) or {}).get("type")
            if typ in ("number", "integer"):
                out[key] = 1
            elif typ == "string":
                out[key] = "texto"
            elif typ == "boolean":
                out[key] = True
            elif typ == "array":
                out[key] = []
            else:
                out[key] = {}
        return out

    def _tools_system(self, system_text, specs, choice):
        lines = ["Herramientas disponibles:"]
        for spec in specs:
            lines.append("- %s: %s" % (self._tool_signature(spec),
                                       spec["description"]))
        lines.append("")
        lines.append("Si necesitas una herramienta, responde SOLO con:")
        lines.append('<tool_call>{"name":"<herramienta>","arguments":{...}}'
                     '</tool_call>')
        lines.append("No agregues texto fuera de la llamada. Si no necesitas "
                     "herramienta, responde normalmente.")
        if isinstance(choice, str) and choice not in ("auto", "none"):
            lines.append("En este turno DEBES llamar a la herramienta '%s'." % choice)
        example = "<tool_call>" + json.dumps(
            {"name": specs[0]["name"],
             "arguments": self._example_args(specs[0]["parameters"])},
            ensure_ascii=False, separators=(",", ":")) + "</tool_call>"
        lines.append("")
        lines.append("Ejemplo:")
        lines.append("Usuario: necesito usar %s" % specs[0]["name"])
        lines.append("Asistente: " + example)
        instr = "\n".join(lines)
        return (system_text + "\n\n" + instr).strip() if system_text else instr

    def _parse_tool_call(self, text, specs, choice):
        start = text.find("<tool_call>")
        if start == -1:
            if isinstance(choice, str) and choice not in ("auto", "none"):
                return None, ["no se encontró la llamada forzada a '%s'" % choice]
            return None, []
        end = text.find("</tool_call>", start)
        if end == -1:
            # Algunos modelos cierran con EOS sin repetir el tag: se intenta
            # parsear el objeto JSON que sigue.
            payload = text[start + len("<tool_call>"):].strip()
        else:
            payload = text[start + len("<tool_call>"):end].strip()
        try:
            data = extract_json(payload)
        except SchemaError:
            try:
                data = relaxed_json(payload)
            except SchemaError as err:
                return None, [str(err)]
        if not isinstance(data, dict):
            return None, ["la llamada debe ser un objeto JSON"]
        name = data.get("name")
        args = data.get("arguments", {})
        if not isinstance(name, str) or not name:
            return None, ["falta 'name' en la llamada"]
        spec = next((s for s in specs if s["name"] == name), None)
        if spec is None:
            return None, ["herramienta desconocida: %s" % name]
        if isinstance(choice, str) and choice not in ("auto", "none") and name != choice:
            return None, ["se esperaba la herramienta '%s', llegó '%s'" %
                          (choice, name)]
        if not isinstance(args, dict):
            return None, ["'arguments' debe ser un objeto"]
        errors = validate(args, spec["parameters"])
        if errors:
            return None, errors
        return {
            "id": "call_" + uuid.uuid4().hex[:24],
            "type": "function",
            "function": {
                "name": name,
                "arguments": json.dumps(args, ensure_ascii=False),
            },
        }, []

    def _tool_corrective(self, errors, specs, choice, raw=""):
        parts = ["Tu respuesta anterior no produjo una llamada de herramienta válida."]
        snippet = (raw or "").strip()
        if snippet:
            if len(snippet) > 300:
                snippet = snippet[:300] + "…"
            parts.append("Respuesta anterior (recortada):\n" + snippet)
        if errors:
            parts.append("Errores:\n" + "\n".join("- " + e for e in errors[:8]))
        parts.append("Herramientas disponibles: " +
                     ", ".join(s["name"] for s in specs))
        if isinstance(choice, str) and choice not in ("auto", "none"):
            parts.append("Debes llamar a '%s'." % choice)
        parts.append('Responde SOLO con <tool_call>{"name":"...",'
                     '"arguments":{...}}</tool_call>.')
        return "\n\n".join(parts)

    def _tool_generation(self, history, system_text, opts, specs, choice, trace):
        max_attempts = 1 + max(0, int(self.state.args.json_retries))
        base_history = list(history)
        attempt = 0
        errors = []
        text = ""
        gen = Generation()
        while attempt < max_attempts:
            attempt += 1
            trace["tool_attempts"] = attempt
            tokens, opts["compact"] = self._tokens(system_text, base_history, opts)
            trace["prompt_tokens"] = len(tokens)
            trace["prompt_sha256"] = canonical_hash(tokens)
            gen = Generation()
            text = "".join(self.state.engine.stream(tokens, opts, gen))
            trace["completion_tokens"] = gen.tokens
            trace["finish_reason"] = gen.finish
            call, errors = self._parse_tool_call(text, specs, choice)
            if not errors:
                return call, text, gen, attempt, tokens
            if attempt < max_attempts:
                base_history = list(history) + [
                    ("user", self._tool_corrective(errors, specs, choice, text))]
        raise ApiError(
            422,
            "no se pudo obtener una llamada de herramienta válida tras %d intento(s)"
            % attempt,
            "invalid_tool_call",
            details={"attempts": attempt, "errors": errors,
                     "content": (text or "")[-2000:]})

    def _corrective(self, errors, fmt, raw):
        parts = ["Tu respuesta anterior no cumplió el formato exigido."]
        snippet = (raw or "").strip()
        if snippet:
            if len(snippet) > 400:
                snippet = snippet[:400] + "…"
            parts.append("Respuesta anterior (recortada):\n" + snippet)
        if errors:
            parts.append("Errores detectados:\n" +
                         "\n".join("- " + e for e in errors[:12]))
        parts.append("Devuelve únicamente el objeto JSON corregido, sin markdown "
                     "ni texto adicional.")
        return "\n\n".join(parts)

    def _stamp(self, trace, gen):
        now = time.monotonic()
        trace["latency_ms"] = round((now - trace["_t0"]) * 1000.0, 1)
        if gen.first_token_at is not None:
            trace["ttft_ms"] = round(
                (gen.first_token_at - trace["_t0"]) * 1000.0, 1)
            gen_ms = (now - gen.first_token_at) * 1000.0
            if gen_ms > 0 and gen.tokens > 1:
                trace["tokens_per_s"] = round(
                    (gen.tokens - 1) / (gen_ms / 1000.0), 1)
        trace["completion_tokens"] = gen.tokens
        trace["finish_reason"] = gen.finish
        trace["queue_ms"] = round(gen.queue_ms, 1)
        if gen.compact:
            trace["compact"] = gen.compact

    def _structured(self, history, system_text, opts, fmt, trace):
        max_attempts = 1 + max(0, int(self.state.args.json_retries))
        base_history = list(history)
        attempt = 0
        errors = []
        raw = ""
        value = None
        tokens = []
        gen = Generation()
        while attempt < max_attempts:
            attempt += 1
            trace["schema_attempts"] = attempt
            tokens, opts["compact"] = self._tokens(system_text, base_history, opts)
            trace["prompt_tokens"] = len(tokens)
            trace["prompt_sha256"] = canonical_hash(tokens)
            gen = Generation()
            raw = "".join(self.state.engine.stream(tokens, opts, gen))
            trace["completion_tokens"] = gen.tokens
            trace["finish_reason"] = gen.finish
            try:
                value = extract_json(raw)
            except SchemaError as err:
                value = None
                errors = [str(err)]
            else:
                if fmt.get("schema") is None:
                    errors = [] if isinstance(value, dict) else [
                        "se esperaba un objeto JSON"]
                else:
                    errors = validate(value, fmt["schema"])
                if not errors:
                    trace["schema_valid"] = True
                    return raw, value, gen, attempt, tokens
            if attempt < max_attempts:
                base_history = list(history) + [
                    ("user", self._corrective(errors, fmt, raw))]
        raise ApiError(
            422,
            "la salida no cumple el esquema tras %d intento(s)" % attempt,
            "invalid_schema_output",
            details={"attempts": attempt, "errors": errors,
                     "content": (raw or "")[-2000:]})

    def _validate_value(self, text, fmt):
        try:
            value = extract_json(text)
        except SchemaError as err:
            return None, [str(err)]
        if fmt.get("schema") is None:
            if isinstance(value, dict):
                return value, []
            return value, ["se esperaba un objeto JSON"]
        return value, validate(value, fmt["schema"])

    def _structured_stream(self, cid, created, history, system_text, opts,
                           fmt, body, trace):
        """Una sola pasada en streaming: emite JSON y valida al final.

        No hay reintentos: el contenido ya salió. La verdad final queda en
        x_aquifer.valid, en la traza y en /metrics. Para garantía estricta
        (422 + reintentos) el cliente debe usar stream=false.
        """
        tokens, opts["compact"] = self._tokens(system_text, history, opts)
        trace["prompt_tokens"] = len(tokens)
        trace["prompt_sha256"] = canonical_hash(tokens)
        trace["schema_attempts"] = 1
        self._begin_sse(self._sse_headers(trace))
        base = self._chunk_base(cid, created)
        self._sse(dict(base, choices=[{
            "index": 0, "delta": {"role": "assistant"}, "finish_reason": None}]))
        gen = Generation()
        filt = JsonStreamFilter()
        parts = []
        streamer = self.state.engine.stream(tokens, opts, gen)
        try:
            for piece in streamer:
                out = filt.feed(piece)
                if not out:
                    continue
                parts.append(out)
                self._sse(dict(base, choices=[{
                    "index": 0, "delta": {"content": out},
                    "finish_reason": None}]))
                if filt.done:
                    break
        finally:
            trace["completion_tokens"] = gen.tokens
            streamer.close()
        text = "".join(parts)
        self._stamp(trace, gen)
        value, errors = self._validate_value(text, fmt)
        valid = not errors
        trace["schema_valid"] = valid
        if not valid:
            trace["schema_invalid"] = True
            trace["schema_errors"] = errors[:10]
        if self.state.trace_content:
            trace["content"] = text
        meta = self._public_trace(trace)
        meta["valid"] = valid
        meta["attempts"] = 1
        if valid:
            meta["value"] = value
        else:
            meta["schema_errors"] = errors[:10]
        if fmt.get("schema_name"):
            meta["schema_name"] = fmt["schema_name"]
        self._finish_sse(base, gen, tokens, body, meta)

    # --- respuesta ---------------------------------------------------------
    def _send_completion(self, cid, created, text, gen, tokens, meta,
                         trace=None, tool_calls=None, finish=None):
        message = {"role": "assistant", "content": text}
        if tool_calls:
            message["tool_calls"] = tool_calls
        payload = {
            "id": cid,
            "object": "chat.completion",
            "created": created,
            "model": self.state.model,
            "choices": [{
                "index": 0,
                "message": message,
                "finish_reason": finish or gen.finish,
            }],
            "usage": {
                "prompt_tokens": len(tokens),
                "completion_tokens": gen.tokens,
                "total_tokens": len(tokens) + gen.tokens,
            },
        }
        if meta is not None:
            payload["x_aquifer"] = meta
        headers = {"x-aquifer-request-id": cid}
        if trace:
            for key, header in (
                    ("latency_ms", "x-aquifer-latency-ms"),
                    ("conversation_sha256", "x-aquifer-conversation-sha256"),
                    ("prompt_sha256", "x-aquifer-prompt-sha256"),
                    ("schema_sha256", "x-aquifer-schema-sha256")):
                if trace.get(key) is not None:
                    headers[header] = trace[key]
        self._send_json(200, payload, headers)

    def _begin_sse(self, headers=None):
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream; charset=utf-8")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "close")
        self.send_header("X-Accel-Buffering", "no")
        for key, value in (headers or {}).items():
            self.send_header(key, str(value))
        self._cors()
        self.end_headers()
        self.close_connection = True

    def _chunk_base(self, cid, created):
        return {
            "id": cid,
            "object": "chat.completion.chunk",
            "created": created,
            "model": self.state.model,
        }

    def _finish_sse(self, base, gen, tokens, body, meta, finish=None):
        final = dict(base, choices=[{
            "index": 0, "delta": {}, "finish_reason": finish or gen.finish}])
        if meta is not None:
            final["x_aquifer"] = meta
        self._sse(final)
        if (body.get("stream_options") or {}).get("include_usage"):
            self._sse(dict(base, choices=[], usage={
                "prompt_tokens": len(tokens),
                "completion_tokens": gen.tokens,
                "total_tokens": len(tokens) + gen.tokens,
            }))
        self.wfile.write(b"data: [DONE]\n\n")
        self.wfile.flush()

    def _sse_headers(self, trace):
        headers = {"x-aquifer-request-id": trace["request_id"]}
        for key, header in (
                ("conversation_sha256", "x-aquifer-conversation-sha256"),
                ("schema_sha256", "x-aquifer-schema-sha256")):
            if trace.get(key) is not None:
                headers[header] = trace[key]
        return headers

    def _chat(self, body, trace):
        if not self.state.engine.alive():
            raise ApiError(503, "el motor no está disponible", "server_error")
        fmt = self._response_format(body)
        opts = self._opts(body)
        system_text, history = self._prompt(body, opts)
        tool_specs, tool_choice = self._tools(body)
        tool_mode = bool(tool_specs) and tool_choice != "none"
        if fmt and tool_mode:
            raise ApiError(400, "response_format y tools no se combinan")
        if tool_specs:
            trace["tools"] = [spec["name"] for spec in tool_specs]
            trace["tool_choice"] = tool_choice
        if tool_mode:
            system_text = self._tools_system(system_text, tool_specs, tool_choice)
        stream = bool(body.get("stream", False))
        cid = trace["request_id"]
        created = int(time.time())

        trace["model"] = self.state.model
        trace["stream"] = stream
        trace["history_turns"] = sum(1 for r, _ in history if r == "user")
        trace["conversation_sha256"] = canonical_hash(
            {"system": system_text, "messages": history})
        trace["settings"] = {k: opts[k] for k in (
            "max_tokens", "temperature", "top_k", "top_p", "seed",
            "repeat_penalty", "repeat_last_n")}
        trace["stop_sequences"] = len(opts.get("stop") or [])
        if fmt:
            trace["response_format"] = fmt["type"]
            if fmt.get("schema") is not None:
                trace["schema_name"] = fmt["schema_name"]
                trace["schema_sha256"] = canonical_hash(fmt["schema"])
            system_text = self._json_system(system_text, fmt)

        if fmt:
            if not stream:
                raw, value, gen, attempts, tokens = self._structured(
                    history, system_text, opts, fmt, trace)
                text = json.dumps(value, ensure_ascii=False)
                self._stamp(trace, gen)
                if self.state.trace_content:
                    trace["content"] = text
                meta = self._public_trace(trace)
                meta["valid"] = True
                meta["attempts"] = attempts
                meta["value"] = value
                if fmt.get("schema_name"):
                    meta["schema_name"] = fmt["schema_name"]
                self._send_completion(cid, created, text, gen, tokens, meta,
                                      trace)
                return
            self._structured_stream(cid, created, history, system_text, opts,
                                    fmt, body, trace)
            return

        if tool_mode:
            call, raw_text, gen, attempts, tokens = self._tool_generation(
                history, system_text, opts, tool_specs, tool_choice, trace)
            self._stamp(trace, gen)
            trace["tool_calls"] = 1 if call else 0
            finish = "tool_calls" if call else gen.finish
            trace["finish_reason"] = finish
            content = None if call else raw_text.strip()
            if call:
                trace["tool_name"] = call["function"]["name"]
            if self.state.trace_content:
                trace["content"] = raw_text
            meta = self._public_trace(trace)
            if not stream:
                self._send_completion(
                    cid, created, content, gen, tokens, meta, trace,
                    tool_calls=[call] if call else None, finish=finish)
                return
            self._begin_sse(self._sse_headers(trace))
            base = self._chunk_base(cid, created)
            self._sse(dict(base, choices=[{
                "index": 0, "delta": {"role": "assistant"},
                "finish_reason": None}]))
            if call:
                self._sse(dict(base, choices=[{
                    "index": 0,
                    "delta": {"tool_calls": [{
                        "index": 0,
                        "id": call["id"],
                        "type": "function",
                        "function": call["function"],
                    }]},
                    "finish_reason": None}]))
            elif content:
                for i in range(0, len(content), 80):
                    self._sse(dict(base, choices=[{
                        "index": 0,
                        "delta": {"content": content[i:i + 80]},
                        "finish_reason": None}]))
            self._finish_sse(base, gen, tokens, body, meta, finish=finish)
            return

        tokens, opts["compact"] = self._tokens(system_text, history, opts)
        trace["prompt_tokens"] = len(tokens)
        trace["prompt_sha256"] = canonical_hash(tokens)
        gen = Generation()
        if not stream:
            text = "".join(self.state.engine.stream(tokens, opts, gen))
            self._stamp(trace, gen)
            if self.state.trace_content:
                trace["content"] = text
            meta = self._public_trace(trace)
            self._send_completion(cid, created, text, gen, tokens, meta, trace)
            return

        self._begin_sse(self._sse_headers(trace))
        base = self._chunk_base(cid, created)
        self._sse(dict(base, choices=[{
            "index": 0, "delta": {"role": "assistant"}, "finish_reason": None}]))
        streamer = self.state.engine.stream(tokens, opts, gen)
        parts = [] if self.state.trace_content else None
        try:
            for piece in streamer:
                if parts is not None:
                    parts.append(piece)
                self._sse(dict(base, choices=[{
                    "index": 0, "delta": {"content": piece}, "finish_reason": None}]))
        finally:
            trace["completion_tokens"] = gen.tokens
            trace["finish_reason"] = gen.finish
            streamer.close()
        self._stamp(trace, gen)
        if parts is not None:
            trace["content"] = "".join(parts)
        self._finish_sse(base, gen, tokens, body, self._public_trace(trace))


def build_parser(repo_dir):
    ap = argparse.ArgumentParser(
        description="Servidor HTTP local de Aquifer (API de chat estilo OpenAI)")
    ap.add_argument("--host", default="127.0.0.1",
                    help="Interfaz de escucha (default: 127.0.0.1)")
    ap.add_argument("--port", type=int, default=8080,
                    help="Puerto (default: 8080)")
    ap.add_argument("--model", default="qwen05",
                    choices=[m["key"] for m in MODELS_CATALOG],
                    help="Modelo servido (default: qwen05)")
    ap.add_argument("--bin", default=os.path.join(repo_dir, "build", "generate_aquifer"),
                    help="Ruta al binario generate_aquifer")
    ap.add_argument("--artifact", default=None)
    ap.add_argument("--tokenizer", default=None)
    ap.add_argument("--system", default=None,
                    help="Prompt de sistema por defecto (si el request no trae uno)")
    ap.add_argument("--ctx", type=int, default=2048)
    ap.add_argument("--n", type=int, default=None,
                    help="Máximo de tokens por respuesta (default: el del modelo)")
    ap.add_argument("--threads", type=int, default=0)
    ap.add_argument("--prefill-batch", type=int, default=8)
    ap.add_argument("--mem-policy", choices=["auto", "paged", "resident"], default="auto")
    ap.add_argument("--kv-dtype", choices=["fp16", "fp32"], default="fp16")
    ap.add_argument("--spec", type=int, default=4)
    ap.add_argument("--spec-min", type=int, default=2)
    ap.add_argument("--spec-max", type=int, default=8)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--top-k", type=int, default=40)
    ap.add_argument("--top-p", type=float, default=0.9)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--repeat-penalty", type=float, default=1.0)
    ap.add_argument("--repeat-last-n", type=int, default=64)
    ap.add_argument("--json-retries", type=int, default=1,
                    help="Reintentos extra cuando la salida no cumple el esquema "
                         "(default: 1 = hasta 2 intentos)")
    ap.add_argument("--api-key", default=None,
                    help="Si se define, exige 'Authorization: Bearer <key>'")
    ap.add_argument("--cors", action="store_true",
                    help="Habilita CORS para UIs locales en el navegador")
    ap.add_argument("--trace-log", default=None, metavar="RUTA",
                    help="Escribe una línea JSON por request (traza auditable)")
    ap.add_argument("--trace-content", action="store_true",
                    help="Incluye el contenido de la respuesta en el trace log")
    ap.add_argument("--trace-hmac-key", default=None, metavar="CLAVE",
                    help="Firma cada entrada del trace con HMAC-SHA256")
    return ap


def main():
    repo_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    default_bin = os.path.join(repo_dir, "build", "generate_aquifer")
    if not os.path.exists(default_bin):
        default_bin = os.path.join(repo_dir, "build", "generate_qwen")

    def resolve_artifact(name):
        local = os.path.join(repo_dir, "build", name)
        if os.path.exists(local):
            return local
        caudal = os.path.join("/home/marco/source/repos/caudal/build", name)
        if os.path.exists(caudal):
            return caudal
        return local

    args = build_parser(repo_dir).parse_args()
    mcfg = get_model_config(args.model, resolve_artifact)
    if args.artifact:
        mcfg["artifact"] = args.artifact
    if args.tokenizer:
        mcfg["tokenizer"] = args.tokenizer
    if args.n is not None:
        mcfg["n"] = args.n

    if not os.path.exists(args.bin):
        print("Error: binario no encontrado en %s. Compílalo con 'make all'." % args.bin)
        return 1
    if not os.path.exists(mcfg["artifact"]):
        print("Error: artefacto no encontrado en %s." % mcfg["artifact"])
        return 1
    tok_path = mcfg["tokenizer"]
    if not os.path.exists(tok_path):
        local_tok = os.path.join(mcfg["artifact"], "tokenizer.json")
        if os.path.exists(local_tok):
            tok_path = local_tok
            mcfg["tokenizer"] = local_tok
        else:
            print("Error: tokenizer no encontrado en %s." % tok_path)
            return 1

    system_text = args.system if args.system is not None else mcfg["system"]
    engine_args = SimpleNamespace(
        model=args.model, bin=args.bin, artifact=args.artifact,
        tokenizer=args.tokenizer, diamond=None, threads=args.threads,
        n=mcfg["n"], ctx=args.ctx, system=system_text,
        prefill_batch=args.prefill_batch, mem_policy=args.mem_policy,
        kv_dtype=args.kv_dtype, spec=args.spec, spec_min=args.spec_min,
        spec_max=args.spec_max, temperature=args.temperature,
        top_k=args.top_k, top_p=args.top_p, seed=args.seed,
        repeat_penalty=args.repeat_penalty, repeat_last_n=args.repeat_last_n,
        session=None,
    )
    if args.trace_log:
        trace_dir = os.path.dirname(os.path.abspath(args.trace_log))
        os.makedirs(trace_dir, exist_ok=True)

    engine = Engine(mcfg, engine_args, repo_dir, default_bin, resolve_artifact)
    state = State(engine, args, mcfg, args.ctx, system_text)

    httpd = Server((args.host, args.port), Handler)
    httpd.state = state

    def _shutdown(signum, frame):  # noqa: ARG001
        threading.Thread(target=httpd.shutdown, daemon=True).start()

    signal.signal(signal.SIGTERM, _shutdown)

    print("\nAquifer local API · %s" % mcfg["title"])
    print("  escucha : http://%s:%d" % (args.host, args.port))
    print("  modelo  : %s   ctx: %d   default n: %d" %
          (args.model, args.ctx, engine_args.n))
    print("  rutas   : GET /health · GET /metrics (JSON) · "
          "GET /metrics/prometheus · GET /v1/models · POST /v1/chat/completions")
    if args.trace_log:
        print("  trazas  : %s%s" % (
            args.trace_log, " (con contenido)" if args.trace_content else ""))
    if args.api_key:
        print("  auth    : Bearer token requerido")
    print("  Ctrl+C para detener.\n", flush=True)

    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nDeteniendo...")
    finally:
        httpd.server_close()
        engine.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
