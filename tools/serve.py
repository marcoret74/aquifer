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
import signal
import sys
import threading
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from types import SimpleNamespace

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
    """Contadores agregados y percentiles simples, con lock."""

    MAX_SAMPLES = 2000

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
        self.tokens_in = 0
        self.tokens_out = 0
        self.by_status = {}
        self.latencies = []
        self.ttfts = []

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
            self.tokens_in += int(trace.get("prompt_tokens") or 0)
            self.tokens_out += int(trace.get("completion_tokens") or 0)
            self._push(self.latencies, trace.get("latency_ms"))
            self._push(self.ttfts, trace.get("ttft_ms"))

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
                "compactions": self.compactions,
            }


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
                if "attempts" in err.details:
                    trace["schema_attempts"] = err.details["attempts"]
                if "errors" in err.details:
                    trace["schema_errors"] = err.details["errors"][:10]
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
                "schema_sha256", "schema_attempts", "compact")
        return {k: trace[k] for k in keys if trace.get(k) is not None}

    # --- rutas -------------------------------------------------------------
    def do_OPTIONS(self):
        self.send_response(204)
        self._cors()
        self.send_header("Content-Length", "0")
        self.end_headers()

    def do_GET(self):
        try:
            if self.path in ("/health", "/healthz"):
                st = self.state
                self._send_json(200, {
                    "status": "ok" if st.engine.alive() else "engine_down",
                    "model": st.model,
                    "ctx": st.ctx,
                    "uptime_s": round(time.time() - st.started, 1),
                })
            elif self.path == "/metrics":
                snap = self.state.metrics.snapshot()
                snap["uptime_s"] = round(time.time() - self.state.started, 1)
                snap["model"] = self.state.model
                snap["ctx"] = self.state.ctx
                self._send_json(200, snap)
            elif self.path.startswith("/v1/") and not self._auth():
                raise ApiError(401, "API key inválida", "authentication_error")
            elif self.path == "/v1/models":
                self._send_json(200, {"object": "list", "data": [{
                    "id": self.state.model,
                    "object": "model",
                    "created": self.state.created,
                    "owned_by": "semantic-geometry",
                }]})
            elif self.path == "/v1/models/catalog":
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
            elif role in ("user", "assistant"):
                history.append((role, text))
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
                         trace=None):
        payload = {
            "id": cid,
            "object": "chat.completion",
            "created": created,
            "model": self.state.model,
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": text},
                "finish_reason": gen.finish,
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

    def _finish_sse(self, base, gen, tokens, body, meta):
        final = dict(base, choices=[{
            "index": 0, "delta": {}, "finish_reason": gen.finish}])
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
    print("  rutas   : GET /health · GET /metrics · GET /v1/models · "
          "POST /v1/chat/completions")
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
