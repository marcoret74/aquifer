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
                  get_model_config, start_engine)

SPECIAL_MARKERS = ("<|im_end|>", "<|endoftext|>", "<｜end", "<think>",
                   "</s>", "<|user|>", "<|assistant|>")


class ApiError(Exception):
    def __init__(self, status, message, kind="invalid_request_error"):
        super().__init__(message)
        self.status = status
        self.message = message
        self.kind = kind


class Generation:
    """Estado de una generación: tokens emitidos y motivo de corte."""

    __slots__ = ("tokens", "finish", "done")

    def __init__(self):
        self.tokens = 0
        self.finish = "stop"
        self.done = False


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
        with self.lock:
            try:
                if not self.alive():
                    raise ApiError(503, "el motor no está disponible", "server_error")
                self.configure(opts)
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

    def _send_json(self, status, payload):
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self._cors()
        self.end_headers()
        self.wfile.write(body)

    def _error(self, err):
        # Si no se consumió el cuerpo, cerrar evita desincronizar keep-alive.
        self.close_connection = True
        self._send_json(err.status, {"error": {
            "message": err.message, "type": err.kind, "code": err.status}})

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
        try:
            if self.path != "/v1/chat/completions":
                raise ApiError(404, "ruta no encontrada", "not_found")
            if not self._auth():
                raise ApiError(401, "API key inválida", "authentication_error")
            body = self._body()
            self._chat(body)
        except ApiError as err:
            try:
                self._error(err)
            except BrokenPipeError:
                pass
        except BrokenPipeError:
            pass
        except Exception as err:  # noqa: BLE001
            try:
                self._error(ApiError(500, "error interno: %s" % err, "server_error"))
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

    def _tokens(self, system_text, history, opts):
        model = self.state.model
        ctx = self.state.ctx
        prompt = format_prompt(history, model, system_text or None)
        tokens = self.state.engine.tok.encode(prompt)
        while len(history) > 1 and len(tokens) + opts["max_tokens"] >= ctx:
            history.pop(0)
            while len(history) > 1 and history[0][0] != "user":
                history.pop(0)
            prompt = format_prompt(history, model, system_text or None)
            tokens = self.state.engine.tok.encode(prompt)
        if len(tokens) + opts["max_tokens"] >= ctx:
            raise ApiError(
                400,
                "la conversación no cabe en ctx=%d; reduce max_tokens o el "
                "historial" % ctx)
        if len(tokens) >= ctx:
            raise ApiError(400, "prompt demasiado largo para ctx=%d" % ctx)
        return tokens

    def _chat(self, body):
        if not self.state.engine.alive():
            raise ApiError(503, "el motor no está disponible", "server_error")
        opts = self._opts(body)
        system_text, history = self._prompt(body, opts)
        tokens = self._tokens(system_text, history, opts)
        stream = bool(body.get("stream", False))
        completion_id = "chatcmpl-" + uuid.uuid4().hex[:24]
        created = int(time.time())
        gen = Generation()

        if not stream:
            parts = []
            for piece in self.state.engine.stream(tokens, opts, gen):
                parts.append(piece)
            text = "".join(parts)
            self._send_json(200, {
                "id": completion_id,
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
            })
            return

        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream; charset=utf-8")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "close")
        self.send_header("X-Accel-Buffering", "no")
        self._cors()
        self.end_headers()
        self.close_connection = True

        base = {
            "id": completion_id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": self.state.model,
        }
        self._sse(dict(base, choices=[{
            "index": 0, "delta": {"role": "assistant"}, "finish_reason": None}]))
        streamer = self.state.engine.stream(tokens, opts, gen)
        try:
            for piece in streamer:
                self._sse(dict(base, choices=[{
                    "index": 0, "delta": {"content": piece}, "finish_reason": None}]))
        finally:
            streamer.close()
        self._sse(dict(base, choices=[{
            "index": 0, "delta": {}, "finish_reason": gen.finish}]))
        if (body.get("stream_options") or {}).get("include_usage"):
            self._sse(dict(base, choices=[], usage={
                "prompt_tokens": len(tokens),
                "completion_tokens": gen.tokens,
                "total_tokens": len(tokens) + gen.tokens,
            }))
        self.wfile.write(b"data: [DONE]\n\n")
        self.wfile.flush()


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
    ap.add_argument("--api-key", default=None,
                    help="Si se define, exige 'Authorization: Bearer <key>'")
    ap.add_argument("--cors", action="store_true",
                    help="Habilita CORS para UIs locales en el navegador")
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
    print("  rutas   : GET /health · GET /v1/models · POST /v1/chat/completions")
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
