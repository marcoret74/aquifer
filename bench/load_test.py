#!/usr/bin/env python3
"""load_test.py — prueba de carga HTTP contra el servidor Aquifer.

Mide latencia, TTFT (streaming), tokens/s y errores desde el cliente, y
opcionalmente los agregados del servidor (/metrics).

Uso:
    python3 bench/load_test.py --requests 12 --concurrency 4 \
        --max-tokens 16 --stream

    ENGINE_CPUS=0-3 CLIENT_CPUS=6,7 ./bench/pilot_sizing.sh
"""

import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed


def percentile(values, p):
    if not values:
        return None
    data = sorted(values)
    if len(data) == 1:
        return float(data[0])
    k = (len(data) - 1) * (p / 100.0)
    lo = int(k)
    hi = min(lo + 1, len(data) - 1)
    if lo == hi:
        return float(data[lo])
    return data[lo] + (data[hi] - data[lo]) * (k - lo)


def http_json(url, body, timeout, api_key):
    data = json.dumps(body).encode("utf-8")
    req = urllib.request.Request(url, data=data,
                                 headers={"Content-Type": "application/json"})
    if api_key:
        req.add_header("Authorization", "Bearer " + api_key)
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def http_stream(url, body, timeout, api_key):
    data = json.dumps(body).encode("utf-8")
    req = urllib.request.Request(url, data=data,
                                 headers={"Content-Type": "application/json"})
    if api_key:
        req.add_header("Authorization", "Bearer " + api_key)
    started = time.monotonic()
    first = None
    completion = None
    finish = None
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        for raw in resp:
            line = raw.decode("utf-8", "replace").strip()
            if not line.startswith("data: "):
                continue
            payload = line[6:]
            if payload == "[DONE]":
                break
            try:
                event = json.loads(payload)
            except ValueError:
                continue
            choice = (event.get("choices") or [{}])[0]
            delta = choice.get("delta") or {}
            if delta.get("content") and first is None:
                first = time.monotonic() - started
            if choice.get("finish_reason"):
                finish = choice["finish_reason"]
            meta = event.get("x_aquifer") or {}
            if meta.get("completion_tokens") is not None:
                completion = meta["completion_tokens"]
    return {
        "ttft_ms": None if first is None else first * 1000.0,
        "completion_tokens": completion,
        "finish_reason": finish,
        "latency_ms": (time.monotonic() - started) * 1000.0,
    }


def one_request(args):
    body = {
        "messages": [{"role": "user", "content": args.prompt}],
        "max_tokens": args.max_tokens,
        "temperature": args.temperature,
    }
    try:
        if args.stream:
            body["stream"] = True
            body["stream_options"] = {"include_usage": True}
            result = http_stream(args.url, body, args.timeout, args.api_key)
            return {"ok": True, **result}
        data = http_json(args.url, body, args.timeout, args.api_key)
        choice = (data.get("choices") or [{}])[0]
        usage = data.get("usage") or {}
        return {
            "ok": True,
            "latency_ms": None,
            "ttft_ms": None,
            "completion_tokens": usage.get("completion_tokens"),
            "finish_reason": choice.get("finish_reason"),
            "_usage": usage,
        }
    except urllib.error.HTTPError as err:
        return {"ok": False, "status": err.code,
                "error": err.read().decode("utf-8", "replace")[:200]}
    except Exception as err:  # noqa: BLE001
        return {"ok": False, "status": None, "error": repr(err)}


def timed_request(args):
    start = time.monotonic()
    result = one_request(args)
    result["wall_ms"] = (time.monotonic() - start) * 1000.0
    if result.get("latency_ms") is None:
        result["latency_ms"] = result["wall_ms"]
    return result


def server_metrics(url, timeout):
    parsed = urllib.parse.urlsplit(url)
    metrics_url = "%s://%s/metrics" % (parsed.scheme, parsed.netloc)
    try:
        with urllib.request.urlopen(metrics_url, timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except Exception:  # noqa: BLE001
        return None


def main():
    ap = argparse.ArgumentParser(description="Carga HTTP contra Aquifer")
    ap.add_argument("--url", default="http://127.0.0.1:8080/v1/chat/completions")
    ap.add_argument("--requests", type=int, default=12)
    ap.add_argument("--concurrency", type=int, default=1)
    ap.add_argument("--warmup", type=int, default=1)
    ap.add_argument("--max-tokens", type=int, default=16)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--prompt",
                    default="Responde en una frase: ¿qué es un KV cache?")
    ap.add_argument("--stream", action="store_true")
    ap.add_argument("--timeout", type=float, default=300.0)
    ap.add_argument("--api-key", default=None)
    ap.add_argument("--json-out", default=None)
    ap.add_argument("--label", default="")
    args = ap.parse_args()

    for _ in range(max(0, args.warmup)):
        timed_request(args)

    before = server_metrics(args.url, args.timeout)
    started = time.monotonic()
    results = []
    with ThreadPoolExecutor(max_workers=max(1, args.concurrency)) as pool:
        futures = [pool.submit(timed_request, args) for _ in range(args.requests)]
        for future in as_completed(futures):
            results.append(future.result())
    elapsed = time.monotonic() - started
    after = server_metrics(args.url, args.timeout)

    ok = [r for r in results if r.get("ok")]
    failed = [r for r in results if not r.get("ok")]
    latencies = [r["latency_ms"] for r in ok if r.get("latency_ms") is not None]
    ttfts = [r["ttft_ms"] for r in ok if r.get("ttft_ms") is not None]
    completion = sum(int(r.get("completion_tokens") or 0) for r in ok)
    throughput = completion / elapsed if elapsed > 0 else 0.0

    summary = {
        "label": args.label,
        "stream": bool(args.stream),
        "requests": args.requests,
        "concurrency": args.concurrency,
        "max_tokens": args.max_tokens,
        "ok": len(ok),
        "errors": len(failed),
        "error_samples": [r.get("error") for r in failed[:3]],
        "wall_s": round(elapsed, 3),
        "completion_tokens": completion,
        "tokens_per_s": round(throughput, 2),
        "latency_ms": {
            "p50": percentile(latencies, 50),
            "p95": percentile(latencies, 95),
            "p99": percentile(latencies, 99),
            "max": max(latencies) if latencies else None,
        },
        "ttft_ms": {
            "p50": percentile(ttfts, 50),
            "p95": percentile(ttfts, 95),
            "max": max(ttfts) if ttfts else None,
        },
        "server": None,
    }
    if before and after:
        try:
            summary["server"] = {
                "requests": after["requests"]["total"] - before["requests"]["total"],
                "prompt_tokens": after["tokens"]["prompt"] - before["tokens"]["prompt"],
                "completion_tokens": after["tokens"]["completion"] - before["tokens"]["completion"],
                "latency_p50": after["latency_ms"]["p50"],
                "latency_p95": after["latency_ms"]["p95"],
                "timeouts": after.get("timeouts", 0) - before.get("timeouts", 0),
            }
        except (KeyError, TypeError):
            summary["server"] = None

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if args.json_out:
        os.makedirs(os.path.dirname(os.path.abspath(args.json_out)),
                    exist_ok=True)
        with open(args.json_out, "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
            f.write("\n")
    return 0 if not failed else 1


if __name__ == "__main__":
    sys.exit(main())
