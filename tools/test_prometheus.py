#!/usr/bin/env python3
"""test_prometheus.py — pruebas del exporter Prometheus (sin modelo).

Uso: python3 tools/test_prometheus.py
"""

import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from serve import Metrics  # noqa: E402

FAILED = []
TOTAL = 0


def check(name, cond, detail=""):
    global TOTAL
    TOTAL += 1
    if not cond:
        FAILED.append(name + ((" :: " + str(detail)) if detail else ""))


def main():
    m = Metrics()
    m.begin()
    m.record({
        "status": 200, "stream": False, "response_format": None,
        "prompt_tokens": 10, "completion_tokens": 5,
        "latency_ms": 120.0, "ttft_ms": 40.0, "schema_attempts": 1,
        "compact": {"status": "ok", "kept": 50},
        "tools": ["sumar"], "tool_calls": 1, "tool_name": "sumar",
        "tool_local_calls": 1,
        "tool_http_calls": 1,
        "receipt": {"alg": "hmac-sha256", "sig": "x", "key_sha256": "y"},
    })
    m.begin()
    m.record({
        "status": 422, "stream": False, "response_format": "json_schema",
        "prompt_tokens": 20, "completion_tokens": 0,
        "latency_ms": 300.0, "schema_attempts": 2,
        "error_type": "invalid_schema_output",
    })
    m.begin()
    m.record({
        "status": 200, "stream": True, "response_format": "json_schema",
        "prompt_tokens": 7, "completion_tokens": 3,
        "latency_ms": 80.0, "ttft_ms": 30.0, "schema_attempts": 1,
        "schema_invalid": True,
    })

    text = m.prometheus('qwen"05', 512, 12.5, 1000000.0)

    check("requests 200", 'aquifer_requests_total{status="200"} 2' in text)
    check("requests 422", 'aquifer_requests_total{status="422"} 1' in text)
    check("ok total", "aquifer_requests_ok_total 2" in text)
    check("errors total", "aquifer_requests_errors_total 1" in text)
    check("in flight", "aquifer_requests_in_flight 0" in text)
    check("tokens prompt", 'aquifer_tokens_total{direction="prompt"} 37' in text)
    check("tokens completion",
          'aquifer_tokens_total{direction="completion"} 8' in text)
    check("compactions", "aquifer_compactions_total 1" in text)
    check("tool requests", "aquifer_tool_requests_total 1" in text)
    check("tool calls", "aquifer_tool_calls_total 1" in text)
    check("tool local calls", "aquifer_tool_local_calls_total 1" in text)
    check("tool http calls", "aquifer_tool_http_calls_total 1" in text)
    check("receipts", "aquifer_receipts_total 1" in text)
    check("schema failed", "aquifer_schema_failed_total 2" in text)
    check("schema streamed", "aquifer_schema_streamed_total 1" in text)
    check("latencia count",
          "aquifer_request_latency_ms_count 3" in text)
    check("latencia inf",
          'aquifer_request_latency_ms_bucket{le="+Inf"} 3' in text)
    check("latencia bucket 100",
          'aquifer_request_latency_ms_bucket{le="100"} 1' in text)
    check("latencia bucket 200",
          'aquifer_request_latency_ms_bucket{le="200"} 2' in text)
    check("latencia bucket 400",
          'aquifer_request_latency_ms_bucket{le="400"} 3' in text)
    check("ttft count", "aquifer_ttft_ms_count 2" in text)
    check("ttft bucket 50",
          'aquifer_ttft_ms_bucket{le="50"} 2' in text)
    check("latencia sum", "aquifer_request_latency_ms_sum 500" in text)
    check("ttft sum", "aquifer_ttft_ms_sum 70" in text)
    check("info", 'aquifer_info{model="qwen\\"05",ctx="512"} 1' in text)
    check("help", "# HELP aquifer_requests_total" in text)
    check("type histogram", "# TYPE aquifer_request_latency_ms histogram" in text)

    # Validación estructural: buckets monótonos y +Inf == count.
    lat = [int(line.rsplit(" ", 1)[1]) for line in text.splitlines()
           if line.startswith("aquifer_request_latency_ms_bucket{")]
    check("buckets monótonos", lat == sorted(lat), lat)
    check("buckets +Inf", lat[-1] == 3, lat)
    ttft = [int(line.rsplit(" ", 1)[1]) for line in text.splitlines()
            if line.startswith("aquifer_ttft_ms_bucket{")]
    check("ttft monótonos", ttft == sorted(ttft), ttft)

    # Snapshots JSON siguen funcionando.
    snap = m.snapshot()
    check("snapshot in_flight", snap["requests"]["in_flight"] == 0)
    check("snapshot receipts", snap["receipts"] == 1, snap.get("receipts"))
    check("snapshot schema", snap["schema"]["failed"] == 2)
    check("snapshot tools", snap["tools"] == {"requests": 1, "calls": 1,
                                              "local_calls": 1,
                                              "http_calls": 1},
          snap.get("tools"))

    if FAILED:
        print("FALLARON %d de %d pruebas:" % (len(FAILED), TOTAL))
        for name in FAILED:
            print("  - " + name)
        return 1
    print("prometheus: %d pruebas OK" % TOTAL)
    return 0


if __name__ == "__main__":
    sys.exit(main())
