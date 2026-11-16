#!/usr/bin/env python3
"""test_watchdog.py — pruebas del watchdog de generación (sin modelo).

Uso: python3 tools/test_watchdog.py
"""

import os
import queue
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from serve import ApiError, Engine, EngineTimeout, Metrics  # noqa: E402

FAILED = []
TOTAL = 0


def check(name, cond, detail=""):
    global TOTAL
    TOTAL += 1
    if not cond:
        FAILED.append(name + ((" :: " + str(detail)) if detail else ""))


def dummy():
    e = Engine.__new__(Engine)
    e.queue = queue.Queue()
    e.command_timeout = 0.2
    e.proc = None
    return e


def test_readline():
    e = dummy()
    e.queue.put("TOKEN 1\n")
    check("line", e._readline(0.1) == "TOKEN 1\n")

    e = dummy()
    start = time.monotonic()
    try:
        e._readline(0.05)
        check("timeout", False)
    except EngineTimeout:
        check("timeout", time.monotonic() - start >= 0.04)

    e = dummy()
    e.queue.put(None)
    try:
        e._readline(0.1)
        check("eof", False)
    except ApiError as err:
        check("eof", err.status == 503)


def test_read_until():
    e = dummy()
    e.queue.put("NOISE\n")
    e.queue.put("CONFIG_OK\n")
    check("read_until", e._read_until(("CONFIG_OK", "CONFIG_ERR"), 0.2).strip()
          == "CONFIG_OK")

    e = dummy()
    try:
        e._read_until(("X",), 0.05)
        check("read_until timeout", False)
    except EngineTimeout:
        check("read_until timeout", True)


def test_drain():
    e = dummy()
    e.queue.put("TOKEN 1\n")
    e.queue.put("DONE\n")
    check("drain done", e._drain_done(0.2))

    e = dummy()
    check("drain timeout", not e._drain_done(0.05))


def test_metrics():
    m = Metrics()
    m.begin()
    m.record({"status": 504, "error_type": "generation_timeout",
              "latency_ms": 10.0})
    text = m.prometheus("qwen05", 512, 1.0, 0.0)
    check("metric timeouts", "aquifer_timeouts_total 1" in text)
    check("snapshot timeouts", m.snapshot()["timeouts"] == 1)
    check("snapshot in flight", m.snapshot()["requests"]["in_flight"] == 0)


def main():
    test_readline()
    test_read_until()
    test_drain()
    test_metrics()
    if FAILED:
        print("FALLARON %d de %d pruebas:" % (len(FAILED), TOTAL))
        for name in FAILED:
            print("  - " + name)
        return 1
    print("watchdog: %d pruebas OK" % TOTAL)
    return 0


if __name__ == "__main__":
    sys.exit(main())
