#!/usr/bin/env python3
"""spec_draft.py — Speculative decoding con modelo draft (experimento).

Arranca dos procesos del motor: target y draft. El draft propone k tokens con
su propio KV; el target los verifica en una pasada M=k y acepta el prefijo
correcto. Mide aceptación y tokens/s. En greedy el resultado debe ser idéntico
al target sin spec.

Uso:
  python3 bench/spec_draft.py \
      --bin build/generate_aquifer \
      --target /ruta/qwen15_affine \
      --draft  /ruta/qwen05_affine \
      --tokens 1000,1037,1074,... --n 16 --k 4
"""
import argparse
import select
import subprocess
import time


def start(binary, artifact, ctx, eos, threads, mem_policy, prefill_batch):
    p = subprocess.Popen(
        [binary, artifact, "--interactive", "--n", "256", "--ctx", str(ctx),
         "--eos", eos, "--threads", str(threads),
         "--mem-policy", mem_policy, "--prefill-batch", str(prefill_batch),
         "--spec", "0"],
        stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True, bufsize=1)
    for line in p.stdout:
        if line.strip() == "READY":
            break
    return p


def cmd(proc, line, timeout=120.0):
    proc.stdin.write(line + "\n")
    proc.stdin.flush()
    ready, _, _ = select.select([proc.stdout], [], [], timeout)
    if not ready:
        raise RuntimeError("timeout esperando: " + line)
    out = proc.stdout.readline().strip()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bin", required=True)
    ap.add_argument("--target", required=True)
    ap.add_argument("--draft", required=True)
    ap.add_argument("--tokens", required=True, help="IDs separados por coma")
    ap.add_argument("--n", type=int, default=16)
    ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--ctx", type=int, default=512)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--eos", default="151643,151645")
    ap.add_argument("--mem-policy", default="auto")
    ap.add_argument("--prefill-batch", type=int, default=8)
    a = ap.parse_args()

    args = (a.ctx, a.eos, a.threads, a.mem_policy, a.prefill_batch)
    tgt = start(a.bin, a.target, *args)
    drf = start(a.bin, a.draft, *args)
    try:
        print(cmd(tgt, "BEGIN " + a.tokens))
        print(cmd(drf, "BEGIN " + a.tokens))

        out, proposed, accepted = [], 0, 0
        eos = {int(x) for x in a.eos.split(",") if x}
        t0 = time.perf_counter()
        while len(out) < a.n:
            nx = cmd(tgt, "NEXT")
            y = int(nx.split()[1])
            dr = cmd(drf, "DRAFT %d" % a.k).split()
            prop = [int(x) for x in dr[1].split(",")] if len(dr) > 1 and dr[1] else []
            if not prop:
                out.append(y)
                cmd(tgt, "COMMIT %d" % y)
                cmd(drf, "COMMIT %d" % y)
            else:
                a_ok = int(cmd(tgt, "VERIFY " + ",".join(map(str, prop))).split()[1])
                proposed += len(prop)
                accepted += a_ok
                if a_ok == 0:
                    out.append(y)
                    cmd(drf, "SYNC 0")
                    cmd(tgt, "COMMIT %d" % y)
                    cmd(drf, "COMMIT %d" % y)
                else:
                    out.extend(prop[:a_ok])
                    cmd(drf, "SYNC %d" % a_ok)
            if any(t in eos for t in out):
                break
        dt = time.perf_counter() - t0
        print("tokens   :", out[:a.n])
        print("aceptados: %d/%d (%.1f%%)" % (accepted, proposed,
              100.0 * accepted / proposed if proposed else 0.0))
        print("tiempo   : %.2f s  ->  %.2f tok/s" % (dt, len(out) / dt))
    finally:
        for p in (tgt, drf):
            try:
                p.stdin.write("QUIT\n")
                p.stdin.flush()
                p.wait(timeout=5)
            except Exception:
                p.kill()


if __name__ == "__main__":
    main()
