#!/usr/bin/env bash
# pilot_sizing.sh — matriz de dimensionamiento con cpuset/taskset.
#
# Arranca el servidor con afinidad de CPU, corre carga secuencial y concurrente,
# y genera un Markdown con latencia, TTFT y tokens/s.
#
# Uso:
#   ./bench/pilot_sizing.sh
#   MODELS="qwen05 qwen" ENGINE_CPUS=0-3 CLIENT_CPUS=6,7 ./bench/pilot_sizing.sh
#   MODELS=qwen05 OUT=/tmp/sizing.md ./bench/pilot_sizing.sh
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

MODELS="${MODELS:-qwen05 qwen qwen3}"
LAT_REQS="${LAT_REQS:-6}"
THR_REQS="${THR_REQS:-8}"
THR_CONC="${THR_CONC:-4}"
MAX_TOK="${MAX_TOK:-16}"
REQ_TIMEOUT="${REQ_TIMEOUT:-120}"

if [ "${QUICK:-0}" = "1" ]; then
  MODELS="${MODELS:-qwen05}"
  LAT_REQS="${LAT_REQS:-3}"
  THR_REQS="${THR_REQS:-3}"
  THR_CONC="${THR_CONC:-2}"
  MAX_TOK="${MAX_TOK:-16}"
fi
OUT="${OUT:-bench/results/sizing-$(date +%Y%m%d-%H%M%S).md}"
TMP="$(mktemp -d "${TMPDIR:-/tmp}/aquifer-sizing-XXXXXX")"
SERVER_PID=""

mkdir -p "$(dirname "$OUT")"

cleanup_server() {
  if [ -n "${SERVER_PID:-}" ]; then
    kill -TERM "$SERVER_PID" 2>/dev/null || true
    for _ in $(seq 1 25); do
      kill -0 "$SERVER_PID" 2>/dev/null || break
      sleep 0.2
    done
    kill -KILL "$SERVER_PID" 2>/dev/null || true
    wait "$SERVER_PID" 2>/dev/null || true
    SERVER_PID=""
  fi
}

cleanup() {
  cleanup_server
}
trap cleanup EXIT

load_now="$(awk '{print $1}' /proc/loadavg)"
cpus_now="$(nproc)"
if awk -v l="$load_now" -v n="$cpus_now" 'BEGIN{exit !(l > n * 0.75)}'; then
  if [ "${ALLOW_LOAD:-0}" != "1" ]; then
    echo "Host cargado: load $load_now sobre $cpus_now CPUs."
    echo "Los números no serían representativos."
    echo "Esperá a que baje o usá ALLOW_LOAD=1 para forzar."
    exit 1
  fi
  echo "Aviso: load $load_now sobre $cpus_now CPUs; se fuerza con ALLOW_LOAD=1."
fi

model_ready() {
  python3 - "$1" <<'PY'
import os
import sys

sys.path.insert(0, "tools")
from chat import get_model_config

def resolve(name):
    for base in (os.path.join(os.getcwd(), "build"),
                 "/home/marco/source/repos/caudal/build"):
        path = os.path.join(base, name)
        if os.path.exists(path):
            return path
    return os.path.join(os.getcwd(), "build", name)

cfg = get_model_config(sys.argv[1], resolve)
ok = os.path.exists(cfg["artifact"]) and os.path.exists(cfg["tokenizer"])
sys.exit(0 if ok else 1)
PY
}

HAS_TASKSET=0
if command -v taskset >/dev/null 2>&1; then
  HAS_TASKSET=1
fi
if [ "$HAS_TASKSET" = "1" ] && [ "$(nproc)" -ge 8 ]; then
  ENGINE_CPUS="${ENGINE_CPUS:-0-3}"
  CLIENT_CPUS="${CLIENT_CPUS:-6,7}"
else
  ENGINE_CPUS="${ENGINE_CPUS:-}"
  CLIENT_CPUS="${CLIENT_CPUS:-}"
fi
if [ "$HAS_TASKSET" != "1" ]; then
  echo "Aviso: taskset no disponible; sin afinidad de CPU."
  ENGINE_CPUS=""
  CLIENT_CPUS=""
fi

run_pinned() {
  local cpus="$1"
  shift
  if [ -n "$cpus" ]; then
    taskset -c "$cpus" "$@"
  else
    "$@"
  fi
}

json_ok() {
  python3 - "$1" <<'PY'
import json
import sys
try:
    with open(sys.argv[1], encoding="utf-8") as f:
        data = json.load(f)
except (OSError, ValueError):
    sys.exit(1)
sys.exit(0 if int(data.get("errors", 1)) == 0 else 1)
PY
}

run_load() {
  local cpus="$1" json="$2" label="$3"
  shift 3
  local attempt
  for attempt in 1 2; do
    run_pinned "$cpus" python3 bench/load_test.py "$@"       --json-out "$json" --label "$label" >/dev/null || true
    if json_ok "$json"; then
      return 0
    fi
    echo "   reintento $label ($attempt)"
    sleep 1
  done
  return 0
}

configs=()
for model in $MODELS; do
  if ! model_ready "$model"; then
    echo "skip: $model (falta artefacto o tokenizer)"
    continue
  fi
  configs+=("$model 512 4")
  configs+=("$model 2048 4")
done
if model_ready qwen05; then
  configs=("qwen05 512 2" "${configs[@]}")
fi

if [ "${#configs[@]}" -eq 0 ]; then
  echo "No hay configuraciones para correr."
  exit 1
fi

echo "== Dimensionamiento Aquifer =="
echo "   motor cpus : ${ENGINE_CPUS:-sin pinning}"
echo "   cliente    : ${CLIENT_CPUS:-sin pinning}"
echo "   modelos    : $MODELS"
echo "   carga      : $LAT_REQS secuenciales / $THR_REQS a concurrencia $THR_CONC"
echo

idx=0
for config in "${configs[@]}"; do
  read -r model ctx threads <<< "$config"
  id="${model}-ctx${ctx}-t${threads}"
  port="$(python3 -c 'import socket
s = socket.socket()
s.bind(("127.0.0.1", 0))
print(s.getsockname()[1])
s.close()')"
  log="$TMP/$id.log"
  echo "== $id (puerto $port)"

  if [ -n "$ENGINE_CPUS" ]; then
    taskset -c "$ENGINE_CPUS" python3 tools/serve.py \
      --model "$model" --host 127.0.0.1 --port "$port" --ctx "$ctx" \
      --threads "$threads" --mem-policy auto >"$log" 2>&1 &
  else
    python3 tools/serve.py \
      --model "$model" --host 127.0.0.1 --port "$port" --ctx "$ctx" \
      --threads "$threads" --mem-policy auto >"$log" 2>&1 &
  fi
  SERVER_PID=$!

  ready=0
  for _ in $(seq 1 160); do
    if curl -sf "http://127.0.0.1:$port/health" >/dev/null 2>&1; then
      ready=1
      break
    fi
    if ! kill -0 "$SERVER_PID" 2>/dev/null; then
      break
    fi
    sleep 0.5
  done
  if [ "$ready" != "1" ]; then
    echo "   no arrancó; se omite"
    tail -3 "$log" 2>/dev/null || true
    cleanup_server
    continue
  fi

  url="http://127.0.0.1:$port/v1/chat/completions"
  run_load "$CLIENT_CPUS" "$TMP/$id-lat.json" "$id-lat" \
    --url "$url" --requests "$LAT_REQS" --concurrency 1 \
    --max-tokens "$MAX_TOK" --stream --warmup 1 --timeout "$REQ_TIMEOUT"
  run_load "$CLIENT_CPUS" "$TMP/$id-thr.json" "$id-thr" \
    --url "$url" --requests "$THR_REQS" --concurrency "$THR_CONC" \
    --max-tokens "$MAX_TOK" --warmup 0 --timeout "$REQ_TIMEOUT"

  cleanup_server
done

python3 - "$TMP" "$OUT" "$ENGINE_CPUS" "$CLIENT_CPUS" <<'PY'
import glob
import json
import os
import platform
import sys

tmp, out, engine_cpus, client_cpus = sys.argv[1:5]


def load(path):
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError):
        return None


def fmt(value, digits=0):
    if value is None:
        return "-"
    return ("%." + str(digits) + "f") % value


def cpu_model():
    try:
        with open("/proc/cpuinfo", encoding="utf-8") as f:
            for line in f:
                if line.startswith("model name"):
                    return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or "?"


rows = []
for lat_path in sorted(glob.glob(os.path.join(tmp, "*-lat.json"))):
    ident = os.path.basename(lat_path)[:-len("-lat.json")]
    lat = load(lat_path) or {}
    thr = load(os.path.join(tmp, ident + "-thr.json")) or {}
    rows.append((ident, lat, thr))

lines = []
lines.append("# Dimensionamiento de Aquifer")
lines.append("")
lines.append("- Fecha: %s" % platform.platform())
lines.append("- CPU: %s (%s hilos)" % (cpu_model(), os.cpu_count()))
lines.append("- Load average: %.2f %.2f %.2f" % os.getloadavg())
lines.append("- Motor cpuset: `%s`" % (engine_cpus or "sin pinning"))
lines.append("- Cliente cpuset: `%s`" % (client_cpus or "sin pinning"))
lines.append("- Carga: %d secuenciales (streaming) y %d a concurrencia %s"
             % (
                 int((rows[0][1] or {}).get("requests", 0)) if rows else 0,
                 int((rows[0][2] or {}).get("requests", 0)) if rows else 0,
                 (rows[0][2] or {}).get("concurrency", "?") if rows else "?",
             ))
lines.append("")
lines.append("## Latencia secuencial (1 request a la vez, streaming)")
lines.append("")
lines.append("| Modelo | ctx | threads | OK/err | p50 ms | p95 ms | TTFT p50 | TTFT p95 | tok/s |")
lines.append("| :--- | ---: | ---: | :---: | ---: | ---: | ---: | ---: | ---: |")
for ident, lat, _thr in rows:
    parts = ident.split("-")
    model = parts[0]
    ctx = parts[1].replace("ctx", "")
    threads = parts[2].replace("t", "")
    lat_ms = lat.get("latency_ms") or {}
    ttft = lat.get("ttft_ms") or {}
    lines.append("| %s | %s | %s | %s/%s | %s | %s | %s | %s | %s |" % (
        model, ctx, threads, lat.get("ok", 0), lat.get("errors", 0),
        fmt(lat_ms.get("p50")), fmt(lat_ms.get("p95")),
        fmt(ttft.get("p50")), fmt(ttft.get("p95")),
        fmt(lat.get("tokens_per_s"), 1)))

lines.append("")
lines.append("## Throughput (%s concurrentes, no-streaming)"
             % ((rows[0][2] or {}).get("concurrency", "?") if rows else "?"))
lines.append("")
lines.append("| Modelo | ctx | threads | OK/err | p50 ms | p95 ms | tok/s |")
lines.append("| :--- | ---: | ---: | :---: | ---: | ---: | ---: |")
for ident, _lat, thr in rows:
    parts = ident.split("-")
    model = parts[0]
    ctx = parts[1].replace("ctx", "")
    threads = parts[2].replace("t", "")
    lat_ms = thr.get("latency_ms") or {}
    lines.append("| %s | %s | %s | %s/%s | %s | %s | %s |" % (
        model, ctx, threads, thr.get("ok", 0), thr.get("errors", 0),
        fmt(lat_ms.get("p50")), fmt(lat_ms.get("p95")),
        fmt(thr.get("tokens_per_s"), 1)))

lines.append("")
lines.append("## Notas")
lines.append("")
lines.append("- `tok/s` es el agregado del lote: tokens de completion / tiempo de pared.")
lines.append("- Con `--stream`, TTFT es el primer chunk de contenido.")
lines.append("- El servidor serializa generaciones: la concurrencia mide cola, no paralelismo.")
lines.append("- La caché de prefijo favorece prompts repetidos; un piloto con preguntas variadas tendrá más prefill.")
lines.append("- Números medidos con la carga indicada; repetir en el hardware del piloto.")
lines.append("- Bajo contención de CPU, el watchdog puede cortar y reiniciar el motor.")
lines.append("")

text = "\n".join(lines) + "\n"
os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
with open(out, "w", encoding="utf-8") as f:
    f.write(text)
print(text)
print("Resultados en: %s" % out)
print("Logs en: %s" % tmp)
PY
