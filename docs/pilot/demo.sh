#!/usr/bin/env bash
# demo.sh — demo de 5 minutos: tool calling HTTP + allowlist + recibos.
#
# Uso:
#   ./docs/pilot/demo.sh
#   PORT=8121 ./docs/pilot/demo.sh
#
# Requiere: python3, curl, modelo qwen05 descargado y `make all` hecho.
set -euo pipefail

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$DIR/../.." && pwd)"
PORT="${PORT:-8120}"
MOCK_PORT=9100
MODEL="${MODEL:-qwen05}"
TMP="$(mktemp -d "${TMPDIR:-/tmp}/aquifer-pilot-XXXXXX")"
MOCK_PID=""
SERVER_PID=""

cleanup() {
  if [ -n "$SERVER_PID" ]; then kill "$SERVER_PID" 2>/dev/null || true; fi
  if [ -n "$MOCK_PID" ]; then kill "$MOCK_PID" 2>/dev/null || true; fi
  wait 2>/dev/null || true
}
trap cleanup EXIT

command -v python3 >/dev/null || { echo "Falta python3"; exit 1; }
command -v curl >/dev/null || { echo "Falta curl"; exit 1; }
[ -x "$ROOT/build/generate_aquifer" ] || {
  echo "Falta el binario. Compilá con: make all"; exit 1; }

echo "== Aquifer · demo de piloto =="
echo "   repo   : $ROOT"
echo "   modelo : $MODEL"
echo "   puerto : $PORT"
echo "   mock   : 127.0.0.1:$MOCK_PORT"
echo

head -c 32 /dev/urandom > "$TMP/receipt.key"
chmod 600 "$TMP/receipt.key"

MOCK_PORT="$MOCK_PORT" python3 "$DIR/mock_service.py" > "$TMP/mock.log" 2>&1 &
MOCK_PID=$!

python3 "$ROOT/tools/serve.py" --model "$MODEL" --port "$PORT" --ctx 512 \
  --http-tools "$DIR/http-tools.json" --http-allow-host 127.0.0.1 \
  --max-tool-rounds 2 --tool-budget-ms 15000 \
  --receipt-key "$TMP/receipt.key" --trace-log "$TMP/traces.jsonl" \
  > "$TMP/serve.log" 2>&1 &
SERVER_PID=$!

ready=0
for _ in $(seq 1 80); do
  if curl -sf "http://127.0.0.1:$PORT/health" >/dev/null 2>&1; then
    ready=1
    break
  fi
  sleep 0.5
done
if [ "$ready" != "1" ]; then
  echo "El servidor no arrancó. Log:"
  tail -20 "$TMP/serve.log"
  exit 1
fi

echo "-- 1) Catálogo de herramientas HTTP (sin URL ni headers) --"
curl -s "http://127.0.0.1:$PORT/v1/http-tools" | python3 -m json.tool
echo

echo "-- 2) Pregunta: el modelo decide usar consultar_clima --"
python3 - "$DIR/http-tools.json" "$TMP/request.json" <<'PY'
import json
import sys

config = json.load(open(sys.argv[1], encoding="utf-8"))
spec = config["tools"]["consultar_clima"]
request = {
    "messages": [{"role": "user",
                  "content": "¿Qué clima hace en Rancagua? Usa consultar_clima."}],
    "tools": [{"type": "function", "function": {
        "name": "consultar_clima",
        "description": spec["description"],
        "parameters": spec["parameters"],
    }}],
    "tool_execution": "local",
    "max_tokens": 100,
    "temperature": 0.0,
}
with open(sys.argv[2], "w", encoding="utf-8") as f:
    json.dump(request, f, ensure_ascii=False)
PY

curl -s -D "$TMP/headers.txt" \
  "http://127.0.0.1:$PORT/v1/chat/completions" \
  -H 'Content-Type: application/json' \
  -d @"$TMP/request.json" > "$TMP/response.json"

python3 - "$TMP/response.json" <<'PY'
import json
import sys

data = json.load(open(sys.argv[1], encoding="utf-8"))
choice = data["choices"][0]
meta = data.get("x_aquifer", {})
print("Respuesta :", choice["message"].get("content"))
print("finish    :", choice["finish_reason"])
print("rondas    :", meta.get("tool_rounds"))
for item in meta.get("tool_trace", []):
    print("  tool    : %s/%s %s" % (item.get("kind"), item.get("name"),
                                    item.get("arguments")))
    print("  status  : %s  (%s ms)" % (item.get("status"),
                                       item.get("duration_ms")))
    print("  result  :", json.dumps(item.get("result"), ensure_ascii=False))
receipt = meta.get("receipt") or {}
print("recibo    :", (receipt.get("sig") or "sin recibo")[:24], "...")
PY
echo

echo "-- 3) Recibo en cabecera --"
grep -i "x-aquifer-receipt" "$TMP/headers.txt" || true
echo

echo "-- 4) Métricas --"
curl -s "http://127.0.0.1:$PORT/metrics" | python3 -c '
import json, sys
data = json.load(sys.stdin)
print("requests :", data["requests"])
print("tools    :", data["tools"])
print("receipts :", data["receipts"])
print("timeouts :", data["timeouts"])
'
echo

echo "-- 5) Verificación de los recibos del trace --"
python3 - "$TMP/traces.jsonl" "$TMP/receipt.key" "$ROOT/tools" <<'PY'
import json
import sys

sys.path.insert(0, sys.argv[3])
import receipts

key = open(sys.argv[2], "rb").read()
total = 0
ok_all = True
for line in open(sys.argv[1], encoding="utf-8"):
    line = line.strip()
    if not line:
        continue
    data = json.loads(line)
    total += 1
    ok, why = receipts.verify(data, key)
    ok_all = ok_all and ok
    print("  %s request=%s %s" % (
        data.get("status"), data.get("request_id"),
        "OK" if ok else "FALLA (%s)" % why))
print("  total:", total, "| resultado:", "OK" if ok_all else "FALLA")
PY
echo

cp "$TMP/traces.jsonl" /tmp/aquifer-demo-traces.jsonl
echo "Traza copiada en /tmp/aquifer-demo-traces.jsonl"
echo "Log del servidor: $TMP/serve.log (se borra al salir)"
echo "Demo OK."
