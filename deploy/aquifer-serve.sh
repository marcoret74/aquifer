#!/usr/bin/env bash
# aquifer-serve.sh — arranque configurable del servidor Aquifer.
#
# Variables principales (todas opcionales):
#   AQUIFER_ROOT, AQUIFER_MODEL, AQUIFER_HOST, AQUIFER_PORT, AQUIFER_CTX
#   AQUIFER_MODELS_DIR, AQUIFER_ARTIFACT, AQUIFER_TOKENIZER, AQUIFER_BIN
#   AQUIFER_THREADS, AQUIFER_N, AQUIFER_KV_DTYPE, AQUIFER_MEM_POLICY
#   AQUIFER_API_KEY, AQUIFER_SYSTEM
#   AQUIFER_LOCAL_TOOLS, AQUIFER_HTTP_TOOLS, AQUIFER_HTTP_ALLOW_HOSTS
#   AQUIFER_MAX_TOOL_ROUNDS, AQUIFER_TOOL_BUDGET_MS
#   AQUIFER_TRACE_LOG, AQUIFER_RECEIPT_KEY, AQUIFER_TRACE_HMAC_KEY
#   AQUIFER_TRACE_CONTENT, AQUIFER_CORS
#   AQUIFER_TOKEN_TIMEOUT, AQUIFER_TTFT_TIMEOUT, AQUIFER_REQUEST_TIMEOUT,
#   AQUIFER_COMMAND_TIMEOUT, AQUIFER_JSON_RETRIES
#
# No usa eval ni ejecuta comandos del entorno: solo arma argumentos.
set -euo pipefail

ROOT="${AQUIFER_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
MODEL="${AQUIFER_MODEL:-qwen05}"
HOST="${AQUIFER_HOST:-127.0.0.1}"
PORT="${AQUIFER_PORT:-8080}"
CTX="${AQUIFER_CTX:-2048}"

args=(python3 "$ROOT/tools/serve.py"
      --model "$MODEL" --host "$HOST" --port "$PORT" --ctx "$CTX")

[ -n "${AQUIFER_MODELS_DIR:-}" ] && args+=(--models-dir "$AQUIFER_MODELS_DIR")
[ -n "${AQUIFER_ARTIFACT:-}" ] && args+=(--artifact "$AQUIFER_ARTIFACT")
[ -n "${AQUIFER_TOKENIZER:-}" ] && args+=(--tokenizer "$AQUIFER_TOKENIZER")
[ -n "${AQUIFER_BIN:-}" ] && args+=(--bin "$AQUIFER_BIN")
[ -n "${AQUIFER_THREADS:-}" ] && args+=(--threads "$AQUIFER_THREADS")
[ -n "${AQUIFER_N:-}" ] && args+=(--n "$AQUIFER_N")
[ -n "${AQUIFER_PREFILL_BATCH:-}" ] && args+=(--prefill-batch "$AQUIFER_PREFILL_BATCH")
[ -n "${AQUIFER_KV_DTYPE:-}" ] && args+=(--kv-dtype "$AQUIFER_KV_DTYPE")
[ -n "${AQUIFER_MEM_POLICY:-}" ] && args+=(--mem-policy "$AQUIFER_MEM_POLICY")
[ -n "${AQUIFER_SPEC:-}" ] && args+=(--spec "$AQUIFER_SPEC")
[ -n "${AQUIFER_TEMPERATURE:-}" ] && args+=(--temperature "$AQUIFER_TEMPERATURE")
[ -n "${AQUIFER_TOP_K:-}" ] && args+=(--top-k "$AQUIFER_TOP_K")
[ -n "${AQUIFER_TOP_P:-}" ] && args+=(--top-p "$AQUIFER_TOP_P")
[ -n "${AQUIFER_SEED:-}" ] && args+=(--seed "$AQUIFER_SEED")
[ -n "${AQUIFER_REPEAT_PENALTY:-}" ] && args+=(--repeat-penalty "$AQUIFER_REPEAT_PENALTY")
[ -n "${AQUIFER_REPEAT_LAST_N:-}" ] && args+=(--repeat-last-n "$AQUIFER_REPEAT_LAST_N")
[ -n "${AQUIFER_API_KEY:-}" ] && args+=(--api-key "$AQUIFER_API_KEY")
[ -n "${AQUIFER_SYSTEM:-}" ] && args+=(--system "$AQUIFER_SYSTEM")

if [ "${AQUIFER_LOCAL_TOOLS:-0}" = "1" ]; then
  args+=(--local-tools)
fi
[ -n "${AQUIFER_HTTP_TOOLS:-}" ] && args+=(--http-tools "$AQUIFER_HTTP_TOOLS")
if [ -n "${AQUIFER_HTTP_ALLOW_HOSTS:-}" ]; then
  IFS=',' read -r -a hosts <<< "$AQUIFER_HTTP_ALLOW_HOSTS"
  for host in "${hosts[@]}"; do
    host="$(echo "$host" | xargs)"
    [ -n "$host" ] && args+=(--http-allow-host "$host")
  done
fi
[ -n "${AQUIFER_MAX_TOOL_ROUNDS:-}" ] && args+=(--max-tool-rounds "$AQUIFER_MAX_TOOL_ROUNDS")
[ -n "${AQUIFER_TOOL_BUDGET_MS:-}" ] && args+=(--tool-budget-ms "$AQUIFER_TOOL_BUDGET_MS")

if [ -n "${AQUIFER_TRACE_LOG:-}" ]; then
  mkdir -p "$(dirname "$AQUIFER_TRACE_LOG")"
  args+=(--trace-log "$AQUIFER_TRACE_LOG")
fi
[ -n "${AQUIFER_RECEIPT_KEY:-}" ] && args+=(--receipt-key "$AQUIFER_RECEIPT_KEY")
[ -n "${AQUIFER_TRACE_HMAC_KEY:-}" ] && args+=(--trace-hmac-key "$AQUIFER_TRACE_HMAC_KEY")
if [ "${AQUIFER_TRACE_CONTENT:-0}" = "1" ]; then
  args+=(--trace-content)
fi
if [ "${AQUIFER_CORS:-0}" = "1" ]; then
  args+=(--cors)
fi

[ -n "${AQUIFER_TOKEN_TIMEOUT:-}" ] && args+=(--token-timeout "$AQUIFER_TOKEN_TIMEOUT")
[ -n "${AQUIFER_TTFT_TIMEOUT:-}" ] && args+=(--ttft-timeout "$AQUIFER_TTFT_TIMEOUT")
[ -n "${AQUIFER_REQUEST_TIMEOUT:-}" ] && args+=(--request-timeout "$AQUIFER_REQUEST_TIMEOUT")
[ -n "${AQUIFER_COMMAND_TIMEOUT:-}" ] && args+=(--command-timeout "$AQUIFER_COMMAND_TIMEOUT")
[ -n "${AQUIFER_JSON_RETRIES:-}" ] && args+=(--json-retries "$AQUIFER_JSON_RETRIES")

exec "${args[@]}"
