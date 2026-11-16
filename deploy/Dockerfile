# syntax=docker/dockerfile:1
# Imagen del servidor Aquifer. El modelo NO va en la imagen: se monta en /models.

FROM gcc:14-bookworm AS builder
WORKDIR /src
COPY include /src/include
COPY tools/generate_aquifer.cpp /src/tools/generate_aquifer.cpp
RUN mkdir -p /out \
 && g++ -std=c++23 -O3 -mavx2 -mfma -mf16c -mbmi2 -pthread \
      -I /src/include /src/tools/generate_aquifer.cpp \
      -o /out/generate_aquifer

FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    AQUIFER_ROOT=/app \
    AQUIFER_HOST=0.0.0.0 \
    AQUIFER_PORT=8080 \
    AQUIFER_MODELS_DIR=/models \
    AQUIFER_TRACE_LOG=/var/lib/aquifer/traces.jsonl

RUN apt-get update \
 && apt-get install -y --no-install-recommends ca-certificates libstdc++6 \
 && rm -rf /var/lib/apt/lists/* \
 && pip install --no-cache-dir --only-binary=:all: "regex>=2024.11.6"

WORKDIR /app

COPY tools/serve.py tools/chat.py tools/oracle_qwen.py tools/structured.py \
     tools/local_tools.py tools/http_tools.py tools/receipts.py /app/tools/
COPY deploy/aquifer-serve.sh /app/deploy/aquifer-serve.sh
COPY --from=builder /out/generate_aquifer /app/build/generate_aquifer

RUN useradd --system --uid 10001 --home /app --shell /usr/sbin/nologin aquifer \
 && mkdir -p /var/lib/aquifer /etc/aquifer /models \
 && chown -R aquifer:aquifer /var/lib/aquifer /app \
 && chmod +x /app/deploy/aquifer-serve.sh /app/build/generate_aquifer

USER aquifer
EXPOSE 8080

HEALTHCHECK --interval=30s --timeout=5s --start-period=40s --retries=3 \
  CMD python3 -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8080/health', timeout=3)"

ENTRYPOINT ["/app/deploy/aquifer-serve.sh"]
