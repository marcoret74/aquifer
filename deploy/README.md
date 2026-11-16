# Despliegue de Aquifer

Empaquetado del **servidor** (`tools/serve.py`) para systemd y Docker. El
modelo no se incluye en la imagen: se monta en modo lectura.

Checklist operativo del piloto:
[`../docs/deploy/CHECKLIST-PILOTO.md`](../docs/deploy/CHECKLIST-PILOTO.md).

## Layout de modelos

```
/srv/aquifer/models/
  qwen05_affine/          # artefacto: weights, scales, index, aux
  qwen05/
    tokenizer.json
```

`--models-dir` busca:

1. `<models-dir>/<artefacto>` (por ejemplo `qwen05_affine`),
2. `<models-dir>/<modelo>/tokenizer.json`,
3. `<models-dir>/tokenizer.json`,
4. `<artefacto>/tokenizer.json`.

Para otros modelos, ajustar `AQUIFER_MODEL` y los nombres de artefacto.

## Docker

### Compose (recomendado para el piloto)

```bash
cd /ruta/al/repo
make all                     # opcional: el Dockerfile compila el motor
docker compose -f deploy/docker-compose.yml build

# El modelo debe ser legible por el UID 10001 del contenedor
sudo chmod -R a+rX /srv/aquifer/models

AQUIFER_MODELS_DIR=/srv/aquifer/models \
  docker compose -f deploy/docker-compose.yml up -d

curl -s http://127.0.0.1:8080/health
docker compose -f deploy/docker-compose.yml logs -f aquifer
```

- El puerto se publica solo en `127.0.0.1`.
- `read_only`, `cap_drop: ALL`, `no-new-privileges` y `/tmp` como tmpfs.
- El estado (trace log) queda en el volumen `aquifer-state`.
- Para recibos o herramientas HTTP, descomentar los `volumes` y `environment`
  correspondientes en `docker-compose.yml`.

### Docker directo

```bash
docker build -f deploy/Dockerfile -t aquifer:local .
docker run --rm -p 127.0.0.1:8080:8080 \
  -v /srv/aquifer/models:/models:ro \
  -v aquifer-state:/var/lib/aquifer \
  -e AQUIFER_MODEL=qwen05 \
  aquifer:local
```

La imagen compila el motor con AVX2 en un stage `gcc:14`. El host debe soportar
AVX2 (x86_64 moderno). Para ARM hay que ajustar los flags del motor.

## systemd

### Servicio del sistema (usuario dedicado)

```bash
sudo useradd --system --home /opt/aquifer --shell /usr/sbin/nologin aquifer
sudo mkdir -p /opt/aquifer /etc/aquifer /srv/aquifer/models

# Copiar el repo y compilar
sudo git clone <repo> /opt/aquifer
cd /opt/aquifer && sudo make all

sudo cp deploy/aquifer.env.example /etc/aquifer/aquifer.env
sudo chmod 600 /etc/aquifer/aquifer.env
sudo cp deploy/systemd/aquifer-serve.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now aquifer-serve
```

Editar `/etc/aquifer/aquifer.env`:

```bash
AQUIFER_ROOT=/opt/aquifer
AQUIFER_MODEL=qwen05
AQUIFER_MODELS_DIR=/srv/aquifer/models
AQUIFER_TRACE_LOG=/var/lib/aquifer/traces.jsonl
AQUIFER_LOCAL_TOOLS=1
```

- `StateDirectory=aquifer` crea `/var/lib/aquifer` con permisos correctos.
- `ProtectSystem=strict` y `ProtectHome=read-only`: el modelo debe estar fuera
  de `/home`.
- `journalctl -u aquifer-serve -f` para logs.

### Servicio de usuario (workstation)

```bash
mkdir -p ~/.config/aquifer ~/.config/systemd/user
cp deploy/aquifer.env.example ~/.config/aquifer/aquifer.env
cp deploy/systemd/aquifer-serve.user.service ~/.config/systemd/user/
systemctl --user daemon-reload
systemctl --user enable --now aquifer-serve
loginctl enable-linger "$USER"
```

En este caso `AQUIFER_ROOT` debe apuntar a tu copia del repo (por ejemplo
`/home/tu-usuario/aquifer`) y el modelo puede estar en `~/.cache/caudal` o en
`/srv`.

## Variables principales

| Variable | Default | Equivale a |
| :--- | :--- | :--- |
| `AQUIFER_MODEL` | `qwen05` | `--model` |
| `AQUIFER_HOST` / `AQUIFER_PORT` | `127.0.0.1` / `8080` | `--host` / `--port` |
| `AQUIFER_CTX` | `2048` | `--ctx` |
| `AQUIFER_MODELS_DIR` | — | `--models-dir` |
| `AQUIFER_ARTIFACT` / `AQUIFER_TOKENIZER` | — | `--artifact` / `--tokenizer` |
| `AQUIFER_BIN` | `build/generate_aquifer` | `--bin` |
| `AQUIFER_THREADS` | `0` (físicos) | `--threads` |
| `AQUIFER_API_KEY` | — | `--api-key` |
| `AQUIFER_LOCAL_TOOLS` | `0` | `--local-tools` |
| `AQUIFER_HTTP_TOOLS` | — | `--http-tools` |
| `AQUIFER_HTTP_ALLOW_HOSTS` | loopback | `--http-allow-host` (coma) |
| `AQUIFER_MAX_TOOL_ROUNDS` | `2` | `--max-tool-rounds` |
| `AQUIFER_TOOL_BUDGET_MS` | `15000` | `--tool-budget-ms` |
| `AQUIFER_TRACE_LOG` | — | `--trace-log` |
| `AQUIFER_RECEIPT_KEY` | — | `--receipt-key` |
| `AQUIFER_TRACE_HMAC_KEY` | — | `--trace-hmac-key` |
| `AQUIFER_TRACE_CONTENT` | `0` | `--trace-content` |
| `AQUIFER_CORS` | `0` | `--cors` |

El wrapper `aquifer-serve.sh` arma los argumentos sin `eval`; cualquier
variable vacía se omite.

## Seguridad

- Escuchar en `127.0.0.1` o publicar el puerto solo en loopback.
- Si hay API key, el archivo de entorno debe ser `600`: el valor es visible en
  `/proc/<pid>/environ` para el mismo usuario/root.
- Claves de recibos y trazas en archivos `600`.
- En Docker: `read_only`, `cap_drop: ALL`, `no-new-privileges`, `/tmp` tmpfs.
- El modelo se monta `:ro`.
- Las herramientas HTTP están deshabilitadas salvo config explícita.
- No hay shell ni ejecución de código arbitrario.

## CPU y contención

El motor usa AVX2 y, por defecto, los núcleos físicos. Si convive con servicios
que saturan la CPU, el `pool.pause()` del motor puede tardar; el watchdog corta
a los `--command-timeout` (15 s), mata el motor y el siguiente request lo
reinicia. Para el piloto:

- reservá núcleos con `cpuset` o `CPUAffinity`, o
- no co-loqués el motor con cargas pesadas.

```yaml
# deploy/docker-compose.yml (opcional)
    cpuset: "0-3"
    # cpus: 4
```

```ini
# systemd (opcional)
CPUAffinity=0-3
```

## Operación

```bash
# Salud y métricas
curl -s http://127.0.0.1:8080/health
curl -s http://127.0.0.1:8080/metrics
curl -s http://127.0.0.1:8080/metrics/prometheus

# Logs
journalctl -u aquifer-serve -f
docker compose -f deploy/docker-compose.yml logs -f aquifer

# Actualizar
git pull && make all && sudo systemctl restart aquifer-serve
docker compose -f deploy/docker-compose.yml build --pull
docker compose -f deploy/docker-compose.yml up -d
```

Monitoreo: ver [`docs/monitoring/`](../docs/monitoring/).
Dimensionamiento: ver [`docs/deploy/SIZING.md`](../docs/deploy/SIZING.md).
Demo de piloto: ver [`docs/pilot/`](../docs/pilot/).
