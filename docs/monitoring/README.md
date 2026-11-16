# Monitoreo de Aquifer (Prometheus + Grafana)

Ejemplos listos para el piloto: scrape de `/metrics/prometheus`, alertas y un
dashboard con latencia, TTFT, tokens, schema y compactaciones.

> Son archivos de ejemplo de operación. No contienen lógica del motor ni del
> Juez: solo consultan las métricas que expone `tools/serve.py`.

## Archivos

| Archivo | Qué hace |
| :--- | :--- |
| `prometheus.yml` | Scrape cada 10 s de `host.docker.internal:8080/metrics/prometheus` |
| `alerts.yml` | Reglas: caída, error rate, p95 de latencia/TTFT, schema, sin tráfico |
| `grafana-dashboard.json` | Dashboard “Aquifer · Motor local” (13 paneles) |
| `docker-compose.yml` | Prometheus + Grafana con provisioning automático |
| `grafana-provisioning/` | Datasource y provider del dashboard |

## Inicio rápido con Docker

1. Levantar Aquifer escuchando en una interfaz alcanzable por Docker:

```bash
python3 tools/serve.py --model qwen05 --host 0.0.0.0 --port 8080 --ctx 2048
```

2. Levantar Prometheus y Grafana:

```bash
cd docs/monitoring
docker compose up -d
```

3. Abrir:

- Prometheus: <http://localhost:9090>
- Grafana: <http://localhost:3000> (usuario `admin`, clave `aquifer`)
- Dashboard: carpeta **Aquifer** → **Aquifer · Motor local**

El datasource y el dashboard se cargan por provisioning; no hace falta
importarlos a mano.

## Opción sin Docker

Si Prometheus corre en el mismo host:

1. Ajustar `prometheus.yml`:

```yaml
- targets: ["127.0.0.1:8080"]
```

2. Arrancar Aquifer en loopback (default) y Prometheus:

```bash
python3 tools/serve.py --model qwen05 --port 8080
prometheus --config.file=prometheus.yml
```

3. Importar `grafana-dashboard.json` en Grafana y apuntar el datasource a
   `http://localhost:9090`.

## Qué muestra el dashboard

- Peticiones/min, tasa de error, in-flight y uptime.
- Peticiones por estado HTTP.
- Latencia p50/p95 (histograma real) y heatmap de latencia.
- TTFT p95.
- Tokens/s de prompt y completion.
- Fallos/reintentos de esquema y llamadas de herramienta.
- Compactaciones de KV y modelo activo.

## Alertas incluidas

| Alerta | Condición | Severidad |
| :--- | :--- | :--- |
| `AquiferDown` | `up == 0` por 1 min | critical |
| `AquiferErrorRate` | 4xx/5xx > 10% por 5 min | warning |
| `AquiferLatencyP95` | p95 > 5 s por 10 min | warning |
| `AquiferTTFTP95` | TTFT p95 > 2 s por 10 min | warning |
| `AquiferSchemaFailures` | alguna salida estructurada inválida en 10 min | warning |
| `AquiferNoTraffic` | sin requests por 30 min | info |

Ajustar umbrales según el modelo y el hardware. En un 0.5B el TTFT normal es
muy bajo; en 3B con `ctx` grande, subir los umbrales.

## Seguridad

`/metrics` y `/metrics/prometheus` **no piden API key**. Mantené Aquifer en
`127.0.0.1` o en una red privada. Si usás Docker con `--host 0.0.0.0`, agregá
firewall o un proxy delante; no lo expongas a Internet.

## Validación local

```bash
python3 -m json.tool docs/monitoring/grafana-dashboard.json > /dev/null
python3 -c "import yaml; [yaml.safe_load(open(p)) for p in [
  'docs/monitoring/prometheus.yml',
  'docs/monitoring/alerts.yml',
  'docs/monitoring/docker-compose.yml',
  'docs/monitoring/grafana-provisioning/datasources/prometheus.yml',
  'docs/monitoring/grafana-provisioning/dashboards/aquifer.yml']]"
```

Si tenés `promtool`:

```bash
promtool check rules docs/monitoring/alerts.yml
promtool check config docs/monitoring/prometheus.yml
```
