# Checklist de despliegue del piloto

Lista operativa para instalar, verificar y cerrar un piloto de Aquifer.
Aplicable a systemd y Docker. Los números de hardware salen de mediciones en
esta máquina (i7-10510U, 4C/8T, AVX2) y de los tamaños reales de los artefactos.

> Esto es operación técnica. No es asesoría legal ni define el alcance del
> piloto: eso lo acuerdan las partes por fuera del repo.

---

## 1. Alcance y supuestos

- [ ] Un solo servidor local/on-prem, sin nube.
- [ ] Un modelo activo a la vez.
- [ ] El motor corre en CPU con AVX2; no se requiere GPU.
- [ ] El modelo y los documentos **no** se usan para entrenar.
- [ ] El Juez/orquestador, si existe, corre fuera del motor.
- [ ] Hay una persona responsable del servidor y otra del negocio.

## 2. Hardware y sistema

### Mínimos por modelo

| Modelo | Artefacto | KV 2048 (fp16) | KV 4096 (fp16) | RAM mínima | RAM recomendada | CPU |
| :--- | ---: | ---: | ---: | ---: | ---: | :--- |
| Qwen 0.5B | 557 MB | ~25 MB | ~50 MB | 4 GB | 8 GB | 4 núcleos AVX2 |
| Qwen 1.5B | 1.4 GB | ~59 MB | ~117 MB | 8 GB | 16 GB | 4–8 núcleos AVX2 |
| Qwen 3B | 2.4 GB | ~75 MB | ~151 MB | 16 GB | 32 GB | 8 núcleos AVX2 |

- [ ] CPU x86_64 con `avx2`, `fma`, `f16c`, `bmi2`.
- [ ] Al menos 4 núcleos **físicos**; el motor usa núcleos físicos por defecto.
- [ ] Disco libre: modelo + **20 GB** para sesiones, bundles y trazas.
- [ ] Fuente estable y temperatura controlada si es un notebook.
- [ ] Reloj sincronizado (NTP/chrony): los recibos y trazas llevan timestamp.

Para medir el hardware con carga controlada, ver
[`SIZING.md`](SIZING.md).

Comandos de verificación:

```bash
lscpu | grep -o -E 'avx2|fma|f16c|bmi2' | sort -u
nproc
free -h
df -h /srv /var/lib/aquifer
timedatectl status
```

### Sistema operativo

- [ ] Linux x86_64 (probado en Ubuntu/Debian).
- [ ] Python 3.11+; `python3 --version`.
- [ ] `regex` instalado para el tokenizer (`python3 -c "import regex"`).
- [ ] `g++` 13+ para compilar el motor.
- [ ] `openssl` y `curl` para claves, healthchecks y verificación.
- [ ] `make`, `git` si se compila en el servidor.

## 3. Red

- [ ] El servidor escucha en `127.0.0.1` (`AQUIFER_HOST=127.0.0.1`).
- [ ] Si hay acceso remoto, va por VPN (WireGuard/Tailscale) o reverse proxy
      con TLS y autenticación.
- [ ] `/metrics` y `/metrics/prometheus` **no** se publican a Internet.
- [ ] Firewall: solo el puerto del proxy/VPN; sin acceso directo a `8080`.
- [ ] Egress restringido: solo actualizaciones y hosts de herramientas HTTP
      allowlisted.
- [ ] Si se usa `--api-key`, se envía por header y se rota.
- [ ] Puertos de monitoreo (9090/3000) solo en loopback o VPN.
- [ ] El mock de demo (`9100`) solo se levanta durante la demo.

Prueba:

```bash
ss -ltnp | grep -E ':8080|:9090|:3000|:9100'
curl -s http://127.0.0.1:8080/health
```

## 4. Usuarios y permisos

- [ ] Usuario dedicado `aquifer`, sin shell, home `/opt/aquifer`.
- [ ] `/srv/aquifer/models`: lectura para el servicio, sin escritura.
- [ ] `/var/lib/aquifer`: `750`, dueño `aquifer` (trazas y estado).
- [ ] `/etc/aquifer`: `750`; archivos de entorno y claves `600`.
- [ ] Binario del motor `755`; repo de solo lectura para el servicio.
- [ ] Docker: UID `10001`, `read_only`, `cap_drop: ALL`,
      `no-new-privileges`, `/tmp` tmpfs.
- [ ] systemd: `NoNewPrivileges`, `ProtectSystem=strict`,
      `ProtectHome=read-only`, `PrivateTmp`.

Comandos:

```bash
sudo -u aquifer test -r /srv/aquifer/models/qwen05_affine && echo "modelo legible"
sudo -u aquifer test -w /var/lib/aquifer && echo "estado escribible"
stat -c '%a %U:%G %n' /etc/aquifer /etc/aquifer/aquifer.env
```

## 5. Modelo y configuración

- [ ] Artefacto y tokenizer presentes en el layout acordado.
- [ ] `sha256sum` del artefacto y del tokenizer registrado en el dossier.
- [ ] Commit de Aquifer, versión de Python y `regex` registrados.
- [ ] `sha256sum build/generate_aquifer` registrado.
- [ ] Licencia del modelo revisada y aceptada.
- [ ] Variables del servicio revisadas (`deploy/aquifer.env.example`).
- [ ] `AQUIFER_CTX` definido según el caso (no “lo más grande posible”).
- [ ] `AQUIFER_THREADS` = núcleos físicos.
- [ ] `AQUIFER_MEM_POLICY` (`auto`/`paged`/`resident`) elegido según RAM.
- [ ] `AQUIFER_KV_DTYPE=fp16` salvo necesidad explícita de fp32.
- [ ] `AQUIFER_MAX_TOOL_ROUNDS` y `AQUIFER_TOOL_BUDGET_MS` definidos.
- [ ] Herramientas HTTP, si se usan, con allowlist mínima y config `600`.

Layout:

```text
/srv/aquifer/models/
  qwen05_affine/
  qwen05/tokenizer.json
```

Registro sugerido:

```bash
git rev-parse HEAD
python3 --version
python3 -c "import regex; print(regex.__version__)"
sha256sum build/generate_aquifer
sha256sum /srv/aquifer/models/qwen05_affine/* /srv/aquifer/models/qwen05/tokenizer.json
```

## 6. Evidencia, claves y retención

- [ ] `AQUIFER_TRACE_LOG` en `/var/lib/aquifer/traces.jsonl`.
- [ ] `AQUIFER_TRACE_HMAC_KEY` con 32 bytes aleatorios (cadena de trazas).
- [ ] `AQUIFER_RECEIPT_KEY` con 32 bytes aleatorios (recibos por request).
      Si no se define, se usa la clave de trazas.
- [ ] Claves en `600`, respaldadas y con responsable asignado.
- [ ] `AQUIFER_TRACE_CONTENT=1` solo si el cliente autoriza guardar contenido.
- [ ] Retención acordada: **30 / 90 / 365 días** (elegir y documentar).
- [ ] Política de borrado al cierre del piloto.
- [ ] Bundles por caso con `tools/bundle.py export`.
- [ ] KV incluido en el bundle solo para casos críticos (pesa 25–150 MB).
- [ ] Verificación con `tools/verify.py` y salida archivada.
- [ ] Hash del bundle y del manifiesto registrados.

Generar claves:

```bash
head -c 32 /dev/urandom > /etc/aquifer/receipt.key
head -c 32 /dev/urandom > /etc/aquifer/trace.key
chmod 600 /etc/aquifer/*.key
```

Exportar y verificar:

```bash
python3 tools/bundle.py export --session ~/sesiones/caso-01 \
  --out ~/bundles/caso-01 --include-kv \
  --traces /var/lib/aquifer/traces.jsonl \
  --hmac-key /etc/aquifer/receipt.key \
  --traces-hmac-key /etc/aquifer/trace.key

python3 tools/verify.py --bundle ~/bundles/caso-01 \
  --hmac-key /etc/aquifer/receipt.key \
  --traces-hmac-key /etc/aquifer/trace.key
```

**Pérdida de clave:** las firmas HMAC viejas no se pueden verificar. La cadena
de hashes sigue siendo comprobable, pero sin la clave no hay firma. Guardar las
claves viejas durante la retención acordada.

## 7. Backup y recuperación

- [ ] Backup diario de: `/etc/aquifer`, claves, sesiones, `traces.jsonl`,
      bundles y configuración de systemd/Docker.
- [ ] Modelo: se puede reconstruir/descargar; registrar su hash.
- [ ] Backup cifrado y con acceso restringido.
- [ ] Prueba de restauración **antes** del piloto.
- [ ] Prueba de verificación de un bundle restaurado.
- [ ] Alertas de disco al 80 %.
- [ ] Rotación de logs de systemd/Docker definida.

Restauración mínima:

```bash
# En un host limpio
cp -a backup/etc/aquifer /etc/
cp -a backup/var/lib/aquifer /var/lib/
cp -a backup/bundles ~/bundles
systemctl restart aquifer-serve
python3 tools/verify.py --bundle ~/bundles/caso-01 --hmac-key ...
```

## 8. Monitoreo

- [ ] `/health` monitoreado cada 30 s.
- [ ] Prometheus scrapeando `/metrics/prometheus`.
- [ ] Dashboard de `docs/monitoring/` importado.
- [ ] Alertas activas: caída, error rate, p95 latencia, p95 TTFT, schema,
      timeouts, sin tráfico.
- [ ] Alerta de disco y de proceso.
- [ ] Watchdog probado: motor colgado → 504/restart → siguiente request OK.
- [ ] CPU: no co-locar con cargas pesadas; considerar `cpuset`/`CPUAffinity`.

Prueba de watchdog (opcional, en ventana de mantenimiento):

```bash
# Con mucha carga, verificar que el servidor se recupera
for i in $(seq 1 20); do
  curl -s -o /dev/null -w '%{http_code} %{time_total}\n' \
    http://127.0.0.1:8080/v1/chat/completions \
    -H 'Content-Type: application/json' \
    -d '{"messages":[{"role":"user","content":"Di ok"}],"max_tokens":4}'
done
```

## 9. Pruebas de aceptación

- [ ] `make test` → **255 pruebas Python + suite C** sin fallos.
- [ ] Servicio arranca tras reboot.
- [ ] `/health` responde `ok`.
- [ ] `/v1/models` lista el modelo correcto.
- [ ] Chat simple responde.
- [ ] Streaming SSE funciona.
- [ ] Salida JSON con esquema funciona y falla con 422 cuando corresponde.
- [ ] Tool calling local (`sumar`) funciona.
- [ ] Herramienta HTTP allowlisted funciona.
- [ ] Recibo HMAC aparece en body/cabecera.
- [ ] `verify.py` valida un bundle.
- [ ] `/metrics` y `/metrics/prometheus` responden.
- [ ] Watchdog recupera un motor colgado.
- [ ] Restauración de backup verificada.

Smoke mínimo:

```bash
curl -s http://127.0.0.1:8080/v1/models
curl -s http://127.0.0.1:8080/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"messages":[{"role":"user","content":"Hola"}],"max_tokens":32}'
curl -s http://127.0.0.1:8080/metrics | python3 -m json.tool | head
```

## 10. Día del piloto

### T-1

- [ ] Instalación y modelo listos.
- [ ] Configuración y claves `600`.
- [ ] Firewall/VPN probados.
- [ ] Backup inicial y restauración probada.
- [ ] Smoke test completo.
- [ ] Dashboard y alertas visibles.
- [ ] Responsable de guardia definido.

### T0

- [ ] Servicio activo (`systemctl status` / `docker ps`).
- [ ] `/health` y `/metrics` OK.
- [ ] Consulta de prueba con el caso real.
- [ ] Recibo y traza visibles.
- [ ] Bundle de prueba exportado y verificado.
- [ ] Registro de hora de inicio y versión.

### Durante

- [ ] Revisar errores, timeouts, disco y latencia cada hora.
- [ ] No cambiar configuración sin registro.
- [ ] Si hay 504 repetidos, revisar carga de CPU y logs del motor.
- [ ] Guardar incidencias con hora, request_id y traza.

### Cierre

- [ ] Exportar bundles de los casos acordados.
- [ ] Verificar cada bundle.
- [ ] Registrar hashes y ubicación.
- [ ] Acordar retención/borrado.
- [ ] Entregar informe de métricas.
- [ ] Firma de cierre.

## 11. Runbook de incidentes

| Síntoma | Qué mirar | Acción |
| :--- | :--- | :--- |
| `/health` no responde | `systemctl status`, `docker ps`, logs | Reiniciar servicio; revisar puerto/proceso |
| 504 `generation_timeout` | logs `[engine]`, CPU, load | Esperar restart; bajar carga; revisar `cpuset` |
| Motor reinicia seguido | `dmesg`, CPU, memoria, disco | Aislar CPU, revisar OOM, reducir `ctx`/threads |
| Modelo no encontrado | `AQUIFER_MODELS_DIR`, permisos | Corregir ruta/permisos; reiniciar |
| Disco lleno | `df -h`, tamaño de traces/bundles | Rotar/borrar según retención; ampliar |
| Firma/recibo no verifica | clave correcta, archivo alterado | Usar la clave de la época; revisar cadena |
| Latencia alta | load, threads, `ctx`, modelo | Ajustar threads/ctx; no co-locar cargas |
| Schema falla | trazas `schema_errors`, prompt | Ajustar esquema/prompt; revisar modelo |

## 12. Cierre y firma

| Rol | Nombre | Fecha | Firma |
| :--- | :--- | :--- | :--- |
| Responsable técnico | | | |
| Responsable de negocio | | | |
| Revisor del cliente | | | |

Versión desplegada: `git rev-parse HEAD` = ______________________

Modelo / hash: ______________________

Claves / ubicación: ______________________

Retención acordada: ______________________

---

## Anexo: tamaños y comandos

### Tamaños reales

| Elemento | Tamaño |
| :--- | ---: |
| Artefacto 0.5B | 557 MB |
| Artefacto 1.5B | 1.4 GB |
| Artefacto 3B | 2.4 GB |
| Tokenizer | 6.8 MB |
| KV 0.5B a 2048 | ~25 MB |
| KV 1.5B a 2048 | ~59 MB |
| KV 3B a 2048 | ~75 MB |
| Trace por request (sin contenido) | ~1–2 KB |
| Trace por request (con contenido) | ~3–10 KB |
| Bundle con KV (2048) | ~25–150 MB + JSON/trazas |

### Comandos útiles

```bash
# systemd
systemctl status aquifer-serve
journalctl -u aquifer-serve -f
systemctl restart aquifer-serve

# Docker
docker compose -f deploy/docker-compose.yml ps
docker compose -f deploy/docker-compose.yml logs -f aquifer
docker compose -f deploy/docker-compose.yml restart aquifer

# Salud y métricas
curl -s http://127.0.0.1:8080/health
curl -s http://127.0.0.1:8080/metrics | python3 -m json.tool
curl -s http://127.0.0.1:8080/metrics/prometheus

# Verificación
python3 tools/verify.py --bundle ~/bundles/caso-01 --json \
  --hmac-key /etc/aquifer/receipt.key \
  --traces-hmac-key /etc/aquifer/trace.key

# Pruebas
make test
```
