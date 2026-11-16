# Dimensionamiento del piloto

Guía para medir y elegir hardware con `bench/load_test.py` y
`bench/pilot_sizing.sh`. Los números de referencia son orientativos: el
resultado que vale es el que midas en el hardware del piloto, con el host
tranquilo.

## Qué mide

`bench/load_test.py` mide desde el cliente:

- latencia total p50/p95/p99,
- TTFT (primer chunk de contenido) en streaming,
- tokens de completion y tokens/s del lote,
- errores/status y, si está disponible, agregados de `/metrics`.

`bench/pilot_sizing.sh` arranca el servidor con afinidad de CPU (`taskset`),
corre una pasada secuencial (streaming) y una concurrente (no-streaming), y
genera un Markdown con la matriz.

## Antes de medir

- [ ] Host tranquilo: sin builds, sin contenedores pesados, sin copias grandes.
- [ ] `load average` menor que ~75 % de los CPUs.
- [ ] Motor fijado a núcleos físicos (`ENGINE_CPUS`).
- [ ] Cliente fijado a otros núcleos (`CLIENT_CPUS`) para no competir.
- [ ] Modelo y tokenizer ya descargados (la primera carga no se mide).
- [ ] Registrar CPU, RAM, commit, modelo y hash.

El script **aborta** si el load supera el 75 % de los CPUs, salvo que se pase
`ALLOW_LOAD=1`:

```bash
$ ./bench/pilot_sizing.sh
Host cargado: load 12.63 sobre 8 CPUs.
Los números no serían representativos.
Esperá a que baje o usá ALLOW_LOAD=1 para forzar.
```

En esta máquina de desarrollo, un servicio ajeno consumía 230–390 % de CPU y el
load estaba entre 10 y 17. La matriz completa se interrumpió en el modelo 3B y
varias corridas mostraban reinicios del watchdog. **Esos números se descartaron**
y no se usan como resultado.

## Cómo correr

Rápido, solo 0.5B:

```bash
QUICK=1 ./bench/pilot_sizing.sh
```

Matriz completa (host quieto):

```bash
MODELS="qwen05 qwen qwen3" ./bench/pilot_sizing.sh
```

Afinidad explícita:

```bash
ENGINE_CPUS=0-3 CLIENT_CPUS=6,7 \
  MODELS="qwen05 qwen" ./bench/pilot_sizing.sh
```

Parámetros:

| Variable | Default | Qué hace |
| :--- | ---: | :--- |
| `MODELS` | `qwen05 qwen qwen3` | Modelos a medir |
| `LAT_REQS` | 6 | Requests secuenciales (streaming) |
| `THR_REQS` | 8 | Requests concurrentes |
| `THR_CONC` | 4 | Concurrencia |
| `MAX_TOK` | 16 | Tokens de salida por request |
| `REQ_TIMEOUT` | 120 | Timeout del cliente por request |
| `OUT` | `bench/results/sizing-<fecha>.md` | Salida Markdown |
| `QUICK` | 0 | `1` = solo 0.5B, 3+3 requests |

Docker: el equivalente al pinning es `cpuset` en el servicio:

```yaml
    cpuset: "0-3"
```

El script usa `taskset`; si no está disponible, corre sin afinidad y avisa.

## Números de referencia

Del README y de una corrida rápida en esta máquina **con load ~12** (por lo
tanto, cota inferior, no un máximo):

| Modelo | Contexto | Hilos | tok/s | TTFT p50 | Latencia p50 (16 tok) |
| :--- | ---: | ---: | ---: | ---: | ---: |
| Qwen 0.5B | 2048 | 4 | ~40 | ~25 ms | ~395 ms |
| Qwen 0.5B | 512 | 2 | ~19 | ~80 ms | ~840 ms |

Referencias del proyecto en chat interactivo (host descargado):

| Modelo | tok/s aproximados | Nota |
| :--- | ---: | :--- |
| Qwen 0.5B | 35–50 | 4 núcleos físicos |
| Qwen 1.5B | 15–20 | 4–8 núcleos |
| Qwen 3B | medir | Más lento; 8 núcleos y 32 GB recomendados |

No son garantías: dependen de CPU, RAM, `ctx`, longitud de salida, carga y caché
de prefijo.

## Fórmula de capacidad

```text
tok/s_necesarios = usuarios × requests_por_minuto × tokens_de_salida / 60
```

Ejemplos:

- 10 usuarios × 1 request/min × 200 tokens = 2000/60 ≈ **33 tok/s**.
- 5 usuarios × 2 requests/min × 400 tokens = 4000/60 ≈ **67 tok/s**.
- 20 usuarios × 1 request/min × 300 tokens = 6000/60 = **100 tok/s**.

El servidor **serializa generaciones**: la concurrencia se encola. Para el
piloto, dimensioná por tokens/minuto y por latencia aceptable, no por cantidad
de conexiones simultáneas.

## Recomendación conservadora

| Escenario | Modelo | CPU | RAM | Disco | Comentario |
| :--- | :--- | :--- | :--- | :--- | :--- |
| 1–3 usuarios, respuestas cortas | 0.5B | 4 núcleos AVX2 | 8 GB | 20 GB | TTFT bajo, buena opción de demo |
| 3–8 usuarios, respuestas medias | 1.5B | 4–8 núcleos AVX2 | 16 GB | 40 GB | Cola por generación |
| 3B o contexto grande | 3B | 8 núcleos AVX2 | 32 GB | 60 GB | Medir en host quieto |

- No se necesita GPU.
- Más `ctx` = más KV y más latencia; no sobredimensionar sin necesidad.
- `mem-policy paged` baja RSS pero paga page faults; `resident` pide RAM para
  los pesos.
- Bajo contención de CPU, el watchdog puede cortar y reiniciar el motor; esos
  runs no sirven para dimensionar.

## Interpretar los resultados

- Descartar corridas con `[engine] reiniciado` o timeouts en los logs.
- Comparar p95, no solo el promedio.
- Revisar TTFT: es lo que percibe el usuario al empezar a leer.
- Repetir 2–3 veces y quedarse con la mediana.
- Guardar el Markdown generado y anotar load, CPU y commit.

## Reproducir y archivar

```bash
MODELS="qwen05 qwen" \
  LAT_REQS=6 THR_REQS=8 THR_CONC=4 MAX_TOK=16 \
  OUT=bench/results/sizing-piloto.md \
  ./bench/pilot_sizing.sh
```

Adjuntar al dossier del piloto:

- `bench/results/sizing-piloto.md`,
- `uptime` y `lscpu` del host,
- commit de Aquifer y hash del modelo,
- decisión de hardware y por qué.

Ver también: [`CHECKLIST-PILOTO.md`](CHECKLIST-PILOTO.md) y
[`deploy/README.md`](../../deploy/README.md).
