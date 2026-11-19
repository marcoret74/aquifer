# Idea: scheduler por capas (time-slicing tipo «FPS»)

**Origen:** idea de Marco (2026-11-18), planteada en términos de cuadros por
segundo: cargar una capa, mantener sus pesos «vivos» ~10 ms, calcular, guardar
las activaciones en memoria espacial (cubo / contenedor), y pasar el turno a la
siguiente capa, dando la sensación de multitarea cuando en realidad hay
secuencialidad por turnos de milisegundos.

**Estado:** idea registrada. El experimento que la decide está **pendiente**
(se corre con la máquina libre, después del empaquetado del 7B).

---

## 1. Reformulación precisa

No es paralelismo de álgebra: es **scheduling**.

1. Una ventana de puntero apunta al bloque de pesos de la capa L (es un índice
   sobre el artefacto, no una copia).
2. Durante el turno se computa la capa L con sus pesos «vivos».
3. Las activaciones (salida de la capa) se guardan en el espacio de trabajo
   cache-residente.
4. Se cambia el puntero a L+1 y se repite.

El «turno» de 10 ms es análogo al time-slice del SO o al warp de una GPU:
existe para **solapar memoria con cómputo**, no para repartir FLOPs.

## 2. Números del hardware (i7-10510U, 4C/8T, 15 W)

| Modelo | Capas | Hidden | Peso por capa (aprox.) | Veces L3 (8 MB) |
|---|---|---|---|---|
| Qwen2.5-0.5B | 24 | 896 | ~23 MB | 3× |
| Qwen2.5-1.5B | 28 | 1536 | ~50 MB | 6× |
| Qwen2.5-3B | 36 | 2048 | ~67 MB | 8× |
| Qwen2.5-7B | 28 | 3584 | ~160 MB | 20× |

Tiempo por capa medido (3B, ~5 tok/s): **~5,5 ms**. Un turno de 10 ms es del
orden correcto.

## 3. Los tres límites

1. **Dependencia secuencial.** L+1 necesita la salida de L. No hay paralelismo
   dentro de un token; solo solapamiento o entrelazado de streams.
2. **Working set > caché.** 23–160 MB por capa contra 8 MB de L3. Los pesos no
   quedan «vivos» en caché: se streamean desde DRAM en cada turno. A 20 GB/s,
   67 MB son ~3,3 ms de lectura: el turno se va en memoria.
3. **El switch es gratis; el dato no.** Cambiar el puntero cuesta nanosegundos;
   traer la capa cuesta milisegundos.

## 4. Dónde la idea sí pega

- **Activaciones en caché** (ya ocurre): ~4 KB por token por capa; 36 capas
  ≈ 144 KB, cabe en L2/L3.
- **Solapar capas** (ya existe): `AsyncPrefetcher` + `MADV_WILLNEED` de L+1
  mientras se computa L. La idea del «turno» es su versión explícita.
- **Entrelazar streams independientes** (la pepita): con dos consultas en
  curso, mientras una espera DRAM la otra computa. Eso sí produce multitarea
  real medida en throughput agregado.

## 5. Experimento que decide

Con Qwen2.5-3B y el artefacto ya empaquetado:

| Paso | Medición |
|---|---|
| 1 request | tok/s, `LLC-load-misses`, `LLC-loads` |
| 2 requests simultáneos | tok/s **agregado**, latencia individual, misses |
| 4 requests simultáneos | idem |

Comando base:

```bash
perf stat -e LLC-load-misses,LLC-loads \
  python3 tools/serve.py --model qwen3 --artifact ... --tokenizer ...
```

**Criterio:** si el tok/s agregado sube **> 1,3×** al pasar de 1 a 2 streams,
la hipótesis tiene base y se implementa un scheduler de streams en
`tools/serve.py` (hoy el motor serializa). Si no sube, el cuello es ancho de
banda puro y ningún turno lo arregla.

## 6. Si el experimento pasa

Diseño mínimo:

- cola de requests con round-robin por turno (no por capa: por **stream**),
- un motor por stream compartiendo el mismo `mmap` del artefacto (page cache
  compartida),
- métrica por request + agregada en `/metrics`,
- límite de streams por RAM disponible (`MemAvailable`).

## 7. Variante «ventana + batching» (la que paga)

La ventana de **una capa** sola no rinde, porque la capa se reutiliza en cada
token. La forma de amortizar la lectura es aplicar la capa a **muchos tokens**
antes de soltarla:

```
por capa:  traer 67 MB  ->  aplicar a M tokens  ->  soltar
costo por token  ~  bytes_capa / (M * ancho_de_banda)
```

| Modelo | Artefacto | 1 token/pasada | M=8 | M=32 |
|---|---|---|---|---|
| 3B | 2,4 GB | ~1,2 s/token | ~0,15 s | ~0,04 s |
| 7B | 4,5 GB | ~2,2 s/token | ~0,28 s | ~0,07 s |

El prefill ya lo hace (`--prefill-batch 8`). Extenderlo al decode significa
**lote de streams**: varios requests avanzan una capa a la vez, y la lectura se
divide entre todos.

### Matiz importante

`MADV_DONTNEED` sobre un `MAP_SHARED` de archivo **no borra la page cache**:
suelta el mapeo. Con RAM libre, el kernel conserva las páginas limpias y el
próximo *seek* las trae de caché. El costo real aparece solo bajo presión.
Por eso `auto` gana cuando el modelo entra, y `paged` gana cuando no entra.

### Política propuesta

1. Si el artefacto entra en `MemAvailable` con holgura: `auto` (nada que ganar).
2. Si no entra: ventana de N capas (N = lo que quepa) + lote de M tokens.
3. Orden de capas: mantener calientes las primeras y las últimas (donde suele
   concentrarse la sensibilidad), paginar las intermedias.
4. Medir siempre `majflt/s`: es la señal de que la ventana se quedó chica.

## 8. Qué NO promete

- No acelera un request individual.
- No reduce el tráfico de DRAM.
- No hace que una capa «quepa» en L3.
- No reemplaza batching/serving (lo complementa).

---

## 9. Revisión externa (Gemini, 2026-11-18)

### Lo que validó (y coincide con lo medido)

- **Literatura**: FlexGen (ICML 2023), LLM in a flash (Apple, 2023), Orca
  (OSDI 2022), PagedAttention/vLLM (SOSP 2023), ZeRO-Inference.
- **Decode M=1 es GEMV**: ~0,5 FLOP/byte ⇒ *bandwidth-bound*. Correcto.
- **Modelo de I/O**: 4,4 GB a 2 GB/s = 2,2 s/token. Medido en `paged`:
  **2,5 s/token**. El modelo acierta.
- **`auto` vive de page cache**: 2,8 tok/s × 4,4 GB = **12,3 GB/s efectivos**,
  consistente con DRAM y no con NVMe.
- **`DONTNEED` no libera RAM**: destruye PTEs; las páginas quedan en page cache
  y el re-uso es *minor fault*.

### Tres correcciones

1. **KV cache mal calculada.** No es `2·L·hidden·2`, es
   `2·L·(kv_heads·head_dim)·2`:

   | Modelo | KV/token real | Gemini asumió | 8 streams × 2048 |
   |---|---|---|---|
   | 3B | **36 KB** | 288 KB (8×) | **0,60 GB** (no 4,8) |
   | 7B | **56 KB** | 392 KB (7×) | **0,94 GB** (no 3,2) |

   Usó `hidden_size` en vez de `kv_heads·head_dim`. La KV sigue siendo un
   problema, pero 7–8 veces menor.

2. **«Miles de minor faults por capa»**: con `MADV_HUGEPAGE` (2 MB) una capa de
   160 MB son ~80 PTEs, no miles. 28 capas ≈ 2.240 faults/token ≈ pocos ms.
   El costo existe, pero 1–2 órdenes de magnitud menor.

3. **«2 streams no dan ganancia porque compiten por DRAM»**: depende de la
   saturación. Hoy el 7B usa 12,3 GB/s efectivos contra un pico de ~20–25 GB/s:
   **hay margen**. La ganancia de entrelazar streams es ocultar latencia, no
   sumar ancho de banda. El experimento (GB/s efectivos) lo decide, no el
   argumento.

### Alternativas que propone (válidas)

- **A. `O_DIRECT` + io_uring + doble buffer**: memoria fija 2×capa, cero page
  faults, I/O asíncrono. Requiere lote grande (80 ms de I/O por capa vs 5,5 ms
  de cómputo en M=1). Detalle práctico: `O_DIRECT` exige alineación (probado:
  `dd iflag=direct` falla sin alinear el buffer).
- **B. Bloques (macro-capas)**: si entra, `auto`; si no, partir en 2–4 bloques
  y batchear por bloque. Reduce switches de 28–36 a 2–4 por token.
- **C. Sparsity de activaciones en MLP** (Apple): 20–30 % de los pesos por
  capa; requiere predictor y validación de calidad.

### Protocolo de medición actualizado

| Métrica | Fuente |
|---|---|
| `majflt/s` | `/proc/vmstat` (`pgmajfault`) |
| `minflt/s` | `/proc/<pid>/stat` |
| ciclos, instrucciones, cache-misses | `perf stat` |
| TTFT e inter-token (p50/p99) | timestamps del cliente |
| GB/s efectivos | `bytes_artefacto × tok/s` |

**Criterio**: agregado > 1,3× **y** GB/s efectivos < 80 % del pico. Si ya
satura, no hay ganancia posible: el cuello es el bus, no el scheduler.

### Segunda ronda (2026-11-18)

**Corrección a la comparación A vs B.** Gemini concluye que A falla (8,8 GB ÷
2 GB/s = 4,4 s/token) y que B lo logra porque amortiza entre M streams. Eso
compara **A con M=1** contra **B con M=8**. Con el mismo lote, ambos leen el
artefacto completo una vez por pasada:

| Esquema | Bytes por pasada | Ventaja real |
|---|---|---|
| A (capa por capa, io_uring) | artefacto completo | cero page faults, memoria fija |
| B (macro-bloques) | artefacto completo | 2–4 switches en vez de 28–48 |

B **no reduce bytes**; reduce *overhead* (cambios de contexto, PTEs, syscalls).
La ventaja de velocidad viene del lote M, no del tamaño del bloque.

**Speculative decoding**: la idea de usar el 0,5B como draft y verificar K
tokens por pasada es correcta en el principio (bytes/token = artefacto / K),
pero «quintuplica» solo con aceptación del 100 %. Medido antes en esta pila:
aceptación 25–54 % con draft de 0,5B sobre 1,5B/3B, y **más lento** que el
target solo. Para el 14B el cociente es más favorable, pero la mejora realista
es 1,5–2,5×, no 5×.

**O_DIRECT (alineación)**: la solución de Gemini es correcta —alinear los
offsets en el empaquetado (padding de 4 KB, desperdicio ~200 KB) y, si el
artefacto ya existe, leer desde `offset & ~4095` y computar desde
`buffer + Δ`. `loadu` en Skylake+ no penaliza de forma apreciable.

**El 14B no corre en esta máquina (dato que faltaba):** el artefacto a 5,0 bpw
pesa ~11 GB (9,2 GB de pesos + 1,6 GB de embed). El NVMe tiene 5,7 GB libres y
el HDD (`/mnt/hdd`, mecánico, 93 MB/s) daría ~115 s/token. No es un problema de
scheduler: no entra. El techo real acá es el 7B (4,4 GB, 2,8 tok/s).
