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

## 7. Qué NO promete

- No acelera un request individual.
- No reduce el tráfico de DRAM.
- No hace que una capa «quepa» en L3.
- No reemplaza batching/serving (lo complementa).
