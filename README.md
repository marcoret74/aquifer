# caudal

> CPU tensor runtime built around the memory wall.

Motor de inferencia de transformers sobre CPU, sin GPU. No es un fork de nada:
es una derivación de una sola ley,

```
tok/s = η · BW_ef / bytes_del_modelo
```

todo lo demás (layouts, kernel, arenas, poda) son consecuencias de esa ecuación.

## Estado

**Fase 1 · Día 0.** Sólo existe el probe que mide el denominador. No hay modelo,
no hay kernel, no hay arena. A propósito.

## Día 0

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
./build/probe0_bw 512 3
```

Mide ancho de banda de **lectura pura** sobre memoria anónima pre-faulteada, a
1..N hilos, con y sin `MADV_HUGEPAGE`. Ese número (`BW_ef`) es el que gobierna
todas las tablas de tok/s.

## Medición · Día 0 (2026-11-13)

i7-10510U · 8 hilos · L3 8 MiB · 10 GB RAM · `probe0_bw 512 3`

| Configuración | Batería | **Enchufado** |
|---|---|---|
| DRAM (512 MiB, lectura pura) | 12.9 GB/s | **18.0 GB/s** |
| L3 (8 MiB, 4 hilos) | — | 107 GB/s |

Lectura del dato:

- La DRAM **satura en ~18 GB/s con 2 hilos**; 8 hilos no mejora (15→18→17→12).
  Plateau ⇒ es límite de memoria, no de tasa de emisión.
- La L3 **sí** escala (18 → 58 → 107 GB/s a 1/2/4 hilos) ⇒ el barrido mide bien y
  el muro es la DRAM. Ratio DRAM/L3 = **5.9×**.
- 18.0 GB/s = **84 % de DDR4-2666 de un solo canal** (21.3 GB/s teóricos).
  Hipótesis: un solo DIMM. Confirmar con `sudo dmidecode -t 17`.
  Si es así, un segundo SODIMM duplica el ancho de banda y vale más que
  cualquier optimización de kernel.
- THP: `anon = [madvise]` (opera, pero sólo 37 % por fragmentación:
  `Committed_AS` 22.8 GB sobre 11.4 GB); **`shmem = [never]`** ⇒ el plan de A1
  (`memfd` + THP) es **no-op** sin root; pool `hugetlbfs` = 0.
  ⇒ El backing de A1 debe ser seleccionable en runtime, no fijo.

### Derivados (η = 0.8 sobre BW_ef)

| Modelo | int4 | tok/s hoy | tok/s con 2º DIMM (~36 GB/s) |
|---|---|---|---|
| Qwen2.5-0.5B | 0.30 GB | 48 | 96 |
| Qwen2.5-1.5B | 0.88 GB | 16 | 33 |
| Qwen2.5-3B | 1.74 GB | 8.3 | 17 |
| Qwen2.5-7B | 4.28 GB | 3.4 | 6.8 |
| Qwen2.5-14B | 8.27 GB | 1.7 | 3.5 |

`M*(int4) = 55` ⇒ en este equipo el ridge es **inalcanzable**: no existe lote
que lo haga compute-bound. La única palanca es leer menos bytes.

## Medición · Día 1 — `probe1_gemv` (2026-11-13)

Forma: N=65536, K=3584 (234.9 M params) · buffer int4 112 MiB · AVX2.

| Config | GB/s | % del techo 1-hilo |
|---|---|---|
| stream (1 hilo) | 13.7 | — |
| **int4 M=1, 1 hilo** | **11.4** | **83.4 %** |
| int4 M=1, 8 hilos | **21.2** | — |

- **Gate 10.8 GB/s: PASA** (11.4), con 83.4 % de eficiencia sobre el techo
  single-thread. El costo de desempacar 4 bits en AVX2 es **~17 %**.
- El agregado a 8 hilos (21.2 GB/s) **supera los 18.0 de `probe0`** ⇒
  `probe0` subestimó: sus 512 MiB con páginas de 4 KiB (THP al 37 %) pagaban
  TLB. El DRAM real es **≥ 21 GB/s**.
- **ASTILLA: el lote no compra ancho de banda.** M=1→8 baja el GB/s
  (11.4 → 3.5) pero sube el cómputo (22.8 → 56.2 GMAC/s, 2.5×). Se compra
  COMPUTO, no bytes. Con 8 hilos ya se está limitado por memoria en M=1.
- **ASTILLA: esta máquina no puede resolver un margen del 2 %.** El mismo
  binario dio M=1 entre 9.0 y 11.6 GB/s según el estado de potencia. Con
  `passes=30` la corrida es estable (min=max en 3 corridas), pero entre
  corridas el reloj deriva. Toda validación fina necesita reloj estable.

`tok/s` proyectado para Qwen2.5-7B int4: **21.2 / 4.28 ≈ 4.9 tok/s**.

## Orden de construcción

| # | Artefacto | Gate |
|---|---|---|
| 1 | `include/caudal/layout.hpp` — ABI congelado | `static_assert` |
| 2 | `bench/probe0_bw.cpp` — `BW_ef` | número medido |
| 3 | `bench/probe1_gemv.cpp` — int4, 3 layouts | ≥ 60 % de `BW_ef` |
| 4 | `tests/test_doublemap.cpp` — memfd doble mapeo | los mapas aliasan |
| 5 | `tools/pack_qwen.py` — safetensors → A0 | `|Δlogit| < 1e-2` |
| 6 | arena | THP efectivo |

El 3 es el go/no-go: si el kernel int4 no alcanza el 60 % del ancho de banda,
todo lo demás cambia de número y hay que arreglarlo antes de escribir una línea
más.

## Relación con los otros repos

Sin dependencias cruzadas.

- `sg` — motor semántico. Congelado. **No se toca.**
- `hiperram` — motor de memoria espacial (proyecto independiente, roadmap propio).
- `caudal` — este repo. Copia vendorizada de `HilbertCurve3D` cuando haga falta
  (MIT), nunca `#include` cruzado.

`caudal` se puede tirar a la basura en cualquier momento sin tocar `sg`.

## Licencia

MIT. Ver `LICENSE`.
