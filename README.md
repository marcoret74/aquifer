# caudal

> CPU tensor runtime built around the memory wall.


## Orden de construcción

| # | Artefacto | Gate |
|---|---|---|
| 1 | `include/caudal/layout.hpp` — ABI congelado | `static_assert` ✅ |
| 2 | `bench/probe0_bw.cpp` — `BW_ef` | número medido ✅ 21 GB/s |
| 3 | `bench/probe1_gemv.cpp` — int4 AVX2 | ≥ 60 % de `BW_ef` ✅ 79 % |
| 4 | `tests/test_doublemap.cpp` — memfd doble mapeo | los mapas aliasan ✅ |
| 5 | `tools/pack_synth.py` + `gemv_int4.hpp` | round-trip ✅ 1e-4 |
| 6 | `tools/pack_qwen.py` — safetensors reales | `|Δlogit| < 1e-2` |
| 7 | arena A0/A1 sobre el artefacto | THP + anillo juntos |

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
