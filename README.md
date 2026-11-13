# caudal

> CPU tensor runtime built around the memory wall.


## Día 12 — Test 2: el Juez audita a la Voz (2026-11-13)

`tools/test_verifier.py`. Pipeline completo: **caudal genera** (Qwen2.5-1.5B int4,
21 tok/s) → se extraen claims con los relatores que SG reconoce → **`sg.py verify`**
audita (daemon, sub-ms).

### Resultado

| métrica | valor | umbral del consejo |
|---|---|---|
| oraciones generadas | 17 | — |
| **con claim extraíble** | **7 (41,2 %)** | > 20 % ✅ |
| **veredicto definitivo** | **4 (57,1 %)** | — |
| CONFIRMED | 3 | |
| CONFIRMED_BY_PATH | 1 | |
| NO_EVIDENCE | 3 | |
| **falsos confirmados** | **0 / 6** | 0 ✅ |

El control anti-alucinación con claims falsos por construcción (`el gato es un
planeta`, `el agua es metal`, `el sol es un mamífero`, `la luna es un pez`,
`el perro es una estrella`, `el hierro es un gas noble`): **6/6 NO_EVIDENCE.**
El Juez no confirma basura.

Y no es ingenuo: `agua es placenta`, `agua es metal`, `agua es roca` dan todas
NO_EVIDENCE. Confirma `agua → líquido`, no cualquier objeto.

### Tres defectos de SG que el test encontró

1. **Objetos multi-palabra truncados al prefijo conocido.**
   `agua es líquido incolorante` resuelve a `obj='líquido'` y confirma. Eso
   infla el veredicto: se confirma el hiperónimo, no el claim completo.
2. **El heurístico de plurales corrompe palabras terminadas en `-s`.**
   `fotosíntesis` → **`fotosíntesi`** → el nodo no resuelve → NO_EVIDENCE sobre
   algo que el grafo sí sabe.
3. **Sujetos compuestos inconsistentes.** `derecho del trabajo es una piedra`
   conserva el sujeto completo, pero `derecho del trabajo es conjunto de normas`
   resuelve a `sujeto='derecho'` — confirma un claim más débil.

### La conclusión que sobrevive a todo

> **La interfaz proposicional funciona.** A 21 tok/s de generación y sub-ms de
> verificación, el Juez audita a la Voz sobre texto real, con 41 % de cobertura
> y **cero falsos positivos**.

Es la única de las tres vías de convergencia que sobrevivió, y sobrevivió porque
no dependía de que el cubo fuera semántico: opera sobre el **grafo**.

## Día 13 — el layout intercalado: la medición sintética era optimista (2026-11-13)

Hipótesis: `weights.bin` + `scales.bin` son **dos streams** por hilo; intercalar
`[pesos fila][escalas fila]` daría **un solo stream**.

### El test sintético dijo que sí

`probe3_ceiling`, 4 hilos:

| layout | GB/s | % del continuo |
|---|---|---|
| A. continuo (techo) | 25,06 | 100 % |
| B. dos arrays (producción) | 22,66 | 90,4 % |
| **D. intercalado** | **24,73** | **98,7 %** |

+2,07 GB/s (+9,1 %). Aprobado.

### Producción dijo que no

| layout | bytes/token | tok/s | GB/s |
|---|---|---|---|
| dos arrays | 868,3 MB | **22,2** | 19,3 |
| **intercalado** | 893,1 MB | **18,0** | 16,1 |

**−19 %.** El texto es idéntico, la matemática es correcta: es puro rendimiento.

### Por qué el test sintético mintió

El buffer de D se construyó **sin relleno** (`stride = K/2 + G*2`, ej. 816 B).
Pero el kernel necesita **stride múltiplo de 64** o las cargas de 32 B cruzan
línea. Al rellenar (816 → 896) aparece un **hueco de 32 B por fila**.

> **El prefetcher de hardware sigue dos streams densos mejor que uno con
> agujeros.** 3,6 % de huecos cuestan más que el 9 % de tener un stream extra.

Y el relleno además sube los bytes: 4,500 → **4,629 bits/param** (+2,9 %).

### Un bug de borde que costó dos corridas

Con el stride rellenado, `G = (stride − K/2)/2` da **288 en vez de 280**. El
índice ahora lleva `G` explícito. Derivar parámetros de un campo que cambió de
semántica es exactamente el error que ya nos mordió con `L.G = K/64`.

### Estado

Se conserva **el layout de dos arrays** (22,2 tok/s) y el código soporta ambos
formatos por si algún día se re-mide con otro relleno.

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
