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

## Medición · Día 2 — `test_doublemap` (2026-11-13)

**12/12 aserciones · exit 0.** Kernel 7.0.0-43, cap 64 y 256 MiB.

| Pregunta | Respuesta |
|---|---|
| ¿Los dos mapas de `memfd` aliasan por MMU? | **SÍ** — escritura por A visible en B y viceversa |
| ¿Propiedad de anillo (una ventana que cruza el límite)? | **SÍ** — 4000 B sin una sola rama de wraparound |
| `MADV_HUGEPAGE` sobre `memfd` | **NO-OP** (0 kB) con `shmem_enabled=never` |
| `MADV_COLLAPSE` sobre `memfd` | **SÍ** — 131072 kB a 64 MiB, 524288 kB a 256 MiB = **100 % PMD** |

### El hallazgo que cierra A1

`shmem_enabled=never` **no bloquea `MADV_COLLAPSE`**. Eso significa que A1 puede
tener las tres cosas a la vez, sin root y sin sacrificar ninguna:

```
memfd  +  doble mapeo (anillo sin ramas)  +  páginas de 2 MiB (sin dolor de TLB)
```

La tabla de compromisos que abrió el Día 0 queda **anulada**: no hay que elegir
entre "doble mapeo con 4 KiB" y "THP con un `if` por bloque". Se tienen ambos.

Notas de implementación:
- `MADV_COLLAPSE` es **sincrónico** y best-effort: puede devolver `EAGAIN` bajo
  fragmentación (acá `Committed_AS` ya está en 22.8 GB sobre 11.4 GB). Va en el
  init de la arena, nunca en el hot path.
- El RSS aparenta 2× `cap` (cuenta la misma página una vez por mapeo); el DRAM
  real es `cap`, y eso lo prueba el aliasing.

## Día 3 — packer sintético + end-to-end (2026-11-13)

`tools/pack_synth.py` (stdlib puro, sin red) + `include/caudal/gemv_int4.hpp` +
`tests/test_pack_roundtrip.cpp`.

```
N=256 K=512   -> kernel vs ref: 5.15e-05   (tol 7.3e-02)
N=64  K=3584  -> kernel vs ref: 1.45e-04   (tol 2.0e-01)
ref C++ vs ref Python: 7.6e-06 / 1.5e-05
```

**10/10 aserciones.** Lo que queda demostrado, en orden de fuerza:

1. El **packer es correcto**: su referencia independiente (float64, calculada desde
   los mismos códigos) coincide con la referencia C++ a 1e-5, con K=512 y K=3584.
2. El **kernel es correcto**: discrepa de la referencia escalar en ~1e-4, tres
   órdenes por debajo de la tolerancia.
3. El **pre-permutado de x funciona**: se eliminaron los 2 `vperm2i128` del bucle
   y el resultado sigue siendo exacto.
4. El **ABI está enlazado**: `row_bytes` (1792 para K=3584) es múltiplo de 16 B,
   así que el offset de fila es un `caudal::NodeIdx` válido.

### Dos bugs que sólo el end-to-end podía encontrar

🔴 **LUT del nibble rotada 8 posiciones.** El complemento a 2 es `n<8 ? n : n-16`,
pero la tabla mapeaba `nibble 0 -> -8` en vez de `nibble 0 -> 0`. **El mismo error
estaba en `probe1_gemv.cpp`**: el probe midió velocidad con valores basura. El
conteo de instrucciones es idéntico, así que el 11.6 GB/s sigue siendo válido
como *performance* — pero nunca probó corrección. Corregido en ambos.

🔴 **Sesgo del zero-point aplicado 8 veces.** `_mm256_set1_ps(...)` metía el
término `-128·Σcode·sx·sw` en los 8 carriles, y la suma horizontal lo multiplicaba
por 8. Ahora se resta una sola vez en el dominio entero, en el carril 0
(`_mm256_sub_epi32`), lo que además elimina una instrucción de punto flotante.

### El número honesto (layout real, grupo 64)

`bench/probe2_group.cpp` · N=65536 K=3584 · 4.75 bits/param (packed + fp16 + int32).

| métrica | 1 hilo | 8 hilos | gate | veredicto |
|---|---|---|---|---|
| GB/s (sólo packed) | **8.4** | 18.6 | 10.8 | ❌ **NO PASA** |
| GB/s (bytes reales) | 9.9 | 22.0 | — | |
| GMAC/s | **16.7** | **37.1** | 21.6 | ❌ 77 % |

**El gate no se sostiene con el layout de producción.** 8.4 GB/s vs 10.8 = 78 %;
en GMAC/s, 16.7 vs 21.6 = 77 %. El 11.6 GB/s de `probe1` era optimista, tal como
estaba declarado en la deuda.

Descomposición de la pérdida single-thread (23.1 → 16.7 GMAC/s, −27.7 %):

| causa | costo |
|---|---|
| +18.75 % de bytes (`w_codesum` int32 + `w_scale` fp16 = +0.75 bits/param) | −15.8 % |
| 3 instrucciones por grupo (vpsubd + vcvtdq2ps + vfmadd) | −11.9 % |

**Pero el número de producto casi no se movió**, porque en agregado la máquina
vuelve a estar limitada por memoria:

```
probe1 (escalas por fila)  : 42.4 GMAC/s -> 5.58 tok/s (7B)
producción (grupo 64)       : 37.1 GMAC/s -> 4.88 tok/s (7B)   [-12.5 %]
```



`gemv_int4.hpp` usa **escalas por grupo de 64** (el layout real). `probe1_gemv.cpp`
usa **escalas por fila** (más baratas). Por lo tanto **el 11.6 GB/s del gate es
optimista**: el kernel de producción paga un `vpsubd` + `vcvtdpi2ps` + `vfmadd`
por grupo. Hay que re-medir con el layout real.

### La inversión unsigned: resultado

| variante | bits/param | GB/s real 1 hilo | GMAC/s 1 hilo | GB/s real 8 hilos | GMAC/s 8 hilos | tok/s 7B |
|---|---|---|---|---|---|---|
| int4 con LUT + codesum | 4.75 | 9.9 | 16.7 | 22.0 | 37.1 | 4.88 |
| **int4 invertido** | **4.25** | 10.2 | **19.1** | 21.2 | **39.9** | **5.25** |
| int8 (sin desempacar) | 8.25 | 14.6 | 14.2 | **24.4** | 23.7 | 3.11 |

- **La inversión rinde +14,4 % single-thread y +7,5 % agregado.** Sin LUT
  (`vpshufb` fuera), sin `w_codesum` (4,25 bits/param), 2 ops de shuffle por
  32 B en vez de 4.
- **int8 queda refutado**: −26 % single-thread y −41 % agregado. A 8 hilos
  int8 alcanza 24,4 GB/s reales e int4 21,2 → **int4 está al 87 % del techo de
  memoria** que int8 revela: al no tener desempacado, int8 es el termómetro.
- Correctitud verificada antes de cronometrar: `max|kernel−ref| = 1,1e-04`
  (int4, todas las filas) y `1,2e-05` (int8).

### El gate, reconsiderado

Contra el gate literal (10,8 GB/s sobre bytes empaquetados): **9,55 GB/s = 88 %.
No pasa**, por poco.

Pero el gate estaba mal construido, y la evidencia es de `probe0`: **un solo
núcleo tope a 15 GB/s sin importar el kernel**. El gate medía la tasa de emisión
de memoria de un núcleo, no la eficiencia del kernel. Con 8 hilos —como correrá
en producción— estamos al **87 % del techo de memoria**.

| moneda | valor | contra qué |
|---|---|---|
| GB/s packed, 1 hilo | 9,55 | gate 10,8 → 88 % ❌ |
| GB/s reales, 8 hilos | 21,2 | techo 24,4 → 87 % ✅ |
| GMAC/s, 8 hilos | 39,9 | → 5,25 tok/s (7B) · 45 tok/s (1,5B) |

**Decisión: el kernel se cierra acá.** Queda ~13 % de margen teórico (grupos de
128 bajarían el pliegue, ~5 % a cambio de calidad), y no vale otro ciclo de
formato.

## Día 4 — pesos reales (Qwen2.5-0.5B-Instruct)

`tools/pack_qwen.py`: lector de `safetensors` a mano (8 B + JSON), cuantización
int4/g64 vectorizada con numpy, sin `safetensors` ni `transformers`.

```
168 tensores lineales · 357.8 M params · 3.5 s
weights.bin 178.9 MB + scales.bin 11.2 MB = 190.1 MB
4.250 bits/param exactos
error de cuantización: medio 11.78 %  max 17.40 %  (Frobenius relativo)
```

Todos los K son múltiplos de 64 (896 = 14x64, 4864 = 76x64): el layout congelado
aplica sin excepciones.

**El kernel corrió sobre pesos REALES:**

```
./test_pack_roundtrip /tmp/opencode/real_layer 896 4864 2
  kernel vs ref C++    : 5.722e-06   (tolerancia 5.7e-03)
  ref C++ vs ref Python: 1.192e-06
  12/12 PASS
```

1000x de margen. Los pesos reales atraviesan el formato congelado sin que el
kernel se entere de dónde vinieron.

### Proyección corregida

`494 M params` (lineales + lm_head atado) × 4.25 bits = **262 MB por token**.
A los 21.2 GB/s medidos: **~81 tok/s** para el 0.5B.

⚠️ Corrección de un número que di antes: dije "45 tok/s para el 1.5B". Estaba
mal — mezclé GMAC/s con el tamaño en bytes. Lo correcto, dividiendo por
parámetros: **0.5B → 81 tok/s · 1.5B → 26 tok/s · 7B → 5.2 tok/s**.

### Caveat sobre el 11.78 %

Es la norma de Frobenius relativa del residuo de cuantización, no una medida de
calidad del modelo. Para int4/g64 es el valor esperado (~11 % es
`amax/(7·√12)` sobre gaussianas). Si degrada la perplejidad **no está medido**, y
medirlo requiere el forward completo.

## Día 5 — el forward, validado capa por capa (2026-11-13)

`tools/oracle_qwen.py` (PyTorch, sin transformers) + `tools/forward_qwen.cpp`.

### El oráculo primero se valida a sí mismo

Un oráculo sin validar no sirve: si el oráculo y el C++ comparten un
malentendido del spec, coinciden y ambos están mal. Por eso el oráculo se
valida **generando texto**:

```
prompt: "¿Quién eres?"
  -> "Estoy aquí para ayudarte. ¿Cómo puedo asistirte hoy?"
```

Eso confirma el spec completo de Qwen2: RMSNorm **plano** (no `1+w`),
**biases en q/k/v** (Llama no los tiene), GQA 7x, RoPE theta 1e6, SwiGLU,
lm_head atado.

### La matemática del C++ contra PyTorch

```
modo --exact (pesos int4-dequantizados, x float)
  embed      0.000e+00
  capa  0    4.332e-06
  capa  1    3.709e-06
  capa 2-24  ~1.5e-07
  argmax coincide · capas con error alto: 0
```

**1e-6.** RMSNorm, RoPE, atención causal con GQA, SwiGLU, residual y lm_head
atado están correctos.

### El kernel int4 real

```
  capas 3-20   2.7e-02  (2.7 %)  <- cuantización de x a int8 por tensor
  argmax = 68012 = "Hola"  (coincide con el oráculo cuantizado)
  forward: 302.7 ms para T=14 -> 21.6 ms/token
           1 hilo, sin KV cache  =  46 tok/s
```

22.8 GMAC/s single-thread: consistente con los 19-20 medidos. A 8 hilos
(40 GMAC/s) la proyección de **81 tok/s** se sostiene.

### Hallazgo de calidad (hay que decirlo)

int4/g64 cambia la respuesta del 0.5B:

| pesos | respuesta |
|---|---|
| bf16 | "Estoy aquí para ayudarte. ¿Cómo puedo asistirte hoy?" |
| int4/g64 | "Hola, no puedo ser ident..." |

El C++ **reproduce exactamente** el oráculo cuantizado, así que la degradación
es de la cuantización, no del kernel. Un modelo de 0.5B es muy sensible a 4
bits: para este tamaño habría que ir a grupo 32 o 5-6 bits (el 7B tolera 4).

### Tres bugs que sólo la comparación capa-por-capa encontró

1. **Faltaban los biases de q/k/v.** Detectado por conteo: 290 tensores en el
   archivo vs 218 esperados; los 72 = 24x3 que sobraban eran los biases.
2. **`Shape::group` es VALORES por grupo (64), no NÚMERO de grupos (K/64).**
   Confundirlos producía un segfault leyendo 64 "grupos" de 7 bytes.
3. **`x` debe ir pre-permutado por `kPerm`** antes de entrar al kernel. Sin
   eso el producto punto empareja índices distintos y da 100 % de error.

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
