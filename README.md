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

## Día 6 — KV cache en anillo + multihilo (2026-11-13)

`tools/generate_qwen.cpp`. Tres piezas nuevas, sobre el forward ya validado a 1e-6.

### El anillo de A1, por fin en producción

El KV de cada capa vive en un `memfd` mapeado **dos veces consecutivas** (96 MB
para 4096 tokens × 24 capas). Escribir en `base + t*stride` y leer la ventana
`[0, n)` es siempre contiguo: **ninguna rama de wraparound** en el bucle de
atención. Es exactamente la propiedad que `test_doublemap` validó.

### Multihilo: la mitad del problema era el contador

| hilos | ms/token | tok/s |
|---|---|---|
| 1 | 34,5 | 29,0 |
| 2 | 20,9 | 47,8 |
| **4** | **18,6** | **53,8** |
| 6 | 19,7 | 50,8 |
| 8 | 22,2 | 45,0 |

Dos hallazgos, ambos contraintuitivos:

1. **`std::barrier` (futex) es demasiado caro aquí.** Se invoca 168 veces por
   token; a ~10 µs por encuentro son milisegundos. Reemplazado por un
   **spin-barrier**.
2. **El hilo principal debe trabajar.** Con T workers + 1 principal ocioso
   (9 hilos en 8 núcleos) el spin-barrier **se estrangula**: 1106 ms/token.
   El principal entra como worker 0 y quedan exactamente T hilos.
3. **Más hilos no es mejor**: 8 rinde peor que 4 por el límite de potencia del
   chip (mismo patrón que `probe0`/`probe2`). El óptimo acá es **4**.

### Resultado

```
prefill  : 226.7 ms (13 tokens)
1er token: 17.5 ms
decode   : 19.11 ms/token  ->  52.3 tok/s
```

> "Hola! Como soy una inteligencia artificial, no tengo un "mejor" o "mejor"
> como ser humano. Soy una herramienta diseñada para ayudar a los usuarios a
> entender, responder y comunicarse. ¿Cómo puedo ayudarte hoy?"

Párrafo completo, coherente, y termina en `<|im_end|>` — el stop token correcto.
**29 → 53,8 tok/s** (1,86× sobre un hilo).

### Lo que el 81 tok/s no era

La proyección de 81 tok/s venía de `probe2` (39,9 GMAC/s a 8 hilos, una sola
matriz grande). Este workload llega a 26,6 GMAC/s porque: (a) 169 GEMV chicos
en vez de uno grande, (b) throttling de potencia, (c) secciones seriales
(RMSNorm, atención, SiLU). **El número honesto es 52-54 tok/s.**

## Día 7 — grupo 32: medido, no opinado (2026-11-13)

Dos muestras de texto no son una medición. Perplejidad con teacher forcing sobre
88 tokens de prosa en español:

| config | bits/param | ppl | Δ vs bf16 | tok/s |
|---|---|---|---|---|
| bf16 | 16 | 4,487 | — | — |
| int4 **g64** | 4,25 | 5,908 | **+31,7 %** | 52,8 |
| int4 **g32** | 4,50 | **5,073** | **+13,1 %** | 49,3 |
| int4 g16 | 4,75 | 4,988 | +11,2 % | — |

**g32 recupera el 59 % del daño de cuantización por un 6,6 % de throughput.**
g16 añade solo 0,6 puntos de bits para ganar 1,9 puntos de ppl: no paga.

Corrige lo que yo había dicho al ver el texto: con dos párrafos parecía que "la
diferencia es capacidad del modelo, no cuantización". Falso — el 0.5B pierde 32 %
de perplejidad a g64, y eso sí se arregla con el grupo.

### Dos bugs que sólo aparecieron al cambiar el grupo

1. **`L.G = K/64` estaba hardcodeado** en los cargadores. Con grupo 32 son el
   doble de grupos, y todos los índices de escala salían corridos → texto basura.
   Ahora se deriva de `scale_bytes / (2·N)`.
2. **El bucle de permutación iteraba por grupo de escala** (28 × 64 = 2K) en vez
   de por bloque de 64 valores → **corrupción de heap**. La permutación es del
   layout, no de la cuantización; son ejes independientes.

### El kernel g32 no cuesta un pliegue extra

Con `kPerm`, los dos grupos caen limpios en las mitades del acumulador:
lane 0-3 = grupo 0, lane 4-7 = grupo 1. Aplicar dos escalas distintas cuesta
**un solo `set_m128`**, no un pliegue por grupo. Por eso g32 pierde solo el 6,6 %
(que es puramente los bytes extra), no el doble.

## Día 8 — Qwen2.5-1.5B en silicio (2026-11-13)

| | 0.5B | **1.5B** |
|---|---|---|
| params | 494 M | **1.544 G** |
| artefacto g32 | 278 MB | **868 MB** (771,8 + 96,5) |
| bits/param | 4,50 | 4,50 |
| err. cuantización | 10,18 % | 10,18 % medio · 12,81 % máx |
| **ppl bf16** | 4,487 | **3,335** |
| ppl g64 | +31,7 % | **+25,0 %** |
| ppl g32 | +13,1 % | **+22,7 %** |
| tok/s (4 hilos) | 54,3 | **17,5** |
| prefill 13 tok | 222 ms | 675 ms |
| 1er token | 17 ms | 54 ms |

```
"Soy un asistente de inteligencia artificial creado por Anthropic para ser
útil, confiable y inofensivo. Estoy diseñado para ayudarte con información
general, responder preguntas y realizar tareas simples, pero no tengo una
identidad personal real. Estoy aquí para ayud..."
```

### Tres correcciones que debo hacer

**1. "Un modelo más grande toleraría peor la compresión" — falso.** La 1.5B
tolerancia es *mejor*: +25,0 % vs +31,7 % a g64. Los modelos grandes son más
robustos a la cuantización, como dice la literatura.

**2. g32 rinde mucho menos en la 1.5B.** De 31,7→13,1 en la 0.5B, pero de
25,0→22,7 en la 1.5B. Para la 1.5B, **g64 puede ser mejor trato**: 1,9 puntos de
ppl por 6 % de throughput. El óptimo depende del tamaño del modelo.

**3. Se desbordó la memoria.** `ppl_qwen` cacheaba los 1,5B en bf16 (3 GB) más
las copias float32. En un forward completo cada peso se lee **una sola vez**, así
que el caché no ahorraba I/O y costaba 3 GB. Desactivado: pico ahora 3,1 GB.

### Dos bugs de la descarga/empaquetado

- **El archivo llegó con 41,7 % de ceros.** El descargador paralelo hizo
  `truncate` al tamaño final y se cortó por timeout: el tamaño mentía. Detectado
  escaneando chunks; se rebajaron 1,297 GB por rangos a 10,9 MB/s.
- **fp16 no representa 1e-8.** `sw[sw<1e-8]=1e-8` (clamp en fp32) se convierte
  en `0` al castear a fp16 → división por cero → `ppl = nan`. Hay 3135 grupos con
  `amax=0` (canales muertos en `down_proj`). El clamp va en dominio fp16 y en el
  mínimo normal (`2^-14`).

## Día 9 — stencil 3D: la hipótesis se confirma en líneas y se refuta en tiempo (2026-11-13)

`bench/stencil_3d.cpp` · volumen 256³ (16,8 MB, 1 B/voxel) · stencil 3×3×3 ·
tres layouts, verificados equivalentes (mismo campo, mismo resultado).

| layout | líneas/stencil | B útiles | ns/voxel | vs ROW |
|---|---|---|---|---|
| **ROW** | 9,21 | 4,6 % | **5,98** | **1,00×** |
| MORTON | 3,36 | 12,6 % | 10,12 | 0,59× |
| BRICK 16³ | 3,36 | 12,6 % | 21,37 | 0,28× |
| ROW + tabla offsets | — | — | 10,99 | — |
| MORTON + tabla offsets | — | — | 12,90 | — |

### Lo que se confirmó

**El conteo de líneas es exacto.** Predije 1,5³ = 3,375 y salió 3,36. Y
9,21 → 3,36 es un factor **2,74×** menos tráfico exigido. La geometría de
Morton hace lo que dice.

### Lo que se refutó

**El layout con 2,74× menos líneas es 2,2× MÁS LENTO en reloj de pared.**

La razón es que *"líneas por stencil" mide el requisito instantáneo, no la tasa
de fallos en régimen*. Barriendo el volumen, el acceso de row-major es un
stream con stride que **el prefetcher de hardware predice perfectamente**; los
3,36 de Morton son dispersos y lo derrotan. El prefetcher vale más que las
líneas.

### Lo que se refutó de mi propia propuesta

**Bricking 16³ es el peor de los tres (21,37 ns, 3,6× más lento).** Resuelve el
problema de MMU (ladrillo = página de 4 KiB) y crea un problema de cómputo:
añade un nivel de indirección sobre Morton. Era mi sugerencia; medida, no paga.

### Corrección del diagnóstico (el consejo tenía razón)

Yo dije que el prefetcher explicaba la derrota de Morton. **Falso.** Claude hizo
la cuenta que yo no hice:

```
3 planos de 256² = 197 KB  ->  caben en L2 (256 KB por núcleo)
DRAM por pasada  = 33,6 MB -> 1,6 ms a 21 GB/s
medido ROW       = 98 ms   ->  DRAM es el 1,6 % del tiempo
```

**El stencil no es un problema de memoria: es un problema de instrucciones.**
Y la prueba definitiva: reescribí row-major sin aritmética de dirección
(9 punteros de fila) y bajó a **0,53 ns — 11,5× más rápido**, verificado
correcto contra el genérico. El binario tiene 78 instrucciones vectoriales:
**el compilador vectorizó, y ése es todo el 11,5×.**

Los layouts basados en mapa (Morton, brick, tiled) **no se pueden vectorizar**:
el índice es opaco. Así que todas las mediciones anteriores medían generación
de direcciones, no el sistema de memoria.

### Resultado completo (256³, 1 B/voxel)

| layout | líneas/stencil | ns/voxel | vs ROW |
|---|---|---|---|
| ROW (mapa genérico) | 9,21 | 6,11 | 1,00× |
| MORTON | 3,36 | 10,94 | 0,56× |
| BRICK 16³ | 3,36 | 22,02 | 0,28× |
| TILED 16³ row-major | 5,02 | 15,67 | 0,39× |
| **ROW sin aritmética (ROWPTR)** | — | **0,53** | **11,51×** |

**TILED era la predicción falsable de kimi (esperaba ≤6,5 ns). Dio 15,67. Falló.**
Y confirma lo que dijo qwen: *no existe un layout con 3,36 líneas Y velocidad de
row-major en un barrido denso.*

### La lección de método

> **No podés preguntar por el layout hasta que el kernel esté vectorizado.**
> A 0,53 ns/voxel no hay presión de memoria que aliviar: el orden de los datos
> es irrelevante porque el cuello está en otro lado.

Tres hipótesis mías murieron en este día: el prefetcher como causa, el bricking
como solución, y el tiled como variante ganadora.



Esta medición y `bench_fase0` no se contradicen: miden **patrones distintos**.

| patrón | gana | evidencia |
|---|---|---|
| barrido secuencial (stencil) | **row-major** | 5,98 vs 10,12 ns/voxel |
| consulta espacial aleatoria (k-NN) | **Morton** | −28,7 % líneas |

En una consulta aleatoria no hay prefetcher que ayude, y ahí el orden espacial
cobra. En un barrido, el prefetcher ya hace el trabajo.

**Conclusión: no construyas un sustrato de memoria espacial para barridos. Si lo
construyes, es para pointer-chasing, no para stencils.**

## Día 10 — fusión de GEMVs y SiLU paralelo (2026-11-13)

Dos cambios en `tools/generate_qwen.cpp`:

1. **GEMV fusionado.** Q/K/V comparten `h`; Gate/Up comparten `h2`. Ahora un
   solo `pool.run` por grupo, con **`x` cuantizado y permutado UNA vez** (antes
   una por tensor) y reparto de hilos sobre el total de filas de todos los jobs
   juntos — así `k_proj` (256 filas) no deja hilos ociosos.
   197 llamadas → 113.
2. **SiLU·Up paralelo.** Los 250k `exp()` por token salen del hilo principal.

### Resultado

| | antes | después |
|---|---|---|
| decode | 57,2 ms/token | **48,2 ms/token** |
| tok/s | 17,5 | **20,7 – 21,0** |
| GB/s efectivos | 15,2 | **18,0** |
| % del techo (24,4) | 62 % | **74 %** |

**+18 %**, y el texto es **idéntico token por token** al de antes (60/60), así
que la fusión no alteró la aritmética.

Óptimo sigue en **4 hilos** (3→19,2 · 4→20,0-21,0 · 5→15,8 · 6→17,4 · 8→18,7):
el presupuesto de 15 W pesa más que el paralelismo a partir de 5.

Yo había proyectado **~23 tok/s**. Se quedó en 21. La diferencia estaba en
sobreestimar cuánto del tiempo era atribuible a las partes seriales: los
`exp()` y las barreras eran reales pero chicos.

### Corrección: el lever de RAM estaba muerto

`dmidecode` mostró **dual-channel asimétrico**: 8 GB en ChannelA + 4 GB en
ChannelB, ambos DDR4-2400. Es **Flex Mode**: los primeros 8 GB se entrelazan
entre canales y los 4 restantes quedan en A. Con un working set de 868 MB ya
corríamos en la zona dual.

Lo empírico ya lo decía: 24,4 GB/s medidos > 19,2 GB/s del teórico
single-channel. **Un segundo SODIMM no habría duplicado nada.** Yo lo recomendé
dos veces; el número lo refutaba desde `probe2`.

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
