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

## Día 14 — un segundo modelo entra sin tocar C++ (2026-11-13)

`deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B`, int4 g32, layout de dos arrays,
mismo packer y mismos kernels. **Cero líneas de C++ modificadas.**

### Lo único que hubo que cambiar fue una plantilla

R1 **no usa ChatML de Qwen**. Su plantilla es DeepSeek:

```
<｜begin▁of▁sentence｜><｜User｜>{pregunta}<｜Assistant｜><think〉\n
```

Con `<|im_start|>user\n…` el modelo **repetía el prompt y emitía
`<｜Assistant｜>` en bucle** — parece "atención rota", pero es un modelo que
nunca vio ese formato. Con la plantilla correcta razona en español, abre y
cierra `</think>`.

Los ids tampoco coinciden con Qwen2.5: aquí **151645 = `<｜Assistant｜>`**, no
`<|im_end|>`. Asumir los ids de un tokenizer hermano es un error de una línea
que se lee como un bug del motor.

### El oráculo descartó el kernel antes de tocar código

Con `tools/probe_r1_logits.py` (PyTorch, fp32, **sin cuantizar**) el token
siguiente a "La capital de Francia es" es `' '` + `' Franc'` — **el mismo que
produce el C++ en int4**. Si el kernel leyera mal los pesos, no coincidirían.

De paso: `oracle_qwen.Qwen.forward` calculaba los logits con `embed_tokens`
atado. R1 declara `tie_word_embeddings=False` y trae `lm_head` propio, así que
el oráculo estaba mal para este modelo (el C++ ya lo resuelve por `index.txt`).

### Rendimiento (mismo binario, mismas 4 hilos, 15 W)

| | 1 secuencia | 4 secuencias |
|---|---|---|
| R1-Distill-1.5B | 23,3 tok/s | **39,9–43,1** tok/s |
| Qwen2.5-1.5B | 22,6 tok/s | 28,6–37,5 tok/s |

Batching ≈ ×1,8–1,9 el agregado; ~10 tok/s por secuencia en lote.

> ⚠️ **La comparación entre modelos no está limpia.** Governor en `powersave`
> y deriva térmica: en la misma sesión Qwen2.5 cayó de 37,5 a 28,6 (−24 %) y
> R1 de 43,1 a 39,9 (−7 %). La diferencia de ~12 tok/s es del orden de la
> deriva, **no** un resultado. Lo válido es el par intra-corrida (1 vs 4).

### Los límites del festejo

R1-Distill-Qwen **es la misma arquitectura** que Qwen2.5
(`Qwen2ForCausalLM`, 1536/28/12/2/8960). Lo demostrado es que un segundo juego
de pesos con diferencias reales (`lm_head` no atado, `rope_theta` 1e4 en vez de
1e6) entra sin tocar el motor. Lo **no** demostrado es generalidad de
arquitectura: eso pide Llama-3.2-1B (sin bias en q/k/v, otro `head_dim`) o un
MoE. Anotado, no corrido.

Y la calidad: la traza de "7 × 8" se va por las ramas. El motor está firme; la
prosa de un 1,5 B en int4, todavía no.

## Día 15 — SmolLM2 destapó un bug de cuantización que Qwen tapaba (2026-11-13)

Premisa de partida: *"SmolLM2 es otra arquitectura (MHA, `NKV == NH`), entra sin
tocar código"*. **Falso.** Entró, corrió, y escupió basura.

### Síntoma

Primer token `7359` (`' controvers'`) **repetido 48 veces**, e idéntico para
prompts distintos (ids `1`, `100`, `1000`, `15319`, `40000`). Es decir: la
salida no dependía de la entrada.

### La bisección (antes de tocar una línea de kernel)

Tres instrumentos nuevos:

| herramienta | para qué |
|---|---|
| `--dump` en `generate_qwen` | normas de `h_in, q, k, v, o_attn, o_proj, gate, up, mlp_out` y `|x|` por capa |
| `tools/probe_logits.py` | oráculo fp32 + `--quant` (int4 al vuelo) + `--xq` (int8 en x) + `--norms` |
| `tools/probe_layer0.py` | recomputa la capa 0 en numpy **leyendo el artefacto empacado** |
| `tools/check_pack.py` | dequantiza el artefacto y lo compara con el safetensors |

Con la capa 0 sola:

| cantidad | C++ (antes) | numpy sobre el artefacto | |
|---|---|---|---|
| `h_in` | 11,08525 | 11,08525 | idéntico |
| `mlp_out` | **51,30** | **79,62** | **−36 %** |

`h_in` exacto descarta embedding, `rmsnorm` y los pesos de norma. La pérdida
aparecía en cada GEMV y **acumulaba** con la profundidad: 24 capas después, el
modelo elegía un token fijo.

### La causa: una escala por token

`caudal` cuantizaba las activaciones a int8 con **una sola escala por token**
(`amax/127`). Replicando esa aritmética en el oráculo sale **exactamente** el
token del C++, con las mismas tres opciones en el mismo orden:

| numerica | primer token | p |
|---|---|---|
| fp32 (referencia) | `La` | 0,783 |
| int4 g32, x sin tocar | `La` | 0,679 |
| + int8 en x, **1 escala por token** | `controvers` | 0,544 |
| + int8 en x, 1 escala por grupo de 32 | `La` | 0,687 |
| + int8 en x, 1 escala por grupo de 64 | `La` | 0,668 |

Un *outlier* en `x` fija la escala de toda la fila y aplasta el resto. SmolLM2
tiene outliers de activación mucho más agresivos que Qwen2.5, y por eso el
esquema llevaba semanas tapado: **el modelo benévolo hacía de testigo mudo.**

### El fix

`xscale` pasa de un escalar por token a `const float*` con **una escala por
grupo** (`M*G`), en `gemv_ref`, `gemv_int4`, `gemv_int4_g32` y
`gemv_int4_g32_M<M>`. La referencia escalar ya multiplicaba por grupo
(`sc[g] * xscale`), así que el kernel solo cambia de dónde lee.

De paso se cerró un agujero: la permutación `kPerm` se aplica en bloques de 64 y
el resto (`K % 64`) **quedaba sin escribir**.

Verificación en la capa 0: `mlp_out` 51,30 → **80,30** (numpy: 79,62), y
SmolLM2 responde `La capital de Francia es Paris.`

### Coste: cero medible

Mejor de 3 corridas, 1 secuencia, 4 hilos:

| modelo | antes | después |
|---|---|---|
| Qwen2.5-1.5B | 23,4 | **23,7** |
| R1-Distill-1.5B | 23,3 | **23,1** |

### Números de SmolLM2-1.7B

1 secuencia **19,1–19,8 tok/s**; 4 secuencias **19,9 tok/s agregados**
(5,0 por secuencia). Amortiza mucho peor que Qwen (×1,8): con `NKV == NH = 32`
el anillo de KV mueve 8× más bytes por token y posición.

### Lo que enseña

- **Un modelo que anda no valida el esquema de cuantización**; solo lo oculta.
- El motor era **numéricamente fiel a su especificación**. La especificación
  estaba mal, y eso solo se ve cuando el modelo tiene outliers.
- La regla de siempre: bisecar con números intermedios, no leer código a ojo.
  El `h_in` idéntico fue lo que puso la lupa dentro de la capa.

## Día 16 — TinyLlama: el cuarto modelo y la ley que los ordena a todos (2026-11-13)

TinyLlama-1.1B-Chat aporta combinaciones que ninguno de los tres anteriores
tenía: **GQA con ratio 8** (`NH=32`, `NKV=4` — Qwen usaba 6, SmolLM2 1),
`lm_head` propio en arquitectura Llama, `DH=64`, vocab 32000, `rope_theta` 1e4.

```
H 2048 / NLY 22 / NH 32 / NKV 4 / DH 64 / INTER 5632 / V 32000
581,9 MB empaquetado · 155 lineales · 45 aux · 4,500 bits/param
```

**Entró sin tocar una línea de motor.** Respondió
`El país de Francia es la capital de Francia, y su nombre es París.` y cierra
con `</s>`. La prosa es la de un 1.1B; el motor hizo su trabajo.

### La ley de ancho de banda

Poniendo los cuatro modelos en la misma tabla (1 secuencia, 4 hilos), lo
interesante no es el ranking sino la constante:

| modelo | artefacto | ms/paso | tok/s | **GB/s efectivos** |
|---|---|---|---|---|
| Qwen2.5-1.5B | 868,3 MB | 42,7 | 23,4 | 20,3 |
| R1-Distill-1.5B | 868,3 MB | 43,3 | 23,1 | 20,1 |
| TinyLlama-1.1B | 581,9 MB | 30,1 | **33,3** | 19,3 |
| SmolLM2-1.7B | 962,6 MB | 50,5 | 19,8 | 19,1 |

**19,1 – 20,3 GB/s en los cuatro**, o sea 76–80 % del techo medido de
25,3 GB/s. No es coincidencia: es el mismo motor leyendo pesos a la misma
velocidad. Que TinyLlama sea el más rápido **no** es que sea mejor modelo, es
que pesa 33 % menos por token.

La métrica honesta para comparar modelos en `caudal` no es tok/s: es
**GB/s efectivos**. tok/s se deduce del tamaño del artefacto.

### Batching

| modelo | 1 seq | 4 seq (agregado) | por secuencia |
|---|---|---|---|
| TinyLlama-1.1B | 31,4–33,3 | **58,2** | 14,6 |
| Qwen2.5-1.5B | 23,7 | 43,1 | 10,8 |
| R1-Distill-1.5B | 23,1 | 39,9 | 10,0 |
| SmolLM2-1.7B | 19,1–19,8 | 19,9 | 5,0 |

TinyLlama es también el que mejor amortiza en lote (×1,85). SmolLM2 sigue
siendo el peor (×1,04) por el anillo de KV de `NKV == NH = 32`.

> En lote la cuenta cambia de naturaleza: los pesos se leen **una vez por paso**
> y salen M tokens. TinyLlama a M=4 mueve 581,9 MB en 68,69 ms = **8,5 GB/s**:
> ya no es memory-bound, es compute-bound. Es el mismo cambio de régimen que
> documentó `probe4_batch`.

### Tokenizer: tercera familia

TinyLlama no usa byte-level sino **SentencePiece** (`pre_tokenizer: null`,
normalizador que antepone `U+2581` y cambia espacios por `U+2581`, decodificador
con `ByteFallback`). El tokenizer casero aprendió esa rama: BPE sobre el texto
normalizado, `<0xXX>` para lo que no está en el vocabulario, y `U+2581` → espacio
al decodificar. Ahora cubre Qwen2, SmolLM2 y Llama-2/Mistral.

**Límite declarado:** al separar por tokens especiales, `_encode_plain` antepone
un `U+2581` a cada segmento, así que tras `</s>` queda un `▁` que HF quizá no
pondría. Los ids pueden diferir de los de HF en ese detalle. La prueba del motor
sigue siendo válida porque **C++ y oráculo usan exactamente los mismos ids**; lo
que no se puede afirmar es que la tokenización sea bit-idéntica a la de `transformers`.

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
