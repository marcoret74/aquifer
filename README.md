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

**Entró sin tocar una línea de motor.** Con su prompt canónico (BOS incluido)
respondió `El nombre de la capital de Francia es París.` y cierra con `</s>`.
La prosa es la de un 1.1B; el motor hizo su trabajo.

### La ley de ancho de banda

Poniendo los cuatro modelos en la misma tabla (1 secuencia, 4 hilos), lo
interesante no es el ranking sino la constante:

| modelo | artefacto | ms/paso | tok/s | **GB/s efectivos** |
|---|---|---|---|---|
| Qwen2.5-1.5B | 868,3 MB | 42,7 | 23,4 | 20,3 |
| R1-Distill-1.5B | 868,3 MB | 43,3 | 23,1 | 20,1 |
| TinyLlama-1.1B | 581,9 MB | 29,6 | **33,8** | 19,7 |
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
| TinyLlama-1.1B | 33,8 | **58,0** | 14,5 |
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

**Verificado contra `tokenizers`**, en un venv aparte (el motor C++ sigue con
cero dependencias; esto es solo banco de pruebas). `tokenizers` 0.23.3:

| modelo | prompt real (ids) | decode | 60 frases del README |
|---|---|---|---|
| Qwen2.5 | idénticos | idéntico | 60/60 |
| R1-Distill | idénticos | idéntico | 60/60 |
| TinyLlama | idénticos | idéntico | 60/60 |
| SmolLM2 | idénticos | idéntico | 59/60 |

Y el contraste encontró **dos bugs reales míos** que el roundtrip solo no veía:

1. **Patrón equivocado para Qwen.** `_pretok_spec` tomaba el `Sequence` como
   «usa GPT-2» y descartaba el `Split` propio del tokenizer. El `ByteLevel` de
   Qwen lleva `use_regex=false` justamente porque el corte ya lo hizo el `Split`.
   Con el patrón de GPT-2 se fusionaban mal los `\n\n`: **35 % de las frases
   coincidían antes, 100 % después.**
2. **Faltaba el `post_processor`.** TinyLlama declara un `TemplateProcessing`
   que antepone `<s>`; sin leerlo, el prompt salía sin BOS. Ahora el prompt es
   el canónico y la respuesta mejoró.

Queda una diferencia conocida: en SmolLM2, `\n\n` seguido de dígito se parte en
dos `Ċ` en vez de fusionarse (`(?!\S)` no backtracks igual en el crate `regex`
de Rust que en el módulo `regex` de Python). No afecta a ningún prompt usado.

## Día 17 — Contra llama.cpp: empate en tok/s, derrota por byte (2026-11-13)

Hasta hoy la consigna del proyecto era explícita: **no usar llama.cpp, ni como
baseline**. La pregunta por la comparación la levanta, así que se midió.

Sin `cmake` en la máquina, se usó el binario prebuilt `b12374`
(`llama-b12374-bin-ubuntu-x64`) y `llama-bench`, que mide generación pura
(`tg128`) igual que nuestro `--n 128`.

### Protocolo

1. Reboot, y esperar a que baje el load (a los 2 min seguía en 2,15).
2. Calentamiento descartado por motor y modelo.
3. **Bloques consecutivos** por motor (5 corridas), no intercalado: `caudal`
   hace `slurp` de 868 MB y **expulsa el GGUF de la caché de página**, así que
   intercalar penalizaba a llama.cpp por un artefacto del protocolo, no por el
   motor.
4. `taskset -c 0-3`, 4 hilos, 128 tokens, mejor de 5.

> ⚠️ **El ruido manda.** El mismo GGUF, mismo comando, dio **22,81 → 18,06 →
> 13,47 tok/s** en tres corridas seguidas. Con governor en `powersave` y 15 W no
> se puede resolver una diferencia de pocos puntos porcentuales. Los números de
> abajo son *mejor de 5*, no promedios.

### Resultados

| modelo | motor | artefacto | bits/peso | tok/s | **GB/s** | % del techo |
|---|---|---|---|---|---|---|
| Qwen2.5-1.5B | caudal | 868 MB | **4,50** | 21,1 | 18,3 | 72 % |
| Qwen2.5-1.5B | llama.cpp Q4_K_M | 1117 MB | 5,79 | 22,8 | 25,5 | 101 % |
| Qwen2.5-1.5B | llama.cpp Q4_0 | 1006 MB | 4,50 | 19,3 | 19,4 | 77 % |
| TinyLlama-1.1B | caudal | 582 MB | **4,50** | 30,9 | 18,0 | 71 % |
| TinyLlama-1.1B | llama.cpp Q4_K_M | 667 MB | 5,16 | 30,9 | 20,6 | 81 % |

El `Q4_0` con embedding a 4 bits se generó con `llama-quantize
--token-embedding-type q4_0 --output-tensor-type q4_0` para emparejar bits/peso.

### Lo que dice y lo que no

- **En tok/s hay empate.** TinyLlama: 30,9 contra 30,86. Qwen: 21,1 contra 22,8,
  pero con ±20 % de ruido, así que no es una diferencia.
- **Por byte, llama.cpp gana.** 77–101 % del techo contra nuestro 71–72 %. El
  kernel de `caudal` **deja entre 10 y 25 % de ancho de banda sin usar**.
- **Nuestra paridad en tok/s no es gratis**: viene de guardar menos bits. Qwen:
  4,50 contra 5,79; TinyLlama: 4,50 contra 5,16. Q4_K_M mantiene el embedding y
  la salida en ~6 bits y nosotros los bajamos a 4, y ese tensor es el `lm_head`
  atado — o sea, toca la calidad directamente. Nuestro ppl medido es 4,092
  (+22,7 % sobre bf16); **el de ellos no se midió**, así que no se puede afirmar
  «misma calidad, menos bytes».
- Con bits/peso emparejados (Q4_0, 4,50), llama.cpp da 19,3 y nosotros 21,1,
  otra vez dentro del ruido.

### Desglose del paso, y el techo que no era techo

Instrumentado el motor (`--time`), el reparto por token es inequívoco:

| modelo | GEMV | atención | ms/paso |
|---|---|---|---|
| Qwen2.5-1.5B | **97,2 %** | 2,8 % | 47,11 |
| TinyLlama-1.1B | **94,9 %** | 5,1 % | 32,65 |

No hay overhead que rascar: el motor entrega 18,4 GB/s y el kernel aislado
(`probe1_gemv`, 4 hilos) da **19,0 GB/s**. Estamos al 97 % de nuestro propio
kernel. La atención es marginal, así que el lote no ayuda por ahí.

**Retirada de una afirmación:** la columna «% del techo» de la tabla se apoyaba
en un techo de 25,3 GB/s que hoy **no se reproduce**. La misma máquina da 16,3
(`probe0`, 2 hilos), 19,0 (`probe1`, 4 hilos) y 25,3 (histórico). Encima
`probe0` reporta 900 MHz con 4 hilos mientras el motor corriendo mantiene
2,7–2,9 GHz: su columna de MHz es un artefacto. Con tres techos incompatibles no
se puede decir «72 % del techo»; lo defendible es: **~9 % por debajo de
llama.cpp en el mismo bloque, y al 97 % de nuestro propio kernel**.

Antes de perseguir «18 → 24 GB/s» hay que construir un banco de ancho de banda
confiable (buffers pre-tocados, muestreo de frecuencia, repeticiones).

### Intento de layout: intercalar sin relleno (negativo)

El intercalado perdió en el Día 13 por los huecos de relleno (stride alineado a
64 B). Se re-empaquetó con **stride exacto** (`--stride-align 0`), mismo tamaño
que `split` (868,3 MB, 4,500 bits/peso), y volvió a perder:

| layout | tok/s (mejor de 4) |
|---|---|
| split (dos arrays) | **19,4** |
| intercalado sin relleno | 18,0 |

**No era el relleno: es el prefetcher**, que sigue dos streams densos mejor que
uno intercalado aunque no tenga agujeros. El layout de dos arrays se queda.

## Día 18 — La cacería del denominador: mis probes mentían (2026-11-13)

Se consultó al panel (Claude, Gemini, Qwen) por la contradicción de los tres
techos. Consenso: el 25,3 GB/s histórico **no puede ser denominador**; hace
falta una lectura pura medida en la misma sesión, mismo buffer y mismo estado
térmico. Disidencia de Qwen, que lo daba por válido; su propio protocolo
(mediana, pasar PL2) implica lo mismo que los otros dos.

Se construyó `bench/probe5_denominator.cpp` con el protocolo acordado.

### Lo que se descartó, con datos

| hipótesis | prueba | resultado |
|---|---|---|
| THP no aplicaba | `maps_rollup` | **1,8 % → 100 %** al hacer `MADV_HUGEPAGE` *antes* de tocar las páginas. El número **no cambió**: no era THP |
| Ubicación física (Flex Mode 8+4) | mismo probe con `malloc` vs `mmap+THP` | 9,44 vs 9,12 GB/s → **no manda la ubicación** |
| Hilos en hermanos SMT | afinidad explícita con `pthread_setaffinity_np` a CPU 0..h-1 | sin cambio |
| Deriva de la máquina | probe y motor, 3 rondas intercaladas | probe **estable 9,0**; motor **estable 17,2**. No hay deriva: discrepan de verdad |
| El motor no lee lo que creo | contador de bytes dentro del GEMV | **868 MB/token confirmados** (el 983 MB aparente era el prefill dividido por los pasos de decode) |

### El experimento que lo cerró

Lo que pidió Gemini: el probe **dentro del mismo proceso y sobre el mismo
puntero**, justo después del GEMV.

```
READ in-process (uint64, 4 hilos, mismo buffer) :   5,66 GB/s
GEMV sobre ese mismo buffer, mismos 4 hilos     :  18,3  GB/s
```

**El probe es el que miente, por 3×.** El de `uint64_t` es ALU-bound: dos
acumuladores con dependencia en cadena, no llega ni a leer. Y la variante AVX2
con 8 acumuladores independientes se queda en 9-13 GB/s sin que lograra aislar
la causa.

### Conclusión

1. **Ninguno de mis probes sirve de denominador.** Un bucle ingenuo de lectura
   no mide ancho de banda: mide el bucle.
2. La única medida creíble que tengo es **el GEMV: 17–18 GB/s** sobre el mismo
   buffer, mismos hilos y mismo estado. El motor está al ~97 % de eso.
3. Por eso **se retira definitivamente la métrica «% del techo»**, y las
   comparaciones con llama.cpp quedan como **ratios dentro del mismo bloque**.
4. Para un denominador externo hace falta `STREAM` bien compilado o `MLC`/
   `likwid-bench`. No están instalados; instalarlos es el paso pendiente si se
   quiere volver a hablar de porcentajes.

Lo aprendido: el `MADV_HUGEPAGE` después de `MAP_POPULATE` no hace nada (1,8 %
de THP), y un probe de ancho de banda mal escrito produce números estables,
reproducibles **y falsos**. La reproducibilidad no valida una medición.

## Día 19 — Atribución con `perf`: el 16 % es el precio de `group=32` (2026-11-13)

Se habilitó `perf` (`kernel.perf_event_paranoid=1`) y se midieron los dos
motores con los mismos eventos, 4 hilos, 128 tokens:

| | caudal | llama.cpp |
|---|---|---|
| instrucciones | 132,9 G | **93,6 G** |
| ciclos | 83,1 G | 80,9 G |
| **IPC** | **1,60** | **1,16** |
| **instr/byte** | **1,05** | **0,65** |
| **ciclos/byte** | **0,656** | **0,566** |

Ejecutamos **60 % más instrucciones por byte**; el IPC mucho mejor (1,60 vs
1,16) absorbe dos tercios del daño y quedan **16 % más ciclos por byte**. Ese
es el hueco real frente a llama.cpp, y no es ni hardware ni sincronización.

### De dónde salen las instrucciones

Contando el bucle por cada 32 B de pesos (64 valores):

| bloque | instrucciones |
|---|---|
| expandir nibbles (`and`/`srli`/`unpacklo`/`unpackhi`) | 6 |
| producto (`maddubs`×2, `madd`×2, `add`) + loads de x | 7 |
| corrección `−8·Σx` (`setr_epi32`) | ~5 |
| **construir la escala del bloque** (`set_m128` + 2 `fp16→fp32` + 2 `mul` + `set1`) | **~8** |
| `cvtepi32` + `fmadd` | 2 |

Q4_K lee la escala **una vez por super-bloque de 256 valores** y aplica un
solo `float` al acumulado. Nosotros la aplicamos **cada 32 valores**, con
conversión fp16→fp32 incluida: 8 de 28 instrucciones por bloque.

**Conclusión:** el 16 % de ciclos por byte es el precio de `group=32`. Esa
granularidad se eligió por calidad (g64 costaba +31,7 % de ppl), así que ahora
el canje está medido en las dos direcciones: **seis puntos de ppl por ~16 % de
velocidad**.

### Optimización pendiente (sin cambiar la matemática)

Procesar **8 grupos (512 valores) por iteración**, acumular los 8 `int32` y
convertir las 8 escalas fp16 en una sola `_mm256_cvtph_ps`, aplicándolas de
golpe. Elimina la construcción por bloque y no toca la precisión.

### Hipótesis refutadas por medición (no por argumento)

| hipótesis | medida | veredicto |
|---|---|---|
| 113 barreras por token cuestan 2–4 ms | `pool.run` vacía = 0,55 µs → 62,5 µs/token | **0,13 %**. Falso |
| THP en los pesos | 0 % → 100 % THP, A/B intercalado | sin efecto |
| La división por elemento al cuantizar x | recíproco por grupo, A/B intercalado | sin efecto |
| Layout intercalado sin relleno | 18,0 vs 19,4 tok/s | peor |

## Día 20 — El bucle de 8 grupos: paridad de ciclos por byte (2026-11-13)

La optimización que salió del `perf` del Día 19: las 8 escalas de un tramo de
512 valores se convierten **fp16 → fp32 con un solo `_mm256_cvtph_ps`** y se
reparten con `_mm256_permutevar8x32_ps`, en vez de construir una escala por
bloque de 32 B (2 conversiones + 2 `mul` + 2 `set1` + `insert`).

### Verificación antes de medir

El kernel `gemv_int4_g32_M<M>` **no tenía test** (el roundtrip cubre el de grupo
64). Se agregó `tests/check_g32M.cpp`: compara contra la referencia escalar con
la misma data empaquetada, en G=16/48/64/256 con M=3. El kernel optimizado da
**error máximo 0,00e+00** — bit-exacto, no "parecido".

### Resultado (instrucciones de `perf`, deterministas)

| | instrucciones | ciclos | IPC | **ciclos/byte** |
|---|---|---|---|---|
| antes | 132,98 G | 75,67 G | 1,76 | 0,597 |
| **después** | **100,59 G** (−24,4 %) | **72,10 G** (−4,7 %) | 1,40 | **0,569** |
| llama.cpp Q4_K_M | 93,59 G | 80,89 G | 1,16 | **0,566** |

**Paridad de ciclos por byte alcanzada**: 0,569 contra 0,566, moviendo un 25 %
menos de bytes. El IPC baja de 1,76 a 1,40 porque se eliminó trabajo: quedamos
más cerca de ser puramente memory-bound, que es donde queremos estar.

En tok/s, A/B intercalado (mismo binario, macro `CAUDAL_LEGACY_SCALE`):

| ronda | nuevo | viejo |
|---|---|---|
| 1 | **23,0** | 20,1 |
| 2 | **19,1** | 17,3 |
| 3 | **22,5** | 20,7 |

**+8,7 % a +14,4 %**, siempre a favor. El texto sigue correcto en los dos
modelos (`La capital de Francia es París.`, `El nombre de la capital de Francia
es París.`) y las dos suites pasan.

La predicción de que esto cerraba en ~0,56 se cumplió. Lo que **no** se cumplió
es la premisa que la originó: no fueron "8 de 28 instrucciones parásitas" en el
sentido de un lane por grupo —el acumulador tiene 4 lanes por grupo y no mapea
a 8—, sino que se amortizó la *conversión* de las escalas.

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
