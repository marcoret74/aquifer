# Aquifer 🌊
**Zero-Copy HiperRAM Paging & 3D Morton Inference Engine in C++23**  
*Sub-100MB VmRSS @ 15W · Hardware-Software Co-Design for Silicon CPUs*

[![C++23](https://img.shields.io/badge/C%2B%2B-23-blue.svg)](https://en.cppreference.com/w/cpp/23)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Architecture: AVX2/BMI2](https://img.shields.io/badge/Architecture-x86__64%20%7C%20ARM64-orange.svg)]()
[![Memory: <100MB VmRSS](https://img.shields.io/badge/Memory-Sub--100MB%20VmRSS-purple.svg)]()

> *"Caudal corre a la velocidad de la luz; Aquifer es la reserva profunda que dosifica el flujo."*  
> **⚡ Powered by Semantic Geometry Chile** · Autor: Marco Retamales

---

## 🏛️ ¿Qué es Aquifer?

**Aquifer** es un motor de inferencia de modelos masivos de lenguaje (LLMs de 0.5B a 70B) escrito en **C++23 estricto sin dependencias externas**, diseñado para superar el **Muro de Von Neumann (Memory Wall)** en procesadores convencionales de 15 Watts (sin GPUs dedicadas ni centros de datos masivos).

A diferencia de los motores tradicionales (llama.cpp, Ollama, vLLM) que obligan a cargar varios gigabytes de pesos permanentemente en memoria física (RAM/VRAM), Aquifer demuestra que **la decodificación autorregresiva es un problema topológico y de física de silicio**.

### Los Pilares de Aquifer:
1. **El Portero (Zero-Copy Kernel Ring-Buffer):** Mediante `mmap(MAP_SHARED)`, el modelo se proyecta en el espacio virtual: la ventana activa son pocos MB y el resto vive en page cache. Mientras se computa la capa $L$, un hilo desacoplado (`AsyncPrefetcher`) notifica al kernel con `madvise(..., MADV_WILLNEED)` para precalentar la capa $L+1$. En modo `--mem-policy paged`, al terminar la capa $L$ el Portero ejecuta `madvise(..., MADV_DONTNEED)` y purga las páginas físicas; en `auto` las conserva si hay RAM libre (el kernel las reclama bajo presión). La memoria residente (**VmRSS**) queda en **< 90 MB en modo `paged` a ctx 512**; el KV cache crece linealmente con el contexto y `auto` puede retener el artefacto completo para evitar page faults.
2. **Indexación Espacial 3D (Curva de Morton & BMI2 `_pdep_u64`):** Proyecta la topología de pesos en el hipercubo $[-1, 1]^3$, mapeando coordenadas euclidianas a claves contiguas de 64 bits en **~48 nanosegundos** directamente en silicio. El reordenamiento espacial por Z-Order preserva la función matemática con **isomorfismo exacto** ($\Delta < 1.34 \times 10^{-9}$, similaridad coseno $= 1.0000000000$).
3. **Mecánica Óptica del Diamante 3D (Experimental):** En capas MLP intermedias, un cono de refracción óptica en $[-1, 1]^3$ selecciona las neuronas alineadas con el haz. El objetivo es saltar el 10 % de neuronas que concentra poca energía y acelerar la decodificación. La implementación actual (`--diamond N`) selecciona filas dispersas sin reordenarlas, así que **todavía no acelera** (ver nota de rendimiento); para que el cono sea un prefijo contiguo hay que permutar las neuronas por alineación al empaquetar.

---

## 📊 Benchmarks Medidos en Silicio Real (15 Watts CPU)

Pruebas empíricas reproducibles en procesadores de consumo alimentados por batería (Intel / AMD AVX2 + BMI2):

| Modelo | Parámetros | RAM Tradicional | RAM Aquifer (VmRSS) | Reducción RAM | Velocidad / Latencia | Modo |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Qwen 2.5-3B Instruct** | **3.09 B** | ~6.200 MB | **75.0 MB - 89.0 MB** | **-98.5 %** | **163.6 ms/tok (6.1 tok/s)** | 💎 Diamante 90% |
| **Qwen 2.5-3B Instruct** | **3.09 B** | ~6.200 MB | **75.0 MB** | **-98.8 %** | 608.9 ms/tok (1.6 tok/s) | Denso Líquido |
| **Qwen 2.5-0.5B Instruct**| **0.49 B** | ~340 MB | **38.0 MB** | **-88.8 %** | **~52.0 tok/s** | Denso Líquido |
| **Qwen 2.5-Coder-1.5B** | **1.54 B** | ~1.060 MB | **74.0 MB** | **-93.0 %** | **13.7 tok/s** | Denso Líquido |
| **SmolLM2-1.7B Instruct** | **1.71 B** | ~1.150 MB | **78.0 MB** | **-93.2 %** | **15.4 tok/s** | Denso Líquido |
| **TinyLlama-1.1B Chat** | **1.10 B** | ~850 MB | **45.0 MB** | **-94.7 %** | **22.5 tok/s** | SentencePiece |
| **Latencia Morton 3D** | — | — | — | — | **~54 ns/op** | Hardware BMI2 |

> La tabla superior son las mejores cifras registradas por el autor en distintas condiciones. La sección siguiente es la mediana reproducible de este equipo, con el mismo `--prompt`, `--threads 4` y `--time`.
>
> Los valores de VmRSS corresponden a `--ctx 512`. El KV cache crece con el contexto (fp32: ~72 KB/token en 3B; fp16: ~36 KB/token), así que a 2k tokens el KV aporta 75 MB adicionales en 3B incluso en modo `paged`.

### Perfil de latencia medido (i7-10510U · 4 núcleos / 8 hilos · 15 W · 4 hilos)

Medición reproducible con `--prompt` de 24 tokens, `--n` corto y `--time`. "Antes" es el motor previo a las optimizaciones de esta revisión; "después" usa `--mem-policy auto --prefill-batch 8 --kv-dtype fp16` y el GEMV con offsets correctos entre jobs.

| Modelo | Prefill 24 tok (antes → después) | Decode (antes → después) | VmRSS `paged` / `auto` |
| :--- | :---: | :---: | :---: |
| Qwen 2.5-0.5B | 610 ms → **108 ms** (5.6×) | 37.4 → **58.4 tok/s** (+56 %) | 35 MB / 312 MB |
| Qwen 2.5-1.5B | 1531 ms → **272 ms** (5.6×) | 15.3 → **22.6 tok/s** (+48 %) | 47 MB / 944 MB |
| Qwen 2.5-3B | 4425 ms → **557 ms** (7.9×) | 5.7 → **10.5 tok/s** (+84 %) | 59 MB / 1.86 GB |
| Qwen 2.5-0.5B @ ctx 2048 | 35.7 s → **16.1 s** | **43.4 tok/s** (24.5 % atención) | — |
| Morton 3D | — | **54.2 ns/op** (este equipo) | — |

**Qué cambió y por qué acelera:**

- **Corrección del offset entre jobs en el GEMV afín.** Las ramas `affine` y `kmin` hacían `continue` sin avanzar `off = hi`. El job siguiente (por ejemplo `up_proj`) arrancaba en offset 0 y **reprocesaba filas ya calculadas**: gate/up se leía 1.5× y dos hilos escribían las mismas filas de `up` (data race). Medido con contadores: 42.7M filas procesadas donde correspondían 32.0M. Corregirlo da el salto grande (~1.5× en MLP) y además balancea q/k/v. Los tokens generados son idénticos antes y después.
- **`--mem-policy auto` (nuevo default).** Si el artefacto cabe en `MemAvailable` con holgura, no se llama `MADV_DONTNEED`: los pesos quedan en page cache y THP, sin 150k page faults por pasada. Si no cabe, se pagina como antes. `paged` conserva el VmRSS mínimo; `resident` fuerza page cache caliente. El tráfico real de DRAM no cambia; cambia cuántas veces se vuelve a tocar la misma página.
- **`--prefill-batch 8`.** El prefill procesa hasta 8 tokens por pasada: cada byte de peso leído se reutiliza 8 veces. La causalidad se mantiene porque la atención de cada token corta en `pos[i]`, aunque el KV del bloque ya esté escrito. Solo se calculan los logits del último token del bloque.
- **Atención paralelizada por (token, cabeza).** Antes corría en un hilo mientras los workers esperaban; ahora usa el pool. A ctx 512 la atención baja de ~19 ms/token a ~4 ms/token (4 hilos).
- **Decodificación especulativa por prompt-lookup (`--spec 4`, default en chat).** Propone hasta 4 tokens copiados de una repetición reciente del contexto, los verifica en una sola pasada `M=4` y acepta el prefijo correcto. En greedy genera **exactamente los mismos tokens** que sin spec. Medido: +5–15 % en texto natural y **1,5–2,2×** en salida repetitiva/código. `--spec 0` lo desactiva. Tuning: `--spec-min 2 --spec-max 8` es el default; k=6/8 y min=3/4 no fueron robustamente mejores (k=8 ayuda en listas periódicas pero castiga prosa), y una propuesta de 1 token se descarta porque no ahorra nada. `--spec-min/--spec-max` quedan expuestos para experimentar.
- **KV en fp16 (default).** La mitad de memoria y de tráfico de KV; mismo texto generado en las pruebas de regresión. `--kv-dtype fp32` lo desactiva.
- **Hilos = núcleos físicos por defecto.** Usar los 8 hilos lógicos de un 4C/8T calienta y compite por los puertos de memoria: en este equipo el default anterior (8) llegó a 104 ms/token contra 26 ms/token con 4. `--threads 0` deja elegir al motor.

**Modo Diamante (triple check):** el aparente 12 % de mejora era el bug de offsets del GEMV, no el diamante. Con el denso corregido, en 3B: denso **138.7 ms/token**, Diamante 90 % **142.0 ms** (2.5 % más lento), Diamante 50 % **134.4 ms** (3 % más rápido). Además:

- A 90 % de neuronas activas el máximo teórico es **1.11×**; el 3.7× es físicamente imposible sin saltar al menos el 73 % de los pesos.
- La máscara cuesta 4-6 ms/token (`nth_element` + proyección 3D) y genera **1.000-5.500 runs** por proyección; cada run es una llamada al kernel y el GEMV se fragmenta.
- La selección por bloques contiguos Morton de 304 filas captura 76-79 % de la señal al 50 % de bloques, 88-89 % al 75 %, 94,3-94,6 % al 87,5 % y 97,0-97,4 % al 93,75 %. El techo de velocidad es 2×, 1,33×, 1,14× y 1,07× respectivamente; la selección dispersa al 50 % captura 70-78 %. No alcanza para justificar el riesgo de un reempaquetado completo.
- Para que el cono acelere de verdad hay que **permutar las neuronas por alineación/Morton al empaquetar** y aceptar un prefijo contiguo (o varios haces ancla); eso es un cambio de packer + evaluación de calidad, no un flag.

Hasta entonces, `--diamond` queda como experimental y no se recomienda para producción.

### Límites medidos (para no repetir el camino)

- **Atención GQA online (experimental):** se implementó con chunks de posición y reducción `(max, sum)`. En 0.5B a ctx 2048 dio **47 ms vs 35 ms** del kernel por cabeza (26,6 % → 31,2 % del decode) y prefill 33,3 s vs 22,7 s. En este CPU la atención ya es compute-bound y el K/V de GQA vive en L2/L3; compartir K/V no reduce los MAC. Queda en la rama `perf/gqa-attn`, no se usa.
- **Afín grupo 64 (4,5 bpw, experimental):** reduce 5,5–10 % los bytes, pero el kernel rinde **18,5 vs 20,9 GB/s** en 1.5B y la PPL sube **20,73 → 22,37 (+7,9 %)**; en 0.5B la PPL sube ~16 %. El balance neto en 1.5B es más lento. La capacidad de empaquetar/consumir grupo 64 quedó en la rama `perf/affine-g64` con `test_affine_g64`, pero **no es el default**.
- **Speculative con modelo draft (evaluado, no adoptado):** Qwen 0.5B como draft de 1.5B/3B, verificación batcheada con el protocolo `BEGIN/NEXT/DRAFT/VERIFY/COMMIT`. En 1.5B la aceptación fue 45–54 % y el resultado **10,8 tok/s vs 22,6** del target solo; en 3B la aceptación cayó a 25 % y también fue más lento. El draft de 0.5B cuesta ~17 ms/token y el target ya está en 43–95 ms/token: no hay margen. El protocolo y el benchmark quedan en la rama `exp/draft-spec` (`bench/spec_draft.py`), no en producción. El spec n-gram sigue siendo el que se usa.
- **Prefill B=16 (no adoptado):** a `--prefill-batch 8` el prefill ya es compute-bound (≈9,6 GB/s contra 20,9 GB/s de decode en 1.5B); M=16 requiere 32 acumuladores YMM (facc+minacc) y spillea. El límite práctico en AVX2 es 8.
- **Diamante:** ver el triple check arriba; al 90 % queda neutral o más lento, y el 3,7× no es alcanzable.

El camino que sí quedó en producción es el **fix de offsets del GEMV** + memoria adaptativa + prefill batcheado + atención paralela + KV fp16 + núcleos físicos.

### ¿Por qué claves Morton de 64 bits y no 128?

- 3 ejes × 21 bits = **63 bits**. Con 21 bits por eje hay 2.097.152 vóxeles por dimensión; el tensor más grande del repo (`INTER=11.008`) usa ~0.5 % de ese rango. 128 bits darían 42 bits/eje (~4.4 billones por eje), una resolución que no compra nada.
- `_pdep_u64` / `_pext_u64` de BMI2 son **de 64 bits**. Una clave de 128 bits exige dos `pdep` por eje, shifts/ors extra y **duplica el tamaño de las claves** en índices y comparaciones: más bytes y peor caché en un motor que ya está memory-bound.
- La arquitectura es **x86-64**, con AVX2 de **256 bits** para los kernels. Bajar a SIMD de 128 bits (SSE) reduce el ancho de banda de cómputo; subir a AVX-512 no es posible en este i7-10510U (Comet Lake) y en 15 W provocaría downclocking.
- La arena virtual reserva 1 TiB (40 bits de dirección); los 64 bits de dirección sobran. El “64 bits” de Aquifer es la **clave Morton**, no el ancho de la ISA.
- Si algún día hiciera falta más resolución, la salida correcta es una clave de dos niveles (64 bits calientes + offset de hoja), no `__int128` en el camino caliente.



---

## 🚀 Inicio Rápido (3 Comandos)

### 1. Compilación
Aquifer es un proyecto C++23 header-only con un binario de inferencia ultraligero:

```bash
# Clonar y compilar la suite de silicio
git clone https://github.com/marcoret74/aquifer.git
cd aquifer
make all       # binarios + tests
make test      # suite completa
```

### 2. Descargar un Modelo (On-Demand)
Los modelos se descargan y verifican automáticamente desde el catálogo:

```bash
# Listar modelos disponibles
python3 tools/pull.py --list

# Descargar Qwen 2.5-0.5B (ultrarrápido, ~330 MB)
python3 tools/pull.py qwen05

# O descargar Qwen 2.5-3B Instruct
python3 tools/pull.py qwen3
```

### 3. Shell Interactiva de Chat Streaming
Inicia una conversación interactiva en tiempo real:

```bash
# Iniciar chat con Qwen 0.5B (usa RAM libre como caché y prefill por bloques)
python3 tools/chat.py --model qwen05

# Decodificación especulativa por prompt-lookup (default: 4 tokens)
python3 tools/chat.py --model qwen05 --spec 4

# Forzar VmRSS mínimo (paga page faults, pero no retiene pesos)
python3 tools/chat.py --model qwen05 --mem-policy paged

# Qwen 3B: 4 hilos físicos, prefill batcheado y KV fp16
python3 tools/chat.py --model qwen3 --threads 4 --prefill-batch 8 --kv-dtype fp16
```

#### Comandos dentro de la Shell:
* `/models` o `/list` $\to$ Ver el catálogo de modelos instalados y disponibles.
* `/switch <modelo>` $\to$ Cambiar de modelo en caliente sin reiniciar el proceso (ej: `/switch coder`).
* `/reset` o `/clear` $\to$ Limpiar historial de conversación y resetear el KV-cache.
* `/exit` o `/quit` $\to$ Salir de la shell.

---

## 📖 Monografía Técnica y Tratado Científico

El tratado científico completo con demostraciones matemáticas, dictamen del panel de expertos y análisis de física del cómputo está disponible en:

📄 **[docs/GEOMETRIA_SEMANTICA_Y_MEMORIA_TOPOLOGICA.pdf](docs/GEOMETRIA_SEMANTICA_Y_MEMORIA_TOPOLOGICA.pdf)**

---

## 🏛️ Arquitectura del Ecosistema (Open Core)

**Open Core** describe el modelo de distribución, no una licencia distinta: **todo el código y la documentación de este repositorio son MIT**. Los componentes privados viven fuera de este repo y tienen su propia licencia.

* **Aquifer (Público · MIT):** Motor de paginado zero-copy en espacio de usuario, Ring-Buffer de kernel y shell interactiva en silicio.
* **Caudal Runtime (Público · MIT):** Los headers de `include/caudal/` (layout ABI, kernels GEMV int4) están en este repositorio y también son MIT.
* **Caudal Compiler / Calibrador (Privado · fuera de este repo):** Compilador tensorial de alta precisión, calibrador de proyección SVD y empaquetador afín. No se distribuye aquí.
* **Semantic Geometry Core (Privado · fuera de este repo):** Motor ontológico continuo en $[-1, 1]^3$, base de conocimiento formal y Merkle Ledger BLAKE3.
* **Pesos de modelos (Terceros):** Se descargan on-demand con `tools/pull.py`; conservan su licencia original (Apache-2.0, MIT, etc.) y **no** se redistribuyen en este repositorio.

---

## 📜 Licencia y Alcance

Este repositorio está liberado bajo la licencia **MIT** (ver [LICENSE](LICENSE)). La licencia cubre el código, los headers, los tests y la documentación aquí presentes.

No cubre:

* el compilador/calibrador de Caudal ni Semantic Geometry Core (no están en este repositorio);
* los pesos de modelos descargados desde el Hub (cada uno tiene su licencia);
* las marcas **Aquifer**, **Caudal** y **Semantic Geometry Chile** (la licencia no concede derechos de marca).

Se solicita conservar el aviso de copyright y la atribución:
`⚡ Powered by Semantic Geometry Chile · Autor: Marco Retamales`

Esa atribución es una solicitud de crédito, no una restricción adicional a los derechos que otorga MIT. El detalle de alcance está en [NOTICE](NOTICE).

Para consultas de arquitectura, licenciamiento corporativo o auditoría ontológica:  
**Semantic Geometry Chile** · Rancagua, Región de O'Higgins, Chile.
