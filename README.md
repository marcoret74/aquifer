# Aquifer 🌊
**Geometric Tensor Paging Runtime & Hypercube Gatekeeper**

> *"Caudal corre a la velocidad de la luz; Aquifer es la reserva profunda que dosifica el flujo."*

---

## 🏛️ ¿Qué es Aquifer?

**Aquifer** es un runtime de memoria virtual geométrica en **C++23** que permite ejecutar modelos de lenguaje masivos (**LLMs de 0.5B a 70B**) en hardware modesto de consumo (laptops y equipos con 8 a 16 GB de RAM), rompiendo la barrera de tener que cargar todos los pesos de golpe en memoria física.

### El Principio de HiperRAM Cúbica
En un modelo Transformer autorregresivo, la CPU ejecuta una sola capa a la vez en estricta secuencia temporal:
- De un modelo de **964 MB (1.5B)**, cada capa pesa solo **~29 MB**.
- De un modelo de **4.5 GB (7B)**, cada capa pesa solo **~150 MB**.
- De un modelo de **35 GB (70B)**, cada capa pesa solo **~1.2 GB**.

**Aquifer** sustituye la carga estática de todo el modelo por dos mecanismos fundamentales:
1. **El Portero (Zero-Copy Kernel Pager & Ring-Buffer):** Mediante `mmap(MAP_SHARED)`, el modelo reside en disco y se proyecta en el espacio virtual sin consumir un solo byte de RAM física al inicio (4 MB VmRSS). Durante la inferencia:
   - Antes de iniciar la Capa $L$, El Portero notifica al kernel con `madvise(..., MADV_WILLNEED)` para precargar en background la Capa $L+1$.
   - Apenas la Capa $L$ termina su cómputo GEMV AVX2, El Portero notifica `madvise(..., MADV_DONTNEED)`, expulsando de inmediato las páginas físicas de la RAM.
2. **Indexación Espacial Cúbica (Z-Order / Morton 3D):** Proyecta la topología de pesos en el hipercubo semántico $[-1, 1]^3$, mapeando cada vóxel de coordenadas a una clave entera de 64 bits (`uint64_t`) en ~48 nanosegundos mediante instrucciones BMI2 del silicio (`_pdep_u64` / `_pext_u64`).

---

## 📊 Resultados Medidos en Silicio Real

Pruebas ejecutadas en Intel Core Comet Lake (4 núcleos / 8 hilos, AVX2 + BMI2):

| Modelo / Métrica | Enfoque Tradicional (Caudal / llama.cpp) | Aquifer (Zero-Copy Paging) | Reducción de RAM | Velocidad |
| :--- | :---: | :---: | :---: | :---: |
| **VmRSS Inicial (Pesos en disco)** | 964.0 MB | **4.0 MB** | **99.6 %** | Instantáneo |
| **Qwen 2.5-1.5B Instruct** | 1,060.0 MB | **73.0 MB** | **93.1 %** | 11.0 tok/s |
| **Qwen 2.5-0.5B Instruct** | 340.0 MB | **38.0 MB** | **88.8 %** | 39.6 tok/s |
| **Qwen 2.5-Coder-1.5B** | 1,060.0 MB | **74.0 MB** | **93.0 %** | 13.7 tok/s |
| **DeepSeek-R1-Distill-1.5B** | 1,060.0 MB | **75.0 MB** | **92.9 %** | 11.9 tok/s |
| **Latencia Morton 3D (Silicio)** | — | **48.5 ns/op** | — | $O(1)$ Hardware |

> **Nota:** La RAM reportada (**73 MB**) incluye las activaciones temporales, embeddings en bf16 y el KV-cache completo. Los pesos de las capas inactivas **no residen en memoria física**.

---

## 🛠️ Compilación y Pruebas

Requisitos: Linux, C++23, `g++` con soporte AVX2 y BMI2.

```bash
# Compilar y correr toda la suite de tests
make test

# Salida esperada:
# === Test Aquifer: Morton 3D Coordinate Indexer ===
#   Roundtrip de 100.000 puntos en el Cubo OK (48.5 ns/op)
# === Test Aquifer: Portero de HiperRAM (Ring-Buffer Pager) ===
#   RAM física: 8 MB (vs 112 MB) -> Ahorro 92.86 %
# === Test Aquifer: Zero-Copy Kernel Paging (MADV_DONTNEED) ===
#   VmRSS tras Portero MADV_DONTNEED: 3 MB
```

---

## 💬 Shell Interactiva con Paging

Aquifer incluye una shell de chat streaming conectada directamente al motor de paginado:

```bash
# Iniciar chat con Qwen 2.5-1.5B
python3 tools/chat.py --model qwen

# Iniciar chat con Qwen 2.5-0.5B (ultra-rápido)
python3 tools/chat.py --model qwen05

# Iniciar chat con Qwen 2.5-Coder-1.5B (asistente de código C++)
python3 tools/chat.py --model coder

# Iniciar chat con DeepSeek-R1-Distill-1.5B (con cadena de pensamiento <think>)
python3 tools/chat.py --model r1
```
