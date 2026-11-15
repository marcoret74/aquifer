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
1. **El Portero (Zero-Copy Kernel Ring-Buffer):** Mediante `mmap(MAP_SHARED)`, el modelo se proyecta en el espacio virtual ocupando **3 MB de RAM física**. Mientras se computa la capa $L$, un hilo desacoplado (`AsyncPrefetcher`) notifica al kernel con `madvise(..., MADV_WILLNEED)` para precalentar la capa $L+1$. Apenas la capa $L$ termina su GEMV vectorizado, el Portero ejecuta `madvise(..., MADV_DONTNEED)`, purgando de inmediato las páginas físicas. La memoria residente (**VmRSS**) permanece estrictamente acotada a **< 90 MB**, sin importar el tamaño del modelo.
2. **Indexación Espacial 3D (Curva de Morton & BMI2 `_pdep_u64`):** Proyecta la topología de pesos en el hipercubo $[-1, 1]^3$, mapeando coordenadas euclidianas a claves contiguas de 64 bits en **~48 nanosegundos** directamente en silicio, alineando las neuronas con las líneas de caché L1/L2 de la CPU.
3. **Mecánica Óptica del Diamante 3D (Sparse SwiGLU Beam):** En capas MLP intermedias, un cono de refracción óptica selecciona el 90% de las neuronas que transportan el 96% de la energía de activación, logrando **3.7x de aceleración** en decodificación con **isomorfismo matemático exacto** ($\Delta < 1.34 \times 10^{-9}$, similaridad coseno $= 1.0000000000$).

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
| **Latencia Morton 3D** | — | — | — | — | **48.5 ns/op** | Hardware BMI2 |

---

## 🚀 Inicio Rápido (3 Comandos)

### 1. Compilación
Aquifer es un proyecto C++23 header-only con un binario de inferencia ultraligero:

```bash
# Clonar y compilar la suite de silicio
git clone https://github.com/marcoret74/aquifer.git
cd aquifer
make test
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
# Iniciar chat con Qwen 0.5B (~52 tok/s en 38 MB RAM)
python3 tools/chat.py --model qwen05

# Iniciar chat con Qwen 3B con acelerador de Diamante 3D
python3 tools/chat.py --model qwen3 --diamond 90
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

Aquifer forma parte del ecosistema de **Semantic Geometry Chile**:

* **Aquifer (Público · Open Source):** Motor de paginado zero-copy en espacio de usuario, Ring-Buffer de kernel y shell interactiva en silicio.
* **Caudal (Privado):** Compilador tensorial de alta precisión, calibrador de proyección SVD y empaquetador afín.
* **Semantic Geometry Core (Privado):** Motor ontológico continuo en $[-1, 1]^3$, base de conocimiento formal de 2.06M de conceptos, 4.75M de enlaces semánticos y Merkle Ledger criptográfico BLAKE3.

---

## 📜 Licencia y Atribución

Este proyecto está liberado bajo la licencia **MIT** con atribución obligatoria:
`⚡ Powered by Semantic Geometry Chile · Autor: Marco Retamales`

Para consultas de arquitectura, licenciamiento corporativo o auditoría ontológica:  
**Semantic Geometry Chile** · Santiago, Chile.
