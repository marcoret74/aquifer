# Aquifer 🌊
**Geometric Tensor Paging Runtime & Hypercube Gatekeeper**

> *"Caudal corre a la velocidad de la luz; Aquifer es la reserva profunda que dosifica el flujo."*

---

## 🏛️ ¿Qué es Aquifer?

**Aquifer** es un runtime de memoria virtual geométrica en **C++23** que permite ejecutar modelos de lenguaje masivos (**LLMs de 7B a 70B**) en hardware modesto de consumo (laptops con 8 a 16 GB de RAM), rompiendo la restricción de tener que cargar todos los pesos de golpe en memoria física.

### El Principio de HiperRAM Cúbica
En un modelo Transformer, la CPU ejecuta una sola capa a la vez en estricta secuencia temporal:
- De un modelo de **964 MB (1.5B)**, cada capa pesa solo **~34 MB**.
- De un modelo de **4.5 GB (7B)**, cada capa pesa solo **~150 MB**.
- De un modelo de **35 GB (70B)**, cada capa pesa solo **~1.2 GB**.

**Aquifer** sustituye la carga estática de todo el modelo por dos mecanismos fundamentales:
1. **El Portero (Ring-Buffer Asynchronous Pager):** Mantiene un búfer circular en RAM de solo 2 capas activas (~70 MB para 1.5B; ~300 MB para 7B). Mientras la CPU ejecuta con AVX2 la Capa $L$ en el Slot 0, un hilo de fondo precarga asíncronamente la Capa $L+1$ en el Slot 1 directamente desde disco.
2. **Indexación Espacial Cúbica (Z-Order / Morton 3D):** Proyecta la topología de pesos en el hipercubo semántico $[-1, 1]^3$, asociando cada vóxel de tensores a una clave entera de 64 bits (`uint64_t`) en ~1 nanosegundo mediante instrucciones BMI2 del silicio (`_pdep_u64`).

---

## 📊 Resultados Medidos (Benchmark Inicial)

| Métrica | Enfoque Tradicional (Caudal / llama.cpp) | Aquifer (Portero HiperRAM) | Ahorro |
| :--- | :---: | :---: | :---: |
| **RAM en 28 capas (112 MB prueba)** | 112.0 MB | **8.0 MB** | **92.86 %** |
| **RAM en Qwen 1.5B** | 964.0 MB | **~70 MB** | **~92.7 %** |
| **RAM en Qwen 7B** | 4,500.0 MB | **~300 MB** | **~93.3 %** |
| **Latencia de indexación Morton** | — | **~50 ns** | **O(1) Silicio** |

---

## 🛠️ Compilación y Pruebas

Requisitos: Linux, C++23, `g++` con AVX2/BMI2.

```bash
# Compilar y ejecutar test del indexador espacial Morton
g++ -std=c++23 -O3 -mavx2 -mbmi2 -I include tests/test_morton.cpp -o build/test_morton
./build/test_morton

# Compilar y ejecutar test del Portero asíncrono
g++ -std=c++23 -O3 -pthread -I include tests/test_portero.cpp -o build/test_portero
./build/test_portero
```
