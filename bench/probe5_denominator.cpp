// probe5_denominator.cpp — denominador de ancho de banda medido EN LA MISMA
// SESION y SOBRE EL MISMO TAMANO DE BUFFER que los pesos del motor.
//
// Por que existe: los numeros historicos (25.3 GB/s) no se reproducen y no
// sirven como denominador. Protocolo acordado con el panel (claude/gemini/qwen):
//   - buffer >= 512 MB, mmap + MADV_HUGEPAGE, pre-tocado (prefault real)
//   - verificar AnonHugePages de verdad, no asumirlo
//   - 4-8 acumuladores independientes y ancla asm para que el compilador no
//     elimine las cargas
//   - calentamiento descartado; mediana de N pasadas, y maximo aparte
//   - barrido de hilos (los nucleos fisicos son 4)
//
//   ./build/probe5_denominator --mb 868 --passes 15
#include <immintrin.h>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <string>
#include <thread>
#include <vector>

#include <pthread.h>
#include <sched.h>

#include <sys/mman.h>
#include <unistd.h>

namespace {

std::string leer(const std::string& ruta) {
    std::ifstream f(ruta);
    std::string s;
    std::getline(f, s);
    return s;
}

std::uint64_t anon_hugepages_kb() {
    std::ifstream f("/proc/self/smaps_rollup");
    std::string k;
    while (f >> k) {
        if (k == "AnonHugePages:") {
            std::uint64_t v = 0;
            f >> v;
            return v;
        }
        f.ignore(4096, '\n');
    }
    return 0;
}

// Lee `n` bytes con 8 acumuladores AVX2 y 4 punteros por hilo (cuatro streams).
double lectura_gb_s(const std::uint8_t* base, std::size_t bytes, unsigned hilos,
                    double& mejor) {
    std::vector<std::thread> ts;
    std::vector<double> parcial(hilos, 0.0);
    const std::size_t por_hilo = bytes / hilos;
    const std::size_t bloque = 1u << 20;  // 1 MB por pasada intermedia

    auto cuerpo = [&](unsigned t) {
        // Afinidad explicita: sin esto el scheduler puede meter dos hilos en el
        // mismo nucleo fisico (SMT) y el agregado se hunde con varianza alta.
        cpu_set_t set;
        CPU_ZERO(&set);
        CPU_SET(t % 8, &set);
        pthread_setaffinity_np(pthread_self(), sizeof(set), &set);
        // UN stream secuencial por hilo: es lo que hace el motor (cada hilo
        // recibe un rango contiguo de filas). Con 4 punteros por hilo el TLB
        // y el prefetcher se ahogan y la medicion no representa al motor.
        const std::uint8_t* p0 = base + t * por_hilo;
        const std::size_t fin = por_hilo & ~std::size_t(255);
        __m256 acc0 = _mm256_setzero_ps(), acc1 = acc0, acc2 = acc0, acc3 = acc0;
        __m256 acc4 = acc0, acc5 = acc0, acc6 = acc0, acc7 = acc0;
        for (std::size_t i = 0; i < fin; i += 256) {
            acc0 = _mm256_add_ps(acc0, _mm256_loadu_ps(reinterpret_cast<const float*>(p0 + i)));
            acc1 = _mm256_add_ps(acc1, _mm256_loadu_ps(reinterpret_cast<const float*>(p0 + i + 32)));
            acc2 = _mm256_add_ps(acc2, _mm256_loadu_ps(reinterpret_cast<const float*>(p0 + i + 64)));
            acc3 = _mm256_add_ps(acc3, _mm256_loadu_ps(reinterpret_cast<const float*>(p0 + i + 96)));
            acc4 = _mm256_add_ps(acc4, _mm256_loadu_ps(reinterpret_cast<const float*>(p0 + i + 128)));
            acc5 = _mm256_add_ps(acc5, _mm256_loadu_ps(reinterpret_cast<const float*>(p0 + i + 160)));
            acc6 = _mm256_add_ps(acc6, _mm256_loadu_ps(reinterpret_cast<const float*>(p0 + i + 192)));
            acc7 = _mm256_add_ps(acc7, _mm256_loadu_ps(reinterpret_cast<const float*>(p0 + i + 224)));
        }
        __m256 s0 = _mm256_add_ps(_mm256_add_ps(acc0, acc1), _mm256_add_ps(acc2, acc3));
        __m256 s1 = _mm256_add_ps(_mm256_add_ps(acc4, acc5), _mm256_add_ps(acc6, acc7));
        __m256 s = _mm256_add_ps(s0, s1);
        __m128 lo = _mm256_castps256_ps128(s);
        __m128 hi = _mm256_extractf128_ps(s, 1);
        __m128 r = _mm_add_ps(lo, hi);
        r = _mm_hadd_ps(r, r);
        r = _mm_hadd_ps(r, r);
        float v = _mm_cvtss_f32(r);
        asm volatile("" : "+x"(v));
        parcial[t] = double(v);
    };

    const auto t0 = std::chrono::steady_clock::now();
    for (unsigned t = 0; t < hilos; ++t) ts.emplace_back(cuerpo, t);
    for (auto& th : ts) th.join();
    const auto t1 = std::chrono::steady_clock::now();
    const double seg = std::chrono::duration<double>(t1 - t0).count();
    const double gb = double(bytes) / 1e9;
    (void)mejor;
    return gb / (seg > 0 ? seg : 1e-9);
}

}  // namespace

int main(int argc, char** argv) {
    std::size_t mb = 868;
    int passes = 15;
    bool usar_malloc = false;
    for (int i = 1; i + 1 < argc; ++i) {
        const std::string a = argv[i];
        if (a == "--mb") mb = std::size_t(std::atoi(argv[++i]));
        else if (a == "--passes") passes = std::atoi(argv[++i]);
        else if (a == "--malloc") usar_malloc = true;
    }

    const std::size_t bytes = mb * 1024 * 1024;
    std::printf("buffer   : %zu MB\n", mb);
    std::printf("governor : %s\n",
                leer("/sys/devices/system/cpu/cpu0/cpufreq/scaling_governor").c_str());
    std::printf("freq     : %s kHz (reportada; NO es la real bajo carga)\n",
                leer("/sys/devices/system/cpu/cpu0/cpufreq/scaling_cur_freq").c_str());

    // ORDEN CRITICO: madvise ANTES de tocar las paginas. Con MAP_POPULATE las
    // paginas ya se crean en 4 KB y MADV_HUGEPAGE llega tarde (medido: 1,8 %
    // de THP y 12,5 GB/s por thrash de TLB).
    void* p = nullptr;
    if (usar_malloc) {
        // Igual que el motor: buffer del heap (new[]), sin madvise. Sirve para
        // probar si lo que manda es la UBICACION FISICA (Flex Mode: unos 4 GB
        // entrelazados y el resto en un solo canal).
        p = std::malloc(bytes);
        if (!p) { std::printf("malloc fallo\n"); return 1; }
        std::printf("modo     : heap (malloc)\n");
    } else {
        p = mmap(nullptr, bytes, PROT_READ | PROT_WRITE,
                 MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
        if (p == MAP_FAILED) { std::printf("mmap fallo\n"); return 1; }
        madvise(p, bytes, MADV_HUGEPAGE);
        std::printf("modo     : mmap + MADV_HUGEPAGE\n");
    }
    std::memset(p, 1, bytes);   // prefault real
    std::printf("hugepages: %llu KB de %zu KB (%.1f %% THP)\n",
                (unsigned long long)anon_hugepages_kb(), bytes / 1024,
                100.0 * double(anon_hugepages_kb() * 1024) / double(bytes));

    const auto* base = static_cast<const std::uint8_t*>(p);
    std::printf("\n hilos    mediana      mejor     peor   (GB/s de lectura pura)\n");
    for (unsigned h : {1u, 2u, 4u, 8u}) {
        std::vector<double> v;
        double dummy = 0.0;
        lectura_gb_s(base, bytes, h, dummy);                  // calentamiento
        for (int k = 0; k < passes; ++k)
            v.push_back(lectura_gb_s(base, bytes, h, dummy));
        std::vector<double> m = v;
        std::sort(m.begin(), m.end());
        std::printf(" %5u  %8.2f  %8.2f  %8.2f\n", h,
                    m[m.size() / 2], m.back(), m.front());
    }
    if (!usar_malloc) munmap(p, bytes);
    std::printf("\nDenominador a usar: la MEDIANA de 4 hilos de ESTA tabla, medida\n"
                "en la misma sesion. No el 38.4 teorico ni el 25.3 historico.\n");
    return 0;
}
