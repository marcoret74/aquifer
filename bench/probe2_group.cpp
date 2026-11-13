// =============================================================================
//  caudal · bench/probe2_group.cpp
//  EL NÚMERO HONESTO: ¿cuánto cuesta el layout REAL (escalas por grupo de 64)?
// =============================================================================
//  probe1 midió un kernel con escalas POR FILA: lee 117.4 MB y hace un solo
//  pliegue por fila. El kernel de producción (gemv_int4.hpp) usa escalas por
//  GRUPO de 64: lee además w_scale y w_codesum, y pliega 56 veces por fila.
//
//  Este probe mide ese kernel, sin trampas, y reporta TRES denominadores para
//  que la comparación no se pueda maquillar:
//
//    GB/s "packed"   : sólo los pesos empaquetados -> comparable con el gate
//    GB/s "real"     : todos los bytes leídos (packed + escalas + codesum)
//    GMAC/s          : MACs por segundo -> invariante a cómo se cuenten bytes
//
//  El gate se juzga con GMAC/s, que es lo que determina tok/s.
//
//  Uso: ./probe2_group [passes]
// =============================================================================

#include "caudal/gemv_int4.hpp"
#include "caudal/layout.hpp"

#include <algorithm>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <thread>
#include <vector>

#include <time.h>

namespace {

constexpr std::size_t kMiB = std::size_t{1} << 20;

// Misma forma que probe1: una rebanada de la FFN de Qwen2.5-7B.
constexpr std::size_t kN = 65536;
constexpr std::size_t kK = 3584;
constexpr std::size_t kGroup = 64;

volatile std::uint64_t g_sink = 0;

double now_s() {
    timespec ts{};
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return static_cast<double>(ts.tv_sec) + static_cast<double>(ts.tv_nsec) * 1e-9;
}

}  // namespace

int main(int argc, char** argv) {
    int passes = 15;
    if (argc > 1) passes = std::max(1, std::atoi(argv[1]));

    caudal::Shape s;
    s.N = kN;
    s.K = kK;
    s.group = kGroup;
    const std::size_t G = s.groups();
    const std::size_t rb = s.row_bytes();

    const std::size_t w_bytes      = s.N * rb;
    const std::size_t scale_bytes  = s.N * G * sizeof(_Float16);
    const std::size_t cs_bytes     = s.N * G * sizeof(std::int32_t);
    const std::size_t total_bytes  = w_bytes + scale_bytes + cs_bytes;

    std::printf("\n");
    std::printf("caudal · probe2_group  (layout REAL, escalas por grupo de 64)\n");
    std::printf("forma        : N=%zu K=%zu  (%.1f M params)\n",
                s.N, s.K, static_cast<double>(s.N * s.K) / 1e6);
    std::printf("grupos/fila  : %zu\n", G);
    std::printf("w_packed     : %8.1f MiB\n", double(w_bytes) / kMiB);
    std::printf("w_scale      : %8.1f MiB  (fp16 por grupo)\n", double(scale_bytes) / kMiB);
    std::printf("w_codesum    : %8.1f MiB  (int32 por grupo)\n", double(cs_bytes) / kMiB);
    std::printf("TOTAL leido  : %8.1f MiB  (%.2f bits/param)\n",
                double(total_bytes) / kMiB,
                8.0 * double(total_bytes) / double(s.N * s.K));
    std::printf("passes       : %d\n\n", passes);

    auto* w     = static_cast<std::uint8_t*>(std::aligned_alloc(64, (w_bytes + 63) & ~std::size_t{63}));
    auto* scale = static_cast<_Float16*>(std::aligned_alloc(64, (scale_bytes + 63) & ~std::size_t{63}));
    auto* codesum = static_cast<std::int32_t*>(std::aligned_alloc(64, (cs_bytes + 63) & ~std::size_t{63}));
    auto* y     = static_cast<float*>(std::aligned_alloc(64, (s.N * sizeof(float) + 63) & ~std::size_t{63}));
    auto* x     = static_cast<std::uint8_t*>(std::aligned_alloc(64, (kK + 63) & ~std::size_t{63}));
    if (!w || !scale || !codesum || !y || !x) { std::printf("sin memoria\n"); return 1; }

    // --- Poblado (y codesum REAL, coherente con los nibbles) -----------------
    std::uint32_t rng = 0x9E3779B9u;
    for (std::size_t i = 0; i < w_bytes; ++i) {
        rng = rng * 1664525u + 1013904223u;
        w[i] = static_cast<std::uint8_t>(rng >> 24);
    }
    for (std::size_t n = 0; n < s.N; ++n)
        for (std::size_t g = 0; g < G; ++g) {
            std::int32_t cs = 0;
            for (std::size_t j = 0; j < kGroup; ++j)
                cs += caudal::code_at(w + n * rb, g * kGroup + j);
            codesum[n * G + g] = cs;
        }
    for (std::size_t i = 0; i < s.N * G; ++i) scale[i] = _Float16(0.35f);
    for (std::size_t i = 0; i < kK; ++i) x[i] = static_cast<std::uint8_t>(100 + (i % 56));

    auto run_rows = [&](std::size_t b, std::size_t e) {
        caudal::Shape sub = s;
        sub.N = e - b;
        caudal::gemv_int4(sub, w + b * rb, scale + b * G, codesum + b * G, x, 0.02f, y + b);
    };

    // --- 1 hilo --------------------------------------------------------------
    const double t0 = now_s();
    for (int p = 0; p < passes; ++p) run_rows(0, s.N);
    const double t1 = now_s();
    const double secs1 = t1 - t0;

    // --- N hilos (partición por filas) ---------------------------------------
    const unsigned T = std::max(1u, std::thread::hardware_concurrency());
    const std::size_t chunk = (s.N / T) & ~std::size_t{15};
    const double t2 = now_s();
    for (int p = 0; p < passes; ++p) {
        std::vector<std::thread> th;
        th.reserve(T);
        for (unsigned t = 0; t < T; ++t) {
            const std::size_t b = t * chunk;
            const std::size_t e = (t + 1 == T) ? s.N : b + chunk;
            th.emplace_back([&, b, e] { run_rows(b, e); });
        }
        for (auto& z : th) z.join();
    }
    const double t3 = now_s();
    const double secsT = t3 - t2;

    g_sink ^= static_cast<std::uint64_t>(y[s.N - 1] * 1000.0f);

    const double macs = double(s.N) * double(s.K) * double(passes);
    const double gbs_packed = double(w_bytes) * passes / secs1 / 1e9;
    const double gbs_real1  = double(total_bytes) * passes / secs1 / 1e9;
    const double gbs_realT  = double(total_bytes) * passes / secsT / 1e9;
    const double gmac1 = macs / secs1 / 1e9;
    const double gmacT = macs / secsT / 1e9;

    std::printf("%-22s %10s %10s %10s\n", "métrica", "1 hilo", "T hilos", "gate");
    std::printf("%-22s %10.1f %10.1f %10.1f\n", "GB/s (sólo packed)", gbs_packed,
                double(w_bytes) * passes / secsT / 1e9, 11.6);
    std::printf("%-22s %10.1f %10.1f %10s\n", "GB/s (bytes reales)", gbs_real1, gbs_realT, "-");
    std::printf("%-22s %10.1f %10.1f %10.1f\n", "GMAC/s", gmac1, gmacT, 23.1);
    std::printf("\n");
    std::printf("hilos usados        : %u\n", T);
    std::printf("tok/s 7B int4 @T    : %.2f\n", gmacT / 7.61);
    std::printf("tok/s 1.5B int4 @T  : %.1f\n", gmacT / 0.88);

    const double rel = gmac1 / 23.1;
    std::printf("\nvs probe1 (23.1 GMAC/s, escalas por fila): %.1f%% del rendimiento\n",
                100.0 * rel);
    std::printf("(la diferencia se paga en bytes extra y en 3 instrucciones por grupo)\n\n");

    std::free(w); std::free(scale); std::free(codesum); std::free(y); std::free(x);
    return 0;
}
