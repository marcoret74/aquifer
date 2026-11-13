// =============================================================================
//  caudal · bench/probe2_group.cpp
//  RADIOGRAFÍA: int4 (4.25 bits/param) vs int8 (8.25 bits/param) en AVX2.
// =============================================================================
//  int4 con la inversión unsigned: sin LUT, sin codesum.
//  int8: sin desempacar nada, ~11 instrucciones por 64 valores y CERO ops de
//        puerto de shuffle, pero lee 2x los bytes.
//
//  Cuál gana depende de si esta máquina está limitada por memoria (int4) o por
//  instrucciones (int8). Eso no se razona: se mide.
//
//  Cada variante se verifica contra su referencia escalar antes de cronometrar.
//  Un benchmark sin aserción de correctitud no mide un kernel, mide su silueta.
//
//  Uso: ./probe2_group [passes]
// =============================================================================

#include "caudal/gemv_int4.hpp"
#include "caudal/layout.hpp"

#include <algorithm>
#include <cstddef>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <thread>
#include <vector>

#include <immintrin.h>
#include <time.h>

namespace {

constexpr std::size_t kMiB = std::size_t{1} << 20;
constexpr std::size_t kN = 65536;
constexpr std::size_t kK = 3584;
constexpr std::size_t kGroup = 64;

volatile std::uint64_t g_sink = 0;

double now_s() {
    timespec ts{};
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return static_cast<double>(ts.tv_sec) + static_cast<double>(ts.tv_nsec) * 1e-9;
}

// ---- Referencia escalar para int8 (w guardado como uint8 = code+8) ----------
double ref_row_i8(const std::uint8_t* w8, const _Float16* sc,
                  const std::int8_t* x, float xs, std::size_t n) {
    const std::uint8_t* row = w8 + n * kK;
    double acc = 0.0;
    for (std::size_t g = 0; g < kK / kGroup; ++g) {
        long long s = 0;
        for (std::size_t j = 0; j < kGroup; ++j) {
            // int8 no está empaquetado: NO hay permutación que aplicar.
            const int code = static_cast<int>(row[g * kGroup + j]) - 8;
            s += static_cast<long long>(code) * x[g * kGroup + j];
        }
        acc += static_cast<double>(s) * static_cast<double>(sc[n * (kK / kGroup) + g])
             * static_cast<double>(xs);
    }
    return acc;
}

// ---- Kernel int8: cero desempacado, cero shuffle ----------------------------
void gemv_i8(std::size_t b, std::size_t e, const std::uint8_t* w8,
             const _Float16* sc, const std::int32_t* xgsum,
             const std::int8_t* x, float xs, float* y) {
    const __m256i ones16 = _mm256_set1_epi16(1);
    const std::size_t G = kK / kGroup;
    for (std::size_t n = b; n < e; ++n) {
        const std::uint8_t* row = w8 + n * kK;
        __m256 facc = _mm256_setzero_ps();
        for (std::size_t g = 0; g < G; ++g) {
            const __m256i w0 = _mm256_loadu_si256(
                reinterpret_cast<const __m256i*>(row + g * kGroup));
            const __m256i w1 = _mm256_loadu_si256(
                reinterpret_cast<const __m256i*>(row + g * kGroup + 32));
            const __m256i x0 = _mm256_loadu_si256(
                reinterpret_cast<const __m256i*>(x + g * kGroup));
            const __m256i x1 = _mm256_loadu_si256(
                reinterpret_cast<const __m256i*>(x + g * kGroup + 32));
            __m256i acc = _mm256_madd_epi16(_mm256_maddubs_epi16(w0, x0), ones16);
            acc = _mm256_add_epi32(
                acc, _mm256_madd_epi16(_mm256_maddubs_epi16(w1, x1), ones16));
            acc = _mm256_sub_epi32(
                acc, _mm256_setr_epi32(8 * xgsum[g], 0, 0, 0, 0, 0, 0, 0));
            const float fs = static_cast<float>(sc[n * G + g]) * xs;
            facc = _mm256_fmadd_ps(_mm256_cvtepi32_ps(acc), _mm256_set1_ps(fs), facc);
        }
        __m128 lo = _mm256_castps256_ps128(facc);
        __m128 hi = _mm256_extractf128_ps(facc, 1);
        __m128 sum = _mm_add_ps(lo, hi);
        sum = _mm_hadd_ps(sum, sum);
        sum = _mm_hadd_ps(sum, sum);
        y[n] = _mm_cvtss_f32(sum);
    }
}

}  // namespace

int main(int argc, char** argv) {
    int passes = 10;
    if (argc > 1) passes = std::max(1, std::atoi(argv[1]));

    caudal::Shape s;
    s.N = kN; s.K = kK; s.group = kGroup;
    const std::size_t G = s.groups();
    const std::size_t rb = s.row_bytes();

    const std::size_t w4_bytes = s.N * rb;
    const std::size_t w8_bytes = s.N * kK;
    const std::size_t sc_bytes = s.N * G * sizeof(_Float16);
    const std::size_t b4 = w4_bytes + sc_bytes;   // 4.25 bits/param
    const std::size_t b8 = w8_bytes + sc_bytes;   // 8.25 bits/param

    std::printf("\ncaudal · probe2_group\n");
    std::printf("N=%zu K=%zu  (%.1f M params)  passes=%d\n\n",
                s.N, s.K, double(s.N * s.K) / 1e6, passes);
    std::printf("%-6s %10s %10s %10s %9s %9s\n",
                "var", "bits/param", "MiB/pasada", "GB/s real", "GMAC/s", "tok/s 7B");

    auto* w4  = static_cast<std::uint8_t*>(std::aligned_alloc(64, (w4_bytes + 63) & ~std::size_t{63}));
    auto* w8  = static_cast<std::uint8_t*>(std::aligned_alloc(64, (w8_bytes + 63) & ~std::size_t{63}));
    auto* sc  = static_cast<_Float16*>(std::aligned_alloc(64, (sc_bytes + 63) & ~std::size_t{63}));
    auto* y   = static_cast<float*>(std::aligned_alloc(64, (s.N * sizeof(float) + 63) & ~std::size_t{63}));
    auto* x   = static_cast<std::int8_t*>(std::aligned_alloc(64, (kK + 63) & ~std::size_t{63}));
    auto* xg  = static_cast<std::int32_t*>(std::aligned_alloc(64, (G * sizeof(std::int32_t) + 63) & ~std::size_t{63}));
    if (!w4 || !w8 || !sc || !y || !x || !xg) { std::printf("sin memoria\n"); return 1; }

    std::uint32_t rng = 0x9E3779B9u;
    for (std::size_t i = 0; i < w4_bytes; ++i) {
        rng = rng * 1664525u + 1013904223u;
        w4[i] = static_cast<std::uint8_t>(rng >> 24);
    }
    // int8 equivalente (mismos códigos, sin empaquetar): w8 = code+8
    for (std::size_t n = 0; n < s.N; ++n)
        for (std::size_t k = 0; k < kK; ++k)
            w8[n * kK + k] = static_cast<std::uint8_t>(caudal::nibble_at(w4 + n * rb, k));
    for (std::size_t i = 0; i < s.N * G; ++i) sc[i] = _Float16(0.35f);
    for (std::size_t i = 0; i < kK; ++i) x[i] = static_cast<std::int8_t>(-60 + (i % 120));
    caudal::compute_x_gsum(s, x, xg);

    const float xs = 0.02f;

    // --- Correctitud antes de cronometrar ------------------------------------
    {
        std::vector<float> yr(s.N);
        caudal::gemv_int4(s, w4, sc, xg, x, xs, y);
        caudal::gemv_ref(s, w4, sc, x, xs, yr.data());
        double worst = 0.0;
        for (std::size_t n = 0; n < s.N; ++n)
            worst = std::max(worst, std::fabs((double)y[n] - (double)yr[n]));
        std::printf("check int4 (todas las filas): max|kernel-ref| = %.2e\n", worst);

        gemv_i8(0, s.N, w8, sc, xg, x, xs, y);
        const double r8 = ref_row_i8(w8, sc, x, xs, 0);
        std::printf("check int8 fila 0 : kernel=%.4f  ref=%.4f  diff=%.2e\n\n",
                    (double)y[0], r8, std::fabs((double)y[0] - r8));
    }

    auto measure = [&](const std::function<void(std::size_t, std::size_t)>& run,
                       std::size_t total, std::size_t threads) -> double {
        const std::size_t chunk = threads > 1 ? ((s.N / threads) & ~std::size_t{15}) : s.N;
        auto once = [&]() {
            if (threads == 1) { run(0, s.N); return; }
            std::vector<std::thread> th;
            th.reserve(threads);
            for (std::size_t t = 0; t < threads; ++t) {
                const std::size_t bnd = t * chunk;
                const std::size_t e = (t + 1 == threads) ? s.N : bnd + chunk;
                th.emplace_back([&, bnd, e] { run(bnd, e); });
            }
            for (auto& z : th) z.join();
        };
        once();
        const double t0 = now_s();
        for (int p = 0; p < passes; ++p) once();
        const double t1 = now_s();
        return double(total) * passes / (t1 - t0) / 1e9;
    };

    const unsigned T = std::max(1u, std::thread::hardware_concurrency());

    auto run4 = [&](std::size_t b, std::size_t e) {
        caudal::Shape sub = s; sub.N = e - b;
        caudal::gemv_int4(sub, w4 + b * rb, sc + b * G, xg, x, xs, y + b);
    };
    auto run8 = [&](std::size_t b, std::size_t e) {
        gemv_i8(b, e, w8, sc, xg, x, xs, y);
    };

    for (unsigned th : {1u, T}) {
        const double g4 = measure(run4, b4, th);
        const double g8 = measure(run8, b8, th);
        // GMAC/s = params * (B/s) / bytes   (los passes se cancelan)
        const double m4 = double(s.N) * double(s.K) * g4 / double(b4);
        const double m8 = double(s.N) * double(s.K) * g8 / double(b8);
        std::printf("%-6s %10s %10.1f %10.1f %9.1f %9.2f\n",
                    th == 1 ? "int4/1" : "int4/T", "4.25",
                    double(b4) / kMiB, g4, m4, m4 / 7.61);
        std::printf("%-6s %10s %10.1f %10.1f %9.1f %9.2f\n",
                    th == 1 ? "int8/1" : "int8/T", "8.25",
                    double(b8) / kMiB, g8, m8, m8 / 7.61);
    }
    std::printf("\n(hilos T = %u)\n\n", T);

    g_sink ^= static_cast<std::uint64_t>(y[s.N - 1] * 1000.0f);
    std::free(w4); std::free(w8); std::free(sc);
    std::free(y); std::free(x); std::free(xg);
    return 0;
}
