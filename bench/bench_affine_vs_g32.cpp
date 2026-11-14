// bench_affine_vs_g32.cpp — Comparacion A/B de ciclos e instrucciones entre g32 y affine
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <random>
#include <vector>

#include "caudal/gemv_int4.hpp"

int main() {
    // Matriz de tamaño típico de capa Qwen MLP: N = 1536, K = 8960
    const std::size_t N = 1536, K = 8960;
    const std::size_t G = K / 32;
    caudal::Shape s; s.N = N; s.K = K; s.group = 32; s.stride = 0;

    std::vector<std::uint8_t> w(N * (K / 2), 0x55);
    std::vector<_Float16> wscale(N * G, _Float16(0.01f));
    std::vector<_Float16> wmin(N * G, _Float16(-0.05f));
    std::vector<std::int8_t> xperm(K, 3);
    std::vector<std::int32_t> xgsum(G, 120);
    std::vector<float> xs(G, 0.02f);
    std::vector<float> xms(G);
    for (std::size_t g = 0; g < G; ++g) xms[g] = float(xgsum[g]) * xs[g];
    std::vector<float> y(N);

    // Warmup
    for (int it = 0; it < 5; ++it) {
        caudal::gemv_int4_g32_M<1>(s, w.data(), wscale.data(), xgsum.data(), xperm.data(), xs.data(), y.data(), N);
        caudal::gemv_int4_affine_M<1>(s, w.data(), wscale.data(), wmin.data(), xms.data(), xperm.data(), xs.data(), y.data(), N);
    }

    const int ITERS = 100;

    // Benchmark g32 (linea base)
    auto t0 = std::chrono::high_resolution_clock::now();
    for (int it = 0; it < ITERS; ++it) {
        caudal::gemv_int4_g32_M<1>(s, w.data(), wscale.data(), xgsum.data(), xperm.data(), xs.data(), y.data(), N);
    }
    auto t1 = std::chrono::high_resolution_clock::now();
    double ms_g32 = std::chrono::duration<double, std::milli>(t1 - t0).count() / ITERS;

    // Benchmark affine (nuevo)
    auto t2 = std::chrono::high_resolution_clock::now();
    for (int it = 0; it < ITERS; ++it) {
        caudal::gemv_int4_affine_M<1>(s, w.data(), wscale.data(), wmin.data(), xms.data(), xperm.data(), xs.data(), y.data(), N);
    }
    auto t3 = std::chrono::high_resolution_clock::now();
    double ms_affine = std::chrono::duration<double, std::milli>(t3 - t2).count() / ITERS;

    double gflops = (2.0 * double(N) * double(K)) / 1e6;
    std::printf("Matriz N=%zu K=%zu (MLP standard):\n", N, K);
    std::printf("  Línea base g32  : %6.3f ms  (%5.1f GFLOPS)\n", ms_g32, gflops / ms_g32);
    std::printf("  Nuevo affine_g32: %6.3f ms  (%5.1f GFLOPS)\n", ms_affine, gflops / ms_affine);
    std::printf("  Ratio ms        : %6.2f %%\n", (ms_affine / ms_g32) * 100.0);

    return 0;
}
