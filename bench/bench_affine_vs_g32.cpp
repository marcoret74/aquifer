// bench_affine_vs_g32.cpp — Comparacion A/B de ciclos e instrucciones entre g32 y affine
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <random>
#include <vector>

#include "caudal/gemv_int4.hpp"

namespace caudal {

template <unsigned M>
inline void gemv_int4_affine_M(const Shape& s,
                               const std::uint8_t* __restrict w,
                               const _Float16* __restrict wscale,
                               const _Float16* __restrict wmin,
                               const std::int32_t* __restrict xgsum,
                               const std::int8_t* __restrict xperm,
                               const float* __restrict xscale,
                               float* __restrict y,
                               std::size_t ystride) {
    const __m256i m0f = _mm256_set1_epi8(0x0F);
    const __m256i ones16 = _mm256_set1_epi16(1);
    const std::size_t G = s.groups();
    const std::size_t blocks = s.K / 64;

    const __m256i p0 = _mm256_setr_epi32(0, 0, 0, 0, 1, 1, 1, 1);
    const __m256i p1 = _mm256_setr_epi32(2, 2, 2, 2, 3, 3, 3, 3);
    const __m256i p2 = _mm256_setr_epi32(4, 4, 4, 4, 5, 5, 5, 5);
    const __m256i p3 = _mm256_setr_epi32(6, 6, 6, 6, 7, 7, 7, 7);

    for (std::size_t n = 0; n < s.N; ++n) {
        const std::uint8_t* row = w + n * (s.K / 2);
        const _Float16* sc = wscale + n * G;
        const _Float16* mn = wmin + n * G;

        __m256 facc[M];
        __m256 minacc[M];
#pragma GCC unroll 8
        for (unsigned m = 0; m < M; ++m) {
            facc[m] = _mm256_setzero_ps();
            minacc[m] = _mm256_setzero_ps();
        }

        std::size_t bb = 0;
        for (; bb + 4 <= blocks; bb += 4) {
            __m256i A[4], B[4];
#pragma GCC unroll 4
            for (int j = 0; j < 4; ++j) {
                const __m256i v = _mm256_loadu_si256(
                    reinterpret_cast<const __m256i*>(row + (bb + std::size_t(j)) * 32));
                const __m256i lo = _mm256_and_si256(v, m0f);
                const __m256i hi = _mm256_and_si256(_mm256_srli_epi16(v, 4), m0f);
                A[j] = _mm256_unpacklo_epi8(lo, hi);
                B[j] = _mm256_unpackhi_epi8(lo, hi);
            }

            const std::size_t gbase = 2 * bb;
            const __m256 w8 = _mm256_cvtph_ps(
                _mm_loadu_si128(reinterpret_cast<const __m128i*>(sc + gbase)));
            const __m256 m8 = _mm256_cvtph_ps(
                _mm_loadu_si128(reinterpret_cast<const __m128i*>(mn + gbase)));

#pragma GCC unroll 8
            for (unsigned m = 0; m < M; ++m) {
                const float* xs_ptr = xscale + std::size_t(m) * G + gbase;
                const __m256 xs8 = _mm256_loadu_ps(xs_ptr);
                const __m256 s8 = _mm256_mul_ps(w8, xs8);

                const std::int32_t* xg_ptr = xgsum + std::size_t(m) * G + gbase;
                const __m256 xg8 = _mm256_cvtepi32_ps(
                    _mm256_loadu_si256(reinterpret_cast<const __m256i*>(xg_ptr)));
                minacc[m] = _mm256_fmadd_ps(_mm256_mul_ps(m8, xs8), xg8, minacc[m]);

                const __m256 sv[4] = {_mm256_permutevar8x32_ps(s8, p0),
                                      _mm256_permutevar8x32_ps(s8, p1),
                                      _mm256_permutevar8x32_ps(s8, p2),
                                      _mm256_permutevar8x32_ps(s8, p3)};

#pragma GCC unroll 4
                for (int j = 0; j < 4; ++j) {
                    const std::int8_t* xm = xperm + std::size_t(m) * s.K
                                            + (bb + std::size_t(j)) * 64;
                    const __m256i x0 = _mm256_loadu_si256(
                        reinterpret_cast<const __m256i*>(xm));
                    const __m256i x1 = _mm256_loadu_si256(
                        reinterpret_cast<const __m256i*>(xm + 32));
                    __m256i acc = _mm256_madd_epi16(_mm256_maddubs_epi16(A[j], x0), ones16);
                    acc = _mm256_add_epi32(
                        acc, _mm256_madd_epi16(_mm256_maddubs_epi16(B[j], x1), ones16));
                    facc[m] = _mm256_fmadd_ps(_mm256_cvtepi32_ps(acc), sv[j], facc[m]);
                }
            }
        }

        auto hsum = [](__m256 v) {
            __m128 l = _mm256_castps256_ps128(v);
            __m128 h = _mm256_extractf128_ps(v, 1);
            __m128 t = _mm_add_ps(l, h);
            t = _mm_hadd_ps(t, t);
            t = _mm_hadd_ps(t, t);
            return _mm_cvtss_f32(t);
        };

#pragma GCC unroll 8
        for (unsigned m = 0; m < M; ++m) {
            y[std::size_t(m) * ystride + n] = hsum(facc[m]) + hsum(minacc[m]);
        }
    }
}

} // namespace caudal

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
    std::vector<float> y(N);

    // Warmup
    for (int it = 0; it < 5; ++it) {
        caudal::gemv_int4_g32_M<1>(s, w.data(), wscale.data(), xgsum.data(), xperm.data(), xs.data(), y.data(), N);
        caudal::gemv_int4_affine_M<1>(s, w.data(), wscale.data(), wmin.data(), xgsum.data(), xperm.data(), xs.data(), y.data(), N);
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
        caudal::gemv_int4_affine_M<1>(s, w.data(), wscale.data(), wmin.data(), xgsum.data(), xperm.data(), xs.data(), y.data(), N);
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
