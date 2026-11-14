// test_affine_g32.cpp — Verifica el esquema afin (escala fp16 + minimo fp16 por grupo de 32, 5.0 bpw)
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <random>
#include <vector>
#include <algorithm>

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
            // 8 escalas y 8 minimos en fp16 -> fp32
            const __m256 w8 = _mm256_cvtph_ps(
                _mm_loadu_si128(reinterpret_cast<const __m128i*>(sc + gbase)));
            const __m256 m8 = _mm256_cvtph_ps(
                _mm_loadu_si128(reinterpret_cast<const __m128i*>(mn + gbase)));

#pragma GCC unroll 8
            for (unsigned m = 0; m < M; ++m) {
                const float* xs_ptr = xscale + std::size_t(m) * G + gbase;
                const __m256 xs8 = _mm256_loadu_ps(xs_ptr);
                const __m256 s8 = _mm256_mul_ps(w8, xs8);

                // Termino del minimo para los 8 grupos: m8 * xs8 * xgsum
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
    std::mt19937 rng(42);
    std::normal_distribution<float> gauss(0.0f, 0.5f);
    int fallos = 0;

    for (unsigned M : {1u, 3u, 4u}) {
        for (std::size_t K : {512u, 1536u, 2048u, 4096u, 8960u}) {
            const std::size_t G = K / 32;
            const std::size_t N = 19;
            caudal::Shape s;
            s.N = N; s.K = K; s.group = 32; s.stride = 0;

            std::vector<float> w(N * K);
            for (auto& v : w) v = gauss(rng) + 0.05f;

            std::vector<std::uint8_t> nib(N * K);
            std::vector<_Float16> wscale(N * G);
            std::vector<_Float16> wmin(N * G);
            std::vector<float> deq(N * K);

            for (std::size_t n = 0; n < N; ++n) {
                for (std::size_t g = 0; g < G; ++g) {
                    float lo = 1e30f, hi = -1e30f;
                    for (std::size_t i = 0; i < 32; ++i) {
                        float val = w[n * K + g * 32 + i];
                        lo = std::min(lo, val);
                        hi = std::max(hi, val);
                    }
                    float sc = (hi - lo) / 15.0f;
                    _Float16 sc16 = static_cast<_Float16>(sc);
                    _Float16 mn16 = static_cast<_Float16>(lo);
                    wscale[n * G + g] = sc16;
                    wmin[n * G + g] = mn16;

                    float sc_f = static_cast<float>(sc16);
                    float mn_f = static_cast<float>(mn16);
                    float inv_sc = (sc_f > 0.0f) ? 1.0f / sc_f : 0.0f;

                    for (std::size_t i = 0; i < 32; ++i) {
                        std::size_t k = g * 32 + i;
                        int q = std::clamp(static_cast<int>(std::lround((w[n * K + k] - mn_f) * inv_sc)), 0, 15);
                        nib[n * K + k] = static_cast<std::uint8_t>(q);
                        deq[n * K + k] = mn_f + sc_f * float(q);
                    }
                }
            }

            // Empaquetado canónico: nibbles consecutivos en bytes contiguos
            std::vector<std::uint8_t> packed(N * (K / 2));
            for (std::size_t n = 0; n < N; ++n) {
                for (std::size_t k = 0; k < K; k += 2) {
                    packed[n * (K / 2) + k / 2] =
                        static_cast<std::uint8_t>(nib[n * K + k] | (nib[n * K + k + 1] << 4));
                }
            }

            std::vector<std::int8_t> x(M * K), xp(M * K);
            for (auto& v : x) v = std::int8_t(int(rng() % 255) - 127);
            for (unsigned m = 0; m < M; ++m) {
                for (std::size_t b = 0; b < K / 64; ++b) {
                    for (std::size_t i = 0; i < 64; ++i) {
                        xp[m * K + b * 64 + i] = x[m * K + b * 64 + caudal::kPerm[i]];
                    }
                }
            }

            std::vector<std::int32_t> xg(M * G);
            std::vector<float> xs(M * G);
            for (auto& v : xs) v = float(rng() % 100) / 4000.0f + 0.001f;
            for (unsigned m = 0; m < M; ++m) {
                caudal::compute_x_gsum(s, x.data() + m * K, xg.data() + m * G);
            }

            std::vector<float> y_kernel(M * N);
            if (M == 1) {
                caudal::gemv_int4_affine_M<1>(s, packed.data(), wscale.data(), wmin.data(),
                                              xg.data(), xp.data(), xs.data(),
                                              y_kernel.data(), N);
            } else if (M == 3) {
                caudal::gemv_int4_affine_M<3>(s, packed.data(), wscale.data(), wmin.data(),
                                              xg.data(), xp.data(), xs.data(),
                                              y_kernel.data(), N);
            } else if (M == 4) {
                caudal::gemv_int4_affine_M<4>(s, packed.data(), wscale.data(), wmin.data(),
                                              xg.data(), xp.data(), xs.data(),
                                              y_kernel.data(), N);
            }

            // Referencia exacta en double
            double peor = 0.0, mag = 0.0;
            for (unsigned m = 0; m < M; ++m) {
                for (std::size_t n = 0; n < N; ++n) {
                    double acc = 0.0;
                    for (std::size_t g = 0; g < G; ++g) {
                        double sub = 0.0;
                        for (std::size_t i = 0; i < 32; ++i) {
                            const std::size_t k = g * 32 + i;
                            sub += double(deq[n * K + k]) * double(x[m * K + k]);
                        }
                        acc += double(xs[m * G + g]) * sub;
                    }
                    const double got = double(y_kernel[m * N + n]);
                    peor = std::max(peor, std::fabs(got - acc));
                    mag = std::max(mag, std::fabs(acc));
                }
            }

            bool ok = (peor <= 1e-3 * mag + 1e-3);
            std::printf("  M=%u K=%5zu G=%3zu N=%2zu |y|=%.2f err_max=%.2e %s\n",
                        M, K, G, N, mag, peor, ok ? "OK" : "FALLA");
            if (!ok) ++fallos;
        }
    }

    std::printf("RESULTADO: %d fallo(s)\n", fallos);
    return fallos != 0;
}
