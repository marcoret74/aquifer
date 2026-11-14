// test_affine_g32.cpp — Verifica el esquema afin (escala fp16 + minimo fp16 por grupo de 32, 5.0 bpw)
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <random>
#include <vector>
#include <algorithm>

#include "caudal/gemv_int4.hpp"

// test uses gemv_int4_affine_M from include/caudal/gemv_int4.hpp

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
            std::vector<float> xms(M * G);
            for (auto& v : xs) v = float(rng() % 100) / 4000.0f + 0.001f;
            for (unsigned m = 0; m < M; ++m) {
                caudal::compute_x_gsum(s, x.data() + m * K, xg.data() + m * G);
            }
            for (std::size_t i = 0; i < M * G; ++i) xms[i] = float(xg[i]) * xs[i];

            std::vector<float> y_kernel(M * N);
            if (M == 1) {
                caudal::gemv_int4_affine_M<1>(s, packed.data(), wscale.data(), wmin.data(),
                                              xms.data(), xp.data(), xs.data(),
                                              y_kernel.data(), N);
            } else if (M == 3) {
                caudal::gemv_int4_affine_M<3>(s, packed.data(), wscale.data(), wmin.data(),
                                              xms.data(), xp.data(), xs.data(),
                                              y_kernel.data(), N);
            } else if (M == 4) {
                caudal::gemv_int4_affine_M<4>(s, packed.data(), wscale.data(), wmin.data(),
                                              xms.data(), xp.data(), xs.data(),
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
