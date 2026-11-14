// test_kmin.cpp — valida el kernel con minimo (super-bloque de 256) contra un
// producto punto en double sobre los pesos reconstruidos, con los MISMOS bytes
// empaquetados. Mismo espiritu que check_g32M: si el unpacking de 6 bits esta
// mal, esto lo grita antes de tocar un modelo.
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

#include "caudal/gemv_int4.hpp"

namespace {

void kmin_pack(const float* w, std::size_t K, std::uint8_t* nib,
               std::uint8_t* hdr, float* deq) {
    const std::size_t SBn = K / 256;
    for (std::size_t sb = 0; sb < SBn; ++sb) {
        float scor[8], mnv[8];
        for (int g = 0; g < 8; ++g) {
            float lo = 1e30f, hi = -1e30f;
            for (int i = 0; i < 32; ++i) {
                const float v = w[sb * 256 + std::size_t(g) * 32 + std::size_t(i)];
                lo = std::min(lo, v);
                hi = std::max(hi, v);
            }
            scor[g] = (hi - lo) / 15.0f;
            mnv[g] = lo;                       // minimo CON SIGNO
        }
        float smax = 0.0f, mmax = 0.0f;
        for (int g = 0; g < 8; ++g) {
            smax = std::max(smax, scor[g]);
            mmax = std::max(mmax, std::fabs(mnv[g]));
        }
        const float d = smax > 0 ? smax / 63.0f : 1e-8f;
        const float dmin = mmax > 0 ? mmax / 32.0f : 1e-8f;
        int scq[8], mq[8];
        for (int g = 0; g < 8; ++g) {
            scq[g] = int(std::lround(scor[g] / d));  scq[g] = std::max(0, std::min(63, scq[g]));
            mq[g]  = int(std::lround(mnv[g] / dmin)) + 32;
            mq[g]  = std::max(0, std::min(63, mq[g]));
        }
        // header
        std::uint8_t* h = hdr + sb * 16;
        const _Float16 dh = _Float16(d), dmh = _Float16(dmin);
        std::memcpy(h, &dh, 2);
        std::memcpy(h + 2, &dmh, 2);
        std::uint64_t lo = 0, hi = 0;
        auto poner = [&](int t, int v) {
            const int bit = 6 * t;
            const std::uint64_t val = std::uint64_t(v & 0x3F);
            if (bit >= 64) hi |= val << (bit - 64);
            else if (bit + 6 <= 64) lo |= val << bit;
            else { lo |= val << bit; hi |= val >> (64 - bit); }
        };
        for (int g = 0; g < 8; ++g) poner(g, scq[g]);
        for (int g = 0; g < 8; ++g) poner(8 + g, mq[g]);
        for (int i = 0; i < 8; ++i) h[4 + i] = std::uint8_t((lo >> (8 * i)) & 0xFF);
        for (int i = 0; i < 4; ++i) h[12 + i] = std::uint8_t((hi >> (8 * i)) & 0xFF);

        for (int g = 0; g < 8; ++g) {
            const float s = d * float(scq[g]);
            const float b = dmin * float(mq[g] - 32);
            for (int i = 0; i < 32; ++i) {
                const std::size_t idx = sb * 256 + std::size_t(g) * 32 + std::size_t(i);
                int q = (s > 0) ? int(std::lround((w[idx] - b) / s)) : 0;
                q = std::max(0, std::min(15, q));
                deq[idx] = s * float(q) + b;
                nib[idx] = std::uint8_t(q);
            }
        }
    }
}

}  // namespace

int main() {
    std::mt19937 rng(2026);
    std::normal_distribution<float> gauss(0.0f, 0.05f);
    int fallos = 0;
    for (std::size_t K : {1024u, 1536u, 2048u, 8960u}) {
        const std::size_t N = 19, M = 3;
        const std::size_t G = K / 32;

        std::vector<float> w(N * K);
        for (auto& v : w) v = gauss(rng) + 0.01f * float(rng() % 7);   // sesgo no nulo
        std::vector<std::uint8_t> nib(N * K);
        std::vector<std::uint8_t> hdr(N * (K / 256) * 16);
        std::vector<float> deq(N * K);
        for (std::size_t n = 0; n < N; ++n)
            kmin_pack(&w[n * K], K, &nib[n * K], &hdr[n * (K / 256) * 16],
                      deq.data() + n * K);

        // empaquetar nibbles de a dos
        std::vector<std::uint8_t> packed(N * (K / 2));
        for (std::size_t n = 0; n < N; ++n)
            for (std::size_t k = 0; k < K; k += 2)
                packed[n * (K / 2) + k / 2] =
                    std::uint8_t(nib[n * K + k] | (nib[n * K + k + 1] << 4));

        caudal::Shape s; s.N = N; s.K = K; s.group = 32; s.stride = 0;
        std::vector<std::int8_t> x(M * K), xp(M * K);
        for (auto& v : x) v = std::int8_t(int(rng() % 255) - 127);
        for (unsigned m = 0; m < M; ++m)
            for (std::size_t b = 0; b < K / 64; ++b)
                for (std::size_t i = 0; i < 64; ++i)
                    xp[m * K + b * 64 + i] = x[m * K + b * 64 + caudal::kPerm[i]];
        std::vector<std::int32_t> xg(M * G);
        std::vector<float> xs(M * G);
        for (auto& v : xs) v = float(rng() % 100) / 4000.0f + 0.001f;
        for (unsigned m = 0; m < M; ++m)
            caudal::compute_x_gsum(s, x.data() + m * K, xg.data() + m * G);

        std::vector<float> y(M * N);
        caudal::gemv_int4_kmin_M<M>(s, packed.data(), hdr.data(), xg.data(),
                                    xp.data(), xs.data(), y.data(), N);

        // referencia en double: Sum_g xscale_g * Sum_i deq*x  (agrupado, como
        // el kernel, para que el redondeo sea comparable)
        double peor = 0.0, mag = 0.0;
        for (unsigned m = 0; m < M; ++m)
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
                const double got = double(y[m * N + n]);
                peor = std::max(peor, std::fabs(got - acc));
                mag = std::max(mag, std::fabs(acc));
            }
        const bool ok = peor <= 1e-3 * mag + 1e-3;
        std::printf("  K=%5zu G=%3zu  N=%zu M=%zu  |y|=%.1f  err_max=%.3e  %s\n",
                    K, G, N, M, mag, peor, ok ? "OK" : "FALLA");
        if (!ok) ++fallos;
    }
    std::printf("RESULTADO: %d fallo(s)\n", fallos);
    return fallos != 0;
}
