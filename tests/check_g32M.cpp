// check_g32M.cpp — verifica gemv_int4_g32_M<M> contra la referencia escalar
// con la MISMA data empaquetada. Nace porque el roundtrip del repo cubre el
// kernel de grupo 64 y no el _M, que es el que usa el motor.
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <random>
#include <vector>

#include "caudal/gemv_int4.hpp"

int main() {
    std::mt19937 rng(12345);
    int fallos = 0;
    for (std::size_t K : {512u, 2048u, 1536u, 8192u}) {
        const std::size_t G = K / 32;         // grupo 32
        const std::size_t N = 37;             // N impar a proposito
        caudal::Shape s;
        s.N = N; s.K = K; s.group = 32; s.stride = 0;
        const std::size_t rb = s.row_bytes();

        std::vector<std::uint8_t> w(N * rb);
        for (auto& b : w) b = std::uint8_t(rng() & 0xFF);
        std::vector<_Float16> sc(N * G);
        for (auto& v : sc) v = _Float16(float(rng() % 1000) / 1000.0f + 0.001f);

        const unsigned M = 3;
        std::vector<std::int8_t> x(M * K), xp(M * K);
        for (auto& v : x) v = std::int8_t(int(rng() % 255) - 127);
        std::vector<std::int32_t> xg(M * G);
        std::vector<float> xs(M * G);
        for (auto& v : xs) v = float(rng() % 1000) / 5000.0f + 0.001f;

        // permutar x por bloques de 64 (como el motor)
        for (unsigned m = 0; m < M; ++m)
            for (std::size_t b = 0; b < K / 64; ++b)
                for (std::size_t i = 0; i < 64; ++i)
                    xp[m * K + b * 64 + i] = x[m * K + b * 64 + caudal::kPerm[i]];

        for (unsigned m = 0; m < M; ++m) caudal::compute_x_gsum(s, x.data() + m * K, xg.data() + m * G);

        std::vector<float> y_new(M * N), y_ref(M * N);
        switch (M) {
            case 3: caudal::gemv_int4_g32_M<3>(s, w.data(), sc.data(), xg.data(),
                                               xp.data(), xs.data(), y_new.data(), N); break;
            default: break;
        }
        // referencia: una fila a la vez con el kernel escalar de grupo 32
        for (unsigned m = 0; m < M; ++m) {
            caudal::gemv_int4_g32(s, w.data(), sc.data(), xg.data() + m * G,
                                  xp.data() + m * K, xs.data() + m * G,
                                  y_ref.data() + m * N);
        }

        double peor = 0.0, mag = 0.0;
        for (int i = 0; i < M * int(N); ++i) {
            peor = std::max(peor, std::fabs(double(y_new[i]) - double(y_ref[i])));
            mag = std::max(mag, std::fabs(double(y_ref[i])));
        }
        const bool ok = peor <= 1e-3 * mag + 1e-4;
        std::printf("  K=%5zu G=%3zu  M=%u  |y|=%.3f  err_max=%.2e  %s\n",
                    K, G, M, mag, peor, ok ? "OK" : "FALLA");
        if (!ok) ++fallos;
    }
    std::printf("RESULTADO: %d fallo(s)\n", fallos);
    return fallos != 0;
}
