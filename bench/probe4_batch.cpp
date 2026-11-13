// =============================================================================
//  caudal · bench/probe4_batch.cpp
//  LA CURVA QUE DECIDE: t(M)/t(1) sobre los pesos REALES del 1.5B.
// =============================================================================
//  Con IPC ~0,8 el núcleo está esperando memoria. Un GEMV con M>1 dequantiza
//  cada peso UNA vez y lo usa M veces: los bytes por token se parten en M.
//  Si t(M) crece mucho menos que M, el tiempo muerto se convierte en tokens.
//
//  Este probe corre los 197 tensores del artefacto con el kernel M genérico y
//  reporta, por cada M: ms por pasada, t(M)/t(1), y el tok/s PROYECTADO:
//
//      tok/s(M) = M / (t_pasada(M) + t_serial)
//
//  donde t_serial = 48,2 ms (token real) - t_pasada(1) = atención, RMSNorm,
//  SiLU y barreras, que NO se reparten entre las M filas.
//
//  Uso: ./probe4_batch <artefacto> [hilos] [reps]
// =============================================================================

#include "caudal/gemv_int4.hpp"

#include <algorithm>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <fstream>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

namespace {

constexpr double kRealToken_ms = 48.2;   // medido en produccion, 4 hilos

volatile std::uint64_t g_sink = 0;

double now_ms() {
    timespec t{};
    clock_gettime(CLOCK_MONOTONIC, &t);
    return double(t.tv_sec) * 1e3 + double(t.tv_nsec) * 1e-6;
}

struct Lin {
    std::size_t N, K, G, stride;
    const std::uint8_t* packed;
    const _Float16* scale;
};

inline void hsum8(__m256 v, float& out) {
    __m128 lo = _mm256_castps256_ps128(v);
    __m128 hi = _mm256_extractf128_ps(v, 1);
    __m128 s = _mm_add_ps(lo, hi);
    s = _mm_hadd_ps(s, s);
    s = _mm_hadd_ps(s, s);
    out = _mm_cvtss_f32(s);
}

// Kernel M genérico: el desempacado se hace UNA vez por bloque de 64 valores.
template <unsigned M>
void gemv_M(const Lin& L, const std::int32_t* xg, const std::int8_t* xp,
            const float* xs, float* y) {
    const __m256i m0f = _mm256_set1_epi8(0x0F);
    const __m256i ones = _mm256_set1_epi16(1);
    const std::size_t blocks = L.K / 64;
    const std::size_t rb = L.K / 2;

    for (std::size_t n = 0; n < L.N; ++n) {
        const std::uint8_t* row = L.packed + n * L.stride;
        const _Float16* rsc = L.scale ? L.scale + n * L.G
                                      : reinterpret_cast<const _Float16*>(row + rb);
        __m256 facc[M];
#pragma GCC unroll 8
        for (unsigned m = 0; m < M; ++m) facc[m] = _mm256_setzero_ps();

        for (std::size_t bb = 0; bb < blocks; ++bb) {
            const __m256i v = _mm256_loadu_si256(
                reinterpret_cast<const __m256i*>(row + bb * 32));
            const __m256i lo = _mm256_and_si256(v, m0f);
            const __m256i hi = _mm256_and_si256(_mm256_srli_epi16(v, 4), m0f);
            const __m256i a = _mm256_unpacklo_epi8(lo, hi);
            const __m256i b = _mm256_unpackhi_epi8(lo, hi);
            const std::size_t g0 = 2 * bb, g1 = 2 * bb + 1;

#pragma GCC unroll 8
            for (unsigned m = 0; m < M; ++m) {
                const std::int8_t* xm = xp + std::size_t(m) * L.K + bb * 64;
                const __m256i x0 = _mm256_loadu_si256(
                    reinterpret_cast<const __m256i*>(xm));
                const __m256i x1 = _mm256_loadu_si256(
                    reinterpret_cast<const __m256i*>(xm + 32));
                __m256i acc = _mm256_madd_epi16(_mm256_maddubs_epi16(a, x0), ones);
                acc = _mm256_add_epi32(
                    acc, _mm256_madd_epi16(_mm256_maddubs_epi16(b, x1), ones));
                acc = _mm256_sub_epi32(acc, _mm256_setr_epi32(
                    8 * xg[std::size_t(m) * L.G + g0], 0, 0, 0,
                    8 * xg[std::size_t(m) * L.G + g1], 0, 0, 0));
                const __m256 sv = _mm256_set_m128(
                    _mm_set1_ps(static_cast<float>(rsc[g1]) * xs[m]),
                    _mm_set1_ps(static_cast<float>(rsc[g0]) * xs[m]));
                facc[m] = _mm256_fmadd_ps(_mm256_cvtepi32_ps(acc), sv, facc[m]);
            }
        }
#pragma GCC unroll 8
        for (unsigned m = 0; m < M; ++m) {
            float o;
            hsum8(facc[m], o);
            y[std::size_t(m) * L.N + n] = o;
        }
    }
}


}  // namespace

int main(int argc, char** argv) {
    if (argc < 2) { std::printf("uso: %s <artefacto> [hilos] [reps]\n", argv[0]); return 1; }
    const std::string A = argv[1];
    const unsigned T = (argc > 2) ? std::atoi(argv[2]) : 4;
    const int reps = (argc > 3) ? std::atoi(argv[3]) : 3;

    auto wb = [&] { std::ifstream f(A + "/w.bin", std::ios::binary | std::ios::ate);
                    if (f) { auto n = (std::size_t)f.tellg(); f.seekg(0);
                             std::vector<std::uint8_t> v(n);
                             f.read((char*)v.data(), (std::streamsize)n); return v; }
                    return std::vector<std::uint8_t>{}; }();
    if (wb.empty()) { std::printf("necesito el artefacto intercalado (w.bin)\n"); return 1; }

    std::vector<Lin> lins;
    {
        std::ifstream fi(A + "/index.txt");
        std::string kind;
        while (fi >> kind) {
            if (kind != "LINEAR") { std::string rest; std::getline(fi, rest); continue; }
            std::string rest; std::getline(fi, rest);
            std::istringstream is(rest);
            std::string name; is >> name;
            std::vector<std::size_t> v; std::size_t x;
            while (is >> x) v.push_back(x);
            if (v.size() < 5) continue;
            Lin L; L.N = v[0]; L.K = v[1]; L.stride = v[3]; L.G = v[4];
            L.packed = wb.data() + v[2];
            L.scale = nullptr;
            lins.push_back(L);
        }
    }

    constexpr unsigned kMaxM = 8;
    // activaciones sintéticas: M vectores de K por tensor (el mayor K)
    std::size_t maxK = 0;
    for (auto& L : lins) maxK = std::max(maxK, L.K);
    std::size_t maxG = 0;
    for (auto& L : lins) maxG = std::max(maxG, L.G);
    std::vector<std::int8_t> xp(kMaxM * maxK);
    std::vector<std::int32_t> xg(kMaxM * maxG);
    std::vector<float> xs(kMaxM);
    for (std::size_t i = 0; i < xp.size(); ++i) xp[i] = std::int8_t(-60 + (i % 120));
    for (std::size_t i = 0; i < xg.size(); ++i) xg[i] = std::int32_t((i % 41) - 20);
    for (unsigned m = 0; m < kMaxM; ++m) xs[m] = 0.02f;

    std::size_t maxN = 0;
    for (auto& L : lins) maxN = std::max(maxN, L.N);
    std::vector<float> y(kMaxM * maxN);

    std::printf("\ncaudal · probe4_batch   tensores=%zu  hilos=%u  reps=%d\n\n",
                lins.size(), T, reps);
    std::printf("%3s %12s %12s %10s %12s\n",
                "M", "ms/pasada", "t(M)/t(1)", "tok/s proy", "GB/s efec");

    // Buffer de salida POR HILO, asignado una sola vez: dentro del bucle eran
    // 788 mallocs por pasada y eso dominaba la medicion de M=1.
    std::vector<std::vector<float>> ybuf(T, std::vector<float>(kMaxM * maxN));

    auto run_all = [&](unsigned M) {
        auto once = [&]() {
            std::vector<std::thread> th;
            for (unsigned t = 0; t < T; ++t) {
                th.emplace_back([&, t] {
                    float* yy = ybuf[t].data();
                    for (std::size_t i = 0; i < lins.size(); ++i) {
                        const auto& L = lins[i];
                        const std::size_t n0 = std::size_t(t) * L.N / T;
                        const std::size_t n1 = std::size_t(t + 1) * L.N / T;
                        if (n1 <= n0) continue;
                        Lin sub = L; sub.N = n1 - n0;
                        sub.packed = L.packed + n0 * L.stride;
                        switch (M) {
                            case 1: gemv_M<1>(sub, xg.data(), xp.data(), xs.data(), yy); break;
                            case 2: gemv_M<2>(sub, xg.data(), xp.data(), xs.data(), yy); break;
                            case 4: gemv_M<4>(sub, xg.data(), xp.data(), xs.data(), yy); break;
                            default: gemv_M<8>(sub, xg.data(), xp.data(), xs.data(), yy); break;
                        }
                        g_sink ^= std::uint64_t(yy[0] * 1000.0f);
                    }
                });
            }
            for (auto& x : th) x.join();
        };
        once();                     // warm-up
        const double t0 = now_ms();
        for (int r = 0; r < reps; ++r) once();
        return (now_ms() - t0) / reps;
    };

    double t1 = 0;
    std::size_t bytes = 0;
    for (auto& L : lins) bytes += L.N * L.stride;

    for (unsigned M : {1u, 2u, 4u, 8u}) {
        const double ms = run_all(M);
        if (M == 1) t1 = ms;
        const double serial = kRealToken_ms - t1;
        const double tps = 1000.0 * M / (ms + std::max(0.0, serial));
        std::printf("%3u %12.2f %12.2f %12.1f %12.1f\n",
                    M, ms, ms / t1, tps, double(bytes) * M / (ms * 1e6));
    }

    std::printf("\nnota: GB/s efec = bytes x M / tiempo -> si NO crece con M, los\n");
    std::printf("      bytes por token se estan amortizando (eso es lo que buscamos).\n\n");
    return 0;
}
