// =============================================================================
//  caudal · bench/probe1_gemv.cpp   —   EL GO/NO-GO DEL KERNEL
// =============================================================================
//  Pregunta única: ¿puede un GEMV int4 escrito a mano acercarse al ancho de
//  banda de DRAM medido por probe0? Si no, no hay diseño: hay que arreglar el
//  kernel antes de escribir una línea más.
//
//  ── Advertencia de ISA (medida, no supuesta) ────────────────────────────────
//  En el i7-10510U (Comet Lake) NO hay AVX-512 ni VNNI: sólo AVX2 + FMA.
//  O sea: 16 registros ymm, no 32 zmm; y nada de vpdpbusd. El "tile 14×32 con
//  28 acumuladores zmm" de la spec es IMPOSIBLE acá. Este kernel está escrito
//  para el ISA que existe: vpshufb (LUT) para desempacar 4 bits y
//  vpmaddubsw + vpmaddwd para el MAC entero.
//
//  ── Tres medidas, porque una sola no es interpretable ───────────────────────
//      stream : lee el MISMO buffer y sólo suma bytes. Es el TECHO del layout.
//      int4   : el kernel real (desempaca + MAC entero).
//  La diferencia entre ambas es el costo de desempacar. Sin el techo, un
//  "11 GB/s" no dice si el layout es malo o el kernel es malo.
//
//  ── Qué NO mide (honestidad de contrato) ────────────────────────────────────
//    · Escalas por fila, no por grupo. El layout real (grupo 64) cuesta ~1
//      multiplicación extra por cada 64 valores. Este probe es más generoso.
//    · No corrige el zero-point de x (bias 128·Σw). Es un probe de velocidad,
//      no de correctitud numérica.
//    · M > 4 en AVX2 presiona los 16 registros; se mide y se reporta.
//
//  Uso:
//      ./probe1_gemv [passes]
// =============================================================================

#include "caudal/layout.hpp"

#include <algorithm>
#include <atomic>
#include <bit>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <thread>
#include <vector>

#include <immintrin.h>

namespace {

constexpr std::size_t kMiB = std::size_t{1} << 20;

// Forma sintética: una rebanada realista de la FFN de Qwen2.5-7B.
//   K = hidden de Qwen2.5-7B
//   N elegido para ~117 MB empaquetado (14x la L3 de 8 MiB => DRAM)
constexpr std::size_t K = 3584;
constexpr std::size_t N = 65536;
constexpr std::size_t kRowBytes = K / 2;            // 2 valores int4 por byte
constexpr std::size_t kBytes = N * kRowBytes;

static_assert(K % 64 == 0, "el bucle desempaca bloques de 64 valores");
static_assert(kBytes > (64 * kMiB), "el buffer tiene que exceder la L3");

volatile std::uint64_t g_sink = 0;

double now_s() {
    timespec ts{};
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return static_cast<double>(ts.tv_sec) + static_cast<double>(ts.tv_nsec) * 1e-9;
}

// -----------------------------------------------------------------------------
//  Techo: leer el buffer y no hacer nada útil. Mismo patrón de acceso exacto.
// -----------------------------------------------------------------------------
#if defined(__AVX2__)
[[gnu::noinline]] std::uint64_t stream_avx2(const std::uint8_t* p, std::size_t n) {
    __m256i a0 = _mm256_setzero_si256(), a1 = _mm256_setzero_si256();
    __m256i a2 = _mm256_setzero_si256(), a3 = _mm256_setzero_si256();
    std::size_t i = 0;
    for (; i + 128 <= n; i += 128) {
        a0 = _mm256_add_epi64(a0, _mm256_loadu_si256(reinterpret_cast<const __m256i*>(p + i)));
        a1 = _mm256_add_epi64(a1, _mm256_loadu_si256(reinterpret_cast<const __m256i*>(p + i + 32)));
        a2 = _mm256_add_epi64(a2, _mm256_loadu_si256(reinterpret_cast<const __m256i*>(p + i + 64)));
        a3 = _mm256_add_epi64(a3, _mm256_loadu_si256(reinterpret_cast<const __m256i*>(p + i + 96)));
    }
    a0 = _mm256_add_epi64(_mm256_add_epi64(a0, a1), _mm256_add_epi64(a2, a3));
    alignas(32) std::uint64_t lanes[4];
    _mm256_store_si256(reinterpret_cast<__m256i*>(lanes), a0);
    std::uint64_t s = lanes[0] + lanes[1] + lanes[2] + lanes[3];
    for (; i < n; ++i) s += p[i];
    std::atomic_signal_fence(std::memory_order_seq_cst);
    return s;
}
#endif

// -----------------------------------------------------------------------------
//  GEMV/GELM int4: M vectores x contra las N filas de W.
//  Acumulación VERTICAL en int32; una sola reducción horizontal por fila.
// -----------------------------------------------------------------------------
#if defined(__AVX2__)
struct Kernel {
    // LUT: nibble 0..15  ->  int8 -8..+7  (vpshufb exige índice = nibble bajo)
    __m256i lut;
    __m256i m0f;
    __m256i ones;

    Kernel() {
        lut = _mm256_setr_epi8(
            -8, -7, -6, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6, 7,
            -8, -7, -6, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6, 7);
        m0f = _mm256_set1_epi8(0x0F);
        ones = _mm256_set1_epi16(1);
    }

    // 32 bytes empaquetados (64 int4) -> w0 = v[0..31], w1 = v[32..63], int8.
    // Con Perm=true se reordena a orden-k (correcto). Con Perm=false se dejan
    // las mitades tal cual las produce unpacklo/unpackhi: NUMÉRICAMENTE
    // INCORRECTO, pero sirve para medir cuánto cuestan los 2 vperm2i128.
    template <bool Perm>
    inline void unpack(const std::uint8_t* p, __m256i& w0, __m256i& w1) const {
        const __m256i v = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(p));
        const __m256i lo = _mm256_and_si256(v, m0f);
        const __m256i hi = _mm256_and_si256(_mm256_srli_epi16(v, 4), m0f);
        const __m256i wl = _mm256_shuffle_epi8(lut, lo);
        const __m256i wh = _mm256_shuffle_epi8(lut, hi);
        const __m256i a = _mm256_unpacklo_epi8(wl, wh);  // [v0..15 | v32..47]
        const __m256i b = _mm256_unpackhi_epi8(wl, wh);  // [v16..31| v48..63]
        if constexpr (Perm) {
            w0 = _mm256_permute2x128_si256(a, b, 0x20);  // [v0..31]
            w1 = _mm256_permute2x128_si256(a, b, 0x31);  // [v32..63]
        } else {
            w0 = a;
            w1 = b;
        }
    }

    // M acumuladores vivos. Cada fila: 64 valores por iteración.
    // n_begin/n_end permiten particionar por filas de salida: cada hilo lee su
    // propia franja contigua de pesos y no hay reducción entre hilos.
    template <unsigned M, bool Perm = true>
    [[gnu::noinline]] void run(const std::uint8_t* __restrict w,
                               const std::uint8_t* __restrict x,  // M x K, uint8
                               const float* __restrict scale,
                               float* __restrict y,
                               std::size_t n_begin = 0,
                               std::size_t n_end = N) const {
        for (std::size_t n = n_begin; n < n_end; ++n) {
            const std::uint8_t* row = w + n * kRowBytes;
            __m256i acc[M];
#pragma GCC unroll 8
            for (unsigned m = 0; m < M; ++m) acc[m] = _mm256_setzero_si256();

            for (std::size_t k = 0; k < K; k += 64) {
                __m256i w0, w1;
                unpack<Perm>(row + k / 2, w0, w1);
#pragma GCC unroll 8
                for (unsigned m = 0; m < M; ++m) {
                    const std::uint8_t* xm = x + static_cast<std::size_t>(m) * K + k;
                    const __m256i x0 = _mm256_loadu_si256(
                        reinterpret_cast<const __m256i*>(xm));
                    const __m256i x1 = _mm256_loadu_si256(
                        reinterpret_cast<const __m256i*>(xm + 32));
                    const __m256i p0 = _mm256_maddubs_epi16(x0, w0);  // u8 * s8
                    const __m256i p1 = _mm256_maddubs_epi16(x1, w1);
                    acc[m] = _mm256_add_epi32(acc[m], _mm256_madd_epi16(p0, ones));
                    acc[m] = _mm256_add_epi32(acc[m], _mm256_madd_epi16(p1, ones));
                }
            }

            const float sx = scale[n];
#pragma GCC unroll 8
            for (unsigned m = 0; m < M; ++m) {
                alignas(32) std::int32_t lanes[8];
                _mm256_store_si256(reinterpret_cast<__m256i*>(lanes), acc[m]);
                const std::int32_t dot = lanes[0] + lanes[1] + lanes[2] + lanes[3]
                                       + lanes[4] + lanes[5] + lanes[6] + lanes[7];
                y[static_cast<std::size_t>(m) * N + n] = static_cast<float>(dot) * sx;
            }
        }
        std::atomic_signal_fence(std::memory_order_seq_cst);
    }
};
#endif

double gbs(std::uint64_t bytes, double s) {
    return static_cast<double>(bytes) / s / 1e9;
}

}  // namespace

int main(int argc, char** argv) {
    int passes = 10;
    if (argc > 1) passes = std::max(1, std::atoi(argv[1]));

    std::printf("\n");
    std::printf("caudal · probe1_gemv\n");
    std::printf("forma             : N=%zu  K=%zu  (%.1f M params)\n",
                N, K, static_cast<double>(N * K) / 1e6);
    std::printf("buffer int4       : %.1f MiB  (%.1fx la L3 de 8 MiB)\n",
                static_cast<double>(kBytes) / static_cast<double>(kMiB),
                static_cast<double>(kBytes) / static_cast<double>(8 * kMiB));
    std::printf("passes            : %d\n", passes);
    std::printf("ISA               : ");
#if defined(__AVX512F__)
    std::printf("AVX-512");
#elif defined(__AVX2__)
    std::printf("AVX2 + FMA  (sin AVX-512, sin VNNI en este equipo)");
#else
    std::printf("escalar (sin AVX2)");
#endif
    std::printf("\n\n");

#if !defined(__AVX2__)
    std::printf("Este probe necesita AVX2. Recompilar con -mavx2.\n");
    return 1;
#else
    // --- Reserva y población --------------------------------------------------
    auto* w = static_cast<std::uint8_t*>(
        std::aligned_alloc(64, (kBytes + 63) & ~std::size_t{63}));
    auto* scale = static_cast<float*>(std::aligned_alloc(64, N * sizeof(float)));
    if (!w || !scale) { std::printf("sin memoria\n"); return 1; }

    // Contenido determinista (no ceros: queremos trabajo real, no página-cero).
    std::uint32_t s = 0x9E3779B9u;
    for (std::size_t i = 0; i < kBytes; ++i) {
        s = s * 1664525u + 1013904223u;
        w[i] = static_cast<std::uint8_t>(s >> 24);
    }
    for (std::size_t n = 0; n < N; ++n) scale[n] = 0.0007f;

    constexpr unsigned kMaxM = 8;
    std::vector<std::uint8_t> x(kMaxM * K);
    for (std::size_t i = 0; i < x.size(); ++i)
        x[i] = static_cast<std::uint8_t>(100 + (i % 56));   // u8, sin ceros

    std::vector<float> y(static_cast<std::size_t>(kMaxM) * N);

    Kernel k;

    // --- Techo: stream --------------------------------------------------------
    auto bench_stream = [&]() -> double {
        (void)stream_avx2(w, kBytes);                       // warm-up
        const double t0 = now_s();
        std::uint64_t acc = 0;
        for (int p = 0; p < passes; ++p) acc ^= stream_avx2(w, kBytes);
        const double t1 = now_s();
        g_sink ^= acc;
        return gbs(static_cast<std::uint64_t>(kBytes) * passes, t1 - t0);
    };

    struct R { double gbs; double gmacs; };

    auto bench_i4 = [&](unsigned M, bool perm) -> R {
        auto once = [&]() {
            switch ((M << 1) | (perm ? 1u : 0u)) {
                case 2:  k.run<1, true >(w, x.data(), scale, y.data()); break;
                case 3:  k.run<1, false>(w, x.data(), scale, y.data()); break;
                case 4:  k.run<2, true >(w, x.data(), scale, y.data()); break;
                case 5:  k.run<2, false>(w, x.data(), scale, y.data()); break;
                case 8:  k.run<4, true >(w, x.data(), scale, y.data()); break;
                case 9:  k.run<4, false>(w, x.data(), scale, y.data()); break;
                case 16: k.run<8, true >(w, x.data(), scale, y.data()); break;
                case 17: k.run<8, false>(w, x.data(), scale, y.data()); break;
                default: break;
            }
        };
        once();                                             // warm-up
        const double t0 = now_s();
        for (int p = 0; p < passes; ++p) once();
        const double t1 = now_s();
        g_sink ^= static_cast<std::uint64_t>(y[M * N - 1] * 1000.0f);

        const double secs = t1 - t0;
        const double macs = static_cast<double>(N) * static_cast<double>(K)
                          * static_cast<double>(M) * static_cast<double>(passes);
        return {gbs(static_cast<std::uint64_t>(kBytes) * passes, secs), macs / secs / 1e9};
    };

    const double bw_stream = bench_stream();

    std::printf("%-9s %3s %9s %12s %11s %9s\n",
                "kernel", "M", "GB/s", "% del stream", "% de 18.0", "GMAC/s");
    std::printf("%-9s %3d %9.1f %11.1f%% %10.1f%% %9s\n",
                "stream", 0, bw_stream, 100.0, 100.0 * bw_stream / 18.0, "-");

    // int4-np: diagnóstico. Quita los 2 vperm2i128 (resultado INCORRECTO);
    // mide cuánto del gate se pierde en el puerto de shuffle.
    {
        const R r = bench_i4(1, false);
        std::printf("%-9s %3d %9.1f %11.1f%% %10.1f%% %9.1f   <- solo timing\n",
                    "int4-np", 1, r.gbs, 100.0 * r.gbs / bw_stream,
                    100.0 * r.gbs / 18.0, r.gmacs);
    }

    const unsigned Ms[] = {1, 2, 4, 8};
    double best_m1 = 0.0;
    for (unsigned M : Ms) {
        const R r = bench_i4(M, true);
        if (M == 1) best_m1 = r.gbs;
        std::printf("%-9s %3u %9.1f %11.1f%% %10.1f%% %9.1f\n",
                    "int4", M, r.gbs, 100.0 * r.gbs / bw_stream,
                    100.0 * r.gbs / 18.0, r.gmacs);
    }

    // --- Agregado multi-hilo (M=1): partición por filas de salida -----------
    auto bench_stream_mt = [&](unsigned T) -> double {
        if (T == 1) return bw_stream;
        auto once = [&]() {
            std::vector<std::thread> th;
            th.reserve(T);
            const std::size_t chunk = (kBytes / T) & ~std::size_t{127};
            for (unsigned t = 0; t < T; ++t) {
                const std::size_t b = static_cast<std::size_t>(t) * chunk;
                const std::size_t e = (t + 1 == T) ? kBytes : b + chunk;
                th.emplace_back([&, b, e] { (void)stream_avx2(w + b, e - b); });
            }
            for (auto& t : th) t.join();
        };
        once();
        const double t0 = now_s();
        for (int p = 0; p < passes; ++p) once();
        const double t1 = now_s();
        return gbs(static_cast<std::uint64_t>(kBytes) * passes, t1 - t0);
    };

    auto bench_mt = [&](unsigned T) -> double {
        if (T == 1) return best_m1;
        auto once = [&]() {
            std::vector<std::thread> th;
            th.reserve(T);
            const std::size_t chunk = (N / T) & ~std::size_t{15};
            for (unsigned t = 0; t < T; ++t) {
                const std::size_t b = static_cast<std::size_t>(t) * chunk;
                const std::size_t e = (t + 1 == T) ? N : b + chunk;
                th.emplace_back([&, b, e] { k.run<1>(w, x.data(), scale, y.data(), b, e); });
            }
            for (auto& t : th) t.join();
        };
        once();
        const double t0 = now_s();
        for (int p = 0; p < passes; ++p) once();
        const double t1 = now_s();
        return gbs(static_cast<std::uint64_t>(kBytes) * passes, t1 - t0);
    };

    std::printf("\n%-9s %5s %9s %10s %8s %10s\n",
                "kernel", "hilos", "GB/s", "stream_mt", "efic.", "% de 18.0");
    unsigned maxT = std::thread::hardware_concurrency();
    if (!maxT) maxT = 1;
    double best_mt = 0.0;
    unsigned best_mt_t = 0;
    for (unsigned T = 1; T <= maxT; T <<= 1) {
        const double s = bench_stream_mt(T);   // mismo buffer, mismos hilos
        const double g = bench_mt(T);
        if (g > best_mt) { best_mt = g; best_mt_t = T; }
        std::printf("%-9s %5u %9.1f %10.1f %7.1f%% %9.1f%%\n",
                    "int4 M=1", T, g, s, 100.0 * g / s, 100.0 * g / 18.0);
    }
    constexpr double kGate = 10.8;      // 60 % de 18.0 GB/s
    std::printf("\n");
    std::printf("caveat: la columna 'efic.' sólo es válida a 1-2 hilos. A partir de 4,\n");
    std::printf("        el stream de referencia NO escala (el presupuesto de 15 W baja\n");
    std::printf("        el reloj al activar núcleos) mientras el GEMV sí: >100%% es un\n");
    std::printf("        artefacto del denominador, no una violación de la física.\n");
    std::printf("\n");
    std::printf("GATE (M=1, 60%% de 18.0 GB/s) : %.1f GB/s  ->  ", kGate);
    if (best_m1 >= kGate) std::printf("PASA\n");
    else                  std::printf("NO PASA (%.1f GB/s, %.0f%% del gate)\n",
                                      best_m1, 100.0 * best_m1 / kGate);

    std::printf("eficiencia sobre el techo     : %.1f%%  (el costo de desempacar)\n",
                100.0 * best_m1 / bw_stream);
    std::printf("agregado multi-hilo (M=1)     : %.1f GB/s con %u hilos  (%.1f%% de 18.0)\n",
                best_mt, best_mt_t, 100.0 * best_mt / 18.0);
    std::printf("\n");
    std::printf("tok/s de un 7B int4 (4.28 GB) : %.2f  (a %.1f GB/s efectivos)\n",
                best_mt / 4.28, best_mt);
    std::printf("\n");
    std::printf("nota: si el GB/s cae con M pero el GMAC/s sube, el lote compra\n");
    std::printf("      COMPUTO, no ancho de banda. En esta máquina DRAM es tan\n");
    std::printf("      lenta que con hilos ya se está limitado por memoria en M=1:\n");
    std::printf("      el lote no es la palanca, leer menos bytes lo es.\n\n");

    std::free(w);
    std::free(scale);
    return 0;
#endif
}
