#ifndef CAUDAL_GEMV_INT4_HPP
#define CAUDAL_GEMV_INT4_HPP

// =============================================================================
//  caudal · include/caudal/gemv_int4.hpp
//  Kernel de producción: int4 SIN LUT, escalas por grupo, x pre-permutado.
// =============================================================================
//  ── La inversión de operandos (por qué este kernel es distinto) ─────────────
//  vpmaddubsw exige que el PRIMER operando sea unsigned y el segundo signed.
//
//  Lo obvio es poner x en unsigned (offset 128) y w en signed int4, pero eso
//  obliga a:
//      · una LUT vpshufb para llevar el nibble two's-complement a int8 con
//        signo  (2 ops de puerto de shuffle por cada 32 B), y
//      · una tabla w_codesum en memoria para corregir el sesgo del offset
//        (+0.5 bits por parámetro, y por lo tanto ~12 % más tráfico de DRAM).
//
//  Invirtiendo:
//      w = code + 8     (nibble 0..15, YA es unsigned: no hay nada que mapear)
//      x = int8 con signo, zero-point 0
//
//      Σ (code+8)·x = Σ code·x + 8·Σx   ⇒   Σ code·x = dot − 8·Σx
//
//  `Σx` por grupo depende del TOKEN, no del modelo: se calcula una vez por
//  token (56 enteros) y no cuesta ni un byte de artefacto. Desaparecen la LUT
//  y `w_codesum`. Presupuesto:
//      4.75 -> 4.25 bits/param   (−10.5 % de tráfico)
//      4 -> 2   ops de shuffle por 32 B  (−50 % en el recurso escaso de AVX2)
//
//  ── Pre-permutación de x ────────────────────────────────────────────────────
//  vpunpcklbw/vpunpckhbw entregan las mitades como
//      a = [v0..15, v32..47]   b = [v16..31, v48..63]
//  Si el packer guarda x en ese mismo orden (kPerm), el bucle empareja directo
//  y no gasta ningún vperm2i128.
//
//  ── Sesgo ───────────────────────────────────────────────────────────────────
//  `8·Σx` se resta UNA vez, en el carril 0 del acumulador entero. Restarlo con
//  set1() lo aplicaría a los 8 carriles y la suma horizontal lo multiplicaría
//  por 8 (bug real que encontró test_pack_roundtrip).
// =============================================================================

#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>

#if defined(__x86_64__) || defined(_M_X64)
#  include <immintrin.h>
#elif defined(__ARM_NEON) || defined(__aarch64__)
#  include <arm_neon.h>
#endif

namespace caudal {

struct Shape {
    std::size_t N{0};      // filas de salida
    std::size_t K{0};      // profundidad; múltiplo de 64
    std::size_t group{64}; // valores por escala
    // Layout INTERCALADO: cada fila es [pesos K/2][escalas groups*2], y las
    // escalas se leen de row + row_bytes(). Medido: 98.7% del techo continuo
    // contra 90.4% del layout de dos arrays (probe3_ceiling).
    std::size_t stride{0};  // 0 = layout antiguo (dos arrays separados)

    [[nodiscard]] constexpr std::size_t groups() const noexcept { return K / group; }
    [[nodiscard]] constexpr std::size_t row_bytes() const noexcept { return K / 2; }
    [[nodiscard]] constexpr std::size_t row_stride() const noexcept {
        return stride ? stride : row_bytes();
    }
};

// Permutación dentro de cada bloque de 64. Orden natural de vpunpcklbw/hbw.
inline constexpr auto kPerm = [] {
    std::array<std::uint8_t, 64> p{};
    for (int i = 0; i < 16; ++i) {
        p[i]      = static_cast<std::uint8_t>(i);
        p[16 + i] = static_cast<std::uint8_t>(32 + i);
        p[32 + i] = static_cast<std::uint8_t>(16 + i);
        p[48 + i] = static_cast<std::uint8_t>(48 + i);
    }
    return p;
}();

// Nibble empaquetado, tal cual: 0..15  (= code + 8).
[[nodiscard]] inline int nibble_at(const std::uint8_t* row, std::size_t j) noexcept {
    const std::uint8_t b = row[j >> 1];
    return (j & 1u) ? static_cast<int>(b >> 4) : static_cast<int>(b & 0x0Fu);
}

// Código int4 con signo: -8..7.
[[nodiscard]] inline int code_at(const std::uint8_t* row, std::size_t j) noexcept {
    return nibble_at(row, j) - 8;
}

// Σx por grupo, un vector por token. Se calcula UNA vez por token, no por fila.
// Al ser la permutación intra-grupo, el orden no altera la suma.
inline void compute_x_gsum(const Shape& s, const std::int8_t* __restrict x,
                           std::int32_t* __restrict out) {
    for (std::size_t g = 0; g < s.groups(); ++g) {
        std::int32_t acc = 0;
        for (std::size_t j = 0; j < s.group; ++j) acc += x[g * s.group + j];
        out[g] = acc;
    }
}

// -----------------------------------------------------------------------------
//  Referencia escalar. Lee los MISMOS bytes empaquetados.
// -----------------------------------------------------------------------------
inline void gemv_ref(const Shape& s,
                     const std::uint8_t* __restrict w,
                     const _Float16* __restrict wscale,
                     const std::int8_t* __restrict xperm,
                     const float* __restrict xscale,   // una por grupo
                     float* __restrict y) {
    const std::size_t G = s.groups(), rb = s.row_bytes();
    for (std::size_t n = 0; n < s.N; ++n) {
        const std::uint8_t* row = w + n * rb;
        double acc = 0.0;
        for (std::size_t g = 0; g < G; ++g) {
            const std::size_t base = g * s.group;
            long long dot = 0;
            for (std::size_t j = 0; j < s.group; ++j) {
                // x_perm[j] corresponde al code kPerm[j], no al code j.
                const int code = code_at(row, base + kPerm[j]);
                dot += static_cast<long long>(code)
                     * static_cast<int>(xperm[base + j]);
            }
            acc += static_cast<double>(dot)
                 * static_cast<double>(wscale[n * G + g])
                 * static_cast<double>(xscale[g]);
        }
        y[n] = static_cast<float>(acc);
    }
}

// -----------------------------------------------------------------------------
//  Kernel AVX2. Cero LUT, cero permutaciones en el bucle.
// -----------------------------------------------------------------------------
#if defined(__AVX2__)
inline void gemv_int4(const Shape& s,
                      const std::uint8_t* __restrict w,
                      const _Float16* __restrict wscale,
                      const std::int32_t* __restrict xgsum,
                      const std::int8_t* __restrict xperm,
                      const float* __restrict xscale,   // una por grupo
                      float* __restrict y) {
    const __m256i m0f = _mm256_set1_epi8(0x0F);
    const __m256i ones16 = _mm256_set1_epi16(1);

    const std::size_t G = s.groups(), rb = s.row_bytes();
    const std::size_t chunk = s.group / 2;   // 32 B por grupo de 64 valores

    for (std::size_t n = 0; n < s.N; ++n) {
        const std::uint8_t* row = w + n * s.row_stride();
        const _Float16* sc = wscale ? wscale + n * G
                                    : reinterpret_cast<const _Float16*>(row + rb);
        __m256 facc = _mm256_setzero_ps();

        for (std::size_t g = 0; g < G; ++g) {
            const __m256i v = _mm256_loadu_si256(
                reinterpret_cast<const __m256i*>(row + g * chunk));
            // El nibble YA es el valor unsigned 0..15: no hace falta LUT.
            const __m256i lo = _mm256_and_si256(v, m0f);
            const __m256i hi = _mm256_and_si256(_mm256_srli_epi16(v, 4), m0f);

            // Sin vperm2i128: x viene en este orden exacto.
            const __m256i a = _mm256_unpacklo_epi8(lo, hi);  // [v0..15 | v32..47]
            const __m256i b = _mm256_unpackhi_epi8(lo, hi);  // [v16..31| v48..63]

            const std::int8_t* xg = xperm + g * s.group;
            const __m256i x0 = _mm256_loadu_si256(
                reinterpret_cast<const __m256i*>(xg));
            const __m256i x1 = _mm256_loadu_si256(
                reinterpret_cast<const __m256i*>(xg + 32));

            // Primer operando unsigned (w), segundo signed (x).
            __m256i acc = _mm256_madd_epi16(_mm256_maddubs_epi16(a, x0), ones16);
            acc = _mm256_add_epi32(
                acc, _mm256_madd_epi16(_mm256_maddubs_epi16(b, x1), ones16));

            // Σcode·x = dot − 8·Σx. Se resta UNA vez, en el carril 0.
            acc = _mm256_sub_epi32(
                acc, _mm256_setr_epi32(8 * xgsum[g], 0, 0, 0, 0, 0, 0, 0));

            const float fs = static_cast<float>(sc[g]) * xscale[g];
            facc = _mm256_fmadd_ps(_mm256_cvtepi32_ps(acc), _mm256_set1_ps(fs), facc);
        }

        __m128 lo128 = _mm256_castps256_ps128(facc);
        __m128 hi128 = _mm256_extractf128_ps(facc, 1);
        __m128 sum = _mm_add_ps(lo128, hi128);
        sum = _mm_hadd_ps(sum, sum);
        sum = _mm_hadd_ps(sum, sum);
        y[n] = _mm_cvtss_f32(sum);
    }
}
#endif

// -----------------------------------------------------------------------------
//  Variante para grupo = 32 (2 grupos por bloque de 64 valores).
// -----------------------------------------------------------------------------
//  La permutación kPerm deja los dos grupos separados en las mitades del
//  acumulador:
//      a = [v0..15 | v32..47]  ->  lanes 0-3 = grupo 0 (k0..15),  4-7 = grupo 1
//      b = [v16..31| v48..63]  ->  lanes 0-3 = grupo 0 (k16..31), 4-7 = grupo 1
//  Sumando a+b, cada mitad del registro es un grupo completo. Entonces una
//  escala distinta por mitad cuesta UN solo vinsertf128 (set_m128), no un
//  pliegue extra por grupo.
#if defined(__AVX2__)
inline void gemv_int4_g32(const Shape& s,
                          const std::uint8_t* __restrict w,
                          const _Float16* __restrict wscale,
                          const std::int32_t* __restrict xgsum,
                          const std::int8_t* __restrict xperm,
                          const float* __restrict xscale,   // una por grupo
                          float* __restrict y) {
    const __m256i m0f = _mm256_set1_epi8(0x0F);
    const __m256i ones16 = _mm256_set1_epi16(1);
    const std::size_t G = s.groups(), rb = s.row_bytes();
    const std::size_t blocks = s.K / 64;   // bloques de 64 valores

    for (std::size_t n = 0; n < s.N; ++n) {
        const std::uint8_t* row = w + n * s.row_stride();
        const _Float16* sc = wscale ? wscale + n * G
                                    : reinterpret_cast<const _Float16*>(row + rb);
        __m256 facc = _mm256_setzero_ps();

        for (std::size_t bb = 0; bb < blocks; ++bb) {
            const __m256i v = _mm256_loadu_si256(
                reinterpret_cast<const __m256i*>(row + bb * 32));
            const __m256i lo = _mm256_and_si256(v, m0f);
            const __m256i hi = _mm256_and_si256(_mm256_srli_epi16(v, 4), m0f);
            const __m256i a = _mm256_unpacklo_epi8(lo, hi);
            const __m256i b = _mm256_unpackhi_epi8(lo, hi);

            const std::int8_t* xg = xperm + bb * 64;
            const __m256i x0 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(xg));
            const __m256i x1 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(xg + 32));

            __m256i acc = _mm256_madd_epi16(_mm256_maddubs_epi16(a, x0), ones16);
            acc = _mm256_add_epi32(
                acc, _mm256_madd_epi16(_mm256_maddubs_epi16(b, x1), ones16));

            const std::size_t g0 = 2 * bb, g1 = 2 * bb + 1;
            acc = _mm256_sub_epi32(acc, _mm256_setr_epi32(
                8 * xgsum[g0], 0, 0, 0, 8 * xgsum[g1], 0, 0, 0));

            // Una escala por mitad del registro.
            const __m256 sv = _mm256_set_m128(
                _mm_set1_ps(static_cast<float>(sc[g1]) * xscale[g1]),
                _mm_set1_ps(static_cast<float>(sc[g0]) * xscale[g0]));
            facc = _mm256_fmadd_ps(_mm256_cvtepi32_ps(acc), sv, facc);
        }

        __m128 lo128 = _mm256_castps256_ps128(facc);
        __m128 hi128 = _mm256_extractf128_ps(facc, 1);
        __m128 sum = _mm_add_ps(lo128, hi128);
        sum = _mm_hadd_ps(sum, sum);
        sum = _mm_hadd_ps(sum, sum);
        y[n] = _mm_cvtss_f32(sum);
    }
}
#endif


// -----------------------------------------------------------------------------
//  Kernel M genérico (batching / verificación especulativa).
// -----------------------------------------------------------------------------
//  El desempacado se hace UNA vez por bloque de 64 valores y alimenta M
//  acumuladores: los bytes de peso por token se parten en M. Medido en
//  probe4_batch sobre el artefacto real del 1.5B (4 hilos):
//
//      M    ms/pasada   t(M)/t(1)   DRAM real
//      1      45.89       1.00      19.5 GB/s
//      2      53.13       1.16      16.8
//      4      80.34       1.75      11.1
//      8     143.53       3.13       6.2
//
//  La DRAM real CAE con M: se deja de ser memory-bound y se pasa a compute-bound.
//  Techo de cómputo leído del coste marginal: ~63 tok/s.
//
//  xperm: M*K int8 pre-permutados;  xgsum: M*G;  xscale: M*G (una por
//  grupo, no una por token: con una sola escala por token los modelos de
//  activaciones con outliers --SmolLM2-- colapsan);  y: M*N.
#if defined(__AVX2__)
template <unsigned M>
inline void gemv_int4_g32_M(const Shape& s,
                            const std::uint8_t* __restrict w,
                            const _Float16* __restrict wscale,
                            const std::int32_t* __restrict xgsum,
                            const std::int8_t* __restrict xperm,
                            const float* __restrict xscale,
                            float* __restrict y,
                            std::size_t ystride) {
    const __m256i m0f = _mm256_set1_epi8(0x0F);
    const __m256i ones16 = _mm256_set1_epi16(1);
    const std::size_t G = s.groups(), rb = s.row_bytes();
    const std::size_t blocks = s.K / 64;

    for (std::size_t n = 0; n < s.N; ++n) {
        const std::uint8_t* row = w + n * s.row_stride();
        const _Float16* sc = wscale ? wscale + n * G
                                    : reinterpret_cast<const _Float16*>(row + rb);
        __m256 facc[M];
#pragma GCC unroll 8
        for (unsigned m = 0; m < M; ++m) facc[m] = _mm256_setzero_ps();

        std::size_t bb = 0;
        // --- 8 grupos (4 bloques) por iteracion ---------------------------
        // Las 8 escalas se convierten fp16->fp32 con UN _mm256_cvtph_ps y se
        // reparten con permutevar8x32_ps. Antes se construia una escala por
        // bloque (2 conversiones + 2 mul + 2 set1 + insert = ~8 instr cada
        // 32 B); medido con perf, era ~1/3 de las instrucciones del bucle.
#ifndef CAUDAL_LEGACY_SCALE
        if (G % 8 == 0) {
#else
        if (false) {
#endif
            const __m256i p0 = _mm256_setr_epi32(0, 0, 0, 0, 1, 1, 1, 1);
            const __m256i p1 = _mm256_setr_epi32(2, 2, 2, 2, 3, 3, 3, 3);
            const __m256i p2 = _mm256_setr_epi32(4, 4, 4, 4, 5, 5, 5, 5);
            const __m256i p3 = _mm256_setr_epi32(6, 6, 6, 6, 7, 7, 7, 7);
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

#pragma GCC unroll 8
                for (unsigned m = 0; m < M; ++m) {
                    const __m256 s8 = _mm256_mul_ps(
                        w8, _mm256_loadu_ps(xscale + std::size_t(m) * G + gbase));
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
                        const std::size_t gg = gbase + std::size_t(2 * j);
                        acc = _mm256_sub_epi32(acc, _mm256_setr_epi32(
                            8 * xgsum[std::size_t(m) * G + gg], 0, 0, 0,
                            8 * xgsum[std::size_t(m) * G + gg + 1], 0, 0, 0));
                        facc[m] = _mm256_fmadd_ps(_mm256_cvtepi32_ps(acc), sv[j], facc[m]);
                    }
                }
            }
        }

        // --- cola (o G no multiplo de 8) ---------------------------------
        for (; bb < blocks; ++bb) {
            const __m256i v = _mm256_loadu_si256(
                reinterpret_cast<const __m256i*>(row + bb * 32));
            const __m256i lo = _mm256_and_si256(v, m0f);
            const __m256i hi = _mm256_and_si256(_mm256_srli_epi16(v, 4), m0f);
            const __m256i a = _mm256_unpacklo_epi8(lo, hi);
            const __m256i b = _mm256_unpackhi_epi8(lo, hi);
            const std::size_t g0 = 2 * bb, g1 = 2 * bb + 1;

#pragma GCC unroll 8
            for (unsigned m = 0; m < M; ++m) {
                const std::int8_t* xm = xperm + std::size_t(m) * s.K + bb * 64;
                const __m256i x0 = _mm256_loadu_si256(
                    reinterpret_cast<const __m256i*>(xm));
                const __m256i x1 = _mm256_loadu_si256(
                    reinterpret_cast<const __m256i*>(xm + 32));
                __m256i acc = _mm256_madd_epi16(_mm256_maddubs_epi16(a, x0), ones16);
                acc = _mm256_add_epi32(
                    acc, _mm256_madd_epi16(_mm256_maddubs_epi16(b, x1), ones16));
                acc = _mm256_sub_epi32(acc, _mm256_setr_epi32(
                    8 * xgsum[std::size_t(m) * G + g0], 0, 0, 0,
                    8 * xgsum[std::size_t(m) * G + g1], 0, 0, 0));
                const __m256 sv = _mm256_set_m128(
                    _mm_set1_ps(static_cast<float>(sc[g1]) * xscale[std::size_t(m) * G + g1]),
                    _mm_set1_ps(static_cast<float>(sc[g0]) * xscale[std::size_t(m) * G + g0]));
                facc[m] = _mm256_fmadd_ps(_mm256_cvtepi32_ps(acc), sv, facc[m]);
            }
        }
#pragma GCC unroll 8
        for (unsigned m = 0; m < M; ++m) {
            __m128 l = _mm256_castps256_ps128(facc[m]);
            __m128 h = _mm256_extractf128_ps(facc[m], 1);
            __m128 t = _mm_add_ps(l, h);
            t = _mm_hadd_ps(t, t);
            t = _mm_hadd_ps(t, t);
            y[std::size_t(m) * ystride + n] = _mm_cvtss_f32(t);
        }
    }
}
#endif


// -----------------------------------------------------------------------------
//  Esquema con minimo por grupo (super-bloque de 256 = 8 grupos de 32).
// -----------------------------------------------------------------------------
//  Header de 16 B por super-bloque (0,5 bits/valor, igual que las escalas fp16
//  por grupo de 32):  [0..1] d fp16   [2..3] dmin fp16   [4..15] 16 campos 6 bits
//  Reconstruccion:  w = d*sc_g*q + b_g ,  b_g = dmin*(m_g - 32)  (CON SIGNO)
//  Producto:        Sum w*x = Sum_g xscale_g * ( s_g*A_g + b_g*B_g )
//  con s_g = d*sc_g, A_g = Sum q*xq, B_g = Sum xq (xgsum, ya existente).
//  El minimo centrado en 32 cubre grupos negativos Y positivos: con m>=0 fijo,
//  un grupo con minimo positivo (>0) se aplastaba (bug encontrado por el test).
inline void kmin_header(const std::uint8_t* h, float& d, float& dmin,
                        int sc[8], int m[8]) {
    _Float16 dh, dmh;
    std::memcpy(&dh, h, 2);
    std::memcpy(&dmh, h + 2, 2);
    d = static_cast<float>(dh);
    dmin = static_cast<float>(dmh);
    std::uint64_t lo = 0, hi = 0;
    std::memcpy(&lo, h + 4, 8);
    std::uint32_t hi32 = 0;
    std::memcpy(&hi32, h + 12, 4);
    hi = hi32;
    int vals[16];
#pragma GCC unroll 16
    for (int t = 0; t < 16; ++t) {
        const int bit = 6 * t;
        std::uint64_t v;
        if (bit >= 64) v = hi >> (bit - 64);
        else if (bit + 6 <= 64) v = lo >> bit;
        else v = (lo >> bit) | (hi << (64 - bit));
        vals[t] = int(v & 0x3F);
    }
#pragma GCC unroll 8
    for (int g = 0; g < 8; ++g) { sc[g] = vals[g]; m[g] = vals[8 + g]; }
}

#if defined(__AVX2__)
template <unsigned M>
inline void gemv_int4_kmin_M(const Shape& s,
                             const std::uint8_t* __restrict w,
                             const std::uint8_t* __restrict hdr,
                             const std::int32_t* __restrict xgsum,
                             const std::int8_t* __restrict xperm,
                             const float* __restrict xscale,
                             float* __restrict y,
                             std::size_t ystride) {
    const __m256i m0f = _mm256_set1_epi8(0x0F);
    const __m256i ones16 = _mm256_set1_epi16(1);
    const std::size_t G = s.K / 32;
    const std::size_t SBn = s.K / 256;

    for (std::size_t n = 0; n < s.N; ++n) {
        const std::uint8_t* row = w + n * (s.K / 2);
        const std::uint8_t* hrow = hdr + n * SBn * 16;
        __m256 facc[M], minv[M];
#pragma GCC unroll 8
        for (unsigned m = 0; m < M; ++m) { facc[m] = _mm256_setzero_ps(); minv[m] = facc[m]; }

        for (std::size_t sb = 0; sb < SBn; ++sb) {
            const std::uint8_t* h = hrow + sb * 16;
            _Float16 dh, dmh;
            std::memcpy(&dh, h, 2);
            std::memcpy(&dmh, h + 2, 2);
            const float d = static_cast<float>(dh);
            const float dmin = static_cast<float>(dmh);
            std::uint64_t lo64;
            std::memcpy(&lo64, h + 4, 8);
            std::uint32_t hi32;
            std::memcpy(&hi32, h + 12, 4);
            const std::uint64_t hi64 = hi32;
            int vals[16];
#pragma GCC unroll 16
            for (int t = 0; t < 16; ++t) {
                const int bit = 6 * t;
                std::uint64_t v;
                if (bit >= 64) v = hi64 >> (bit - 64);
                else if (bit + 6 <= 64) v = lo64 >> bit;
                else v = (lo64 >> bit) | (hi64 << (64 - bit));
                vals[t] = int(v & 0x3F);
            }
            const __m256 b8 = _mm256_mul_ps(
                _mm256_cvtepi32_ps(_mm256_setr_epi32(
                    vals[8] - 32, vals[9] - 32, vals[10] - 32, vals[11] - 32,
                    vals[12] - 32, vals[13] - 32, vals[14] - 32, vals[15] - 32)),
                _mm256_set1_ps(dmin));
#pragma GCC unroll 8
            for (unsigned m = 0; m < M; ++m) {
                const std::size_t base = std::size_t(m) * G + sb * 8;
                const __m256 prod = _mm256_mul_ps(
                    _mm256_loadu_ps(xscale + base),
                    _mm256_cvtepi32_ps(_mm256_loadu_si256(
                        reinterpret_cast<const __m256i*>(xgsum + base))));
                minv[m] = _mm256_fmadd_ps(b8, prod, minv[m]);
            }
#pragma GCC unroll 4
            for (int j = 0; j < 4; ++j) {
                const std::size_t bb = sb * 4 + std::size_t(j);
                const __m256i v = _mm256_loadu_si256(
                    reinterpret_cast<const __m256i*>(row + bb * 32));
                const __m256i lo = _mm256_and_si256(v, m0f);
                const __m256i hi = _mm256_and_si256(_mm256_srli_epi16(v, 4), m0f);
                const __m256i a = _mm256_unpacklo_epi8(lo, hi);
                const __m256i b = _mm256_unpackhi_epi8(lo, hi);
                const int g0 = 2 * j, g1 = 2 * j + 1;
                const float s0 = d * float(vals[g0]);
                const float s1 = d * float(vals[g1]);
#pragma GCC unroll 8
                for (unsigned m = 0; m < M; ++m) {
                    const std::int8_t* xm = xperm + std::size_t(m) * s.K + bb * 64;
                    const __m256i x0 = _mm256_loadu_si256(
                        reinterpret_cast<const __m256i*>(xm));
                    const __m256i x1 = _mm256_loadu_si256(
                        reinterpret_cast<const __m256i*>(xm + 32));
                    __m256i acc = _mm256_madd_epi16(_mm256_maddubs_epi16(a, x0), ones16);
                    acc = _mm256_add_epi32(
                        acc, _mm256_madd_epi16(_mm256_maddubs_epi16(b, x1), ones16));
                    const std::size_t o0 = std::size_t(m) * G + sb * 8 + std::size_t(g0);
                    const __m256 sv = _mm256_set_m128(
                        _mm_set1_ps(s1 * xscale[o0 + 1]),
                        _mm_set1_ps(s0 * xscale[o0]));
                    facc[m] = _mm256_fmadd_ps(_mm256_cvtepi32_ps(acc), sv, facc[m]);
                }
            }
        }
#pragma GCC unroll 8
        for (unsigned m = 0; m < M; ++m) {
            auto hsum = [](__m256 v) {
                __m128 l = _mm256_castps256_ps128(v);
                __m128 h = _mm256_extractf128_ps(v, 1);
                __m128 t = _mm_add_ps(l, h);
                t = _mm_hadd_ps(t, t);
                t = _mm_hadd_ps(t, t);
                return _mm_cvtss_f32(t);
            };
            y[std::size_t(m) * ystride + n] = hsum(facc[m]) + hsum(minv[m]);
        }
    }
}
#endif  // defined(__AVX2__)
// -----------------------------------------------------------------------------
//  Afín con grupo 64: escala fp16 + minimo fp16 por grupo de 64 (4.50 bpw).
//  Misma matemática que affine_M, pero cada bloque de 64 valores usa UNA
//  escala; el registro de 8 carriles completo pertenece al mismo grupo.
// -----------------------------------------------------------------------------
template <unsigned M>
inline void gemv_int4_affine_g64_M(const Shape& s,
                                   const std::uint8_t* __restrict w,
                                   const _Float16* __restrict wscale,
                                   const _Float16* __restrict wmin,
                                   const float* __restrict xmin_scale,
                                   const std::int8_t* __restrict xperm,
                                   const float* __restrict xscale,
                                   float* __restrict y,
                                   std::size_t ystride) {
#if defined(__AVX2__) && defined(__F16C__)
    const __m256i m0f = _mm256_set1_epi8(0x0F);
    const __m256i ones16 = _mm256_set1_epi16(1);
    const std::size_t G = s.groups();
    const std::size_t blocks = s.K / 64;
    const __m256i p0 = _mm256_setr_epi32(0, 0, 0, 0, 0, 0, 0, 0);
    const __m256i p1 = _mm256_setr_epi32(1, 1, 1, 1, 1, 1, 1, 1);
    const __m256i p2 = _mm256_setr_epi32(2, 2, 2, 2, 2, 2, 2, 2);
    const __m256i p3 = _mm256_setr_epi32(3, 3, 3, 3, 3, 3, 3, 3);

    auto hsum = [](__m256 v) {
        __m128 l = _mm256_castps256_ps128(v);
        __m128 h = _mm256_extractf128_ps(v, 1);
        __m128 t = _mm_add_ps(l, h);
        t = _mm_hadd_ps(t, t); t = _mm_hadd_ps(t, t);
        return _mm_cvtss_f32(t);
    };

    for (std::size_t n = 0; n < s.N; ++n) {
        const std::uint8_t* row = w + n * s.row_stride();
        const _Float16* sc = wscale + n * G;
        const _Float16* mn = wmin + n * G;
        __m256 facc[M], minacc[M];
#pragma GCC unroll 8
        for (unsigned m = 0; m < M; ++m) { facc[m] = _mm256_setzero_ps(); minacc[m] = facc[m]; }

        std::size_t bb = 0;
        for (; bb + 4 <= blocks; bb += 4) {
            __m256i A[4], B[4];
#pragma GCC unroll 4
            for (int j = 0; j < 4; ++j) {
                const __m256i v = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(
                    row + (bb + std::size_t(j)) * 32));
                const __m256i lo = _mm256_and_si256(v, m0f);
                const __m256i hi = _mm256_and_si256(_mm256_srli_epi16(v, 4), m0f);
                A[j] = _mm256_unpacklo_epi8(lo, hi);
                B[j] = _mm256_unpackhi_epi8(lo, hi);
            }
            const std::size_t gb = bb;
            const __m256 w8 = _mm256_cvtph_ps(_mm_loadl_epi64(
                reinterpret_cast<const __m128i*>(sc + gb)));
            const __m256 m8 = _mm256_cvtph_ps(_mm_loadl_epi64(
                reinterpret_cast<const __m128i*>(mn + gb)));
#pragma GCC unroll 8
            for (unsigned m = 0; m < M; ++m) {
                const __m256 xs4 = _mm256_set_m128(_mm_setzero_ps(),
                                                   _mm_loadu_ps(xscale + std::size_t(m) * G + gb));
                const __m256 xms4 = _mm256_set_m128(_mm_setzero_ps(),
                                                    _mm_loadu_ps(xmin_scale + std::size_t(m) * G + gb));
                const __m256 s8 = _mm256_mul_ps(w8, xs4);
                const __m256 sv[4] = {_mm256_permutevar8x32_ps(s8, p0),
                                      _mm256_permutevar8x32_ps(s8, p1),
                                      _mm256_permutevar8x32_ps(s8, p2),
                                      _mm256_permutevar8x32_ps(s8, p3)};
                minacc[m] = _mm256_fmadd_ps(m8, xms4, minacc[m]);
                const std::int8_t* xm = xperm + std::size_t(m) * s.K;
#pragma GCC unroll 4
                for (int j = 0; j < 4; ++j) {
                    const __m256i x0 = _mm256_loadu_si256(
                        reinterpret_cast<const __m256i*>(xm + (bb + std::size_t(j)) * 64));
                    const __m256i x1 = _mm256_loadu_si256(
                        reinterpret_cast<const __m256i*>(xm + (bb + std::size_t(j)) * 64 + 32));
                    __m256i acc = _mm256_madd_epi16(
                        _mm256_maddubs_epi16(A[j], x0), ones16);
                    acc = _mm256_add_epi32(acc,
                        _mm256_madd_epi16(_mm256_maddubs_epi16(B[j], x1), ones16));
                    facc[m] = _mm256_fmadd_ps(_mm256_cvtepi32_ps(acc), sv[j], facc[m]);
                }
            }
        }
        for (; bb < blocks; ++bb) {
            const __m256i v = _mm256_loadu_si256(
                reinterpret_cast<const __m256i*>(row + bb * 32));
            const __m256i lo = _mm256_and_si256(v, m0f);
            const __m256i hi = _mm256_and_si256(_mm256_srli_epi16(v, 4), m0f);
            const __m256i a = _mm256_unpacklo_epi8(lo, hi);
            const __m256i b = _mm256_unpackhi_epi8(lo, hi);
#pragma GCC unroll 8
            for (unsigned m = 0; m < M; ++m) {
                const float s0 = static_cast<float>(sc[bb]) * xscale[std::size_t(m) * G + bb];
                const float m0 = static_cast<float>(mn[bb]) * xmin_scale[std::size_t(m) * G + bb];
                const __m256 minv = _mm256_setr_ps(m0, 0, 0, 0, 0, 0, 0, 0);
                const __m256 sv = _mm256_set1_ps(s0);
                const std::int8_t* xm = xperm + std::size_t(m) * s.K + bb * 64;
                const __m256i x0 = _mm256_loadu_si256(
                    reinterpret_cast<const __m256i*>(xm));
                const __m256i x1 = _mm256_loadu_si256(
                    reinterpret_cast<const __m256i*>(xm + 32));
                __m256i acc = _mm256_madd_epi16(_mm256_maddubs_epi16(a, x0), ones16);
                acc = _mm256_add_epi32(acc,
                    _mm256_madd_epi16(_mm256_maddubs_epi16(b, x1), ones16));
                facc[m] = _mm256_fmadd_ps(_mm256_cvtepi32_ps(acc), sv, facc[m]);
                minacc[m] = _mm256_add_ps(minacc[m], minv);
            }
        }
#pragma GCC unroll 8
        for (unsigned m = 0; m < M; ++m)
            y[std::size_t(m) * ystride + n] = hsum(facc[m]) + hsum(minacc[m]);
    }
#else
    // Fallback escalar (SSE/ARM): mismo layout y misma matemática.
    const std::size_t G = s.groups(), K = s.K, rb = s.row_stride();
    for (std::size_t n = 0; n < s.N; ++n) {
        const std::uint8_t* row = w + n * rb;
        const _Float16* sc = wscale + n * G;
        const _Float16* mn = wmin + n * G;
        for (unsigned m = 0; m < M; ++m) {
            const std::int8_t* xm = xperm + std::size_t(m) * K;
            double acc = 0.0;
            for (std::size_t g = 0; g < G; ++g) {
                long long dot = 0, sumx = 0;
                for (std::size_t j = 0; j < 64; ++j) {
                    const int q = nibble_at(row, g * 64 + kPerm[j]);
                    const int xv = xm[g * 64 + j];
                    dot += static_cast<long long>(q) * xv;
                    sumx += xv;
                }
                acc += static_cast<double>(xscale[g])
                     * (static_cast<double>(sc[g]) * dot
                        + static_cast<double>(mn[g]) * sumx);
            }
            y[std::size_t(m) * ystride + n] = static_cast<float>(acc);
        }
    }
#endif
}

// -----------------------------------------------------------------------------
//  Esquema afin directo: escala fp16 + minimo fp16 por grupo de 32 (5.00 bpw).
//  w = s * q + m  con q in [0, 15] unsigned.
//  Sum w*x = Sum_g [ s_g * xscale_g * Sum_i(q*x) + m_g * xscale_g * xgsum_g ]
//  CERO decode, CERO shifts, el termino de m_g se vectoriza 1 vez cada 8 grupos.
// -----------------------------------------------------------------------------
template <unsigned M>
inline void gemv_int4_affine_M(const Shape& s,
                               const std::uint8_t* __restrict w,
                               const _Float16* __restrict wscale,
                               const _Float16* __restrict wmin,
                               const float* __restrict xmin_scale,
                               const std::int8_t* __restrict xperm,
                               const float* __restrict xscale,
                               float* __restrict y,
                               std::size_t ystride) {
#if defined(__AVX2__)
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

                const float* xms_ptr = xmin_scale + std::size_t(m) * G + gbase;
                const __m256 xms8 = _mm256_loadu_ps(xms_ptr);
                minacc[m] = _mm256_fmadd_ps(m8, xms8, minacc[m]);

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

        // --- cola (bloques restantes si blocks no es multiplo de 4, ej. Qwen-0.5B con K=896) ---
        for (; bb < blocks; ++bb) {
            const __m256i v = _mm256_loadu_si256(
                reinterpret_cast<const __m256i*>(row + bb * 32));
            const __m256i lo = _mm256_and_si256(v, m0f);
            const __m256i hi = _mm256_and_si256(_mm256_srli_epi16(v, 4), m0f);
            const __m256i A_block = _mm256_unpacklo_epi8(lo, hi);
            const __m256i B_block = _mm256_unpackhi_epi8(lo, hi);
            const std::size_t g0 = 2 * bb, g1 = 2 * bb + 1;

            const float w0 = static_cast<float>(sc[g0]);
            const float w1 = static_cast<float>(sc[g1]);
            const float mn0 = static_cast<float>(mn[g0]);
            const float mn1 = static_cast<float>(mn[g1]);

            const __m256 m_tail = _mm256_setr_ps(mn0, mn1, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f);

#pragma GCC unroll 8
            for (unsigned m = 0; m < M; ++m) {
                const float s0 = w0 * xscale[std::size_t(m) * G + g0];
                const float s1 = w1 * xscale[std::size_t(m) * G + g1];
                const __m256 sv = _mm256_set_m128(_mm_set1_ps(s1), _mm_set1_ps(s0));

                const float xms0 = xmin_scale[std::size_t(m) * G + g0];
                const float xms1 = xmin_scale[std::size_t(m) * G + g1];
                const __m256 xms_tail = _mm256_setr_ps(xms0, xms1, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f);
                minacc[m] = _mm256_fmadd_ps(m_tail, xms_tail, minacc[m]);

                const std::int8_t* xm = xperm + std::size_t(m) * s.K + bb * 64;
                const __m256i x0 = _mm256_loadu_si256(
                    reinterpret_cast<const __m256i*>(xm));
                const __m256i x1 = _mm256_loadu_si256(
                    reinterpret_cast<const __m256i*>(xm + 32));
                __m256i acc = _mm256_madd_epi16(_mm256_maddubs_epi16(A_block, x0), ones16);
                acc = _mm256_add_epi32(
                    acc, _mm256_madd_epi16(_mm256_maddubs_epi16(B_block, x1), ones16));
                facc[m] = _mm256_fmadd_ps(_mm256_cvtepi32_ps(acc), sv, facc[m]);
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
#elif defined(__x86_64__) || defined(_M_X64)
    // --- Implementación SSE4.1 / SSE4.2 (128 bits para Intel Pentium N3700 / Acer) ---
    const __m128i m0f = _mm_set1_epi8(0x0F);
    const __m128i ones16 = _mm_set1_epi16(1);
    const std::size_t G = s.groups();
    const std::size_t blocks = s.K / 64;

    for (std::size_t n = 0; n < s.N; ++n) {
        const std::uint8_t* row = w + n * (s.K / 2);
        const _Float16* sc = wscale + n * G;
        const _Float16* mn = wmin + n * G;

        float facc[M]{0.0f};

        for (std::size_t bb = 0; bb < blocks; ++bb) {
            const __m128i v0 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(row + bb * 32));
            const __m128i v1 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(row + bb * 32 + 16));

            const __m128i lo0 = _mm_and_si128(v0, m0f);
            const __m128i hi0 = _mm_and_si128(_mm_srli_epi16(v0, 4), m0f);
            const __m128i A0 = _mm_unpacklo_epi8(lo0, hi0);
            const __m128i B0 = _mm_unpackhi_epi8(lo0, hi0);

            const __m128i lo1 = _mm_and_si128(v1, m0f);
            const __m128i hi1 = _mm_and_si128(_mm_srli_epi16(v1, 4), m0f);
            const __m128i A1 = _mm_unpacklo_epi8(lo1, hi1);
            const __m128i B1 = _mm_unpackhi_epi8(lo1, hi1);

            const std::size_t g0 = 2 * bb, g1 = 2 * bb + 1;
            const float w0 = static_cast<float>(sc[g0]);
            const float w1 = static_cast<float>(sc[g1]);
            const float mn0 = static_cast<float>(mn[g0]);
            const float mn1 = static_cast<float>(mn[g1]);

            for (unsigned m = 0; m < M; ++m) {
                const std::int8_t* xm = xperm + std::size_t(m) * s.K + bb * 64;
                const __m128i x0_0 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(xm));
                const __m128i x0_1 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(xm + 16));
                const __m128i x1_0 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(xm + 32));
                const __m128i x1_1 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(xm + 48));

                __m128i d0 = _mm_madd_epi16(_mm_maddubs_epi16(A0, x0_0), ones16);
                d0 = _mm_add_epi32(d0, _mm_madd_epi16(_mm_maddubs_epi16(B0, x1_0), ones16));

                __m128i d1 = _mm_madd_epi16(_mm_maddubs_epi16(A1, x0_1), ones16);
                d1 = _mm_add_epi32(d1, _mm_madd_epi16(_mm_maddubs_epi16(B1, x1_1), ones16));

                alignas(16) std::int32_t r0[4], r1[4];
                _mm_store_si128(reinterpret_cast<__m128i*>(r0), d0);
                _mm_store_si128(reinterpret_cast<__m128i*>(r1), d1);

                const float dot0 = float(r0[0] + r0[1] + r0[2] + r0[3]);
                const float dot1 = float(r1[0] + r1[1] + r1[2] + r1[3]);

                facc[m] += dot0 * (w0 * xscale[std::size_t(m) * G + g0])
                         + dot1 * (w1 * xscale[std::size_t(m) * G + g1])
                         + mn0 * xmin_scale[std::size_t(m) * G + g0]
                         + mn1 * xmin_scale[std::size_t(m) * G + g1];
            }
        }

        for (unsigned m = 0; m < M; ++m) {
            y[std::size_t(m) * ystride + n] = facc[m];
        }
    }
#elif defined(__ARM_NEON) || defined(__aarch64__)
    // --- Implementación ARM NEON (128 bits para ARMv8-A / Exynos / Galaxy Note 10 Lite) ---
    const uint8x16_t m0f = vdupq_n_u8(0x0F);
    const std::size_t G = s.groups();
    const std::size_t blocks = s.K / 64;

    for (std::size_t n = 0; n < s.N; ++n) {
        const std::uint8_t* row = w + n * (s.K / 2);
        const _Float16* sc = wscale + n * G;
        const _Float16* mn = wmin + n * G;

        float facc[M]{0.0f};

        for (std::size_t bb = 0; bb < blocks; ++bb) {
            const uint8x16_t v0 = vld1q_u8(row + bb * 32);
            const uint8x16_t v1 = vld1q_u8(row + bb * 32 + 16);

            const uint8x16_t lo0 = vandq_u8(v0, m0f);
            const uint8x16_t hi0 = vandq_u8(vshrq_n_u8(v0, 4), m0f);
            const uint8x16_t A0 = vzip1q_u8(lo0, hi0);
            const uint8x16_t B0 = vzip2q_u8(lo0, hi0);

            const uint8x16_t lo1 = vandq_u8(v1, m0f);
            const uint8x16_t hi1 = vandq_u8(vshrq_n_u8(v1, 4), m0f);
            const uint8x16_t A1 = vzip1q_u8(lo1, hi1);
            const uint8x16_t B1 = vzip2q_u8(lo1, hi1);

            const std::size_t g0 = 2 * bb, g1 = 2 * bb + 1;
            const float w0 = static_cast<float>(sc[g0]);
            const float w1 = static_cast<float>(sc[g1]);
            const float mn0 = static_cast<float>(mn[g0]);
            const float mn1 = static_cast<float>(mn[g1]);

            for (unsigned m = 0; m < M; ++m) {
                const std::int8_t* xm = xperm + std::size_t(m) * s.K + bb * 64;
                const int8x16_t x0_0 = vld1q_s8(xm);
                const int8x16_t x0_1 = vld1q_s8(xm + 16);
                const int8x16_t x1_0 = vld1q_s8(xm + 32);
                const int8x16_t x1_1 = vld1q_s8(xm + 48);

                int16x8_t d0 = vmull_s8(vget_low_s8(vreinterpretq_s8_u8(A0)), vget_low_s8(x0_0));
                d0 = vmlal_s8(d0, vget_high_s8(vreinterpretq_s8_u8(A0)), vget_high_s8(x0_0));
                d0 = vmlal_s8(d0, vget_low_s8(vreinterpretq_s8_u8(B0)), vget_low_s8(x1_0));
                d0 = vmlal_s8(d0, vget_high_s8(vreinterpretq_s8_u8(B0)), vget_high_s8(x1_0));

                int16x8_t d1 = vmull_s8(vget_low_s8(vreinterpretq_s8_u8(A1)), vget_low_s8(x0_1));
                d1 = vmlal_s8(d1, vget_high_s8(vreinterpretq_s8_u8(A1)), vget_high_s8(x0_1));
                d1 = vmlal_s8(d1, vget_low_s8(vreinterpretq_s8_u8(B1)), vget_low_s8(x1_1));
                d1 = vmlal_s8(d1, vget_high_s8(vreinterpretq_s8_u8(B1)), vget_high_s8(x1_1));

                const float dot0 = float(vaddvq_s32(vpaddlq_s16(d0)));
                const float dot1 = float(vaddvq_s32(vpaddlq_s16(d1)));

                facc[m] += dot0 * (w0 * xscale[std::size_t(m) * G + g0])
                         + dot1 * (w1 * xscale[std::size_t(m) * G + g1])
                         + mn0 * xmin_scale[std::size_t(m) * G + g0]
                         + mn1 * xmin_scale[std::size_t(m) * G + g1];
            }
        }

        for (unsigned m = 0; m < M; ++m) {
            y[std::size_t(m) * ystride + n] = facc[m];
        }
    }
#endif
}

}  // namespace caudal

#endif  // CAUDAL_GEMV_INT4_HPP
