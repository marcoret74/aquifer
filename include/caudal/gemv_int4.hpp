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

#if defined(__AVX2__)
#  include <immintrin.h>
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

}  // namespace caudal

#endif  // CAUDAL_GEMV_INT4_HPP
