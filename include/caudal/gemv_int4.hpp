#ifndef CAUDAL_GEMV_INT4_HPP
#define CAUDAL_GEMV_INT4_HPP

// =============================================================================
//  caudal · include/caudal/gemv_int4.hpp
//  GEMV int4 con escalas por grupo y x PRE-PERMUTADO.
// =============================================================================
//  Este es el kernel "de verdad" (no el del probe). Diferencias con
//  `bench/probe1_gemv.cpp`:
//
//    1. Escalas por GRUPO de 64, no por fila. Cuesta un cvt + fmadd + add por
//       grupo, y obliga a corregir el sesgo del zero-point. El probe medía con
//       escalas por fila: su 11.4 GB/s es por lo tanto OPTIMISTA.
//    2. x viene PRE-PERMUTADO por el packer, así que el bucle NO gasta los dos
//       vperm2i128 que gastaba el probe.
//
//  ── Por qué la permutación es legal ─────────────────────────────────────────
//  El desempaquetado AVX2 entrega las mitades como
//      a = [v0..15, v32..47]   b = [v16..31, v48..63]
//  Si x se guarda en ese mismo orden (PERM), el emparejamiento es directo.
//  El producto punto es conmutativo: cualquier permutación CONSISTENTE entre
//  los dos operandos da el mismo resultado. El packer la paga una vez.
//
//  ── Aritmética del zero-point ───────────────────────────────────────────────
//  x_q = round(x/sx) + 128  (unsigned, para vpmaddubsw)
//  w   = code * sw          (int4 con signo, -8..7)
//
//      y = Σ_k (x_q - 128)·w·sx = sx·sw·( Σ_k x_q·code  −  128·Σ_k code )
//
//  `Σ code` por grupo se precalcula en el packer (`codesum`).
// =============================================================================

#include <array>
#include <cstddef>
#include <cstdint>

#if defined(__AVX2__)
#  include <immintrin.h>
#endif

namespace caudal {

// Permutación dentro de cada bloque de 64 valores. Es el orden que produce
// naturalmente vpunpcklbw/vpunpckhbw sobre las mitades de nibbles:
//     a = [0..15, 32..47]   b = [16..31, 48..63]
// El packer guarda x en ESTE orden; el kernel empareja directo.
// Si esto cambia, cambia en los dos lados o el round-trip falla.
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

struct Shape {
    std::size_t N{0};      // filas de salida
    std::size_t K{0};      // profundidad; múltiplo de 64
    std::size_t group{64}; // valores por escala

    [[nodiscard]] constexpr std::size_t groups() const noexcept { return K / group; }
    [[nodiscard]] constexpr std::size_t row_bytes() const noexcept { return K / 2; }
};

// Byte empaquetado -> code int4 con signo (-8..7). nibble = complemento a 2.
[[nodiscard]] inline int code_at(const std::uint8_t* row, std::size_t j) noexcept {
    const std::uint8_t b = row[j >> 1];
    const unsigned nib = (j & 1u) ? (static_cast<unsigned>(b) >> 4)
                                  : (static_cast<unsigned>(b) & 0x0Fu);
    return (nib < 8u) ? static_cast<int>(nib) : static_cast<int>(nib) - 16;
}

// -----------------------------------------------------------------------------
//  Referencia escalar. Lee los MISMOS bytes empaquetados, así que valida el
//  layout y el empaquetado, no sólo el kernel.
// -----------------------------------------------------------------------------
inline void gemv_ref(const Shape& s,
                     const std::uint8_t* __restrict w,
                     const _Float16* __restrict wscale,
                     const std::int32_t* __restrict codesum,
                     const std::uint8_t* __restrict xperm,
                     float xscale,
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
                const int xq = static_cast<int>(xperm[base + j]) - 128;
                dot += static_cast<long long>(code) * xq;
            }
            (void)codesum;  // la referencia no lo necesita: resta el zero-point
            acc += static_cast<double>(dot)
                 * static_cast<double>(wscale[n * G + g])
                 * static_cast<double>(xscale);
        }
        y[n] = static_cast<float>(acc);
    }
}

// -----------------------------------------------------------------------------
//  Kernel AVX2. Sin permutaciones en el bucle (x ya viene permutado).
// -----------------------------------------------------------------------------
#if defined(__AVX2__)
inline void gemv_int4(const Shape& s,
                      const std::uint8_t* __restrict w,
                      const _Float16* __restrict wscale,
                      const std::int32_t* __restrict codesum,
                      const std::uint8_t* __restrict xperm,
                      float xscale,
                      float* __restrict y) {
    const __m256i lut = _mm256_setr_epi8(
        0, 1, 2, 3, 4, 5, 6, 7, -8, -7, -6, -5, -4, -3, -2, -1,
        0, 1, 2, 3, 4, 5, 6, 7, -8, -7, -6, -5, -4, -3, -2, -1);
    const __m256i m0f = _mm256_set1_epi8(0x0F);
    const __m256i ones16 = _mm256_set1_epi16(1);

    const std::size_t G = s.groups(), rb = s.row_bytes();
    const std::size_t chunk = s.group / 2;   // bytes por grupo (32 para 64)

    for (std::size_t n = 0; n < s.N; ++n) {
        const std::uint8_t* row = w + n * rb;
        __m256 facc = _mm256_setzero_ps();

        for (std::size_t g = 0; g < G; ++g) {
            const std::uint8_t* p = row + g * chunk;
            const __m256i v = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(p));
            const __m256i lo = _mm256_and_si256(v, m0f);
            const __m256i hi = _mm256_and_si256(_mm256_srli_epi16(v, 4), m0f);
            const __m256i wl = _mm256_shuffle_epi8(lut, lo);
            const __m256i wh = _mm256_shuffle_epi8(lut, hi);
            // Sin vperm2i128: x viene en este orden exacto.
            const __m256i a = _mm256_unpacklo_epi8(wl, wh);  // [v0..15 | v32..47]
            const __m256i b = _mm256_unpackhi_epi8(wl, wh);  // [v16..31| v48..63]

            const std::uint8_t* xg = xperm + g * s.group;
            const __m256i x0 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(xg));
            const __m256i x1 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(xg + 32));

            __m256i acc = _mm256_madd_epi16(_mm256_maddubs_epi16(x0, a), ones16);
            acc = _mm256_add_epi32(
                acc, _mm256_madd_epi16(_mm256_maddubs_epi16(x1, b), ones16));

            // Sesgo del zero-point: se resta UNA vez, en el carril 0. Restarlo
            // con set1() lo aplicaría a los 8 carriles y la suma horizontal lo
            // multiplicaría por 8 (bug encontrado por test_pack_roundtrip).
            acc = _mm256_sub_epi32(
                acc, _mm256_setr_epi32(128 * codesum[n * G + g], 0, 0, 0, 0, 0, 0, 0));

            const float fs = static_cast<float>(wscale[n * G + g]) * xscale;
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

}  // namespace caudal

#endif  // CAUDAL_GEMV_INT4_HPP
