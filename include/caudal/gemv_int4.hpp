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
                dot += static_cast<long long>(code)
                     * static_cast<int>(xperm[base + j]);
            }
            acc += static_cast<double>(dot)
                 * static_cast<double>(wscale[n * G + g])
                 * static_cast<double>(xscale);
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
                      float xscale,
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

            const float fs = static_cast<float>(sc[g]) * xscale;
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
                          float xscale,
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
                _mm_set1_ps(static_cast<float>(sc[g1]) * xscale),
                _mm_set1_ps(static_cast<float>(sc[g0]) * xscale));
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

}  // namespace caudal

#endif  // CAUDAL_GEMV_INT4_HPP
