#include "caudal/layout.hpp"
#include <iostream>
#include <vector>
#include <array>
#include <cmath>
#include <cassert>
#include <immintrin.h>

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

// Implementación SSE4.1 de referencia para comparar
void sse4_gemv_row(const uint8_t* row,
                   const _Float16* sc,
                   const _Float16* mn,
                   const float* xmin_scale,
                   const int8_t* xperm,
                   const float* xscale,
                   float* y,
                   size_t K,
                   size_t G) {
    const __m128i m0f = _mm_set1_epi8(0x0F);
    const __m128i ones16 = _mm_set1_epi16(1);
    const size_t blocks = K / 64;

    float acc = 0.0f;

    for (size_t bb = 0; bb < blocks; ++bb) {
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

        const size_t g0 = 2 * bb, g1 = 2 * bb + 1;
        const float w0 = static_cast<float>(sc[g0]);
        const float w1 = static_cast<float>(sc[g1]);
        const float mn0 = static_cast<float>(mn[g0]);
        const float mn1 = static_cast<float>(mn[g1]);

        const int8_t* xm = xperm + bb * 64;
        const __m128i x0_0 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(xm));
        const __m128i x0_1 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(xm + 16));
        const __m128i x1_0 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(xm + 32));
        const __m128i x1_1 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(xm + 48));

        // A0 * xm[0..15] + B0 * xm[32..47]
        __m128i d0 = _mm_madd_epi16(_mm_maddubs_epi16(A0, x0_0), ones16);
        d0 = _mm_add_epi32(d0, _mm_madd_epi16(_mm_maddubs_epi16(B0, x1_0), ones16));

        // A1 * xm[16..31] + B1 * xm[48..63]
        __m128i d1 = _mm_madd_epi16(_mm_maddubs_epi16(A1, x0_1), ones16);
        d1 = _mm_add_epi32(d1, _mm_madd_epi16(_mm_maddubs_epi16(B1, x1_1), ones16));

        alignas(16) int32_t r0[4], r1[4];
        _mm_store_si128(reinterpret_cast<__m128i*>(r0), d0);
        _mm_store_si128(reinterpret_cast<__m128i*>(r1), d1);

        float dot0 = float(r0[0] + r0[1] + r0[2] + r0[3]);
        float dot1 = float(r1[0] + r1[1] + r1[2] + r1[3]);

        acc += dot0 * (w0 * xscale[g0]) + dot1 * (w1 * xscale[g1])
             + mn0 * xmin_scale[g0] + mn1 * xmin_scale[g1];
    }
    *y = acc;
}

int main() {
    std::cout << "=== Test SSE4.1 Affine int4 GEMV Kernel ===\n";
    const size_t K = 896;
    const size_t G = K / 32;

    std::vector<uint8_t> w(K / 2);
    for (size_t i = 0; i < w.size(); ++i) w[i] = uint8_t((i * 17) & 0xFF);

    std::vector<_Float16> sc(G), mn(G);
    for (size_t g = 0; g < G; ++g) {
        sc[g] = _Float16(0.05f * float(g + 1));
        mn[g] = _Float16(-0.02f * float(g + 1));
    }

    std::vector<int8_t> x(K), xperm(K);
    for (size_t i = 0; i < K; ++i) x[i] = int8_t((int(i) % 250) - 125);

    const size_t blocks = K / 64;
    for (size_t bb = 0; bb < blocks; ++bb)
        for (size_t i = 0; i < 64; ++i)
            xperm[bb * 64 + i] = x[bb * 64 + kPerm[i]];

    std::vector<float> xs(G), xms(G);
    for (size_t g = 0; g < G; ++g) {
        xs[g] = 0.01f;
        int32_t sum = 0;
        for (size_t j = 0; j < 32; ++j) sum += x[g * 32 + j];
        xms[g] = float(sum) * xs[g];
    }

    float y_sse = 0.0f;
    sse4_gemv_row(w.data(), sc.data(), mn.data(), xms.data(), xperm.data(), xs.data(), &y_sse, K, G);

    // Referencia escalar pura
    float y_ref = 0.0f;
    for (size_t g = 0; g < G; ++g) {
        float sum_q_x = 0.0f;
        float sum_x = 0.0f;
        for (size_t j = 0; j < 32; ++j) {
            size_t idx = g * 32 + j;
            uint8_t byte = w[idx / 2];
            int q = (idx % 2 == 0) ? (byte & 0x0F) : (byte >> 4);
            sum_q_x += float(q) * float(x[idx]);
            sum_x += float(x[idx]);
        }
        y_ref += float(sc[g]) * xs[g] * sum_q_x + float(mn[g]) * xs[g] * sum_x;
    }

    std::cout << "  Resultado SSE4: " << y_sse << "\n";
    std::cout << "  Resultado Ref:  " << y_ref << "\n";
    std::cout << "  Diferencia:     " << std::abs(y_sse - y_ref) << "\n";

    assert(std::abs(y_sse - y_ref) < 1e-3f);
    std::cout << "=== TEST SSE4 EXITOSO: COINCIDENCIA EXACTA ===\n";
    return 0;
}
