#pragma once
// =============================================================================
//  aquifer · include/aquifer/morton.hpp
//  Indexador Espacial Cúbico 3D a nivel de bit (Z-Order / Morton Code).
// =============================================================================
//  Proyecta coordenadas normalizadas del cubo [-1.0, 1.0]³ a un índice entero
//  de 64 bits (uint64_t) en ~1 nanosegundo usando instrucciones BMI2 del CPU
//  (_pdep_u64 / _pext_u64).
//
//  Propiedad fundamental:
//    La cercanía geométrica 3D se preserva como contigüidad espacial en 1D,
//    alineando los vóxeles de tensores con las líneas de caché L3 y páginas THP.
// =============================================================================

#include <algorithm>
#include <cstdint>
#include <cmath>
#if defined(__x86_64__) || defined(_M_X64)
#include <immintrin.h>
#endif

namespace aquifer {

// Máscaras de bits para entrelazado 3D (Z-Order):
//   bit_pos(x) = 3*i
//   bit_pos(y) = 3*i + 1
//   bit_pos(z) = 3*i + 2
inline constexpr std::uint64_t kMaskX = 0x9249249249249249ULL; // bits 0, 3, 6, 9...
inline constexpr std::uint64_t kMaskY = 0x2492492492492492ULL; // bits 1, 4, 7, 10...
inline constexpr std::uint64_t kMaskZ = 0x4924924924924924ULL; // bits 2, 5, 8, 11...

// 21 bits por eje -> 21 * 3 = 63 bits útiles (caben en uint64_t)
inline constexpr std::uint32_t kMaxCoord21 = (1u << 21) - 1u; // 2.097.151

/// Normaliza un valor flotante en [-1.0, 1.0] al rango entero [0, 2^21 - 1].
inline std::uint32_t quantize_coord21(double v) noexcept {
    const double clamped = std::clamp(v, -1.0, 1.0);
    const double norm = (clamped + 1.0) * 0.5; // [0.0, 1.0]
    return static_cast<std::uint32_t>(std::round(norm * double(kMaxCoord21)));
}

/// Desnormaliza un entero [0, 2^21 - 1] a coordenada flotante [-1.0, 1.0].
inline double dequantize_coord21(std::uint32_t u) noexcept {
    const double norm = double(u) / double(kMaxCoord21);
    return (norm * 2.0) - 1.0;
}

#if defined(__BMI2__)
/// Codificación Morton 3D por hardware (3 ciclos de reloj en x86-64).
inline std::uint64_t encode_morton3d_u21(std::uint32_t x, std::uint32_t y, std::uint32_t z) noexcept {
    return _pdep_u64(x, kMaskX) | _pdep_u64(y, kMaskY) | _pdep_u64(z, kMaskZ);
}

/// Decodificación Morton 3D por hardware (3 ciclos de reloj en x86-64).
inline void decode_morton3d_u21(std::uint64_t code, std::uint32_t& x, std::uint32_t& y, std::uint32_t& z) noexcept {
    x = static_cast<std::uint32_t>(_pext_u64(code, kMaskX));
    y = static_cast<std::uint32_t>(_pext_u64(code, kMaskY));
    z = static_cast<std::uint32_t>(_pext_u64(code, kMaskZ));
}
#else
// Fallback portable sin BMI2 por expansión de bits
inline std::uint64_t split_by_3(std::uint32_t a) noexcept {
    std::uint64_t x = a & 0x1fffff; // 21 bits
    x = (x | (x << 32)) & 0x1f00000000ffffULL;
    x = (x | (x << 16)) & 0x1f0000ff0000ffULL;
    x = (x | (x << 8))  & 0x100f00f00f00f00fULL;
    x = (x | (x << 4))  & 0x10c30c30c30c30c3ULL;
    x = (x | (x << 2))  & 0x1249249249249249ULL;
    return x;
}

inline std::uint64_t encode_morton3d_u21(std::uint32_t x, std::uint32_t y, std::uint32_t z) noexcept {
    return split_by_3(x) | (split_by_3(y) << 1) | (split_by_3(z) << 2);
}

inline std::uint32_t compact_by_3(std::uint64_t x) noexcept {
    x &= 0x1249249249249249ULL;
    x = (x ^ (x >> 2))  & 0x10c30c30c30c30c3ULL;
    x = (x ^ (x >> 4))  & 0x100f00f00f00f00fULL;
    x = (x ^ (x >> 8))  & 0x1f0000ff0000ffULL;
    x = (x ^ (x >> 16)) & 0x1f00000000ffffULL;
    x = (x ^ (x >> 32)) & 0x1fffffULL;
    return static_cast<std::uint32_t>(x);
}

inline void decode_morton3d_u21(std::uint64_t code, std::uint32_t& x, std::uint32_t& y, std::uint32_t& z) noexcept {
    x = compact_by_3(code);
    y = compact_by_3(code >> 1);
    z = compact_by_3(code >> 2);
}
#endif

/// Codifica directamente coordenadas del cubo [-1, 1]³ a clave de Vóxel uint64_t.
inline std::uint64_t cube_to_morton(double x, double y, double z) noexcept {
    return encode_morton3d_u21(quantize_coord21(x), quantize_coord21(y), quantize_coord21(z));
}

/// Decodifica clave de Vóxel uint64_t a coordenadas del cubo [-1, 1]³.
inline void morton_to_cube(std::uint64_t code, double& x, double& y, double& z) noexcept {
    std::uint32_t qx, qy, qz;
    decode_morton3d_u21(code, qx, qy, qz);
    x = dequantize_coord21(qx);
    y = dequantize_coord21(qy);
    z = dequantize_coord21(qz);
}

/// Identificador de un Vóxel de Tensores en el hiperespacio.
struct VoxelKey {
    std::uint64_t code{0};

    constexpr VoxelKey() = default;
    constexpr explicit VoxelKey(std::uint64_t c) : code(c) {}
    VoxelKey(double x, double y, double z) : code(cube_to_morton(x, y, z)) {}

    [[nodiscard]] bool operator==(const VoxelKey& o) const noexcept { return code == o.code; }
    [[nodiscard]] bool operator<(const VoxelKey& o) const noexcept { return code < o.code; }
};

} // namespace aquifer
