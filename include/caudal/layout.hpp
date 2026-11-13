#ifndef CAUDAL_LAYOUT_HPP
#define CAUDAL_LAYOUT_HPP

// =============================================================================
//  caudal · include/caudal/layout.hpp
//  CONGELAMIENTO DEL ABI DE LA ARENA VIRTUAL
// =============================================================================
//  Este archivo es el CONTRATO, no la implementación. No contiene I/O, ni
//  política, ni lógica de negocio. Es lo único que el packer (Python) y los
//  kernels (C++) tienen que respetar para poder hablarse.
//
//  Si cambiás kNodeSize o kRegionShift, TODO lo empaquetado offline queda
//  inválido: hay que reempaquetar 4.3 GB. Por eso los static_assert de abajo.
//
//  Diseño (spec corregida tras auditoría multi-modelo, 2026-11-13):
//
//    · 1 TiB de espacio virtual reservado UNA sola vez.
//      Base en runtime (mmap anónimo o memfd). NUNCA MAP_FIXED sobre una
//      dirección arbitraria: puede pisar heap/PIE. El patrón correcto es
//      reservar con PROT_NONE y rellenar con MAP_FIXED *dentro* de esa reserva.
//
//    · 16 regiones fijas de 64 GiB. El nibble alto del offset es la región.
//
//    · Offset de nodo = uint32_t en UNIDADES DE 16 BYTES.
//          2^32 × 16 B = 68'719'476'736 B = 64 GiB EXACTOS
//      Esto evita el error detectado en auditoría: un uint32 como byte-offset
//      sólo cubre 4 GiB, insuficiente incluso para los pesos (4.3 GB) y para
//      el KV a 128k (7.52 GB). Escalado, cubre la región entera sin residuo.
//      Y coincide con el tamaño de celda de HiperRAM: el offset ES el índice.
//
//  Invariante del nodo: 16 B alineados, nunca cruzando línea de caché si se
//  empaqueta de a dos por línea de 32 B (ver packer).
// =============================================================================

#include <cstddef>
#include <cstdint>

#if __cplusplus < 202100L
#  error "caudal requiere C++23 (-std=c++23)"
#endif

namespace caudal {

// -----------------------------------------------------------------------------
//  Grano de direccionamiento
// -----------------------------------------------------------------------------
inline constexpr std::size_t kNodeSize  = 16;   // bytes por celda
inline constexpr std::size_t kNodeShift = 4;    // log2(kNodeSize)
inline constexpr std::size_t kNodeMask  = kNodeSize - 1;

// -----------------------------------------------------------------------------
//  Regiones
// -----------------------------------------------------------------------------
inline constexpr std::size_t kRegionShift  = 36;                        // 64 GiB
inline constexpr std::size_t kRegionSize   = std::size_t{1} << kRegionShift;
inline constexpr std::size_t kRegionMask   = kRegionSize - 1;
inline constexpr std::size_t kArenaCount   = 16;
inline constexpr std::size_t kTotalVirtual = kRegionSize * kArenaCount; // 1 TiB

static_assert((std::size_t{1} << kNodeShift) == kNodeSize);
static_assert(kRegionSize   == (std::size_t{64} << 30), "región = 64 GiB");
static_assert(kTotalVirtual == (std::size_t{1} << 40),  "arena total = 1 TiB");

// El invariante central. Si esto falla, alguien tocó el grano o el tamaño de
// región y el ABI de 32 bits ya no cierra.
static_assert((std::uint64_t{1} << 32) * kNodeSize == kRegionSize,
              "uint32 * 16 B debe cubrir 64 GiB exactos, sin residuo");

// -----------------------------------------------------------------------------
//  Identidad de arena
// -----------------------------------------------------------------------------
enum class Arena : std::uint8_t {
    Weights = 0,   // A0: pesos target, int4 lineal contiguo (decode)
    KvCache = 1,   // A1: KV paginada, memfd con doble mapeo (circular sin ramas)
    Index   = 2,   // A2: índice Quest: cajas min/max por bloque de 256 tokens
    Draft   = 3,   // A3: modelo draft 0.5B (speculative decoding)
};

inline constexpr std::uint8_t kArenaNamed = 4;

static_assert(kArenaCount <= 256, "RegionId tiene que caber en uint8_t");
static_assert(static_cast<std::uint8_t>(Arena::Draft) < kArenaCount);

// -----------------------------------------------------------------------------
//  Tipos de referencia
// -----------------------------------------------------------------------------
using RegionId = std::uint8_t;    // 0..15
using NodeIdx  = std::uint32_t;   // índice de nodo DENTRO de una región

// -----------------------------------------------------------------------------
//  Aritmética de direcciones
//  `base` se obtiene en runtime. Nunca se asume una dirección concreta.
// -----------------------------------------------------------------------------

[[nodiscard]] constexpr std::size_t region_offset(RegionId r) noexcept {
    return static_cast<std::size_t>(r) << kRegionShift;
}

[[nodiscard]] constexpr std::size_t node_bytes(NodeIdx n) noexcept {
    return static_cast<std::size_t>(n) << kNodeShift;
}

// Offset de byte relativo a `base`. Es de 64 bits a propósito: puede exceder
// 4 GiB (los pesos solos son 4.3 GB).
[[nodiscard]] constexpr std::size_t offset_of(Arena a, NodeIdx n) noexcept {
    return region_offset(static_cast<RegionId>(a)) + node_bytes(n);
}

[[nodiscard]] constexpr std::byte* at(std::byte* base, Arena a, NodeIdx n) noexcept {
    return base + offset_of(a, n);
}

[[nodiscard]] constexpr const std::byte* at(const std::byte* base,
                                            Arena a, NodeIdx n) noexcept {
    return base + offset_of(a, n);
}

// Inversa.
[[nodiscard]] constexpr RegionId region_of(std::size_t off) noexcept {
    return static_cast<RegionId>(off >> kRegionShift);
}

[[nodiscard]] constexpr NodeIdx node_of(std::size_t off) noexcept {
    return static_cast<NodeIdx>((off & kRegionMask) >> kNodeShift);
}

// Cuántos nodos caben en una región. Son 2^32, que NO cabe en NodeIdx: el
// índice máximo válido es 2^32 - 1 y el conteo es uno más. Por eso devuelve
// uint64_t — es el off-by-one clásico de este diseño.
[[nodiscard]] constexpr std::uint64_t nodes_per_region() noexcept {
    return kRegionSize / kNodeSize;
}

static_assert(nodes_per_region() == (std::uint64_t{1} << 32));
static_assert(static_cast<std::uint64_t>(~NodeIdx{0}) + 1 == nodes_per_region());

}  // namespace caudal

#endif  // CAUDAL_LAYOUT_HPP
