// =============================================================================
//  caudal · bench/stencil_3d.cpp
//  STENCIL 3x3x3 SOBRE UN VOLUMEN: row-major vs Morton global vs bricked 16^3.
// =============================================================================
//  Pregunta: ¿cuánto vale realmente el ordenamiento espacial de la memoria?
//
//  Tres layouts, 1 byte por voxel (para que 4x4x4 = 64 B = una línea exacta y
//  16^3 = 4096 B = una página exacta):
//
//    ROW     idx = (z*N + y)*N + x
//    MORTON  idx = morton3(x,y,z)           (8 bits/eje = 24 bits)
//    BRICK   idx = morton3_4(bx,by,bz)*4096 + morton3_4(x&15,y&15,z&15)
//
//  Se miden DOS cosas distintas y complementarias:
//    1. LÍNEAS POR STENCIL: promedio de líneas de 64 B distintas que tocan los
//       27 vecinos de un voxel. Es independiente del tamaño de caché: mide
//       cuánto tráfico EXIGE el layout, no cuánto acierta.
//    2. ns/VOXEL: la evaluación completa, incluyendo el costo del mapeo
//       (Morton no es gratis: 3 lookups + shifts por dirección).
//
//  Y se corre a dos tamaños: 64^3 (256 KB, cabe en L2) y 256^3 (16 MB, DRAM).
//  La predicción falsable es que el layout importa en L2 y deja de importar
//  cuando el working set desborda a DRAM.
//
//  Uso: ./stencil_3d [side] [reps]
// =============================================================================

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <vector>

namespace {

volatile std::uint64_t g_sink = 0;

std::uint32_t g_part[256];

inline std::uint32_t part1by2(std::uint32_t n) {
    n &= 0x000003ffu;
    n = (n ^ (n << 16)) & 0xff0000ffu;
    n = (n ^ (n <<  8)) & 0x0300f00fu;
    n = (n ^ (n <<  4)) & 0x030c30c3u;
    n = (n ^ (n <<  2)) & 0x09249249u;
    return n;
}

double now_ns() {
    timespec t{};
    clock_gettime(CLOCK_MONOTONIC, &t);
    return double(t.tv_sec) * 1e9 + double(t.tv_nsec);
}

// --- mapas de dirección ------------------------------------------------------
struct RowMap {
    int N;
    inline std::size_t operator()(int x, int y, int z) const {
        return (std::size_t(z) * N + y) * N + x;
    }
};

struct MortonMap {
    inline std::size_t operator()(int x, int y, int z) const {
        return std::size_t(g_part[x] | (g_part[y] << 1) | (g_part[z] << 2));
    }
};

struct BrickMap {
    inline std::size_t operator()(int x, int y, int z) const {
        const std::uint32_t b =
            (g_part[x >> 4] | (g_part[y >> 4] << 1) | (g_part[z >> 4] << 2));
        const std::uint32_t l = (g_part[x & 15] | (g_part[y & 15] << 1) |
                                 (g_part[z & 15] << 2));
        return (std::size_t(b) << 12) | l;
    }
};

// --- 1. líneas de caché exigidas por un stencil ------------------------------
template <class F>
double lines_per_stencil(F map, int N, int samples) {
    const std::size_t total = std::size_t(N) * N * N;
    std::vector<std::uint64_t> lines;
    lines.reserve(27);

    std::uint32_t rng = 12345u;
    long long sum = 0;
    for (int s = 0; s < samples; ++s) {
        rng = rng * 1664525u + 1013904223u;
        const int x = int(rng % std::uint32_t(N - 2)) + 1;
        rng = rng * 1664525u + 1013904223u;
        const int y = int(rng % std::uint32_t(N - 2)) + 1;
        rng = rng * 1664525u + 1013904223u;
        const int z = int(rng % std::uint32_t(N - 2)) + 1;

        lines.clear();
        for (int dz = -1; dz <= 1; ++dz)
            for (int dy = -1; dy <= 1; ++dy)
                for (int dx = -1; dx <= 1; ++dx) {
                    const std::size_t a = map(x + dx, y + dy, z + dz);
                    lines.push_back(a >> 6);          // línea de 64 B
                }
        std::sort(lines.begin(), lines.end());
        sum += long(std::unique(lines.begin(), lines.end()) - lines.begin());
    }
    (void)total;
    return double(sum) / double(samples);
}

// --- 2. evaluación completa del stencil --------------------------------------
template <class F>
double stencil_ns(std::vector<std::uint8_t>& vol, std::vector<std::uint8_t>& out,
                  F map, int N, int reps) {
    const int n = N - 2;
    const double t0 = now_ns();
    for (int r = 0; r < reps; ++r) {
        for (int z = 1; z < N - 1; ++z)
            for (int y = 1; y < N - 1; ++y)
                for (int x = 1; x < N - 1; ++x) {
                    unsigned s = 0;
                    for (int dz = -1; dz <= 1; ++dz)
                        for (int dy = -1; dy <= 1; ++dy)
                            for (int dx = -1; dx <= 1; ++dx)
                                s += vol[map(x + dx, y + dy, z + dz)];
                    out[map(x, y, z)] = std::uint8_t(s);
                }
        g_sink ^= out[std::size_t(r) % out.size()];
    }
    const double t1 = now_ns();
    return (t1 - t0) / (double(reps) * double(n) * double(n) * double(n));
}

// --- campo logico: el MISMO dato en los tres layouts -------------------------
// Llenar cada layout con bytes aleatorios secuenciales daria TRES volumenes
// distintos (una permutacion de datos i.i.d. es otro dato). El campo se genera
// por coordenada y se escribe en cada layout con su propio mapa.
inline std::uint8_t field(int x, int y, int z) {
    std::uint32_t h = std::uint32_t(x) * 73856093u ^ std::uint32_t(y) * 19349663u
                    ^ std::uint32_t(z) * 83492791u;
    h ^= h >> 13;
    return std::uint8_t((h * 2654435761u) >> 24);
}


// --- version OPTIMIZADA: tabla de 27 offsets por posicion local (mod 4) ------
// El mapa se calcula UNA vez por voxel; los 27 vecinos son base+delta.
// Vale para ROW y MORTON: un +-1 nunca cambia mas que el bit 2 de la
// coordenada, que en Morton ocupa una posicion fija, asi que el delta solo
// depende de (x&3, y&3, z&3). BRICK no lo admite: cruzar ladrillo salta de
// forma dependiente de la posicion del ladrillo en la rejilla.
template <class F>
void build_delta(F map, std::vector<std::int32_t>& d) {
    d.assign(64 * 27, 0);
    for (int pp = 0; pp < 64; ++pp) {
        const int x0 = pp & 3, y0 = (pp >> 2) & 3, z0 = (pp >> 4) & 3;
        const int x = 4 * 16 + x0, y = 4 * 16 + y0, z = 4 * 16 + z0;
        const std::int32_t b = std::int32_t(map(x, y, z));
        int k = 0;
        for (int dz = -1; dz <= 1; ++dz)
            for (int dy = -1; dy <= 1; ++dy)
                for (int dx = -1; dx <= 1; ++dx)
                    d[pp * 27 + (k++)] =
                        std::int32_t(map(x + dx, y + dy, z + dz)) - b;
    }
}

template <class F>
double stencil_ns_opt(std::vector<std::uint8_t>& vol, std::vector<std::uint8_t>& out,
                      F map, int N, int reps, const std::vector<std::int32_t>& d) {
    const int n = N - 2;
    const double t0 = now_ns();
    for (int r = 0; r < reps; ++r) {
        for (int z = 1; z < N - 1; ++z)
            for (int y = 1; y < N - 1; ++y)
                for (int x = 1; x < N - 1; ++x) {
                    const std::size_t b = map(x, y, z);
                    const int pp = (x & 3) | ((y & 3) << 2) | ((z & 3) << 4);
                    const std::int32_t* dd = &d[pp * 27];
                    unsigned s = 0;
                    for (int k = 0; k < 27; ++k)
                        s += vol[std::size_t(std::ptrdiff_t(b) + dd[k])];
                    out[b] = std::uint8_t(s);
                }
        g_sink ^= out[std::size_t(r) % out.size()];
    }
    const double t1 = now_ns();
    return (t1 - t0) / (double(reps) * double(n) * double(n) * double(n));
}

}  // namespace

int main(int argc, char** argv) {
    int N = 256;
    int reps = 3;
    if (argc > 1) N = std::atoi(argv[1]);
    if (argc > 2) reps = std::atoi(argv[2]);
    for (int i = 0; i < 256; ++i) g_part[i] = part1by2(std::uint32_t(i));

    const std::size_t total = std::size_t(N) * N * N;
    std::printf("\ncaudal · stencil_3d   N=%d  (%.1f MB, 1 B/voxel)  reps=%d\n\n",
                N, double(total) / 1e6, reps);

    const RowMap    row{N};
    const MortonMap mort;
    const BrickMap  brick;

    // Tres volumenes, el MISMO campo logico escrito con cada mapa.
    std::vector<std::uint8_t> v_row(total), v_mor(total), v_bri(total);
    std::vector<std::uint8_t> o_row(total), o_mor(total), o_bri(total);
    for (int z = 0; z < N; ++z)
        for (int y = 0; y < N; ++y)
            for (int x = 0; x < N; ++x) {
                const std::uint8_t f = field(x, y, z);
                v_row[row(x, y, z)] = f;
                v_mor[mort(x, y, z)] = f;
                v_bri[brick(x, y, z)] = f;
            }

    // --- verificación 1: los mapas son biyecciones consistentes -------------
    bool maps_ok = true;
    for (int s = 0; s < 100000 && maps_ok; ++s) {
        std::uint32_t r = std::uint32_t(s) * 2654435761u;
        const int x = int(r % std::uint32_t(N));
        const int y = int((r >> 8) % std::uint32_t(N));
        const int z = int((r >> 16) % std::uint32_t(N));
        const std::uint8_t f = field(x, y, z);
        if (v_row[row(x,y,z)] != f || v_mor[mort(x,y,z)] != f ||
            v_bri[brick(x,y,z)] != f) maps_ok = false;
    }
    std::printf("mapas biyectivos: %s\n", maps_ok ? "OK" : "FALLA");
    if (!maps_ok) return 1;

    // --- líneas por stencil --------------------------------------------------
    const int samples = 200000;
    const double l_row = lines_per_stencil(row, N, samples);
    const double l_mor = lines_per_stencil(mort, N, samples);
    const double l_bri = lines_per_stencil(brick, N, samples);

    // --- verificación 2: mismo resultado en coordenadas logicas -------------
    auto run_into = [&](auto map, std::vector<std::uint8_t>& vol,
                        std::vector<std::uint8_t>& o) {
        for (int z = 1; z < N - 1; ++z)
            for (int y = 1; y < N - 1; ++y)
                for (int x = 1; x < N - 1; ++x) {
                    unsigned s = 0;
                    for (int dz = -1; dz <= 1; ++dz)
                        for (int dy = -1; dy <= 1; ++dy)
                            for (int dx = -1; dx <= 1; ++dx)
                                s += vol[map(x + dx, y + dy, z + dz)];
                    o[map(x, y, z)] = std::uint8_t(s);
                }
    };
    run_into(row, v_row, o_row);
    run_into(mort, v_mor, o_mor);
    run_into(brick, v_bri, o_bri);
    bool same_ok = true;
    for (int s = 0; s < 100000 && same_ok; ++s) {
        std::uint32_t r = std::uint32_t(s) * 40503u + 12345u;
        const int x = int(r % std::uint32_t(N - 2)) + 1;
        const int y = int((r >> 7) % std::uint32_t(N - 2)) + 1;
        const int z = int((r >> 14) % std::uint32_t(N - 2)) + 1;
        const std::uint8_t a = o_row[row(x,y,z)];
        if (o_mor[mort(x,y,z)] != a || o_bri[brick(x,y,z)] != a) same_ok = false;
    }
    std::printf("equivalencia     : %s\n\n", same_ok ? "OK (mismo stencil)" : "FALLA");
    if (!same_ok) return 1;

    // --- ns/voxel ------------------------------------------------------------
    const double t_row = stencil_ns(v_row, o_row, row, N, reps);
    const double t_mor = stencil_ns(v_mor, o_mor, mort, N, reps);
    const double t_bri = stencil_ns(v_bri, o_bri, brick, N, reps);

    std::vector<std::int32_t> d_row, d_mor;
    build_delta(row, d_row);
    build_delta(mort, d_mor);
    const double o_row_t = stencil_ns_opt(v_row, o_row, row, N, reps, d_row);
    const double o_mor_t = stencil_ns_opt(v_mor, o_mor, mort, N, reps, d_mor);

    std::printf("%-8s %10s %10s %10s %12s\n",
                "layout", "lineas/st", "B utiles%", "ns/voxel", "vs row-major");
    auto line = [&](const char* nm, double l, double t) {
        const double eff = 100.0 * 27.0 / (l * 64.0);
        std::printf("%-8s %10.2f %9.1f%% %10.2f %11.2fx\n",
                    nm, l, eff, t, t_row / t);
    };
    line("ROW", l_row, t_row);
    line("MORTON", l_mor, t_mor);
    line("BRICK", l_bri, t_bri);

    std::printf("\n");
    std::printf("%-14s %10s %12s\n", "OPTIMIZADO", "ns/voxel", "vs row-opt");
    std::printf("%-14s %10.2f %11.2fx\n", "ROW+tabla", o_row_t, 1.0);
    std::printf("%-14s %10.2f %11.2fx\n", "MORTON+tabla", o_mor_t, o_row_t / o_mor_t);

    std::printf("\n");
    std::printf("nota: 27 B utiles por stencil. Techo teorico = 1 linea (4x4x4).\n");
    std::printf("      El tamano del working set (%zu MB) decide si el orden paga.\n\n",
                (2 * total) / 1000000);
    return 0;
}
