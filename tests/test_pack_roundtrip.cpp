// =============================================================================
//  caudal · tests/test_pack_roundtrip.cpp
//  END-TO-END: packer sintético -> layout A0 -> kernel AVX2 -> referencia.
// =============================================================================
//  Tres comparaciones, en orden de fuerza:
//    1. kernel AVX2   == referencia escalar C++   (mismos bytes empaquetados)
//       => valida desempacado unsigned, pre-permutado de x, escalas por grupo
//          y la corrección del sesgo 8·Σx.
//    2. referencia C++ == ref.bin de Python       (calculada en float64)
//       => valida el PACKER, no sólo el kernel.
//    3. tamaños de archivo == lo esperado
//       => valida que packer y test no se desincronicen.
//
//  Uso: ./test_pack_roundtrip <dir_artefacto> [N] [K] [M]
// =============================================================================

#include "caudal/gemv_int4.hpp"
#include "caudal/layout.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <string>
#include <vector>

namespace {

int g_fail = 0;

void check(bool cond, const char* what) {
    std::printf("  [%s] %s\n", cond ? "PASS" : "FAIL", what);
    if (!cond) ++g_fail;
}

std::vector<std::uint8_t> slurp(const std::string& p, bool& ok) {
    std::ifstream f(p, std::ios::binary | std::ios::ate);
    if (!f) { ok = false; return {}; }
    const auto n = static_cast<std::size_t>(f.tellg());
    f.seekg(0);
    std::vector<std::uint8_t> v(n);
    f.read(reinterpret_cast<char*>(v.data()), static_cast<std::streamsize>(n));
    ok = static_cast<bool>(f);
    return v;
}

}  // namespace

int main(int argc, char** argv) {
    const std::string dir = (argc > 1) ? argv[1] : "build/artefacto";
    caudal::Shape s;
    s.N = (argc > 2) ? static_cast<std::size_t>(std::strtoull(argv[2], nullptr, 10)) : 256;
    s.K = (argc > 3) ? static_cast<std::size_t>(std::strtoull(argv[3], nullptr, 10)) : 512;
    const std::size_t M = (argc > 4)
        ? static_cast<std::size_t>(std::strtoull(argv[4], nullptr, 10)) : 4;
    s.group = 64;

    std::printf("\n");
    std::printf("caudal · test_pack_roundtrip\n");
    std::printf("artefacto : %s\n", dir.c_str());
    std::printf("forma     : N=%zu K=%zu M=%zu grupo=%zu  (%zu grupos/fila)\n\n",
                s.N, s.K, M, s.group, s.groups());

    bool ok = true;
    auto w      = slurp(dir + "/w_packed.bin", ok);
    auto wscale = slurp(dir + "/w_scale.bin", ok);
    auto xperm  = slurp(dir + "/x_perm.bin", ok);
    auto xscale = slurp(dir + "/x_scale.bin", ok);
    auto refbin = slurp(dir + "/ref.bin", ok);
    check(ok, "los 5 archivos del artefacto existen y se leen");

    const std::size_t G = s.groups();
    check(w.size() == s.N * s.row_bytes(), "w_packed: N * K/2");
    check(wscale.size() == s.N * G * sizeof(_Float16), "w_scale: N * G * 2 B");
    check(xperm.size() == M * s.K, "x_perm: M * K");
    check(xscale.size() == M * sizeof(float), "x_scale: M * 4 B");
    check(refbin.size() == M * s.N * sizeof(float), "ref: M * N * 4 B");
    check(s.row_bytes() % caudal::kNodeSize == 0,
          "row_bytes es múltiplo de 16 B (offset de fila = NodeIdx válido)");
    if (g_fail) { std::printf("\nRESULTADO: %d fallo(s) de tamaño\n\n", g_fail); return 1; }

    const auto* ws  = reinterpret_cast<const _Float16*>(wscale.data());
    const auto* xs  = reinterpret_cast<const float*>(xscale.data());
    const auto* rf  = reinterpret_cast<const float*>(refbin.data());
    const auto* xp  = reinterpret_cast<const std::int8_t*>(xperm.data());

    // El sesgo 8·Σx se calcula por token, como hará el motor real.
    std::vector<std::int32_t> xgsum(G);

    std::vector<float> y_ref(s.N), y_fast(s.N);
    double max_err_fast = 0.0, max_err_pack = 0.0, max_mag = 0.0;

    for (std::size_t m = 0; m < M; ++m) {
        const std::int8_t* xm = xp + m * s.K;
        caudal::compute_x_gsum(s, xm, xgsum.data());

        caudal::gemv_ref(s, w.data(), ws, xm, xs[m], y_ref.data());
#if defined(__AVX2__)
        caudal::gemv_int4(s, w.data(), ws, xgsum.data(), xm, xs[m], y_fast.data());
#else
        std::printf("  (sin AVX2: se omite la comparacion de kernel)\n");
        for (std::size_t n = 0; n < s.N; ++n) y_fast[n] = y_ref[n];
#endif

        for (std::size_t n = 0; n < s.N; ++n) {
            const double r = static_cast<double>(rf[m * s.N + n]);
            max_mag = std::max(max_mag, std::fabs(r));
            max_err_fast = std::max(max_err_fast,
                std::fabs(static_cast<double>(y_fast[n]) - static_cast<double>(y_ref[n])));
            max_err_pack = std::max(max_err_pack,
                std::fabs(static_cast<double>(y_ref[n]) - r));
        }
    }

    const double tol = 1e-3 * std::max(1.0, max_mag);
    std::printf("\n");
    std::printf("|y| maximo           : %.4f\n", max_mag);
    std::printf("kernel vs ref C++   : %.3e  (tolerancia %.1e)\n", max_err_fast, tol);
    std::printf("ref C++ vs ref Python: %.3e\n", max_err_pack);
    std::printf("\n");

    check(max_err_fast <= tol, "kernel AVX2 == referencia (unsigned + permutacion + sesgo)");
    check(max_err_pack <= tol * 100.0, "referencia C++ == referencia Python (packer correcto)");

    std::printf("\nRESULTADO: %d fallo(s)\n\n", g_fail);
    return g_fail == 0 ? 0 : 1;
}
