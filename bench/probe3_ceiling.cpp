// =============================================================================
//  caudal · bench/probe3_ceiling.cpp
//  ACCIÓN #1: el techo REAL a T hilos, con el patrón real (197 tensores).
// =============================================================================
//  El 24,4 GB/s de probe2 se midió con 8 hilos sobre UNA matriz de 231 MB y
//  lectura pura. Este probe hace lo mismo que la producción: recorre los 197
//  tensores del artefacto (weights.bin + scales.bin, 868 MB) con el MISMO
//  reparto por filas, y sólo suma bytes — sin MAC, sin atención, sin barreras.
//
//  Tres números, y la diferencia entre ellos es la información:
//    A. stream puro sobre 868 MB contiguos     (techo de la máquina)
//    B. stream por los 197 tensores            (techo del PATRÓN)
//    C. generate_qwen real                     (18,0 GB/s, medido aparte)
//
//  A-B = costo de la fragmentación en 197 tensores.
//  B-C = costo del cómputo y las partes seriales.
//
//  Uso: ./probe3_ceiling <artefacto> [hilos] [reps]
// =============================================================================

#include <algorithm>
#include <atomic>
#include <barrier>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <fstream>
#include <string>
#include <thread>
#include <vector>

namespace {

volatile std::uint64_t g_sink = 0;

double now_s() {
    timespec t{};
    clock_gettime(CLOCK_MONOTONIC, &t);
    return double(t.tv_sec) + double(t.tv_nsec) * 1e-9;
}

std::vector<std::uint8_t> slurp(const std::string& p) {
    std::ifstream f(p, std::ios::binary | std::ios::ate);
    if (!f) { std::printf("no abrio %s\n", p.c_str()); std::exit(1); }
    const auto n = static_cast<std::size_t>(f.tellg());
    f.seekg(0);
    std::vector<std::uint8_t> v(n);
    f.read(reinterpret_cast<char*>(v.data()), static_cast<std::streamsize>(n));
    return v;
}

// Suma de bytes, 4 acumuladores independientes para no dejar una cadena serial.
[[gnu::noinline]] std::uint64_t scan(const std::uint8_t* p, std::size_t n) {
    std::uint64_t a0 = 0, a1 = 0, a2 = 0, a3 = 0;
    std::size_t i = 0;
    for (; i + 32 <= n; i += 32) {
        a0 += *reinterpret_cast<const std::uint64_t*>(p + i);
        a1 += *reinterpret_cast<const std::uint64_t*>(p + i + 8);
        a2 += *reinterpret_cast<const std::uint64_t*>(p + i + 16);
        a3 += *reinterpret_cast<const std::uint64_t*>(p + i + 24);
    }
    for (; i < n; ++i) a0 += p[i];
    std::atomic_signal_fence(std::memory_order_seq_cst);
    return a0 + a1 + a2 + a3;
}

struct Seg { std::size_t off, bytes; };

}  // namespace

int main(int argc, char** argv) {
    const std::string A = (argc > 1) ? argv[1] : "";
    unsigned T = (argc > 2) ? std::atoi(argv[2]) : 4;
    int reps = (argc > 3) ? std::atoi(argv[3]) : 5;
    if (A.empty()) { std::printf("uso: %s <artefacto> [hilos] [reps]\n", argv[0]); return 1; }

    auto wb = slurp(A + "/weights.bin");
    auto sb = slurp(A + "/scales.bin");

    // lista de tensores desde index.txt
    std::vector<Seg> tens;
    {
        std::ifstream fi(A + "/index.txt");
        std::string kind;
        while (fi >> kind) {
            if (kind == "LINEAR") {
                std::string name; std::size_t N, K, po, pb, so, szb;
                fi >> name >> N >> K >> po >> pb >> so >> szb;
                tens.push_back({po, pb});           // pesos
                tens.push_back({so, szb});          // escalas
            } else if (kind == "AUX") {
                std::string name; std::size_t cnt, off; fi >> name >> cnt >> off;
            } else if (kind == "EMBED") {
                std::string name; std::size_t a, b2; fi >> name >> a >> b2;
            }
        }
    }
    std::size_t total = 0;
    for (auto& t : tens) if (t.off + t.bytes <= wb.size() + sb.size()) total += t.bytes;

    // Reparto DISJUNTO por bytes acumulados: cada hilo lee solo su franja.
    auto scan_tensors = [&](unsigned tid) {
        const std::size_t lo = total * tid / T, hi = total * (tid + 1) / T;
        std::size_t acc = 0;
        std::uint64_t s = 0;
        for (const auto& t : tens) {
            const std::size_t a = acc, b = acc + t.bytes;
            acc = b;
            if (b <= lo) continue;
            if (a >= hi) break;
            if (t.off + t.bytes > wb.size() + sb.size()) continue;
            const std::uint8_t* p = (t.off < wb.size())
                ? wb.data() + t.off
                : sb.data() + (t.off - wb.size());
            const std::size_t s0 = (lo > a ? lo : a) - a;
            const std::size_t s1 = (hi < b ? hi : b) - a;
            if (s1 > s0) s ^= scan(p + s0, s1 - s0);
        }
        g_sink ^= s;
    };

    std::printf("\ncaudal · probe3_ceiling   hilos=%u reps=%d\n", T, reps);
    std::printf("artefacto: %zu tensores de pesos/escalas, %.1f MB\n\n",
                tens.size(), double(total) / 1e6);

    // --- A: stream continuo sobre weights+scales -----------------------------
    std::vector<std::uint8_t> all;
    all.insert(all.end(), wb.begin(), wb.end());
    all.insert(all.end(), sb.begin(), sb.end());
    const std::size_t N = all.size();
    auto stream_once = [&](unsigned tid) {
        const std::size_t b = std::size_t(tid) * N / T;
        const std::size_t e = std::size_t(tid + 1) * N / T;
        g_sink ^= scan(all.data() + b, e - b);
    };
    auto run_mt = [&](auto fn) {
        std::vector<std::thread> th;
        for (unsigned i = 0; i < T; ++i) th.emplace_back([&, i] { fn(i); });
        for (auto& t : th) t.join();
    };
    // warm-up
    run_mt(stream_once);
    double t0 = now_s();
    for (int r = 0; r < reps; ++r) run_mt(stream_once);
    double t1 = now_s();
    const double A_gbs = double(N) * reps / (t1 - t0) / 1e9;

    // --- B: stream por los 197 tensores --------------------------------------
    run_mt(scan_tensors);
    t0 = now_s();
    for (int r = 0; r < reps; ++r) run_mt(scan_tensors);
    t1 = now_s();
    const double B_gbs = double(total) * reps / (t1 - t0) / 1e9;

    std::printf("%-34s %8.2f GB/s\n", "A. stream continuo (868 MB)", A_gbs);
    std::printf("%-34s %8.2f GB/s  (%.1f %% de A)\n",
                "B. stream por 197 tensores", B_gbs, 100.0 * B_gbs / A_gbs);
    std::printf("%-34s %8.2f GB/s  (%.1f %% de B)\n",
                "C. generate_qwen real (medido)", 18.0, 100.0 * 18.0 / B_gbs);
    std::printf("\n");
    std::printf("brecha A->B (fragmentacion) : %5.2f GB/s\n", A_gbs - B_gbs);
    std::printf("brecha B->C (computo+serial): %5.2f GB/s\n", B_gbs - 18.0);
    std::printf("techo del modelo 1.5B a A   : %.1f tok/s\n", A_gbs / 0.8683);
    std::printf("techo del modelo 1.5B a B   : %.1f tok/s\n\n", B_gbs / 0.8683);
    return 0;
}
