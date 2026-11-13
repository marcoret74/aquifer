// =============================================================================
//  caudal · bench/probe0_bw.cpp     —   DÍA 0: medir el DENOMINADOR
// =============================================================================
//  Todo el diseño son derivaciones de una sola ley:
//
//      tok/s  = η · BW_ef / bytes_del_modelo
//      ridge  = Pico_FLOPS / BW
//      M*     = ridge · bits / 16          (int4  ->  M* = ridge / 4)
//
//  BW_ef es el único término que no se puede derivar de un datasheet: hay que
//  medirlo en este silicio. Este programa mide eso, y nada más.
//
//  ── Por qué LECTURA PURA y no STREAM ────────────────────────────────────────
//  El decode de un transformer lee cada peso ~1 vez por token: la mezcla es
//  ~99.9 % lectura. STREAM-triad (2 lecturas + 1 escritura) mide otra cosa y
//  subestima justo el número que limita tok/s.
//
//  ── La trampa que este programa evita ───────────────────────────────────────
//  Con MAP_ANONYMOUS | MAP_NORESERVE las páginas NO están respaldadas por RAM.
//  Leerlas devuelve la página-cero compartida del kernel: cientos de GB/s
//  falsos, con cero tráfico de DRAM. Por eso se PRE-FAULTEAN escribiendo 1 byte
//  por página, y ese tiempo se reporta aparte (es costo de arranque, no decode).
//
//  ── Por qué varios hilos ────────────────────────────────────────────────────
//  Un solo núcleo casi nunca satura DRAM en x86. El número que gobierna tok/s
//  es el AGREGADO, porque los kernels particionan las columnas de N entre hilos
//  y cada hilo hace streaming de su propia franja de pesos. Medir sólo 1 hilo
//  reporta un BW_ef falsamente bajo.
//
//  Uso:
//      ./probe0_bw [MB] [passes] [--no-thp]
//          MB        tamaño del buffer        (default 512; > L3 en casi todo)
//          passes    pasadas por medición     (default 3)
//          --no-thp  desactiva MADV_HUGEPAGE  (para medir su efecto)
// =============================================================================

#include "caudal/layout.hpp"

#include <algorithm>
#include <atomic>
#include <barrier>
#include <cerrno>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <string_view>
#include <thread>
#include <vector>

#include <sched.h>
#include <sys/mman.h>
#include <sys/resource.h>
#include <time.h>

namespace {

constexpr std::size_t kMiB = std::size_t{1} << 20;
constexpr std::size_t kTwoMiB = std::size_t{1} << 21;
constexpr std::size_t kPage = std::size_t{1} << 12;

// Evita que el compilador elimine el trabajo medido.
volatile std::uint64_t g_sink = 0;

double now_s() noexcept {
    timespec ts{};
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return static_cast<double>(ts.tv_sec) + static_cast<double>(ts.tv_nsec) * 1e-9;
}

// Lee un campo (en kB) de /proc/self/smaps_rollup. 0 si no existe.
std::uint64_t smaps_kb(std::string_view field) {
    std::FILE* f = std::fopen("/proc/self/smaps_rollup", "r");
    if (!f) return 0;
    char line[256];
    std::uint64_t value = 0;
    while (std::fgets(line, sizeof line, f)) {
        if (std::strncmp(line, field.data(), field.size()) == 0) {
            const char* p = line + field.size();
            while (*p == ':' || *p == ' ' || *p == '\t') ++p;
            value = std::strtoull(p, nullptr, 10);
            break;
        }
    }
    std::fclose(f);
    return value;
}

void pin_to(unsigned cpu) noexcept {
    cpu_set_t set;
    CPU_ZERO(&set);
    CPU_SET(cpu, &set);
    (void)sched_setaffinity(0, sizeof set, &set);  // best-effort
}

// Lee la primera línea de un archivo pequeño (config del kernel).
std::string read_line(const char* path) {
    std::FILE* f = std::fopen(path, "r");
    if (!f) return "(n/d)";
    char buf[160] = {0};
    const bool ok = std::fgets(buf, sizeof buf, f) != nullptr;
    std::fclose(f);
    if (!ok) return "(n/d)";
    std::string s(buf);
    while (!s.empty() && (s.back() == '\n' || s.back() == ' ')) s.pop_back();
    return s;
}

struct Freq { unsigned lo_mhz; unsigned hi_mhz; };

// Frecuencia instantánea por CPU. Se muestrea INMEDIATAMENTE después de la
// ronda medida: un número de ancho de banda sin su frecuencia es indecidible
// (¿DRAM lenta, o CPU estrangulada?). Es la trampa de AGENTS.md §9.11.
Freq freq_mhz() {
    unsigned lo = ~0u, hi = 0;
    bool any = false;
    for (unsigned c = 0; c < 64; ++c) {
        char path[96];
        std::snprintf(path, sizeof path,
                      "/sys/devices/system/cpu/cpu%u/cpufreq/scaling_cur_freq", c);
        std::FILE* f = std::fopen(path, "r");
        if (!f) break;
        unsigned khz = 0;
        if (std::fscanf(f, "%u", &khz) == 1) {
            const unsigned mhz = khz / 1000;
            lo = std::min(lo, mhz);
            hi = std::max(hi, mhz);
            any = true;
        }
        std::fclose(f);
    }
    return any ? Freq{lo, hi} : Freq{0, 0};
}

unsigned cpu_max_mhz() {
    std::FILE* f = std::fopen(
        "/sys/devices/system/cpu/cpu0/cpufreq/scaling_max_freq", "r");
    if (!f) return 0;
    unsigned khz = 0;
    const bool ok = std::fscanf(f, "%u", &khz) == 1;
    std::fclose(f);
    return ok ? khz / 1000 : 0;
}

// Barrido de lectura puro. 8 acumuladores independientes para que el compilador
// vectorice sin crear una cadena de dependencia serial por la suma.
[[gnu::noinline]] std::uint64_t scan_readonly(const std::byte* p, std::size_t n) {
    const auto* q = reinterpret_cast<const std::uint64_t*>(p);
    const std::size_t words = n / sizeof(std::uint64_t);

    std::uint64_t a0 = 0, a1 = 0, a2 = 0, a3 = 0, a4 = 0, a5 = 0, a6 = 0, a7 = 0;
    std::size_t i = 0;
    for (; i + 8 <= words; i += 8) {
        a0 += q[i + 0]; a1 += q[i + 1]; a2 += q[i + 2]; a3 += q[i + 3];
        a4 += q[i + 4]; a5 += q[i + 5]; a6 += q[i + 6]; a7 += q[i + 7];
    }
    for (; i < words; ++i) a0 += q[i];

    std::uint64_t s = a0 + a1 + a2 + a3 + a4 + a5 + a6 + a7;
    std::atomic_signal_fence(std::memory_order_seq_cst);
    return s;
}

// Fuerza el respaldo real de cada página. Sin esto la medición es ficción.
void prefault(std::byte* p, std::size_t n) {
    for (std::size_t o = 0; o < n; o += kPage) p[o] = std::byte{1};
    if (n) p[n - 1] = std::byte{1};
}

double gb_per_s(std::uint64_t bytes, double seconds) {
    return static_cast<double>(bytes) / seconds / 1e9;  // GB decimales
}

unsigned max_threads() {
    unsigned hw = std::thread::hardware_concurrency();
    return hw ? hw : 1;
}

}  // namespace

int main(int argc, char** argv) {
    std::size_t mb = 512;
    int passes = 3;
    bool use_thp = true;

    int pos = 0;
    for (int i = 1; i < argc; ++i) {
        std::string_view a = argv[i];
        if (a == "--no-thp") { use_thp = false; continue; }
        if (a == "-h" || a == "--help") {
            std::printf("uso: %s [MB] [passes] [--no-thp]\n", argv[0]);
            return 0;
        }
        if (pos == 0) mb = std::strtoull(argv[i], nullptr, 10);
        else if (pos == 1) passes = std::atoi(argv[i]);
        ++pos;
    }
    if (mb == 0) mb = 512;
    if (passes < 1) passes = 1;

    const std::size_t n = mb * kMiB;

    std::printf("\n");
    std::printf("caudal · probe0_bw\n");
    std::printf("buffer            : %zu MiB\n", mb);
    std::printf("passes            : %d\n", passes);
    std::printf("THP (MADV_HUGEPAGE): %s\n", use_thp ? "on" : "off");
    std::printf("hilos disponibles : %u\n", max_threads());
    std::printf("------------ config del kernel (contexto de la medición) ------------\n");
    std::printf("thp anon          : %s\n",
                read_line("/sys/kernel/mm/transparent_hugepage/enabled").c_str());
    std::printf("thp defrag        : %s\n",
                read_line("/sys/kernel/mm/transparent_hugepage/defrag").c_str());
    std::printf("thp shmem         : %s\n",
                read_line("/sys/kernel/mm/transparent_hugepage/shmem_enabled").c_str());
    std::printf("governor          : %s\n",
                read_line("/sys/devices/system/cpu/cpu0/cpufreq/scaling_governor").c_str());
    std::printf("freq max posible  : %u MHz\n\n", cpu_max_mhz());

    // Reserva. MAP_NORESERVE para no comprometer swap por 1 TiB de virtual.
    void* raw = mmap(nullptr, n, PROT_READ | PROT_WRITE,
                     MAP_PRIVATE | MAP_ANONYMOUS | MAP_NORESERVE, -1, 0);
    if (raw == MAP_FAILED) {
        std::perror("mmap");
        return 1;
    }
    auto* base = static_cast<std::byte*>(raw);

    if (use_thp) {
        if (madvise(base, n, MADV_HUGEPAGE) != 0) {
            std::printf("aviso: MADV_HUGEPAGE rechazado (%s)\n", std::strerror(errno));
        }
    }

    // --- Pre-fault (si se omite, se mide la página-cero del kernel) ------------
    rusage ru0{}, ru1{};
    getrusage(RUSAGE_SELF, &ru0);
    const std::uint64_t huge_before = smaps_kb("AnonHugePages:");

    const double p0 = now_s();
    prefault(base, n);
    const double p1 = now_s();

    getrusage(RUSAGE_SELF, &ru1);
    const std::uint64_t huge_after = smaps_kb("AnonHugePages:");

    std::printf("pre-fault         : %.1f ms  (%.1f GB/s de escritura)\n",
                (p1 - p0) * 1e3, gb_per_s(n, p1 - p0));
    std::printf("minor faults      : %ld\n",
                static_cast<long>(ru1.ru_minflt - ru0.ru_minflt));
    std::printf("AnonHugePages     : %llu kB -> %llu kB",
                static_cast<unsigned long long>(huge_before),
                static_cast<unsigned long long>(huge_after));
    if (huge_after * 1024 >= n / 2) {
        std::printf("   [%.0f%% del buffer con hugepages]\n",
                    100.0 * static_cast<double>(huge_after) * 1024.0 / static_cast<double>(n));
    } else if (huge_after > 0) {
        std::printf("   [hugepages PARCIALES, %.0f%%: fragmentación de memoria]\n",
                    100.0 * static_cast<double>(huge_after) * 1024.0 / static_cast<double>(n));
    } else {
        std::printf("   [sin hugepages: revisar thp enabled/defrag]\n");
    }
    std::printf("\n");

    // --- Medición -------------------------------------------------------------
    std::printf("%7s %12s %12s %14s %7s\n",
                "hilos", "GB/s", "GiB/s", "GB/s por hilo", "MHz");

    std::vector<unsigned> counts;
    for (unsigned t = 1; t <= max_threads(); t <<= 1) counts.push_back(t);
    if (counts.back() != max_threads()) counts.push_back(max_threads());
    counts.erase(std::unique(counts.begin(), counts.end()), counts.end());

    double best_gbs = 0.0;
    unsigned best_t = 0;
    unsigned peak_mhz = 0;

    for (unsigned T : counts) {
        const std::size_t slice = (n / T) & ~(kTwoMiB - 1);
        if (slice < 1 * kMiB) continue;

        auto run_round = [&](int pass_count) -> double {
            std::barrier<> start(static_cast<std::ptrdiff_t>(T) + 1);
            std::barrier<> done (static_cast<std::ptrdiff_t>(T) + 1);
            std::atomic<std::uint64_t> sink{0};

            std::vector<std::thread> th;
            th.reserve(T);
            for (unsigned t = 0; t < T; ++t) {
                th.emplace_back([&, t] {
                    pin_to(t);
                    start.arrive_and_wait();
                    std::uint64_t acc = 0;
                    for (int p = 0; p < pass_count; ++p)
                        acc += scan_readonly(base + static_cast<std::size_t>(t) * slice, slice);
                    sink.fetch_add(acc, std::memory_order_relaxed);
                    done.arrive_and_wait();
                });
            }
            const double t0 = now_s();
            start.arrive_and_wait();
            done.arrive_and_wait();
            const double t1 = now_s();
            for (auto& x : th) x.join();
            g_sink ^= sink.load(std::memory_order_relaxed);
            return t1 - t0;
        };

        (void)run_round(1);                      // warm-up: TLB + branch predictor
        const double secs = run_round(passes);
        const Freq f = freq_mhz();               // inmediatamente después: reloj bajo carga
        if (f.hi_mhz > peak_mhz) peak_mhz = f.hi_mhz;

        const std::uint64_t bytes = static_cast<std::uint64_t>(slice) * T
                                  * static_cast<std::uint64_t>(passes);
        const double gbs = gb_per_s(bytes, secs);
        if (gbs > best_gbs) { best_gbs = gbs; best_t = T; }

        std::printf("%7u %12.1f %12.1f %14.1f %7u\n",
                    T, gbs, gbs * 1e9 / 1073741824.0, gbs / static_cast<double>(T),
                    f.hi_mhz);
    }

    std::printf("\n");
    std::printf("BW_ef (mejor, %u hilos) : %.1f GB/s  (%.1f GiB/s)\n",
                best_t, best_gbs, best_gbs * 1e9 / 1073741824.0);

    // Guardia anti-falso-negativo: un BW bajo puede ser DRAM lenta O un CPU
    // estrangulado. Sin este aviso, el número es indecidible.
    const unsigned fmax = cpu_max_mhz();
    if (fmax && peak_mhz && static_cast<unsigned long>(peak_mhz) * 100ul <
                                static_cast<unsigned long>(fmax) * 60ul) {
        std::printf("\n");
        std::printf("  ⚠  MEDICIÓN LIMITADA POR FRECUENCIA\n");
        std::printf("     pico observado %u MHz vs %u MHz posibles (%.0f%%)  —  governor: %s\n",
                    peak_mhz, fmax,
                    100.0 * static_cast<double>(peak_mhz) / static_cast<double>(fmax),
                    read_line("/sys/devices/system/cpu/cpu0/cpufreq/scaling_governor").c_str());
        std::printf("     Este BW_ef es de ESTA sesión, no del hardware. En un\n");
        std::printf("     portátil 15W con batería descargando es lo esperado.\n");
    }

    // Derivados del diseño. M* = ridge · bits/16; con int4, M* = ridge/4.
    const double ridge4 = 4000.0 / best_gbs;   // 4 TFLOP/s
    const double ridge8 = 8000.0 / best_gbs;   // 8 TFLOP/s
    std::printf("ridge @ 4 TFLOP/s       : %.1f FLOP/byte  ->  M*(int4) = %.1f\n",
                ridge4, ridge4 / 4.0);
    std::printf("ridge @ 8 TFLOP/s       : %.1f FLOP/byte  ->  M*(int4) = %.1f\n",
                ridge8, ridge8 / 4.0);
    std::printf("\nnota: en un desktop (BW bajo) M* sale grande; eso NO invalida el\n");
    std::printf("      speculative decoding. Su ganancia viene de amortizar bytes por\n");
    std::printf("      token ACEPTADO, no de cruzar el ridge.\n\n");

    munmap(raw, n);
    return 0;
}
