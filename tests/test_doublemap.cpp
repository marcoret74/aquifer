// =============================================================================
//  caudal · tests/test_doublemap.cpp
//  ASERCIÓN, no medición: el doble mapeo de memfd debe aliasar por MMU.
// =============================================================================
//  Por qué esto es un TEST y no un bench: el resultado no es un número que el
//  gobernador de energía pueda bailar. O los dos mapas apuntan a la misma
//  memoria física, o no. Si no, A1 cambia de diseño y punto.
//
//  Qué se prueba:
//    1. memfd_create + ftruncate + reserva PROT_NONE + 2x MAP_FIXED.
//    2. Aliasing real: escribir por el mapa A se ve por el mapa B.
//    3. La propiedad de ANILLO: una ventana que empieza cerca del final del
//       archivo y cruza el límite sigue leyendo desde el principio, SIN
//       ramas de wraparound. Esa es toda la razón del doble mapeo.
//    4. (informativo) Si MADV_HUGEPAGE / MADV_COLLAPSE producen páginas de
//       2 MiB en shmem con `shmem_enabled=never`.
//
//  Uso:  ./test_doublemap [MiB]
//  Salida: 0 si todas las aserciones pasan, 1 si alguna falla.
// =============================================================================

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>

#include <sys/mman.h>
#include <sys/types.h>
#include <unistd.h>

namespace {

constexpr std::size_t kMiB = std::size_t{1} << 20;

int g_fail = 0;

void check(bool cond, const char* what) {
    std::printf("  [%s] %s\n", cond ? "PASS" : "FAIL", what);
    if (!cond) ++g_fail;
}

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

std::uint64_t smaps_kb(const char* field) {
    std::FILE* f = std::fopen("/proc/self/smaps_rollup", "r");
    if (!f) return 0;
    char line[256];
    std::uint64_t v = 0;
    const std::size_t n = std::strlen(field);
    while (std::fgets(line, sizeof line, f)) {
        if (std::strncmp(line, field, n) == 0) {
            const char* p = line + n;
            while (*p == ':' || *p == ' ' || *p == '\t') ++p;
            v = std::strtoull(p, nullptr, 10);
            break;
        }
    }
    std::fclose(f);
    return v;
}

std::uint64_t meminfo_kb(const char* field) {
    std::FILE* f = std::fopen("/proc/meminfo", "r");
    if (!f) return 0;
    char line[256];
    std::uint64_t v = 0;
    const std::size_t n = std::strlen(field);
    while (std::fgets(line, sizeof line, f)) {
        if (std::strncmp(line, field, n) == 0) {
            const char* p = line + n;
            while (*p == ':' || *p == ' ' || *p == '\t') ++p;
            v = std::strtoull(p, nullptr, 10);
            break;
        }
    }
    std::fclose(f);
    return v;
}

// Patrón verificable byte a byte, periódico en 251 (primo, no divide potencias).
inline std::uint8_t pat(std::size_t i) {
    return static_cast<std::uint8_t>(i % 251u);
}

}  // namespace

int main(int argc, char** argv) {
    std::size_t cap = 64 * kMiB;
    if (argc > 1) {
        const unsigned mb = static_cast<unsigned>(std::strtoul(argv[1], nullptr, 10));
        if (mb) cap = static_cast<std::size_t>(mb) * kMiB;
    }

    std::printf("\n");
    std::printf("caudal · test_doublemap\n");
    std::printf("cap               : %zu MiB\n", cap / kMiB);
    std::printf("kernel            : %s\n",
                read_line("/proc/sys/kernel/osrelease").c_str());
    std::printf("thp anon          : %s\n",
                read_line("/sys/kernel/mm/transparent_hugepage/enabled").c_str());
    std::printf("thp shmem         : %s\n",
                read_line("/sys/kernel/mm/transparent_hugepage/shmem_enabled").c_str());
    std::printf("hugepages total   : %llu kB (pool hugetlbfs)\n",
                static_cast<unsigned long long>(meminfo_kb("HugePages_Total")));
    std::printf("\n");

    // --- 1. memfd ------------------------------------------------------------
    const int fd = ::memfd_create("caudal_kv", MFD_CLOEXEC);
    check(fd >= 0, "memfd_create");
    if (fd < 0) { std::perror("memfd_create"); return 1; }

    check(::ftruncate(fd, static_cast<off_t>(cap)) == 0, "ftruncate(cap)");

    // --- 2. Reserva + doble MAP_FIXED DENTRO de la reserva -------------------
    // MAP_FIXED sobre una dirección arbitraria puede pisar heap/PIE. Reservar
    // primero con PROT_NONE y rellenar después es lo que lo hace seguro.
    void* res = ::mmap(nullptr, 2 * cap, PROT_NONE,
                       MAP_PRIVATE | MAP_ANONYMOUS | MAP_NORESERVE, -1, 0);
    check(res != MAP_FAILED, "reserva 2*cap con PROT_NONE");
    if (res == MAP_FAILED) { std::perror("mmap reserve"); return 1; }

    auto* base = static_cast<std::uint8_t*>(res);
    void* a = ::mmap(base, cap, PROT_READ | PROT_WRITE,
                     MAP_SHARED | MAP_FIXED, fd, 0);
    check(a == base, "mapa A (MAP_SHARED|MAP_FIXED, offset 0)");
    void* b = ::mmap(base + cap, cap, PROT_READ | PROT_WRITE,
                     MAP_SHARED | MAP_FIXED, fd, 0);
    check(b == base + cap, "mapa B (MAP_SHARED|MAP_FIXED, offset 0)");
    if (a != base || b != base + cap) { std::perror("mmap MAP_FIXED"); return 1; }

    std::printf("  A = %p   B = %p   delta = %zu MiB\n\n",
                static_cast<void*>(base), static_cast<void*>(base + cap), cap / kMiB);

    // --- 3. THP sobre memfd --------------------------------------------------
    const std::uint64_t pmd0 = smaps_kb("ShmemPmdMapped:");

    const int hg_a = ::madvise(base, cap, MADV_HUGEPAGE);
    const int hg_b = ::madvise(base + cap, cap, MADV_HUGEPAGE);
    std::printf("MADV_HUGEPAGE     : A=%d B=%d%s\n", hg_a, hg_b,
                (hg_a == 0 && hg_b == 0) ? "  (aceptado)" : "  (rechazado)");

    // Poblar Escribiendo: sin esto no hay páginas que puedan ser grandes.
    auto* A = base;
    auto* B = base + cap;
    for (std::size_t i = 0; i < cap; ++i) A[i] = pat(i);

    const std::uint64_t pmd1 = smaps_kb("ShmemPmdMapped:");
    std::printf("ShmemPmdMapped    : %llu kB -> %llu kB",
                static_cast<unsigned long long>(pmd0),
                static_cast<unsigned long long>(pmd1));
    if (pmd1 > pmd0) std::printf("   [THP efectivo]\n");
    else             std::printf("   [sin PMD de 2 MiB]\n");

#ifdef MADV_COLLAPSE
    const int col = ::madvise(base, 2 * cap, MADV_COLLAPSE);
    const std::uint64_t pmd2 = smaps_kb("ShmemPmdMapped:");
    std::printf("MADV_COLLAPSE     : rc=%d  -> ShmemPmdMapped %llu kB\n",
                col, static_cast<unsigned long long>(pmd2));
#else
    std::printf("MADV_COLLAPSE     : no disponible en este kernel/headers\n");
#endif

    std::printf("\naserciones:\n");

    // --- 4. Aliasing ---------------------------------------------------------
    check(A != B, "A y B son direcciones virtuales distintas");

    bool same0 = true, same_mid = true, same_end = true;
    for (std::size_t i = 0; i < 64; ++i) {
        if (A[i] != B[i]) same0 = false;
        if (A[cap / 2 + i] != B[cap / 2 + i]) same_mid = false;
        if (A[cap - 64 + i] != B[cap - 64 + i]) same_end = false;
    }
    check(same0, "alias en el inicio (A[0..64) == B[0..64))");
    check(same_mid, "alias en el medio");
    check(same_end, "alias en el final");

    // Escritura por A visible en B.
    // OJO: hay que restaurar el patrón. Si no, la aserción del anillo falla
    // justo cuando la ventana llega al byte mutado (bug real que tuvo este test).
    const std::uint8_t save1234 = A[1234];
    A[1234] = 0x5A;
    check(B[1234] == 0x5A, "escritura por A visible en B (memoria compartida)");
    A[1234] = save1234;

    const std::uint8_t save4321 = B[4321];
    B[4321] = 0xA5;
    check(A[4321] == 0xA5, "escritura por B visible en A");
    B[4321] = save4321;

    // --- 5. La propiedad de ANILLO (la razón de todo esto) -------------------
    // Una ventana de L bytes que arranca en p sigue leyendo desde el principio
    // del archivo al cruzar el límite, SIN ninguna rama de wraparound.
    const std::size_t p = cap - 100;
    const std::size_t L = 4000;
    bool ring_ok = true;
    std::size_t first_bad = 0;
    for (std::size_t j = 0; j < L; ++j) {
        const std::size_t file_off = (p + j) % cap;
        if (A[p + j] != pat(file_off)) { ring_ok = false; first_bad = j; break; }
    }
    check(ring_ok, "ventana circular de 4000 B cruzando el límite, sin ramas");
    if (!ring_ok) std::printf("        primer byte distinto en j=%zu\n", first_bad);

    // Y que el anillo realmente se dobla: los primeros 100 B de la ventana son
    // la COLA del archivo, y el resto es la CABEZA. Sin ninguna rama.
    bool tail_ok = true, head_ok = true;
    for (std::size_t j = 0; j < 100; ++j)
        if (A[p + j] != pat(cap - 100 + j)) { tail_ok = false; break; }
    for (std::size_t j = 100; j < L; ++j)
        if (A[p + j] != pat(j - 100)) { head_ok = false; break; }
    check(tail_ok, "ventana: los primeros 100 B son la COLA del archivo");
    check(head_ok, "ventana: el resto sigue en la CABEZA (wraparound por MMU)");
    check(A[cap - 1] == pat(cap - 1) && A[cap] == pat(0),
          "frontera exacta: A[cap-1] = cola, A[cap] = offset 0");

    std::printf("\nRESULTADO: %d fallo(s)\n\n", g_fail);

    ::munmap(base, 2 * cap);
    ::close(fd);
    return g_fail == 0 ? 0 : 1;
}
