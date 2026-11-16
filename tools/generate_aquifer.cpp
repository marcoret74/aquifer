// =============================================================================
//  caudal · tools/generate_qwen.cpp
//  GENERADOR con KV cache en anillo + batching de N secuencias (M genérico).
// =============================================================================
//  Tres mecanismos:
//   1. KV CACHE en anillo: memfd mapeado DOS VECES por secuencia -> una ventana
//      [0,n) es siempre contigua, sin ramas de wraparound. (test_doublemap)
//   2. MULTIHILO con SpinBarrier y el principal como worker 0.
//   3. BATCHING: M secuencias independientes por pasada. Cada peso leído sirve
//      a M filas, así que los bytes por token se parten en M. Medido:
//          M=1 20.7 tok/s | M=2 35.4 | M=4 52.3 | M=8 58.8
//
//  Uso:
//    ./generate_qwen <artefacto> --n 40 --threads 4 --ctx 512
//                    --prompt 151644,872,198,... --prompt 151644,872,...
// =============================================================================

#include "caudal/gemv_int4.hpp"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <fstream>
#include <functional>
#include <iostream>
#include <map>
#include <mutex>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

#if defined(__x86_64__) || defined(_M_X64)
#  include <immintrin.h>
#elif defined(__ARM_NEON) || defined(__aarch64__)
#  include <arm_neon.h>
#endif
#include <fcntl.h>
#include <poll.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

// Instrumentacion opcional (--time).
static double g_t_gemv = 0.0, g_t_attn = 0.0;
static double g_bytes = 0.0;
static bool g_probe = false;
static bool g_bench_pool = false;
static bool g_slow_quant = false;
static bool g_time = false;
// Política de residencia de pesos:
//   paged    = MADV_DONTNEED por capa (VmRSS mínimo, pero paga page faults)
//   resident = deja los pesos en page cache/RSS (máxima velocidad si hay RAM)
//   auto     = resident si el modelo cabe con holgura; paged si no
enum class MemPolicy { Auto, Paged, Resident };
static MemPolicy g_mem_policy = MemPolicy::Auto;
static std::size_t g_mem_reserve_mb = 512;
static unsigned g_prefill_batch = 8;
static bool g_evict_layers = true;   // lo decide la política en runtime
static bool g_kv_fp16 = true;        // KV en fp16: mitad de tráfico en atención

namespace {

constexpr unsigned kMaxM = 8;          // secuencias por pasada

struct Cfg {
    std::size_t H{0}, NLY{0}, NH{0}, NKV{0}, DH{0}, INTER{0}, V{0};
    float eps{1e-6f}, theta{1e6f};
    float inv_sqrt_d{1.0f};
};

struct Lin {
    std::size_t N{0}, K{0}, G{0}, stride{0};
    const std::uint8_t* packed{nullptr};
    const std::uint8_t* kmin{nullptr};      // headers de 16 B por super-bloque
    const _Float16* scale{nullptr};
    const _Float16* min{nullptr};           // esquema afin (5.00 bpw)
    std::size_t row_bytes{0};
};

struct Job { const Lin* L; float* y; };

std::vector<std::uint8_t> slurp(const std::string& p) {
    std::ifstream f(p, std::ios::binary | std::ios::ate);
    if (!f) { std::printf("no abrio %s\n", p.c_str()); std::exit(1); }
    const auto n = static_cast<std::size_t>(f.tellg());
    f.seekg(0);
    std::vector<std::uint8_t> v(n);
    f.read(reinterpret_cast<char*>(v.data()), static_cast<std::streamsize>(n));
    return v;
}

inline float bf16_to_f32(std::uint16_t u) {
    const std::uint32_t x = static_cast<std::uint32_t>(u) << 16;
    float f; std::memcpy(&f, &x, 4); return f;
}

double now_ms() {
    timespec t{}; clock_gettime(CLOCK_MONOTONIC, &t);
    return double(t.tv_sec) * 1e3 + double(t.tv_nsec) * 1e-6;
}

// --- barrera spin lock-free pura (cero syscalls, latencia 0 en GEMVs) ---
class SpinBarrier {
public:
    explicit SpinBarrier(unsigned t) : T_(t) {}
    void wait() {
        const unsigned g = gen_.load(std::memory_order_acquire);
        if (cnt_.fetch_add(1, std::memory_order_acq_rel) + 1 == T_) {
            cnt_.store(0, std::memory_order_relaxed);
            gen_.fetch_add(1, std::memory_order_release);
        } else {
            while (gen_.load(std::memory_order_acquire) == g) {
#if defined(__ARM_NEON) || defined(__aarch64__)
                asm volatile("yield");
#else
                _mm_pause();
#endif
            }
        }
    }
private:
    unsigned T_;
    std::atomic<unsigned> cnt_{0}, gen_{0};
};

class Pool {
public:
    explicit Pool(unsigned T) : T_(T ? T : 1), go_(T_), fin_(T_) {
        for (unsigned i = 1; i < T_; ++i)
            th_.emplace_back([this, i] {
                for (;;) {
                    if (sleeping_.load(std::memory_order_acquire)) {
                        std::unique_lock<std::mutex> lk(sleep_m_);
                        sleep_ack_.fetch_add(1, std::memory_order_release);
                        sleep_cv_.wait(lk, [this] {
                            return !sleeping_.load(std::memory_order_acquire) || stop_;
                        });
                        wake_ack_.fetch_add(1, std::memory_order_release);
                    }
                    go_.wait();
                    if (stop_) return;
                    job_(i);
                    fin_.wait();
                }
            });
    }
    ~Pool() { stop_ = true; resume(); go_.wait(); for (auto& t : th_) t.join(); }
    Pool(const Pool&) = delete;
    Pool& operator=(const Pool&) = delete;
    unsigned size() const { return T_; }

    void pause() {
        if (T_ <= 1 || sleeping_.load(std::memory_order_acquire)) return;
        sleep_ack_.store(0, std::memory_order_relaxed);
        sleeping_.store(true, std::memory_order_release);
        run([](unsigned) {}); // despierta workers de go_ para que entren en sleep
        // Esperar a que TODOS estén dormidos: si no, un resume seguido de un
        // pause (comandos sin inferencia) puede perder el wake y colgar.
        while (sleep_ack_.load(std::memory_order_acquire) < T_ - 1) _mm_pause();
    }

    void resume() {
        if (!sleeping_.exchange(false, std::memory_order_acq_rel)) return;
        wake_ack_.store(0, std::memory_order_relaxed);
        sleep_cv_.notify_all();
        while (wake_ack_.load(std::memory_order_acquire) < T_ - 1) _mm_pause();
    }

    void run(std::function<void(unsigned)> f) {
        job_ = std::move(f); go_.wait(); job_(0); fin_.wait();
    }
private:
    unsigned T_;
    SpinBarrier go_, fin_;
    std::vector<std::thread> th_;
    std::function<void(unsigned)> job_;
    bool stop_ = false;
    std::atomic<bool> sleeping_{false};
    std::atomic<unsigned> sleep_ack_{0}, wake_ack_{0};
    std::mutex sleep_m_;
    std::condition_variable sleep_cv_;
};

// --- anillos de KV, uno por secuencia ---------------------------------------
struct KvRings {
    std::uint8_t* base{nullptr};
    std::size_t cap_tok{0}, per_seq{0}, stride{0};
    std::size_t elem{sizeof(float)};
    // per_seq = layers*2*cap_bytes

    void init(std::size_t tokens, std::size_t layers, std::size_t nkv,
              std::size_t dh, unsigned nseq, bool fp16) {
        cap_tok = tokens;
        elem = fp16 ? sizeof(_Float16) : sizeof(float);
        stride = nkv * dh * elem;
        const std::size_t cap_bytes = cap_tok * stride;
        per_seq = layers * 2 * cap_bytes;
        const std::size_t total = per_seq * nseq;
        const int fd = ::memfd_create("caudal_kv", MFD_CLOEXEC);
        if (fd < 0 || ::ftruncate(fd, static_cast<off_t>(total)) != 0) { std::perror("memfd"); std::exit(1); }
        void* res = ::mmap(nullptr, 2 * total, PROT_NONE,
                           MAP_PRIVATE | MAP_ANONYMOUS | MAP_NORESERVE, -1, 0);
        if (res == MAP_FAILED) { std::perror("reserva"); std::exit(1); }
        base = static_cast<std::uint8_t*>(res);
        if (::mmap(base, total, PROT_READ | PROT_WRITE, MAP_SHARED | MAP_FIXED, fd, 0) != base ||
            ::mmap(base + total, total, PROT_READ | PROT_WRITE, MAP_SHARED | MAP_FIXED, fd, 0) != base + total) {
            std::perror("mapa"); std::exit(1);
        }
        ::madvise(base, 2 * total, MADV_HUGEPAGE);
#ifdef MADV_COLLAPSE
        ::madvise(base, 2 * total, MADV_COLLAPSE);
#endif
    }
    std::uint8_t* slot(unsigned seq, std::size_t layer, bool is_v, std::size_t t) const {
        const std::size_t cap_bytes = cap_tok * stride;
        std::uint8_t* b = base + std::size_t(seq) * per_seq
                        + layer * 2 * cap_bytes + (is_v ? cap_bytes : 0);
        return b + (t % cap_tok) * stride;
    }
};

// El buffer de pesos usa mmap + MADV_HUGEPAGE en MMapFile. Medido en este
// kernel: FilePmdMapped sube a ~742 MB de 969 MB, así que las páginas grandes
// sí se materializan para el mapping de archivo read-only.
static std::size_t get_vm_rss_mb() {
    std::ifstream f("/proc/self/status");
    std::string k;
    while (f >> k) {
        if (k == "VmRSS:") {
            std::size_t v = 0;
            f >> v;
            return v / 1024;
        }
        f.ignore(4096, '\n');
    }
    return 0;
}

// MemAvailable es la métrica que usa el kernel para decidir si puede
// satisfacer una asignación sin empezar a reclamar. Si el modelo cabe ahí,
// no tiene sentido expulsar páginas que el sistema puede cachear gratis.
static std::size_t get_mem_available_mb() {
    std::ifstream f("/proc/meminfo");
    std::string k;
    while (f >> k) {
        if (k == "MemAvailable:") {
            std::size_t v = 0;
            f >> v;   // kB
            return v / 1024;
        }
        f.ignore(4096, '\n');
    }
    return 0;
}

// Núcleos físicos, no hilos lógicos. En un 15 W con SMT, usar los 2T/core
// agrega contención de puertos y calor sin comprar ancho de banda.
static unsigned physical_cores() {
    std::ifstream f("/proc/cpuinfo");
    std::string line;
    unsigned cores = 0, siblings = 0;
    auto value_after_colon = [](const std::string& s) -> unsigned {
        const auto p = s.find(':');
        return p == std::string::npos ? 0u : unsigned(std::strtoul(s.c_str() + p + 1, nullptr, 10));
    };
    while (std::getline(f, line)) {
        if (line.rfind("cpu cores", 0) == 0) cores = value_after_colon(line);
        else if (line.rfind("siblings", 0) == 0) siblings = value_after_colon(line);
        if (cores && siblings) break;
    }
    if (cores && (!siblings || cores <= siblings)) return cores;
    const unsigned hc = std::thread::hardware_concurrency();
    return hc ? std::max(1u, hc / 2) : 1u;
}

struct MMapFile {
    std::uint8_t* ptr{nullptr};
    std::size_t size{0};
    int fd{-1};

    void open_ro(const std::string& path) {
        fd = ::open(path.c_str(), O_RDONLY);
        if (fd < 0) {
            std::perror(("Aquifer MMap: no pudo abrir " + path).c_str());
            std::exit(1);
        }
        struct stat st;
        if (::fstat(fd, &st) < 0) { std::perror("fstat"); std::exit(1); }
        size = st.st_size;
        void* p = ::mmap(nullptr, size ? size : 1, PROT_READ, MAP_SHARED, fd, 0);
        if (p == MAP_FAILED) { std::perror("mmap"); std::exit(1); }
        ptr = static_cast<std::uint8_t*>(p);
        ::madvise(ptr, size, MADV_HUGEPAGE);
    }

    ~MMapFile() {
        if (ptr && ptr != MAP_FAILED) ::munmap(ptr, size);
        if (fd >= 0) ::close(fd);
    }
};

struct LayerSpan {
    std::size_t w_start{std::size_t(-1)}, w_end{0};
    std::size_t s_start{std::size_t(-1)}, s_end{0};
};

struct Ctx {
    MMapFile w_file;
    MMapFile s_file;
    MMapFile e_file;
    std::vector<std::uint8_t> auxbin;
    std::vector<LayerSpan> layer_spans;
    std::map<std::string, Lin> lin;
    std::map<std::string, const float*> aux;
    const float* aux_of(const std::string& n) const {
        auto it = aux.find(n); return it == aux.end() ? nullptr : it->second;
    }
};

class AsyncPrefetcher {
public:
    explicit AsyncPrefetcher(const Ctx& ctx) : ctx_(ctx) {
        worker_ = std::thread([this]() { run(); });
    }

    ~AsyncPrefetcher() {
        stop();
    }

    void stop() {
        {
            std::lock_guard<std::mutex> lock(mu_);
            if (stopped_) return;
            stopped_ = true;
            cv_.notify_one();
        }
        if (worker_.joinable()) {
            worker_.join();
        }
    }

    void prefetch(std::size_t layer_idx) {
        if (layer_idx >= ctx_.layer_spans.size()) return;
        {
            std::lock_guard<std::mutex> lock(mu_);
            requested_layer_ = layer_idx;
            cv_.notify_one();
        }
    }

private:
    void run() {
        while (true) {
            std::size_t target = std::size_t(-1);
            {
                std::unique_lock<std::mutex> lock(mu_);
                cv_.wait(lock, [this]() { return stopped_ || requested_layer_ != std::size_t(-1); });
                if (stopped_) break;
                target = requested_layer_;
                requested_layer_ = std::size_t(-1);
            }

            if (target < ctx_.layer_spans.size()) {
                const auto& sp = ctx_.layer_spans[target];
                if (sp.w_end > sp.w_start) {
                    const std::size_t page_start = sp.w_start & ~4095UL;
                    const std::size_t page_len = ((sp.w_end + 4095UL) & ~4095UL) - page_start;
                    ::madvise(ctx_.w_file.ptr + page_start, page_len, MADV_WILLNEED);
                    if (ctx_.w_file.fd >= 0) {
                        ::posix_fadvise(ctx_.w_file.fd, page_start, page_len, POSIX_FADV_WILLNEED);
                    }
                }
                if (sp.s_end > sp.s_start) {
                    const std::size_t page_start = sp.s_start & ~4095UL;
                    const std::size_t page_len = ((sp.s_end + 4095UL) & ~4095UL) - page_start;
                    ::madvise(ctx_.s_file.ptr + page_start, page_len, MADV_WILLNEED);
                    if (ctx_.s_file.fd >= 0) {
                        ::posix_fadvise(ctx_.s_file.fd, page_start, page_len, POSIX_FADV_WILLNEED);
                    }
                }
            }
        }
    }

    const Ctx& ctx_;
    std::thread worker_;
    std::mutex mu_;
    std::condition_variable cv_;
    std::size_t requested_layer_{std::size_t(-1)};
    bool stopped_{false};
};

// Producto punto AVX2/SSE4 float.
inline float dot_f32(const float* __restrict a, const float* __restrict b, std::size_t n) {
#if defined(__AVX2__)
    __m256 a0 = _mm256_setzero_ps(), a1 = a0, a2 = a0, a3 = a0;
    std::size_t i = 0;
    for (; i + 32 <= n; i += 32) {
        a0 = _mm256_fmadd_ps(_mm256_loadu_ps(a + i),      _mm256_loadu_ps(b + i),      a0);
        a1 = _mm256_fmadd_ps(_mm256_loadu_ps(a + i + 8),  _mm256_loadu_ps(b + i + 8),  a1);
        a2 = _mm256_fmadd_ps(_mm256_loadu_ps(a + i + 16), _mm256_loadu_ps(b + i + 16), a2);
        a3 = _mm256_fmadd_ps(_mm256_loadu_ps(a + i + 24), _mm256_loadu_ps(b + i + 24), a3);
    }
    for (; i + 8 <= n; i += 8)
        a0 = _mm256_fmadd_ps(_mm256_loadu_ps(a + i), _mm256_loadu_ps(b + i), a0);
    a0 = _mm256_add_ps(_mm256_add_ps(a0, a1), _mm256_add_ps(a2, a3));
    __m128 lo = _mm256_castps256_ps128(a0), hi = _mm256_extractf128_ps(a0, 1);
    __m128 t = _mm_add_ps(lo, hi);
    t = _mm_hadd_ps(t, t); t = _mm_hadd_ps(t, t);
    float r = _mm_cvtss_f32(t);
    for (; i < n; ++i) r += a[i] * b[i];
    return r;
#elif defined(__x86_64__) || defined(_M_X64)
    __m128 a0 = _mm_setzero_ps(), a1 = a0, a2 = a0, a3 = a0;
    std::size_t i = 0;
    for (; i + 16 <= n; i += 16) {
        a0 = _mm_add_ps(a0, _mm_mul_ps(_mm_loadu_ps(a + i),      _mm_loadu_ps(b + i)));
        a1 = _mm_add_ps(a1, _mm_mul_ps(_mm_loadu_ps(a + i + 4),  _mm_loadu_ps(b + i + 4)));
        a2 = _mm_add_ps(a2, _mm_mul_ps(_mm_loadu_ps(a + i + 8),  _mm_loadu_ps(b + i + 8)));
        a3 = _mm_add_ps(a3, _mm_mul_ps(_mm_loadu_ps(a + i + 12), _mm_loadu_ps(b + i + 12)));
    }
    a0 = _mm_add_ps(_mm_add_ps(a0, a1), _mm_add_ps(a2, a3));
    a0 = _mm_hadd_ps(a0, a0); a0 = _mm_hadd_ps(a0, a0);
    float r = _mm_cvtss_f32(a0);
    for (; i < n; ++i) r += a[i] * b[i];
    return r;
#elif defined(__ARM_NEON) || defined(__aarch64__)
    float32x4_t a0 = vdupq_n_f32(0.0f), a1 = a0, a2 = a0, a3 = a0;
    std::size_t i = 0;
    for (; i + 16 <= n; i += 16) {
        a0 = vmlaq_f32(a0, vld1q_f32(a + i),      vld1q_f32(b + i));
        a1 = vmlaq_f32(a1, vld1q_f32(a + i + 4),  vld1q_f32(b + i + 4));
        a2 = vmlaq_f32(a2, vld1q_f32(a + i + 8),  vld1q_f32(b + i + 8));
        a3 = vmlaq_f32(a3, vld1q_f32(a + i + 12), vld1q_f32(b + i + 12));
    }
    a0 = vaddq_f32(vaddq_f32(a0, a1), vaddq_f32(a2, a3));
    float r = vaddvq_f32(a0);
    for (; i < n; ++i) r += a[i] * b[i];
    return r;
#else
    float r = 0.0f;
    for (std::size_t i = 0; i < n; ++i) r += a[i] * b[i];
    return r;
#endif
}

// Producto punto float·fp16 para el KV. La mitad de bytes en atención es la
// diferencia entre memory-bound y compute-bound a ctx largo.
inline float dot_f32_f16(const float* __restrict a, const _Float16* __restrict b,
                         std::size_t n) {
#if defined(__AVX2__) && defined(__F16C__)
    __m256 a0 = _mm256_setzero_ps(), a1 = a0, a2 = a0, a3 = a0;
    std::size_t i = 0;
    for (; i + 32 <= n; i += 32) {
        a0 = _mm256_fmadd_ps(_mm256_loadu_ps(a + i),
            _mm256_cvtph_ps(_mm_loadu_si128(reinterpret_cast<const __m128i*>(b + i))), a0);
        a1 = _mm256_fmadd_ps(_mm256_loadu_ps(a + i + 8),
            _mm256_cvtph_ps(_mm_loadu_si128(reinterpret_cast<const __m128i*>(b + i + 8))), a1);
        a2 = _mm256_fmadd_ps(_mm256_loadu_ps(a + i + 16),
            _mm256_cvtph_ps(_mm_loadu_si128(reinterpret_cast<const __m128i*>(b + i + 16))), a2);
        a3 = _mm256_fmadd_ps(_mm256_loadu_ps(a + i + 24),
            _mm256_cvtph_ps(_mm_loadu_si128(reinterpret_cast<const __m128i*>(b + i + 24))), a3);
    }
    for (; i + 8 <= n; i += 8)
        a0 = _mm256_fmadd_ps(_mm256_loadu_ps(a + i),
            _mm256_cvtph_ps(_mm_loadu_si128(reinterpret_cast<const __m128i*>(b + i))), a0);
    a0 = _mm256_add_ps(_mm256_add_ps(a0, a1), _mm256_add_ps(a2, a3));
    __m128 lo = _mm256_castps256_ps128(a0), hi = _mm256_extractf128_ps(a0, 1);
    __m128 t = _mm_add_ps(lo, hi);
    t = _mm_hadd_ps(t, t); t = _mm_hadd_ps(t, t);
    float r = _mm_cvtss_f32(t);
    for (; i < n; ++i) r += a[i] * static_cast<float>(b[i]);
    return r;
#else
    float r = 0.0f;
    for (std::size_t i = 0; i < n; ++i) r += a[i] * static_cast<float>(b[i]);
    return r;
#endif
}

// o += p * v, con v en fp16 (la mitad de tráfico en la fase de valores).
inline void axpy_f32_f16(float* __restrict o, const _Float16* __restrict v,
                         float p, std::size_t n) {
#if defined(__AVX2__) && defined(__F16C__)
    const __m256 pv = _mm256_set1_ps(p);
    std::size_t i = 0;
    for (; i + 16 <= n; i += 16) {
        _mm256_storeu_ps(o + i, _mm256_fmadd_ps(pv,
            _mm256_cvtph_ps(_mm_loadu_si128(reinterpret_cast<const __m128i*>(v + i))),
            _mm256_loadu_ps(o + i)));
        _mm256_storeu_ps(o + i + 8, _mm256_fmadd_ps(pv,
            _mm256_cvtph_ps(_mm_loadu_si128(reinterpret_cast<const __m128i*>(v + i + 8))),
            _mm256_loadu_ps(o + i + 8)));
    }
    for (; i + 8 <= n; i += 8)
        _mm256_storeu_ps(o + i, _mm256_fmadd_ps(pv,
            _mm256_cvtph_ps(_mm_loadu_si128(reinterpret_cast<const __m128i*>(v + i))),
            _mm256_loadu_ps(o + i)));
    for (; i < n; ++i) o[i] += p * static_cast<float>(v[i]);
#else
    for (std::size_t i = 0; i < n; ++i) o[i] += p * static_cast<float>(v[i]);
#endif
}

void rmsnorm(const float* x, const float* w, float* out, std::size_t n, float eps) {
    double ss = 0.0;
    for (std::size_t i = 0; i < n; ++i) ss += double(x[i]) * double(x[i]);
    const float r = 1.0f / std::sqrt(float(ss / double(n)) + eps);
    for (std::size_t i = 0; i < n; ++i) out[i] = x[i] * r * w[i];
}

// --- GEMV por grupo, M vectores de entrada -----------------------------------
void gemv_group_M(Pool& pool, const std::vector<Job>& jobs, unsigned M,
                  const float* x, std::size_t K) {
    const auto t0 = std::chrono::steady_clock::now();
    const Lin& L0 = *jobs[0].L;
    const std::size_t G = L0.G, grp = K / G;

    // La cuantizacion de x es M*K elementos: en serie costaba ~10 ms a M=4
    // (era el 10 % del paso). Se reparte entre los hilos por rango de m.
    std::vector<std::int8_t> xq(M * K), xqp(M * K);
    std::vector<std::int32_t> xg(M * G);
    std::vector<float> xs(M * G);
    std::vector<float> xms(M * G);
    pool.run([&](unsigned tid) {
        const unsigned T0 = pool.size();
        const unsigned a = unsigned(std::size_t(tid) * M / T0);
        const unsigned b = unsigned(std::size_t(tid + 1) * M / T0);
        for (unsigned m = a; m < b; ++m) {
            const float* xm = x + std::size_t(m) * K;
            std::int8_t* q = xq.data() + std::size_t(m) * K;
            float* sm = xs.data() + std::size_t(m) * G;
            // Una escala por GRUPO, no por token. Con una sola escala por
            // token, un outlier en x arruina el resto de la fila: medido en
            // SmolLM2 (id 7359 "controvers" en bucle, p=0.54) contra 0.69
            // en 'La' con escala por grupo.
            for (std::size_t g = 0; g < G; ++g) {
                const std::size_t b0 = g * grp;
                float amax = 0.0f;
                for (std::size_t i = 0; i < grp; ++i)
                    amax = std::max(amax, std::fabs(xm[b0 + i]));
                const float sx = (amax > 0.0f ? amax / 127.0f : 1.0f);
                sm[g] = sx;
                if (g_slow_quant) {
                    // version vieja: una DIVISION por elemento (231k por token)
                    for (std::size_t i = 0; i < grp; ++i)
                        q[b0 + i] = static_cast<std::int8_t>(
                            std::clamp(std::round(xm[b0 + i] / sx), -127.0f, 127.0f));
                } else {
                    // reciproco una vez por grupo: mismos valores, sin divides
                    const float inv = (sx > 0.0f ? 1.0f / sx : 0.0f);
                    for (std::size_t i = 0; i < grp; ++i)
                        q[b0 + i] = static_cast<std::int8_t>(
                            std::clamp(std::round(xm[b0 + i] * inv), -127.0f, 127.0f));
                }
            }
            std::int8_t* qp = xqp.data() + std::size_t(m) * K;
            const std::size_t bloques = K / 64;
            for (std::size_t bb = 0; bb < bloques; ++bb)
                for (std::size_t i = 0; i < 64; ++i)
                    qp[bb * 64 + i] = q[bb * 64 + caudal::kPerm[i]];
            // Resto que no completa un bloque de 64: se copia sin permutar
            // (kPerm no esta definido para bloques parciales). Ningun modelo
            // actual lo ejercita, pero antes quedaba sin escribir.
            for (std::size_t i = bloques * 64; i < K; ++i) qp[i] = q[i];
            caudal::Shape s0; s0.N = 0; s0.K = K; s0.group = grp;
            caudal::compute_x_gsum(s0, q, xg.data() + std::size_t(m) * G);
            for (std::size_t g = 0; g < G; ++g) {
                xms[std::size_t(m) * G + g] = float(xg[std::size_t(m) * G + g]) * sm[g];
            }
        }
    });

    std::size_t total = 0;
    for (const auto& j : jobs) total += j.L->N;

    const unsigned T = pool.size();
    pool.run([&](unsigned tid) {
        const std::size_t a = std::size_t(tid) * total / T;
        const std::size_t b = std::size_t(tid + 1) * total / T;
        std::size_t off = 0;
        for (const auto& j : jobs) {
            const std::size_t lo = off, hi = off + j.L->N;
            if (b <= lo) break;
            const std::size_t n0 = (a > lo ? a : lo) - lo;
            const std::size_t n1 = (b < hi ? b : hi) - lo;
            if (n1 > n0) {
                caudal::Shape s; s.N = n1 - n0; s.K = K; s.group = grp;
                s.stride = j.L->stride;
                const std::uint8_t* pk = j.L->packed + n0 * (j.L->stride ? j.L->stride : j.L->row_bytes);
                // OJO: al partir filas hay que avanzar TAMBIEN el puntero de
                // escalas por n0; el kernel solo suma el indice local.
                const _Float16* sc = j.L->stride ? nullptr
                                                 : (j.L->scale + n0 * j.L->G);
                float* yy = j.y;   // M*N, fila m en yy + m*N
                if (j.L->min) {
                    const _Float16* mn = j.L->min + n0 * j.L->G;
                    switch (M) {
                        case 1: caudal::gemv_int4_affine_M<1>(s, pk, sc, mn, xms.data(), xqp.data(), xs.data(), yy + n0, j.L->N); break;
                        case 2: caudal::gemv_int4_affine_M<2>(s, pk, sc, mn, xms.data(), xqp.data(), xs.data(), yy + n0, j.L->N); break;
                        case 3: caudal::gemv_int4_affine_M<3>(s, pk, sc, mn, xms.data(), xqp.data(), xs.data(), yy + n0, j.L->N); break;
                        case 4: caudal::gemv_int4_affine_M<4>(s, pk, sc, mn, xms.data(), xqp.data(), xs.data(), yy + n0, j.L->N); break;
                        case 5: caudal::gemv_int4_affine_M<5>(s, pk, sc, mn, xms.data(), xqp.data(), xs.data(), yy + n0, j.L->N); break;
                        case 6: caudal::gemv_int4_affine_M<6>(s, pk, sc, mn, xms.data(), xqp.data(), xs.data(), yy + n0, j.L->N); break;
                        case 7: caudal::gemv_int4_affine_M<7>(s, pk, sc, mn, xms.data(), xqp.data(), xs.data(), yy + n0, j.L->N); break;
                        default: caudal::gemv_int4_affine_M<8>(s, pk, sc, mn, xms.data(), xqp.data(), xs.data(), yy + n0, j.L->N); break;
                    }
                    off = hi;   // no saltear el offset del siguiente job
                    continue;
#if defined(__AVX2__)
                } else if (j.L->kmin) {
                    const std::uint8_t* hp = j.L->kmin + n0 * (K / 256) * 16;
                    switch (M) {
                        case 1: caudal::gemv_int4_kmin_M<1>(s, pk, hp, xg.data(), xqp.data(), xs.data(), yy + n0, j.L->N); break;
                        case 2: caudal::gemv_int4_kmin_M<2>(s, pk, hp, xg.data(), xqp.data(), xs.data(), yy + n0, j.L->N); break;
                        case 3: caudal::gemv_int4_kmin_M<3>(s, pk, hp, xg.data(), xqp.data(), xs.data(), yy + n0, j.L->N); break;
                        case 4: caudal::gemv_int4_kmin_M<4>(s, pk, hp, xg.data(), xqp.data(), xs.data(), yy + n0, j.L->N); break;
                        case 5: caudal::gemv_int4_kmin_M<5>(s, pk, hp, xg.data(), xqp.data(), xs.data(), yy + n0, j.L->N); break;
                        case 6: caudal::gemv_int4_kmin_M<6>(s, pk, hp, xg.data(), xqp.data(), xs.data(), yy + n0, j.L->N); break;
                        case 7: caudal::gemv_int4_kmin_M<7>(s, pk, hp, xg.data(), xqp.data(), xs.data(), yy + n0, j.L->N); break;
                        default: caudal::gemv_int4_kmin_M<8>(s, pk, hp, xg.data(), xqp.data(), xs.data(), yy + n0, j.L->N); break;
                    }
                    off = hi;   // idem: el proximo job arranca en hi
                    continue;
                }
                switch (M) {
                    case 1: caudal::gemv_int4_g32_M<1>(s, pk, sc, xg.data(), xqp.data(), xs.data(), yy + n0, j.L->N); break;
                    case 2: caudal::gemv_int4_g32_M<2>(s, pk, sc, xg.data(), xqp.data(), xs.data(), yy + n0, j.L->N); break;
                    case 3: caudal::gemv_int4_g32_M<3>(s, pk, sc, xg.data(), xqp.data(), xs.data(), yy + n0, j.L->N); break;
                    case 4: caudal::gemv_int4_g32_M<4>(s, pk, sc, xg.data(), xqp.data(), xs.data(), yy + n0, j.L->N); break;
                    case 5: caudal::gemv_int4_g32_M<5>(s, pk, sc, xg.data(), xqp.data(), xs.data(), yy + n0, j.L->N); break;
                    case 6: caudal::gemv_int4_g32_M<6>(s, pk, sc, xg.data(), xqp.data(), xs.data(), yy + n0, j.L->N); break;
                    case 7: caudal::gemv_int4_g32_M<7>(s, pk, sc, xg.data(), xqp.data(), xs.data(), yy + n0, j.L->N); break;
                    default: caudal::gemv_int4_g32_M<8>(s, pk, sc, xg.data(), xqp.data(), xs.data(), yy + n0, j.L->N); break;
                }
#else
                } else {
                    std::fprintf(stderr, "Error: solo la cuantizacion afin es soportada en SSE4.2\n");
                    std::abort();
                }
#endif
            }
            off = hi;
        }
    });
    if (g_time) {
        g_t_gemv += std::chrono::duration<double, std::milli>(
                        std::chrono::steady_clock::now() - t0).count();
        // bytes que este GEMV recorre de verdad: filas*(K/2) + filas*G*2
        for (const auto& j : jobs)
            g_bytes += double(j.L->N) * (double(j.L->K) / 2.0 + double(j.L->G) * 2.0);
    }
}

void add_bias(float* y, const float* b, std::size_t n) {
    if (b) for (std::size_t i = 0; i < n; ++i) y[i] += b[i];
}

void rope_one(float* v, std::size_t pos, const Cfg& c) {
    const std::size_t half = c.DH / 2;
    for (std::size_t i = 0; i < half; ++i) {
        const float freq = 1.0f / std::pow(c.theta, float(2 * i) / float(c.DH));
        const float ang = float(pos) * freq;
        const float co = std::cos(ang), si = std::sin(ang);
        const float a = v[i], b = v[i + half];
        v[i] = a * co - b * si;
        v[i + half] = b * co + a * si;
    }
}

// offsets para que y[m*N + n] funcione con el kernel, que escribe y[m*N+n]
// (el kernel usa s.N como stride de M, por eso se le pasa el bloque completo).

// --- Prompt-lookup drafter ---------------------------------------------------
// Busca el sufijo mas largo del contexto que ya aparecio antes y propone los
// tokens que le siguieron. Coste ~0 (memcmp sobre el historico), sin modelo.
// Es la unica forma de batching con UN SOLO usuario: los k tokens propuestos se
// verifican en UNA pasada (M=k) y se acepta el prefijo correcto.
std::vector<int> prompt_lookup(const std::vector<int>& ctx, int k) {
    const int N = int(ctx.size());
    for (int n = std::min(8, N - 1); n >= 2; --n) {
        for (int p = N - n - 1; p >= 0; --p) {
            bool eq = true;
            for (int i = 0; i < n; ++i)
                if (ctx[p + i] != ctx[N - n + i]) { eq = false; break; }
            if (!eq) continue;
            std::vector<int> prop;
            for (int i = 0; i < k && p + n + i < N; ++i) prop.push_back(ctx[p + n + i]);
            if (!prop.empty()) return prop;
        }
    }
    return {};
}

}  // namespace

static double vnorm(const float* p, std::size_t n) {
    double s = 0.0;
    for (std::size_t i = 0; i < n; ++i) s += double(p[i]) * double(p[i]);
    return std::sqrt(s);
}

static bool g_dump = false;
static int g_stepn = 0;

int main(int argc, char** argv) {
    if (argc < 2) {
        std::printf("uso: %s <artefacto> --n <N> [--threads T] [--ctx C] "
                    "--prompt id,id,.. [--prompt id,id,..]\n", argv[0]);
        return 1;
    }
    const std::string A = argv[1];
    int ngen = 32, ctx = 512, spec = 0;
    unsigned reqT = 0;
    bool g_interactive = false;
    std::vector<int> eos_tokens = {151643, 151645};
    std::vector<std::vector<int>> prompts;
    for (int i = 2; i < argc; ++i) {
        const std::string a = argv[i];
        auto next = [&]() -> std::string { return (i + 1 < argc) ? argv[++i] : std::string(); };
        if (a == "--n") ngen = std::atoi(next().c_str());
        else if (a == "--threads") reqT = std::atoi(next().c_str());
        else if (a == "--ctx") ctx = std::atoi(next().c_str());
        else if (a == "--spec") spec = std::atoi(next().c_str());
        else if (a == "--dump") g_dump = true;
        else if (a == "--time") g_time = true;
        else if (a == "--probe-read") g_probe = true;
        else if (a == "--bench-pool") g_bench_pool = true;
        else if (a == "--slow-quant") g_slow_quant = true;
        else if (a == "--interactive") g_interactive = true;
        else if (a == "--prefill-batch") g_prefill_batch = unsigned(std::max(1, std::atoi(next().c_str())));
        else if (a == "--mem-policy") {
            const std::string v = next();
            if (v == "paged") g_mem_policy = MemPolicy::Paged;
            else if (v == "resident") g_mem_policy = MemPolicy::Resident;
            else if (v == "auto") g_mem_policy = MemPolicy::Auto;
            else { std::printf("mem-policy invalida: %s (auto|paged|resident)\n", v.c_str()); return 1; }
        }
        else if (a == "--keep-resident") g_mem_policy = MemPolicy::Resident;
        else if (a == "--kv-dtype") {
            const std::string v = next();
            if (v == "fp16") g_kv_fp16 = true;
            else if (v == "fp32") g_kv_fp16 = false;
            else { std::printf("kv-dtype invalido: %s (fp16|fp32)\n", v.c_str()); return 1; }
        }
        else if (a == "--eos") {
            eos_tokens.clear();
            std::stringstream ss(next()); std::string t;
            while (std::getline(ss, t, ',')) if (!t.empty()) eos_tokens.push_back(std::atoi(t.c_str()));
        }
        else if (a == "--prompt") {
            std::vector<int> v; std::stringstream ss(next()); std::string t;
            while (std::getline(ss, t, ',')) if (!t.empty()) v.push_back(std::atoi(t.c_str()));
            if (!v.empty()) prompts.push_back(v);
        }
    }
    if (prompts.empty() && !g_interactive) { std::printf("falta --prompt\n"); return 1; }
    if (prompts.size() > kMaxM) {
        std::printf("maximo %u secuencias, pediste %zu\n", kMaxM, prompts.size());
        return 1;
    }
    if (ctx < 1) { std::printf("--ctx tiene que ser >= 1\n"); return 1; }
    g_prefill_batch = std::min(g_prefill_batch, unsigned(kMaxM));
    spec = std::max(0, std::min(spec, int(kMaxM)));
    // Cada secuencia conserva SU longitud: truncar al minimo hacia que el
    // modelo recibiera prompts cortados a mitad de frase (y repitiera el texto).
    std::size_t plen = prompts.empty() ? 0 : prompts[0].size();
    for (auto& p : prompts) plen = std::max(plen, p.size());
    if (plen > std::size_t(ctx)) {
        std::printf("el prompt (%zu tokens) no cabe en --ctx %d\n", plen, ctx);
        return 1;
    }

    Cfg cfg;
    {
        std::ifstream fm(A + "/model.txt");
        if (!fm) { std::printf("no abrio model.txt\n"); return 1; }
        std::string k;
        while (fm >> k) {
            if (k == "H") fm >> cfg.H;
            else if (k == "NLY") fm >> cfg.NLY;
            else if (k == "NH") fm >> cfg.NH;
            else if (k == "NKV") fm >> cfg.NKV;
            else if (k == "DH") fm >> cfg.DH;
            else if (k == "INTER") fm >> cfg.INTER;
            else if (k == "V") fm >> cfg.V;
            else if (k == "EPS") fm >> cfg.eps;
            else if (k == "THETA") fm >> cfg.theta;
        }
        cfg.inv_sqrt_d = 1.0f / std::sqrt(float(cfg.DH));
    }

    Ctx c;
    std::string w_path = A + "/weights.bin";
    std::string s_path = A + "/scales.bin";
    if (!std::ifstream(w_path).good()) {
        w_path = A + "/w.bin";
        s_path = A + "/w.bin";
    }
    c.w_file.open_ro(w_path);
    c.s_file.open_ro(s_path);
    c.e_file.open_ro(A + "/embed.bf16");
    c.auxbin = slurp(A + "/aux.bin");

    // Decisión de residencia: si el artefacto cabe en la RAM disponible con
    // holgura, expulsar páginas solo agrega page faults y consumo de CPU.
    {
        const std::size_t model_mb = (c.w_file.size + c.s_file.size) >> 20;
        const std::size_t avail_mb = get_mem_available_mb();
        if (g_mem_policy == MemPolicy::Resident) g_evict_layers = false;
        else if (g_mem_policy == MemPolicy::Paged) g_evict_layers = true;
        else g_evict_layers = (model_mb + g_mem_reserve_mb > avail_mb);
        std::printf("  Memoria        : %zu MB de pesos, %zu MB disponibles -> %s\n",
                    model_mb, avail_mb,
                    g_evict_layers ? "paged (MADV_DONTNEED)"
                                   : "resident (page cache caliente)");
    }

    std::printf("\n=================================================================\n");
    std::printf("      Aquifer Paging Engine · HiperRAM & El Portero Activados     \n");
    std::printf("      ⚡ Powered by Semantic Geometry Chile                       \n");
    std::printf("  Pesos en disco : %.1f MB (weights) + %.1f MB (scales)\n",
                double(c.w_file.size) / (1024 * 1024), double(c.s_file.size) / (1024 * 1024));
    std::printf("  VmRSS Inicial  : %zu MB (¡Pesos NO cargados en RAM!)\n", get_vm_rss_mb());
    std::printf("=================================================================\n\n");

    {
        std::ifstream fi(A + "/index.txt");
        std::string kind;
        while (fi >> kind) {
            if (kind == "LINEARA") {              // afin: escala fp16 + minimo fp16
                std::string rest; std::getline(fi, rest);
                std::istringstream is(rest);
                std::string name; is >> name;
                std::size_t N2, K2, off2, pb2, so2, mo2;
                is >> N2 >> K2 >> off2 >> pb2 >> so2 >> mo2;
                Lin L; L.N = N2; L.K = K2; L.row_bytes = K2 / 2;
                L.packed = c.w_file.ptr + off2;
                L.scale = reinterpret_cast<const _Float16*>(c.s_file.ptr + so2);
                L.min = reinterpret_cast<const _Float16*>(c.s_file.ptr + mo2);
                L.stride = 0; L.G = K2 / 32;
                c.lin[name] = L;

                if (name.find("model.layers.") == 0) {
                    std::size_t dot1 = 13;
                    std::size_t dot2 = name.find('.', dot1);
                    if (dot2 != std::string::npos) {
                        std::size_t l_idx = std::stoul(name.substr(dot1, dot2 - dot1));
                        if (l_idx >= c.layer_spans.size()) c.layer_spans.resize(l_idx + 1);
                        auto& sp = c.layer_spans[l_idx];
                        sp.w_start = std::min(sp.w_start, off2);
                        sp.w_end   = std::max(sp.w_end, off2 + pb2);
                        sp.s_start = std::min(sp.s_start, so2);
                        sp.s_end   = std::max(sp.s_end, mo2 + (N2 * (K2 / 32) * 2));
                    }
                }
            } else if (kind == "LINEARK") {              // super-bloque + minimo
                std::string rest; std::getline(fi, rest);
                std::istringstream is(rest);
                std::string name; is >> name;
                std::size_t N2, K2, off2, pb2, so2, sb2;
                is >> N2 >> K2 >> off2 >> pb2 >> so2 >> sb2;
                Lin L; L.N = N2; L.K = K2; L.row_bytes = K2 / 2;
                L.packed = c.w_file.ptr + off2;
                L.scale = nullptr;
                L.kmin = c.s_file.ptr + so2;
                L.stride = 0; L.G = K2 / 32;
                c.lin[name] = L;
            } else if (kind == "LINEAR") {
                std::string rest; std::getline(fi, rest);
                std::istringstream is(rest);
                std::string name; is >> name;
                std::vector<std::size_t> v; std::size_t x;
                while (is >> x) v.push_back(x);
                if (v.size() < 2) continue;
                Lin L; L.N = v[0]; L.K = v[1]; L.row_bytes = L.K / 2;
                if (v.size() == 5) {                 // N K off stride G
                    L.packed = c.w_file.ptr + v[2];
                    L.stride = v[3]; L.G = v[4];
                } else if (v.size() == 6) {          // N K po pb so sb
                    L.packed = c.w_file.ptr + v[2];
                    L.scale = reinterpret_cast<const _Float16*>(c.s_file.ptr + v[4]);
                    L.stride = 0; L.G = v[5] / (2 * L.N);
                } else continue;
                c.lin[name] = L;
            } else if (kind == "AUX") {
                std::string name; std::size_t cnt, off;
                fi >> name >> cnt >> off;
                c.aux[name] = reinterpret_cast<const float*>(c.auxbin.data() + off * 4);
            } else {
                std::string rest; std::getline(fi, rest);
            }
        }
    }

    const unsigned T = reqT ? reqT : physical_cores();
    Pool pool(T);
    if (g_bench_pool) {
        const int N = 20000;
        for (int i = 0; i < 1000; ++i) pool.run([](unsigned) {});
        const auto p0 = std::chrono::steady_clock::now();
        for (int i = 0; i < N; ++i) pool.run([](unsigned) {});
        const double us = std::chrono::duration<double, std::micro>(
                              std::chrono::steady_clock::now() - p0).count() / N;
        std::printf("POOL: %.2f us por run vacia  ->  %.1f us/token con 113 llamadas\n",
                    us, us * 113.0);
    }
    const unsigned NS = g_interactive ? 1 : unsigned(prompts.size());
    KvRings kv;
    kv.init(std::size_t(ctx), cfg.NLY, cfg.NKV, cfg.DH, NS, g_kv_fp16);

    std::printf("\ncaudal · generate_qwen (batching)\n");
    std::printf("modelo   : H=%zu capas=%zu heads=%zu/%zu d=%zu inter=%zu\n",
                cfg.H, cfg.NLY, cfg.NH, cfg.NKV, cfg.DH, cfg.INTER);
    std::printf("hilos    : %u\n", T);
    std::printf("secuencias: %u   prompt=%zu tokens   ctx=%d\n", NS, plen, ctx);
    std::printf("KV total : %.1f MB (%.0f MB por secuencia)\n\n",
                NS * double(kv.per_seq) / (1024 * 1024),
                double(kv.per_seq) / (1024 * 1024));

    (void)NS;
    // Dimensionados para el MAXIMO de filas: el paso especulativo usa M=k
    // aunque haya una sola secuencia (era el segfault).
    const unsigned MB = kMaxM;
    std::vector<float> x(MB * cfg.H), h(MB * cfg.H), tmp(MB * cfg.H);
    std::vector<float> q(MB * cfg.NH * cfg.DH), kt(MB * cfg.NKV * cfg.DH), vt(MB * cfg.NKV * cfg.DH);
    // att por (token, cabeza): permite paralelizar la atención sin carreras.
    std::vector<float> att(std::size_t(MB) * cfg.NH * std::size_t(ctx)), o(MB * cfg.H);
    std::vector<float> gate(MB * cfg.INTER), up(MB * cfg.INTER);
    std::vector<float> logits(MB * cfg.V);

    // Instancia el Portero Asíncrono (hilo de I/O desacoplado)
    AsyncPrefetcher prefetcher(c);

    // Un paso para M secuencias en posiciones propias.
    auto step = [&](const std::vector<unsigned>& seq,
                   const std::vector<int>& toks,
                   const std::vector<std::size_t>& pos,
                   bool last_logits_only = false) {
        const unsigned m = unsigned(toks.size());
        const std::uint16_t* e = reinterpret_cast<const std::uint16_t*>(c.e_file.ptr);
        for (unsigned i = 0; i < m; ++i)
            for (std::size_t j = 0; j < cfg.H; ++j)
                x[i * cfg.H + j] = bf16_to_f32(e[std::size_t(toks[i]) * cfg.H + j]);

        if (g_dump && g_stepn == 0)
            std::printf("DUMP m=%u tok0=%d pos0=%zu  x[0..3]=%.6f %.6f %.6f %.6f\n",
                        m, toks[0], std::size_t(pos[0]), x[0], x[1], x[2], x[3]);

        // El Portero Asíncrono asegura la precarga de la Capa 0
        prefetcher.prefetch(0);

        for (std::size_t L = 0; L < cfg.NLY; ++L) {
            // El Portero Asíncrono precarga la Capa L+1 en background mientras AVX2 computa L
            if (L + 1 < cfg.NLY) {
                prefetcher.prefetch(L + 1);
            } else {
                prefetcher.prefetch(0); // Listo para el siguiente token
            }
            const std::string p = "model.layers." + std::to_string(L) + ".";
            for (unsigned i = 0; i < m; ++i)
                rmsnorm(x.data() + i * cfg.H, c.aux_of(p + "input_layernorm.weight"),
                        h.data() + i * cfg.H, cfg.H, cfg.eps);

            if (g_dump && g_stepn == 0 && L <= 1)
                std::printf("DUMP L%zu |h_in|=%.5f\n", L, vnorm(h.data(), cfg.H));

            const Job jq{&c.lin.at(p + "self_attn.q_proj.weight"), q.data()};
            const Job jk{&c.lin.at(p + "self_attn.k_proj.weight"), kt.data()};
            const Job jv{&c.lin.at(p + "self_attn.v_proj.weight"), vt.data()};
              gemv_group_M(pool, {jq, jk, jv}, m, h.data(), cfg.H);

            if (g_dump && g_stepn == 0 && L <= 1)
                std::printf("DUMP L%zu |q|=%.5f |k|=%.5f |v|=%.5f\n",
                            L,
                            vnorm(q.data(), cfg.NH * cfg.DH),
                            vnorm(kt.data(), cfg.NKV * cfg.DH),
                            vnorm(vt.data(), cfg.NKV * cfg.DH));


            for (unsigned i = 0; i < m; ++i) {
                add_bias(q.data() + i * cfg.NH * cfg.DH, c.aux_of(p + "self_attn.q_proj.bias"), cfg.NH * cfg.DH);
                add_bias(kt.data() + i * cfg.NKV * cfg.DH, c.aux_of(p + "self_attn.k_proj.bias"), cfg.NKV * cfg.DH);
                add_bias(vt.data() + i * cfg.NKV * cfg.DH, c.aux_of(p + "self_attn.v_proj.bias"), cfg.NKV * cfg.DH);
                for (std::size_t hh = 0; hh < cfg.NH; ++hh)
                    rope_one(q.data() + (i * cfg.NH + hh) * cfg.DH, pos[i], cfg);
                for (std::size_t hh = 0; hh < cfg.NKV; ++hh)
                    rope_one(kt.data() + (i * cfg.NKV + hh) * cfg.DH, pos[i], cfg);
                const std::size_t kn = cfg.NKV * cfg.DH;
                if (g_kv_fp16) {
                    auto* kd = reinterpret_cast<_Float16*>(kv.slot(seq[i], L, false, pos[i]));
                    auto* vd = reinterpret_cast<_Float16*>(kv.slot(seq[i], L, true, pos[i]));
                    const float* ks = kt.data() + i * kn;
                    const float* vs = vt.data() + i * kn;
                    for (std::size_t j = 0; j < kn; ++j) {
                        kd[j] = _Float16(ks[j]);
                        vd[j] = _Float16(vs[j]);
                    }
                } else {
                    std::memcpy(kv.slot(seq[i], L, false, pos[i]), kt.data() + i * kn, kn * sizeof(float));
                    std::memcpy(kv.slot(seq[i], L, true, pos[i]), vt.data() + i * kn, kn * sizeof(float));
                }
            }

            const auto ta = std::chrono::steady_clock::now();
            const std::size_t rep = cfg.NH / cfg.NKV;
            // La atención es O(ctx) por cabeza y antes corría en un solo hilo
            // mientras los workers miraban. Se reparte por (token, cabeza):
            // cada tarea escribe su propia fila de att y su propio vector o.
            const unsigned atasks = m * unsigned(cfg.NH);
            const unsigned aT = pool.size();
            pool.run([&](unsigned tid) {
                const unsigned t0 = unsigned(std::size_t(tid) * atasks / aT);
                const unsigned t1 = unsigned(std::size_t(tid + 1) * atasks / aT);
                for (unsigned task = t0; task < t1; ++task) {
                    const unsigned i = task / unsigned(cfg.NH);
                    const std::size_t hh = task % cfg.NH;
                    const std::size_t kvi = hh / rep;
                    const std::size_t n = std::min<std::size_t>(pos[i] + 1, std::size_t(ctx));
                    const std::uint8_t* kb = kv.slot(seq[i], L, false, 0);
                    const std::uint8_t* vb = kv.slot(seq[i], L, true, 0);
                    const std::size_t rowb = cfg.NKV * cfg.DH * kv.elem;
                    const std::size_t colb = kvi * cfg.DH * kv.elem;
                    const float* qh = q.data() + (i * cfg.NH + hh) * cfg.DH;
                    float* arow = att.data() + (std::size_t(i) * cfg.NH + hh) * std::size_t(ctx);
                    float mx = -1e30f;
                    for (std::size_t s = 0; s < n; ++s) {
                        const std::uint8_t* ksb = kb + s * rowb + colb;
                        const float d = (g_kv_fp16
                            ? dot_f32_f16(qh, reinterpret_cast<const _Float16*>(ksb), cfg.DH)
                            : dot_f32(qh, reinterpret_cast<const float*>(ksb), cfg.DH))
                            * cfg.inv_sqrt_d;
                        arow[s] = d;
                        mx = std::max(mx, d);
                    }
                    float sum = 0.0f;
                    for (std::size_t s = 0; s < n; ++s) {
                        const float ex = std::exp(arow[s] - mx);
                        arow[s] = ex;
                        sum += ex;
                    }
                    const float inv = 1.0f / sum;
                    float* oacc = o.data() + (i * cfg.NH + hh) * cfg.DH;
                    std::fill(oacc, oacc + cfg.DH, 0.0f);
                    for (std::size_t s = 0; s < n; ++s) {
                        const float a = arow[s] * inv;
                        const std::uint8_t* vsb = vb + s * rowb + colb;
                        if (g_kv_fp16) {
                            axpy_f32_f16(oacc, reinterpret_cast<const _Float16*>(vsb), a, cfg.DH);
                            continue;
                        }
                        const float* vs = reinterpret_cast<const float*>(vsb);
#if defined(__AVX2__)
                        const __m256 p = _mm256_set1_ps(a);
                        for (std::size_t j = 0; j + 8 <= cfg.DH; j += 8)
                            _mm256_storeu_ps(oacc + j, _mm256_fmadd_ps(
                                p, _mm256_loadu_ps(vs + j), _mm256_loadu_ps(oacc + j)));
#elif defined(__x86_64__) || defined(_M_X64)
                        const __m128 p = _mm_set1_ps(a);
                        std::size_t j = 0;
                        for (; j + 4 <= cfg.DH; j += 4) {
                            __m128 cur = _mm_loadu_ps(oacc + j);
                            __m128 v = _mm_loadu_ps(vs + j);
                            _mm_storeu_ps(oacc + j, _mm_add_ps(cur, _mm_mul_ps(p, v)));
                        }
                        for (; j < cfg.DH; ++j) oacc[j] += a * vs[j];
#elif defined(__ARM_NEON) || defined(__aarch64__)
                        const float32x4_t p = vdupq_n_f32(a);
                        std::size_t j = 0;
                        for (; j + 4 <= cfg.DH; j += 4) {
                            float32x4_t cur = vld1q_f32(oacc + j);
                            float32x4_t v = vld1q_f32(vs + j);
                            vst1q_f32(oacc + j, vmlaq_f32(cur, p, v));
                        }
                        for (; j < cfg.DH; ++j) oacc[j] += a * vs[j];
#else
                        for (std::size_t j = 0; j < cfg.DH; ++j) oacc[j] += a * vs[j];
#endif
                    }
                }
            });

            if (g_dump && g_stepn == 0 && L <= 1) {
                std::printf("DUMP L%zu |o_attn|=%.5f\n", L, vnorm(o.data(), cfg.H));
                std::printf("DUMP o[0..5]: %f %f %f %f %f\n", o[0], o[1], o[2], o[3], o[4]);
            }

            if (g_time)
                g_t_attn += std::chrono::duration<double, std::milli>(
                                std::chrono::steady_clock::now() - ta).count();

              const Job jo{&c.lin.at(p + "self_attn.o_proj.weight"), tmp.data()};
              gemv_group_M(pool, {jo}, m, o.data(), cfg.H);

            if (g_dump && g_stepn == 0 && L <= 1) {
                std::printf("DUMP L%zu |o_proj|=%.5f\n", L, vnorm(tmp.data(), cfg.H));
                std::printf("DUMP oproj[0..5]: %f %f %f %f %f\n", tmp[0], tmp[1], tmp[2], tmp[3], tmp[4]);
            }

            for (std::size_t j = 0; j < m * cfg.H; ++j) x[j] += tmp[j];

            for (unsigned i = 0; i < m; ++i)
                rmsnorm(x.data() + i * cfg.H, c.aux_of(p + "post_attention_layernorm.weight"),
                        h.data() + i * cfg.H, cfg.H, cfg.eps);

            if (g_dump && g_stepn == 0 && L <= 1) {
                std::printf("DUMP L%zu |h_mlp|=%.5f\n", L, vnorm(h.data(), cfg.H));
                std::printf("DUMP h_mlp[0..5]: %f %f %f %f %f\n", h[0], h[1], h[2], h[3], h[4]);
            }
            { const Job jg{&c.lin.at(p + "mlp.gate_proj.weight"), gate.data()};
              const Job ju{&c.lin.at(p + "mlp.up_proj.weight"), up.data()};
              gemv_group_M(pool, {jg, ju}, m, h.data(), cfg.H); }

            if (g_dump && g_stepn == 0 && L <= 1) {
                std::printf("DUMP L%zu |gate|=%.5f |up|=%.5f\n",
                            L, vnorm(gate.data(), cfg.INTER), vnorm(up.data(), cfg.INTER));
                std::printf("DUMP gate[0..5]: %f %f %f %f %f\n", gate[0], gate[1], gate[2], gate[3], gate[4]);
                std::printf("DUMP up[0..5]: %f %f %f %f %f\n", up[0], up[1], up[2], up[3], up[4]);
            }

            {
                const unsigned T2 = pool.size();
                pool.run([&](unsigned tid) {
                    const std::size_t a2 = std::size_t(tid) * (m * cfg.INTER) / T2;
                    const std::size_t b2 = std::size_t(tid + 1) * (m * cfg.INTER) / T2;
                    for (std::size_t i = a2; i < b2; ++i)
                        gate[i] = (gate[i] / (1.0f + std::exp(-gate[i]))) * up[i];
                });
            }
            { const Job jd{&c.lin.at(p + "mlp.down_proj.weight"), tmp.data()};
              gemv_group_M(pool, {jd}, m, gate.data(), cfg.INTER); }

            if (g_dump && g_stepn == 0 && L <= 1) {
                std::printf("DUMP L%zu |mlp_out|=%.5f\n", L, vnorm(tmp.data(), cfg.H));
                std::printf("DUMP out[0..5]: %f %f %f %f %f\n", tmp[0], tmp[1], tmp[2], tmp[3], tmp[4]);
            }

            for (std::size_t j = 0; j < m * cfg.H; ++j) x[j] += tmp[j];

            // El Portero desaloja la Capa L si la política es paged (alineado a pagina)
            if (g_evict_layers && L < c.layer_spans.size()) {
                const auto& cur = c.layer_spans[L];
                if (cur.w_end > cur.w_start) {
                    const std::size_t p_start = cur.w_start & ~4095UL;
                    const std::size_t p_len = ((cur.w_end + 4095UL) & ~4095UL) - p_start;
                    ::madvise(c.w_file.ptr + p_start, p_len, MADV_DONTNEED);
                }
                if (cur.s_end > cur.s_start) {
                    const std::size_t p_start = cur.s_start & ~4095UL;
                    const std::size_t p_len = ((cur.s_end + 4095UL) & ~4095UL) - p_start;
                    ::madvise(c.s_file.ptr + p_start, p_len, MADV_DONTNEED);
                }
            }

            if (g_dump && g_stepn == 0) {
                double s2 = 0.0;
                for (std::size_t j = 0; j < m * cfg.H; ++j) s2 += double(x[j]) * x[j];
                std::printf("DUMP L=%2zu |x|=%.5f x0=%.5f x1=%.5f\n",
                            L, std::sqrt(s2), x[0], x[1]);
            }
        }

        // En prefill solo los logits del último token importan: calcular los
        // demás cuesta M-1 pasadas completas por lm_head (116 MB en Qwen).
        const unsigned l0 = last_logits_only ? (m - 1) : 0;
        for (unsigned i = l0; i < m; ++i)
            rmsnorm(x.data() + i * cfg.H, c.aux_of("model.norm.weight"),
                    h.data() + std::size_t(i - l0) * cfg.H, cfg.H, cfg.eps);
          // lm_head puede tener nombre propio (tie_word_embeddings=False).
          const std::string lmn = c.lin.count("lm_head.weight") ? "lm_head.weight"
                                                              : "model.embed_tokens.weight";
          const Job jh{&c.lin.at(lmn), logits.data()};
          gemv_group_M(pool, {jh}, m - l0, h.data(), cfg.H);

          // El Portero desaloja lm_head tras calcular los logits, con la
          // dirección alineada a página (madvise devuelve EINVAL si no).
          if (g_evict_layers && c.lin.count(lmn)) {
              const auto& Lhead = c.lin.at(lmn);
              const std::uintptr_t a0 = reinterpret_cast<std::uintptr_t>(Lhead.packed) & ~std::uintptr_t(4095);
              const std::uintptr_t a1 = (reinterpret_cast<std::uintptr_t>(Lhead.packed)
                                         + Lhead.row_bytes * Lhead.N + 4095) & ~std::uintptr_t(4095);
              ::madvise(reinterpret_cast<void*>(a0), a1 - a0, MADV_DONTNEED);
          }


        std::vector<int> out(m, -1);
        for (unsigned i = 0; i < m - l0; ++i) {
            const float* lg = logits.data() + std::size_t(i) * cfg.V;
            std::size_t best = 0;
            for (std::size_t j = 1; j < cfg.V; ++j) if (lg[j] > lg[best]) best = j;
            out[l0 + i] = int(best);
        }
        if (g_dump && g_stepn == 0) {
            const float* lg = logits.data();
            std::vector<std::size_t> idx(cfg.V);
            for (std::size_t j = 0; j < cfg.V; ++j) idx[j] = j;
            std::partial_sort(idx.begin(), idx.begin() + 5, idx.end(),
                              [&](std::size_t a2, std::size_t b2) { return lg[a2] > lg[b2]; });
            std::printf("DUMP logits top5:");
            for (int k = 0; k < 5; ++k) std::printf(" %zu:%.4f", idx[k], lg[idx[k]]);
            std::printf("\n");
        }
        ++g_stepn;
        return out;
    };

    // Prefill por bloques: cada token del bloque usa el mismo peso leído una
    // vez. La causalidad se mantiene porque la atención de cada token corta en
    // su propia posición pos[i], aunque el KV del bloque ya esté escrito.
    auto prefill_range = [&](unsigned s, const std::vector<int>& toks,
                             std::size_t begin, std::size_t end) {
        const unsigned B = std::max(1u, std::min(g_prefill_batch, unsigned(kMaxM)));
        for (std::size_t i = begin; i < end; i += B) {
            const std::size_t n = std::min<std::size_t>(B, end - i);
            std::vector<unsigned> sq(n, s);
            std::vector<int> tt(toks.begin() + std::ptrdiff_t(i),
                                toks.begin() + std::ptrdiff_t(i + n));
            std::vector<std::size_t> pp(n);
            for (std::size_t j = 0; j < n; ++j) pp[j] = i + j;
            (void)step(sq, tt, pp, true);
        }
    };

    if (g_interactive) {
        std::printf("READY\n");
        std::fflush(stdout);
        std::vector<int> cached_tokens;
        std::size_t cur_pos = 0;
        int last_tok = -1;
        bool state_ready = false, have_logits = false;
        std::size_t draft_initial = 0;
        std::vector<int> last_draft;
        std::vector<float> saved_logits;
        auto parse_ids = [](const std::string& s) {
            std::vector<int> v;
            std::stringstream pss(s);
            std::string t;
            while (std::getline(pss, t, ',')) if (!t.empty()) v.push_back(std::atoi(t.c_str()));
            return v;
        };
        auto argmax_row = [&](const float* lg) {
            std::size_t best = 0;
            for (std::size_t j = 1; j < cfg.V; ++j) if (lg[j] > lg[best]) best = j;
            return int(best);
        };
        auto ensure_logits = [&]() {
            if (state_ready && !have_logits) {
                (void)step({0u}, {last_tok}, {cur_pos});
                if (cached_tokens.size() <= cur_pos) cached_tokens.resize(cur_pos + 1);
                cached_tokens[cur_pos] = last_tok;
                have_logits = true;
            }
        };
        std::string line;
        for (;;) {
            pool.pause(); // Pone los workers a dormir mientras espera comando (0% CPU, ventilador silencioso)
            if (!std::getline(std::cin, line)) break;
            pool.resume(); // Despierta workers para inferencia con latencia 0 y máxima velocidad
            if (line.empty()) continue;
            if (line == "QUIT" || line == "EXIT") break;
            std::stringstream ss(line);
            std::string cmd;
            ss >> cmd;
            if (cmd == "RESET") {
                cached_tokens.clear();
                state_ready = false; have_logits = false;
                last_draft.clear(); saved_logits.clear();
                std::printf("RESET_OK\n");
                std::fflush(stdout);
                continue;
            }
            if (cmd == "PREWARM") {
                std::string ids_str;
                ss >> ids_str;
                std::vector<int> ptokens;
                std::stringstream pss(ids_str);
                std::string t;
                while (std::getline(pss, t, ',')) {
                    if (!t.empty()) ptokens.push_back(std::atoi(t.c_str()));
                }
                if (ptokens.size() > std::size_t(ctx)) {
                    std::fprintf(stderr, "PREWARM: %zu tokens > ctx %d, se trunca\n",
                                 ptokens.size(), ctx);
                    ptokens.resize(std::size_t(ctx));
                }
                prefill_range(0, ptokens, 0, ptokens.size());
                cached_tokens = ptokens;
                state_ready = false; have_logits = false;
                std::printf("PREWARMED\n");
                std::fflush(stdout);
                continue;
            }
            if (cmd == "BEGIN") {
                std::string ids_str;
                ss >> ids_str;
                std::vector<int> ptokens = parse_ids(ids_str);
                if (ptokens.empty() || ptokens.size() > std::size_t(ctx)) {
                    std::printf("BEGIN_ERR\n"); std::fflush(stdout); continue;
                }
                std::size_t common = 0;
                const std::size_t limit = std::min(cached_tokens.size(), ptokens.size() - 1);
                while (common < limit && cached_tokens[common] == ptokens[common]) ++common;
                prefill_range(0, ptokens, common, ptokens.size() - 1);
                cached_tokens.resize(ptokens.size() - 1);
                for (std::size_t i = common; i + 1 < ptokens.size(); ++i)
                    cached_tokens[i] = ptokens[i];
                last_tok = ptokens.back();
                cur_pos = ptokens.size() - 1;
                (void)step({0u}, {last_tok}, {cur_pos});
                cached_tokens.push_back(last_tok);
                state_ready = true; have_logits = true;
                std::printf("BEGIN_OK\n"); std::fflush(stdout);
                continue;
            }
            if (cmd == "NEXT") {
                if (!state_ready) { std::printf("NEXT_ERR\n"); std::fflush(stdout); continue; }
                ensure_logits();
                std::printf("NEXT %d\n", argmax_row(logits.data()));
                std::fflush(stdout);
                continue;
            }
            if (cmd == "VERIFY") {
                std::string ids_str;
                ss >> ids_str;
                std::vector<int> prop = parse_ids(ids_str);
                if (!state_ready || prop.empty() || prop.size() > kMaxM) {
                    std::printf("ACCEPT 0\n"); std::fflush(stdout); continue;
                }
                ensure_logits();
                const int y = argmax_row(logits.data());
                unsigned a = 0;
                if (prop[0] == y) {
                    std::vector<unsigned> sq(prop.size(), 0u);
                    std::vector<std::size_t> ps(prop.size());
                    for (std::size_t j = 0; j < prop.size(); ++j) ps[j] = cur_pos + 1 + j;
                    (void)step(sq, prop, ps);
                    a = 1;
                    for (std::size_t j = 1; j < prop.size(); ++j) {
                        if (argmax_row(logits.data() + (j - 1) * cfg.V) == prop[j]) ++a;
                        else break;
                    }
                    std::memcpy(logits.data(),
                                logits.data() + std::size_t(a - 1) * cfg.V,
                                cfg.V * sizeof(float));
                    for (unsigned j = 0; j < a; ++j) cached_tokens.push_back(prop[j]);
                    cached_tokens.resize(cur_pos + 1 + a);
                    cur_pos += a;
                    last_tok = prop[a - 1];
                    have_logits = true;
                }
                std::printf("ACCEPT %u\n", a); std::fflush(stdout);
                continue;
            }
            if (cmd == "COMMIT") {
                int tok = -1;
                ss >> tok;
                if (!state_ready || cur_pos + 1 >= std::size_t(ctx)) {
                    std::printf("COMMIT_ERR\n"); std::fflush(stdout); continue;
                }
                (void)step({0u}, {tok}, {cur_pos + 1});
                ++cur_pos;
                last_tok = tok;
                cached_tokens.push_back(tok);
                have_logits = true;
                std::printf("COMMIT_OK\n"); std::fflush(stdout);
                continue;
            }
            if (cmd == "DRAFT") {
                unsigned k = 0;
                ss >> k;
                if (!state_ready || k == 0 || k > kMaxM) {
                    std::printf("DRAFT\n"); std::fflush(stdout); continue;
                }
                ensure_logits();
                draft_initial = cur_pos;
                last_draft.clear();
                saved_logits.resize(std::size_t(k) * cfg.V);
                for (unsigned i = 0; i < k && cur_pos + 1 < std::size_t(ctx); ++i) {
                    const int y = argmax_row(logits.data());
                    last_draft.push_back(y);
                    (void)step({0u}, {y}, {cur_pos + 1});
                    ++cur_pos;
                    have_logits = true;
                    cached_tokens.push_back(y);
                    std::memcpy(saved_logits.data() + std::size_t(i) * cfg.V,
                                logits.data(), cfg.V * sizeof(float));
                }
                std::printf("DRAFT ");
                for (std::size_t i = 0; i < last_draft.size(); ++i)
                    std::printf("%s%d", i ? "," : "", last_draft[i]);
                std::printf("\n"); std::fflush(stdout);
                continue;
            }
            if (cmd == "SYNC") {
                unsigned a = 0;
                ss >> a;
                if (!state_ready || a > last_draft.size()) {
                    std::printf("SYNC_ERR\n"); std::fflush(stdout); continue;
                }
                if (a == 0) {
                    cached_tokens.resize(draft_initial + 1);
                    cur_pos = draft_initial;
                    have_logits = false;
                } else {
                    cached_tokens.resize(draft_initial + 1 + a);
                    cur_pos = draft_initial + a;
                    last_tok = last_draft[a - 1];
                    std::memcpy(logits.data(),
                                saved_logits.data() + std::size_t(a - 1) * cfg.V,
                                cfg.V * sizeof(float));
                    have_logits = true;
                }
                std::printf("SYNC_OK\n"); std::fflush(stdout);
                continue;
            }
            if (cmd == "PROMPT") {
                std::string ids_str;
                ss >> ids_str;
                std::vector<int> ptokens;
                std::stringstream pss(ids_str);
                std::string t;
                while (std::getline(pss, t, ',')) {
                    if (!t.empty()) ptokens.push_back(std::atoi(t.c_str()));
                }
                if (ptokens.empty()) {
                    std::printf("DONE\n");
                    std::fflush(stdout);
                    continue;
                }
                // Encuentra prefijo ya presente en el KV cache
                std::size_t common = 0;
                while (common < cached_tokens.size() &&
                       common + 1 < ptokens.size() &&
                       cached_tokens[common] == ptokens[common]) {
                    ++common;
                }

                if (ptokens.size() > std::size_t(ctx)) {
                    std::fprintf(stderr, "PROMPT: %zu tokens > ctx %d\n",
                                 ptokens.size(), ctx);
                    std::printf("DONE\n");
                    std::fflush(stdout);
                    continue;
                }

                // Prefill unicamente de los tokens nuevos (los anteriores ya estan en KV)
                prefill_range(0, ptokens, common, ptokens.size() - 1);

                cached_tokens.resize(ptokens.size() - 1);
                for (std::size_t i = common; i + 1 < ptokens.size(); ++i) {
                    cached_tokens[i] = ptokens[i];
                }

                int last_tok = ptokens.back();
                std::size_t cur_pos = ptokens.size() - 1;
                // El último token del prompt todavía no está procesado: el
                // primer step lo procesa y deja logits para elegir el siguiente.
                bool have_logits = false;
                int generated = 0;
                bool stop = false;
                auto argmax_row = [&](const float* lg) {
                    std::size_t best = 0;
                    for (std::size_t j = 1; j < cfg.V; ++j)
                        if (lg[j] > lg[best]) best = j;
                    return int(best);
                };
                auto emit = [&](int tok) {
                    std::printf("TOKEN %d\n", tok);
                    std::fflush(stdout);
                    ++generated;
                    for (int e : eos_tokens) if (tok == e) stop = true;
                };
                while (generated < ngen && !stop) {
                    if (cur_pos >= std::size_t(ctx)) break;
                    if (!have_logits) {
                        (void)step({0u}, {last_tok}, {cur_pos});
                        if (cached_tokens.size() <= cur_pos) cached_tokens.resize(cur_pos + 1);
                        cached_tokens[cur_pos] = last_tok;
                        have_logits = true;
                    }
                    const int y = argmax_row(logits.data());

                    bool speculated = false;
                    if (spec > 0 && cur_pos + 1 < std::size_t(ctx)) {
                        auto prop = prompt_lookup(cached_tokens, spec);
                        if (!prop.empty() && prop[0] == y) {
                            std::size_t k = prop.size();
                            k = std::min(k, std::size_t(ngen - generated));
                            k = std::min(k, std::size_t(ctx) - 1 - cur_pos);
                            if (k > 0) {
                                prop.resize(k);
                                std::vector<unsigned> sq(k, 0u);
                                std::vector<std::size_t> ps(k);
                                for (std::size_t j = 0; j < k; ++j) ps[j] = cur_pos + 1 + j;
                                (void)step(sq, prop, ps);   // verifica k tokens en una pasada
                                unsigned a = 1;             // prop[0] == y ya está aceptado
                                for (std::size_t j = 1; j < k; ++j) {
                                    const int pred = argmax_row(logits.data() + (j - 1) * cfg.V);
                                    if (pred == prop[j]) ++a; else break;
                                }
                                for (unsigned j = 0; j < a && !stop; ++j) {
                                    emit(prop[j]);
                                    cached_tokens.push_back(prop[j]);
                                }
                                cur_pos += a;
                                last_tok = prop[a - 1];
                                if (!stop && generated < ngen) {
                                    // R[a-1] es el último token procesado y
                                    // predice el siguiente (sea aceptado o no).
                                    std::memcpy(logits.data(),
                                                logits.data() + std::size_t(a - 1) * cfg.V,
                                                cfg.V * sizeof(float));
                                    have_logits = true;
                                }
                                speculated = true;
                            }
                        }
                    }
                    if (speculated) {
                        if (stop || generated >= ngen) break;
                        struct pollfd pfd{STDIN_FILENO, POLLIN, 0};
                        if (::poll(&pfd, 1, 0) > 0) {
                            std::string ctl;
                            std::getline(std::cin, ctl);
                            if (ctl == "STOP") break;
                        }
                        continue;
                    }

                    // Sin propuesta: emite y y procesa y en la próxima vuelta.
                    emit(y);
                    cached_tokens.push_back(y);
                    if (stop || generated >= ngen) break;
                    struct pollfd pfd{STDIN_FILENO, POLLIN, 0};
                    if (::poll(&pfd, 1, 0) > 0) {
                        std::string ctl;
                        std::getline(std::cin, ctl);
                        if (ctl == "STOP") break;
                    }
                    ++cur_pos;
                    last_tok = y;
                    have_logits = false;
                }
                std::printf("DONE\n");
                std::fflush(stdout);
            }
        }
        return 0;
    }

    // --- prefill batcheado (una secuencia a la vez, B tokens por pasada) ------
    const double tp0 = now_ms();
    for (unsigned s = 0; s < NS; ++s) {
        const std::size_t n_s = prompts[s].size();
        prefill_range(s, prompts[s], 0, n_s - 1);
    }
    const double tp1 = now_ms();
    // --time debe medir SOLO decode: el prefill tiene otra mezcla de M y de
    // tráfico, y sumarlo inflaba los bytes/token reportados.
    if (g_time) { g_t_gemv = 0.0; g_t_attn = 0.0; g_bytes = 0.0; }

    // --- decode batcheado ----------------------------------------------------
    std::vector<unsigned> seqids(NS);
    for (unsigned s = 0; s < NS; ++s) seqids[s] = s;
    std::vector<int> last(NS);
    std::vector<std::size_t> pos(NS);
    for (unsigned s = 0; s < NS; ++s) { pos[s] = prompts[s].size() - 1; last[s] = prompts[s].back(); }

    std::vector<std::vector<int>> gen(NS);
    const int steps = ngen;
    const double td0 = now_ms();
    for (int t = 0; t < steps; ++t) {
        auto out = step(seqids, last, pos);
        for (unsigned s = 0; s < NS; ++s) {
            gen[s].push_back(out[s]);
            last[s] = out[s];
            ++pos[s];
        }
    }
    const double td1 = now_ms();

    std::printf("prefill  : %.1f ms (%u secuencias x %zu tokens)\n", tp1 - tp0, NS, plen);
    if (g_probe) {
        // Lectura pura, EN ESTE PROCESO y SOBRE EL MISMO BUFFER de pesos, justo
        // despues del GEMV: es el unico denominador que comparte estado termico,
        // politica de potencia y paginas fisicas con el motor.
        std::vector<double> v;
        for (int r = 0; r < 4; ++r) {
            const auto t0 = std::chrono::steady_clock::now();
            const std::size_t n = c.w_file.size;
            const std::uint64_t* w = reinterpret_cast<const std::uint64_t*>(c.w_file.ptr);
            const std::size_t nw = n / 8;
            std::uint64_t suma = 0;
            pool.run([&](unsigned tid) {
                const std::size_t T3 = pool.size();
                const std::size_t a = std::size_t(tid) * nw / T3;
                const std::size_t b = std::size_t(tid + 1) * nw / T3;
                std::uint64_t s0 = 0, s1 = 0;
                for (std::size_t i = a; i + 8 <= b; i += 8) { s0 += w[i]; s1 += w[i + 4]; }
                for (std::size_t i = a; i < b; ++i) s0 += w[i] & 1u;
                if (tid == 0) suma = s0 ^ s1; else suma ^= s0 ^ s1;
            });
            asm volatile("" : "+r"(suma));
            const double seg = std::chrono::duration<double>(
                std::chrono::steady_clock::now() - t0).count();
            v.push_back(double(n) / 1e9 / (seg > 0 ? seg : 1e-9));
        }
        std::sort(v.begin(), v.end());
        std::printf("READ in-process: %.2f GB/s (mediana de 4, mismo buffer, %u hilos)\n",
                    v[2], pool.size());
    }
    std::printf("decode   : %.1f ms para %d pasos x %u secuencias\n", td1 - td0, steps, NS);
    if (g_time) {
        // El cierre de gemv_group_M no puede tocar g_t_gemv sin un objeto;
        // se acumula aqui con una lambda al salir de cada llamada.
        std::printf("TIME bytes=%.1f MB/token  (%.1f GB/s reales)\n",
                    g_bytes / 1e6 / double(steps), g_bytes / 1e6 / g_t_gemv);
        std::printf("TIME gemv=%.0f ms  attn=%.0f ms  (%.1f%% / %.1f%%)\n",
                    g_t_gemv, g_t_attn,
                    100.0 * g_t_gemv / (g_t_gemv + g_t_attn > 0 ? g_t_gemv + g_t_attn : 1),
                    100.0 * g_t_attn / (g_t_gemv + g_t_attn > 0 ? g_t_gemv + g_t_attn : 1));
    }
    std::printf("AQUIFER  : VmRSS real en silicio = %zu MB  (%s)\n",
                get_vm_rss_mb(),
                g_evict_layers ? "paged: pesos desalojados"
                               : "resident: pesos en page cache/RSS");
    std::printf("           %.2f ms/paso  ->  %.1f tok/s AGREGADOS  (%.1f por secuencia)\n\n",
                (td1 - td0) / steps,
                1000.0 * steps * NS / (td1 - td0),
                1000.0 * steps / (td1 - td0));

    for (unsigned s = 0; s < NS; ++s) {
        std::printf("SEQ%u ", s);
        for (int v : gen[s]) std::printf("%d ", v);
        std::printf("\n");
    }

    std::printf("\n");
    return 0;
}
