// =============================================================================
//  caudal · tools/generate_qwen.cpp
//  GENERACIÓN con KV cache en ANILLO (memfd doble mapeo) + pool de hilos.
// =============================================================================
//  Tres cosas nuevas respecto de forward_qwen.cpp (que queda como referencia
//  O(T^2) ya validada contra PyTorch a 1e-6):
//
//   1. KV CACHE. El estado de atención de cada capa vive en un anillo de tamaño
//      fijo. Se escribe en `base + t*stride`; como el mismo memfd está mapeado
//      DOS VECES consecutivas, una ventana [start, end] con end-start <= cap es
//      SIEMPRE contigua: ninguna rama de wraparound en el bucle de atención.
//      (Esa es exactamente la propiedad que test_doublemap valida.)
//
//   2. MULTIHILO. Pool persistente con barreras: se evita el costo de crear
//      169 GEMV x 8 hilos por token. Partición por filas de salida, sin
//      reducción entre hilos.
//
//   3. DECODE INCREMENTAL. Prefill del prompt una vez; cada token nuevo cuesta
//      O(1) GEMVs + O(T) atención (no O(T) capas completas).
//
//  Uso: ./generate_qwen <artefacto> <id> [<id> ...] --n <N>
// =============================================================================

#include "caudal/gemv_int4.hpp"
#include "caudal/layout.hpp"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <fstream>
#include <functional>
#include <map>
#include <string>
#include <thread>
#include <vector>

#include <immintrin.h>
#include <sys/mman.h>
#include <unistd.h>

namespace {

constexpr std::size_t kH = 896, kNLY = 24, kNH = 14, kNKV = 2, kDH = 64;
constexpr float kEps = 1e-6f, kTheta = 1000000.0f, kInvSqrtD = 1.0f / 8.0f;
constexpr std::size_t kMaxTok = 4096;

struct Lin {
    std::size_t N{0}, K{0}, G{0};
    const std::uint8_t* packed{nullptr};
    const _Float16* scale{nullptr};
    std::size_t row_bytes{0};
};

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

// --- barrera de spin ---------------------------------------------------------
// std::barrier bloquea en futex: ~5-20 us por encuentro. Con 168 GEMV por token
// eso son milisegundos de puro overhead. Con 16 hilos o menos, girar gana.
class SpinBarrier {
public:
    explicit SpinBarrier(unsigned t) : T_(t) {}
    void wait() noexcept {
        const unsigned g = gen_.load(std::memory_order_acquire);
        if (cnt_.fetch_add(1, std::memory_order_acq_rel) + 1 == T_) {
            cnt_.store(0, std::memory_order_relaxed);
            gen_.fetch_add(1, std::memory_order_release);
        } else {
            while (gen_.load(std::memory_order_acquire) == g) _mm_pause();
        }
    }
private:
    unsigned T_;
    std::atomic<unsigned> cnt_{0}, gen_{0};
};

// --- pool persistente --------------------------------------------------------
class Pool {
public:
    // T = TOTAL de hilos, contando al principal. Con T+1 (principal ocioso)
    // el spin-barrier se estrangula: alguien gira mientras otro fue desalojado.
    explicit Pool(unsigned T) : T_(T ? T : 1), go_(T_), fin_(T_) {
        for (unsigned i = 1; i < T_; ++i)
            th_.emplace_back([this, i] {
                for (;;) {
                    go_.wait();
                    if (stop_) return;
                    job_(i);
                    fin_.wait();
                }
            });
    }
    ~Pool() { stop_ = true; go_.wait(); for (auto& t : th_) t.join(); }
    Pool(const Pool&) = delete;
    Pool& operator=(const Pool&) = delete;
    unsigned size() const { return T_; }
    void run(std::function<void(unsigned)> f) {
        job_ = std::move(f);
        go_.wait();
        job_(0);              // el principal TAMBIEN trabaja
        fin_.wait();
    }
private:
    unsigned T_;
    SpinBarrier go_, fin_;
    std::vector<std::thread> th_;
    std::function<void(unsigned)> job_;
    bool stop_ = false;
};

// --- anillo de KV (memfd doble mapeo) ----------------------------------------
struct KvRing {
    std::uint8_t* base{nullptr};
    std::size_t cap_tok{0}, cap_bytes{0}, stride{0}, total{0};
    int fd{-1};

    void init(std::size_t tokens, std::size_t layers) {
        cap_tok = tokens;
        stride = kNKV * kDH * sizeof(float);       // bytes por token por cabeza-set
        cap_bytes = cap_tok * stride;
        total = layers * 2 * cap_bytes;
        fd = ::memfd_create("caudal_kv", MFD_CLOEXEC);
        if (fd < 0 || ::ftruncate(fd, static_cast<off_t>(total)) != 0) {
            std::perror("memfd"); std::exit(1);
        }
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
    float* k(std::size_t layer, std::size_t t) const {
        return reinterpret_cast<float*>(base + layer * 2 * cap_bytes) + (t % cap_tok) * (stride / 4);
    }
    float* v(std::size_t layer, std::size_t t) const {
        return reinterpret_cast<float*>(base + layer * 2 * cap_bytes + cap_bytes) + (t % cap_tok) * (stride / 4);
    }
};

struct Ctx {
    std::vector<std::uint8_t> wbin, sbin, auxbin, embed;
    std::size_t V{0}, H{0};
    std::map<std::string, Lin> lin;
    std::map<std::string, const float*> aux;
    const float* aux_of(const std::string& n) const {
        auto it = aux.find(n); return it == aux.end() ? nullptr : it->second;
    }
};

void rmsnorm(const float* x, const float* w, float* out, std::size_t n) {
    double ss = 0.0;
    for (std::size_t i = 0; i < n; ++i) ss += double(x[i]) * double(x[i]);
    const float r = 1.0f / std::sqrt(float(ss / double(n)) + kEps);
    for (std::size_t i = 0; i < n; ++i) out[i] = x[i] * r * w[i];
}

// GEMV int4 multihilo. x se cuantiza UNA vez; los hilos solo hacen sus filas.
void gemv_mt(Pool& pool, const Lin& L, const float* x, float* y) {
    std::vector<std::int8_t> xq(L.K), xqp(L.K);
    float amax = 0.0f;
    for (std::size_t i = 0; i < L.K; ++i) amax = std::max(amax, std::fabs(x[i]));
    const float sx = (amax > 0.0f ? amax / 127.0f : 1.0f);
    for (std::size_t i = 0; i < L.K; ++i) {
        const float q = std::clamp(std::round(x[i] / sx), -127.0f, 127.0f);
        xq[i] = static_cast<std::int8_t>(q);
    }
    for (std::size_t g = 0; g < L.G; ++g)
        for (std::size_t i = 0; i < 64; ++i)
            xqp[g * 64 + i] = xq[g * 64 + caudal::kPerm[i]];

    caudal::Shape s0; s0.N = 0; s0.K = L.K; s0.group = 64;
    std::vector<std::int32_t> xg(L.G);
    caudal::compute_x_gsum(s0, xq.data(), xg.data());

    const unsigned T = pool.size();
    pool.run([&](unsigned tid) {
        const std::size_t n0 = std::size_t(tid) * L.N / T;
        const std::size_t n1 = std::size_t(tid + 1) * L.N / T;
        if (n1 <= n0) return;
        caudal::Shape s; s.N = n1 - n0; s.K = L.K; s.group = 64;
        caudal::gemv_int4(s, L.packed + n0 * L.row_bytes, L.scale + n0 * L.G,
                          xg.data(), xqp.data(), sx, y + n0);
    });
}

void add_bias(float* y, const float* b, std::size_t n) {
    if (b) for (std::size_t i = 0; i < n; ++i) y[i] += b[i];
}

void rope_one(float* v, std::size_t pos) {
    const std::size_t half = kDH / 2;
    for (std::size_t i = 0; i < half; ++i) {
        const float freq = 1.0f / std::pow(kTheta, float(2 * i) / float(kDH));
        const float ang = float(pos) * freq;
        const float c = std::cos(ang), s = std::sin(ang);
        const float a = v[i], b = v[i + half];
        v[i] = a * c - b * s;
        v[i + half] = b * c + a * s;
    }
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 3) {
        std::printf("uso: %s <artefacto> <id> [<id>...] --n <N>\n", argv[0]);
        return 1;
    }
    const std::string A = argv[1];
    std::vector<int> prompt;
    int ngen = 16;
    unsigned reqT = 0;
    for (int i = 2; i < argc; ++i) {
        if (std::strcmp(argv[i], "--n") == 0 && i + 1 < argc) { ngen = std::atoi(argv[++i]); continue; }
        if (std::strcmp(argv[i], "--threads") == 0 && i + 1 < argc) { reqT = std::atoi(argv[++i]); continue; }
        prompt.push_back(std::atoi(argv[i]));
    }
    if (prompt.empty()) { std::printf("falta el prompt\n"); return 1; }

    Ctx c;
    c.wbin = slurp(A + "/weights.bin");
    c.sbin = slurp(A + "/scales.bin");
    c.auxbin = slurp(A + "/aux.bin");
    c.embed = slurp(A + "/embed.bf16");

    std::ifstream fi(A + "/index.txt");
    std::string kind;
    while (fi >> kind) {
        if (kind == "LINEAR") {
            std::string name; std::size_t N, K, po, pb, so, sb;
            fi >> name >> N >> K >> po >> pb >> so >> sb;
            Lin L; L.N = N; L.K = K; L.G = K / 64; L.row_bytes = K / 2;
            L.packed = c.wbin.data() + po;
            L.scale = reinterpret_cast<const _Float16*>(c.sbin.data() + so);
            c.lin[name] = L;
        } else if (kind == "AUX") {
            std::string name; std::size_t cnt, off;
            fi >> name >> cnt >> off;
            c.aux[name] = reinterpret_cast<const float*>(c.auxbin.data() + off * 4);
        } else if (kind == "EMBED") {
            std::string name; fi >> name >> c.V >> c.H;
        }
    }

    unsigned T = reqT ? reqT : std::max(1u, std::thread::hardware_concurrency());
    Pool pool(T);

    KvRing kv;
    kv.init(kMaxTok, kNLY);

    std::printf("\ncaudal · generate_qwen\n");
    std::printf("artefacto : %s  (%zu lineales)\n", A.c_str(), c.lin.size());
    std::printf("hilos     : %u\n", T);
    std::printf("KV ring   : %zu tokens x %zu capas = %.1f MB (memfd doble mapeo)\n",
                kMaxTok, kNLY, double(kv.total) / (1024.0 * 1024.0));
    std::printf("prompt    : %zu tokens\n\n", prompt.size());

    std::vector<float> x(kH), h(kH), tmp(kH);
    std::vector<float> q(kNH * kDH), kt(kNKV * kDH), vt(kNKV * kDH);
    std::vector<float> att(kNH * kMaxTok), o(kH), gate(4864), up(4864);

    auto step = [&](int tok, std::size_t pos) {
        const std::uint16_t* e = reinterpret_cast<const std::uint16_t*>(c.embed.data());
        for (std::size_t i = 0; i < kH; ++i) x[i] = bf16_to_f32(e[std::size_t(tok) * kH + i]);

        for (std::size_t L = 0; L < kNLY; ++L) {
            const std::string p = "model.layers." + std::to_string(L) + ".";
            rmsnorm(x.data(), c.aux_of(p + "input_layernorm.weight"), h.data(), kH);

            gemv_mt(pool, c.lin.at(p + "self_attn.q_proj.weight"), h.data(), q.data());
            gemv_mt(pool, c.lin.at(p + "self_attn.k_proj.weight"), h.data(), kt.data());
            gemv_mt(pool, c.lin.at(p + "self_attn.v_proj.weight"), h.data(), vt.data());
            add_bias(q.data(), c.aux_of(p + "self_attn.q_proj.bias"), kNH * kDH);
            add_bias(kt.data(), c.aux_of(p + "self_attn.k_proj.bias"), kNKV * kDH);
            add_bias(vt.data(), c.aux_of(p + "self_attn.v_proj.bias"), kNKV * kDH);

            for (std::size_t hh = 0; hh < kNH; ++hh) rope_one(q.data() + hh * kDH, pos);
            for (std::size_t hh = 0; hh < kNKV; ++hh) rope_one(kt.data() + hh * kDH, pos);

            std::memcpy(kv.k(L, pos), kt.data(), kNKV * kDH * sizeof(float));
            std::memcpy(kv.v(L, pos), vt.data(), kNKV * kDH * sizeof(float));

            const std::size_t rep = kNH / kNKV;
            const std::size_t n = pos + 1;
            for (std::size_t hh = 0; hh < kNH; ++hh) {
                const std::size_t kvi = hh / rep;
                const float* kb = kv.k(L, 0);   // el anillo: ventana [0,n) contigua
                const float* vb = kv.v(L, 0);
                double mx = -1e30;
                for (std::size_t s = 0; s < n; ++s) {
                    double d = 0.0;
                    const float* ks = kb + s * (kNKV * kDH) + kvi * kDH;
                    for (std::size_t i = 0; i < kDH; ++i)
                        d += double(q[hh * kDH + i]) * double(ks[i]);
                    d *= double(kInvSqrtD);
                    att[hh * kMaxTok + s] = float(d);
                    mx = std::max(mx, d);
                }
                double sum = 0.0;
                for (std::size_t s = 0; s < n; ++s) {
                    const double e2 = std::exp(double(att[hh * kMaxTok + s]) - mx);
                    att[hh * kMaxTok + s] = float(e2);
                    sum += e2;
                }
                for (std::size_t i = 0; i < kDH; ++i) {
                    double acc = 0.0;
                    for (std::size_t s = 0; s < n; ++s)
                        acc += double(att[hh * kMaxTok + s]) / sum *
                               double(vb[s * (kNKV * kDH) + kvi * kDH + i]);
                    o[hh * kDH + i] = float(acc);
                }
            }

            gemv_mt(pool, c.lin.at(p + "self_attn.o_proj.weight"), o.data(), tmp.data());
            for (std::size_t i = 0; i < kH; ++i) x[i] += tmp[i];

            rmsnorm(x.data(), c.aux_of(p + "post_attention_layernorm.weight"), h.data(), kH);
            gemv_mt(pool, c.lin.at(p + "mlp.gate_proj.weight"), h.data(), gate.data());
            gemv_mt(pool, c.lin.at(p + "mlp.up_proj.weight"), h.data(), up.data());
            for (std::size_t i = 0; i < 4864; ++i)
                gate[i] = (gate[i] / (1.0f + std::exp(-gate[i]))) * up[i];
            gemv_mt(pool, c.lin.at(p + "mlp.down_proj.weight"), gate.data(), tmp.data());
            for (std::size_t i = 0; i < kH; ++i) x[i] += tmp[i];
        }

        rmsnorm(x.data(), c.aux_of("model.norm.weight"), h.data(), kH);
        std::vector<float> logits(c.V);
        gemv_mt(pool, c.lin.at("model.embed_tokens.weight"), h.data(), logits.data());
        std::size_t best = 0;
        for (std::size_t i = 1; i < c.V; ++i) if (logits[i] > logits[best]) best = i;
        return int(best);
    };

    // --- prefill -------------------------------------------------------------
    const double t0 = now_ms();
    for (std::size_t i = 0; i + 1 < prompt.size(); ++i) (void)step(prompt[i], i);
    const double t1 = now_ms();
    int next = step(prompt.back(), prompt.size() - 1);
    const double t2 = now_ms();

    std::vector<int> out;
    out.push_back(next);

    // --- decode --------------------------------------------------------------
    const double t3 = now_ms();
    for (int i = 1; i < ngen; ++i) {
        out.push_back(step(out.back(), prompt.size() + i - 1));
    }
    const double t4 = now_ms();

    std::printf("prefill  : %.1f ms (%zu tokens)\n", t1 - t0, prompt.size() - 1);
    std::printf("1er token: %.1f ms\n", t2 - t1);
    std::printf("decode   : %.1f ms para %d tokens -> %.2f ms/token -> %.1f tok/s\n\n",
                t4 - t3, ngen - 1, (t4 - t3) / double(ngen - 1),
                1000.0 * double(ngen - 1) / (t4 - t3));

    std::printf("IDS ");
    for (int v : out) std::printf("%d ", v);
    std::printf("\n\n");
    return 0;
}
