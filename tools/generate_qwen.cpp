// =============================================================================
//  caudal · tools/generate_qwen.cpp
//  GENERACIÓN con KV cache en ANILLO (memfd doble mapeo) + pool de hilos.
//  Dimensiones en RUNTIME: sirve para 0.5B, 1.5B y cualquier Qwen2.
// =============================================================================
//   1. KV CACHE en anillo: el memfd está mapeado DOS VECES consecutivas, así una
//      ventana [start,end] con end-start<=cap es SIEMPRE contigua. Ninguna rama
//      de wraparound en el bucle de atención. (Validado en test_doublemap.)
//   2. MULTIHILO con SpinBarrier y el hilo principal como worker 0: con T+1
//      hilos en T núcleos el barrier se estrangula (~1 s/token).
//   3. DECODE INCREMENTAL: prefill una vez, luego O(1) GEMVs por token.
//
//  Uso: ./generate_qwen <artefacto> <id> [<id>...] --n <N> [--threads <T>]
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

constexpr std::size_t kMaxTok = 4096;

// --- hiperparámetros, leídos de model.txt ------------------------------------
struct Cfg {
    std::size_t H{0}, NLY{0}, NH{0}, NKV{0}, DH{0}, INTER{0}, V{0};
    float eps{1e-6f}, theta{1e6f};
    float inv_sqrt_d{1.0f};
};

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
// std::barrier bloquea en futex: ~10 us por encuentro x 168 GEMV por token.
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

class Pool {
public:
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
        job_(0);
        fin_.wait();
    }
private:
    unsigned T_;
    SpinBarrier go_, fin_;
    std::vector<std::thread> th_;
    std::function<void(unsigned)> job_;
    bool stop_ = false;
};

// --- anillo de KV ------------------------------------------------------------
struct KvRing {
    std::uint8_t* base{nullptr};
    std::size_t cap_tok{0}, cap_bytes{0}, stride{0}, total{0};

    void init(std::size_t tokens, std::size_t layers, std::size_t nkv, std::size_t dh) {
        cap_tok = tokens;
        stride = nkv * dh * sizeof(float);
        cap_bytes = cap_tok * stride;
        total = layers * 2 * cap_bytes;
        const int fd = ::memfd_create("caudal_kv", MFD_CLOEXEC);
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
        return reinterpret_cast<float*>(base + layer * 2 * cap_bytes)
             + (t % cap_tok) * (stride / 4);
    }
    float* v(std::size_t layer, std::size_t t) const {
        return reinterpret_cast<float*>(base + layer * 2 * cap_bytes + cap_bytes)
             + (t % cap_tok) * (stride / 4);
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

void rmsnorm(const float* x, const float* w, float* out, std::size_t n, float eps) {
    double ss = 0.0;
    for (std::size_t i = 0; i < n; ++i) ss += double(x[i]) * double(x[i]);
    const float r = 1.0f / std::sqrt(float(ss / double(n)) + eps);
    for (std::size_t i = 0; i < n; ++i) out[i] = x[i] * r * w[i];
}

// --- GEMV FUSIONADO ----------------------------------------------------------
// Varios tensores que comparten el mismo vector de entrada. La cuantizacion y
// la permutacion de x se hacen UNA vez (antes eran una por tensor) y el reparto
// entre hilos se hace sobre el total de filas de todos los jobs juntos, asi
// las matrices chicas (k_proj: 256 filas) no dejan hilos ociosos.
struct Job { const Lin* L; float* y; };

void gemv_fused(Pool& pool, const std::vector<Job>& jobs, const float* x) {
    const std::size_t K = jobs[0].L->K, G = jobs[0].L->G, grp = K / G;
    std::vector<std::int8_t> xq(K), xqp(K);
    float amax = 0.0f;
    for (std::size_t i = 0; i < K; ++i) amax = std::max(amax, std::fabs(x[i]));
    const float sx = (amax > 0.0f ? amax / 127.0f : 1.0f);
    for (std::size_t i = 0; i < K; ++i) {
        const float q = std::clamp(std::round(x[i] / sx), -127.0f, 127.0f);
        xq[i] = static_cast<std::int8_t>(q);
    }
    for (std::size_t bb = 0; bb < K / 64; ++bb)
        for (std::size_t i = 0; i < 64; ++i)
            xqp[bb * 64 + i] = xq[bb * 64 + caudal::kPerm[i]];

    caudal::Shape s0; s0.N = 0; s0.K = K; s0.group = grp;
    std::vector<std::int32_t> xg(G);
    caudal::compute_x_gsum(s0, xq.data(), xg.data());

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
                const std::uint8_t* pk = j.L->packed + n0 * j.L->row_bytes;
                const _Float16* sc = j.L->scale + n0 * G;
                if (grp == 32)
                    caudal::gemv_int4_g32(s, pk, sc, xg.data(), xqp.data(), sx, j.y + n0);
                else
                    caudal::gemv_int4(s, pk, sc, xg.data(), xqp.data(), sx, j.y + n0);
            }
            off = hi;
        }
    });
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

}  // namespace

int main(int argc, char** argv) {
    if (argc < 3) {
        std::printf("uso: %s <artefacto> <id> [<id>...] --n <N> [--threads <T>]\n", argv[0]);
        return 1;
    }
    const std::string A = argv[1];
    std::vector<int> prompt;
    int ngen = 16;
    unsigned reqT = 0;
    for (int i = 2; i < argc; ++i) {
        if (!std::strcmp(argv[i], "--n") && i + 1 < argc) { ngen = std::atoi(argv[++i]); continue; }
        if (!std::strcmp(argv[i], "--threads") && i + 1 < argc) { reqT = std::atoi(argv[++i]); continue; }
        prompt.push_back(std::atoi(argv[i]));
    }
    if (prompt.empty()) { std::printf("falta el prompt\n"); return 1; }

    // --- hiperparámetros -----------------------------------------------------
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
            Lin L; L.N = N; L.K = K; L.row_bytes = K / 2;
            L.G = sb / (2 * N);                    // número de grupos, NO K/64
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
    kv.init(kMaxTok, cfg.NLY, cfg.NKV, cfg.DH);

    std::printf("\ncaudal · generate_qwen\n");
    std::printf("artefacto : %s  (%zu lineales)\n", A.c_str(), c.lin.size());
    std::printf("modelo    : H=%zu capas=%zu heads=%zu/%zu d=%zu inter=%zu vocab=%zu\n",
                cfg.H, cfg.NLY, cfg.NH, cfg.NKV, cfg.DH, cfg.INTER, cfg.V);
    std::printf("hilos     : %u\n", T);
    std::printf("KV ring   : %zu tok x %zu capas = %.1f MB (memfd doble mapeo)\n",
                kMaxTok, cfg.NLY, double(kv.total) / (1024.0 * 1024.0));
    std::printf("prompt    : %zu tokens\n\n", prompt.size());

    std::vector<float> x(cfg.H), h(cfg.H), tmp(cfg.H);
    std::vector<float> q(cfg.NH * cfg.DH), kt(cfg.NKV * cfg.DH), vt(cfg.NKV * cfg.DH);
    std::vector<float> att(cfg.NH * kMaxTok), o(cfg.H), gate(cfg.INTER), up(cfg.INTER);
    std::vector<float> logits(cfg.V);

    auto step = [&](int tok, std::size_t pos) {
        const std::uint16_t* e = reinterpret_cast<const std::uint16_t*>(c.embed.data());
        for (std::size_t i = 0; i < cfg.H; ++i)
            x[i] = bf16_to_f32(e[std::size_t(tok) * cfg.H + i]);

        for (std::size_t L = 0; L < cfg.NLY; ++L) {
            const std::string p = "model.layers." + std::to_string(L) + ".";
            rmsnorm(x.data(), c.aux_of(p + "input_layernorm.weight"), h.data(), cfg.H, cfg.eps);

            // Q/K/V comparten h: un solo GEMV fusionado, x cuantizado una vez.
            const Job jq{&c.lin.at(p + "self_attn.q_proj.weight"), q.data()};
            const Job jk{&c.lin.at(p + "self_attn.k_proj.weight"), kt.data()};
            const Job jv{&c.lin.at(p + "self_attn.v_proj.weight"), vt.data()};
            gemv_fused(pool, {jq, jk, jv}, h.data());
            add_bias(q.data(), c.aux_of(p + "self_attn.q_proj.bias"), cfg.NH * cfg.DH);
            add_bias(kt.data(), c.aux_of(p + "self_attn.k_proj.bias"), cfg.NKV * cfg.DH);
            add_bias(vt.data(), c.aux_of(p + "self_attn.v_proj.bias"), cfg.NKV * cfg.DH);

            for (std::size_t hh = 0; hh < cfg.NH; ++hh) rope_one(q.data() + hh * cfg.DH, pos, cfg);
            for (std::size_t hh = 0; hh < cfg.NKV; ++hh) rope_one(kt.data() + hh * cfg.DH, pos, cfg);

            std::memcpy(kv.k(L, pos), kt.data(), cfg.NKV * cfg.DH * sizeof(float));
            std::memcpy(kv.v(L, pos), vt.data(), cfg.NKV * cfg.DH * sizeof(float));

            const std::size_t rep = cfg.NH / cfg.NKV;
            const std::size_t n = pos + 1;
            for (std::size_t hh = 0; hh < cfg.NH; ++hh) {
                const std::size_t kvi = hh / rep;
                const float* kb = kv.k(L, 0);      // ventana [0,n) contigua
                const float* vb = kv.v(L, 0);
                double mx = -1e30;
                for (std::size_t s = 0; s < n; ++s) {
                    double dd = 0.0;
                    const float* ks = kb + s * (cfg.NKV * cfg.DH) + kvi * cfg.DH;
                    for (std::size_t i = 0; i < cfg.DH; ++i)
                        dd += double(q[hh * cfg.DH + i]) * double(ks[i]);
                    dd *= double(cfg.inv_sqrt_d);
                    att[hh * kMaxTok + s] = float(dd);
                    mx = std::max(mx, dd);
                }
                double sum = 0.0;
                for (std::size_t s = 0; s < n; ++s) {
                    const double ex = std::exp(double(att[hh * kMaxTok + s]) - mx);
                    att[hh * kMaxTok + s] = float(ex);
                    sum += ex;
                }
                for (std::size_t i = 0; i < cfg.DH; ++i) {
                    double acc = 0.0;
                    for (std::size_t s = 0; s < n; ++s)
                        acc += double(att[hh * kMaxTok + s]) / sum *
                               double(vb[s * (cfg.NKV * cfg.DH) + kvi * cfg.DH + i]);
                    o[hh * cfg.DH + i] = float(acc);
                }
            }

            { const Job jo{&c.lin.at(p + "self_attn.o_proj.weight"), tmp.data()};
              gemv_fused(pool, {jo}, o.data()); }
            for (std::size_t i = 0; i < cfg.H; ++i) x[i] += tmp[i];

            rmsnorm(x.data(), c.aux_of(p + "post_attention_layernorm.weight"), h.data(), cfg.H, cfg.eps);
            // Gate/Up comparten h2: fusionados. SiLU*Up paralelo (250k exp/token).
            { const Job jg{&c.lin.at(p + "mlp.gate_proj.weight"), gate.data()};
              const Job ju{&c.lin.at(p + "mlp.up_proj.weight"), up.data()};
              gemv_fused(pool, {jg, ju}, h.data()); }
            {
                const unsigned T2 = pool.size();
                pool.run([&](unsigned tid) {
                    const std::size_t a = std::size_t(tid) * cfg.INTER / T2;
                    const std::size_t b = std::size_t(tid + 1) * cfg.INTER / T2;
                    for (std::size_t i = a; i < b; ++i)
                        gate[i] = (gate[i] / (1.0f + std::exp(-gate[i]))) * up[i];
                });
            }
            { const Job jd{&c.lin.at(p + "mlp.down_proj.weight"), tmp.data()};
              gemv_fused(pool, {jd}, gate.data()); }
            for (std::size_t i = 0; i < cfg.H; ++i) x[i] += tmp[i];
        }

        rmsnorm(x.data(), c.aux_of("model.norm.weight"), h.data(), cfg.H, cfg.eps);
        { const Job jh{&c.lin.at("model.embed_tokens.weight"), logits.data()};
          gemv_fused(pool, {jh}, h.data()); }
        std::size_t best = 0;
        for (std::size_t i = 1; i < cfg.V; ++i) if (logits[i] > logits[best]) best = i;
        return int(best);
    };

    const double t0 = now_ms();
    for (std::size_t i = 0; i + 1 < prompt.size(); ++i) (void)step(prompt[i], i);
    const double t1 = now_ms();
    std::vector<int> out;
    out.push_back(step(prompt.back(), prompt.size() - 1));
    const double t2 = now_ms();

    const double t3 = now_ms();
    for (int i = 1; i < ngen; ++i) out.push_back(step(out.back(), prompt.size() + i - 1));
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
