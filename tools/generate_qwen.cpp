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
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <fstream>
#include <functional>
#include <map>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

#include <immintrin.h>
#include <sys/mman.h>
#include <unistd.h>

namespace {

constexpr unsigned kMaxM = 4;          // secuencias por pasada

struct Cfg {
    std::size_t H{0}, NLY{0}, NH{0}, NKV{0}, DH{0}, INTER{0}, V{0};
    float eps{1e-6f}, theta{1e6f};
    float inv_sqrt_d{1.0f};
};

struct Lin {
    std::size_t N{0}, K{0}, G{0}, stride{0};
    const std::uint8_t* packed{nullptr};
    const _Float16* scale{nullptr};
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

// --- barrera de spin ---------------------------------------------------------
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
                for (;;) { go_.wait(); if (stop_) return; job_(i); fin_.wait(); }
            });
    }
    ~Pool() { stop_ = true; go_.wait(); for (auto& t : th_) t.join(); }
    Pool(const Pool&) = delete;
    Pool& operator=(const Pool&) = delete;
    unsigned size() const { return T_; }
    void run(std::function<void(unsigned)> f) {
        job_ = std::move(f); go_.wait(); job_(0); fin_.wait();
    }
private:
    unsigned T_;
    SpinBarrier go_, fin_;
    std::vector<std::thread> th_;
    std::function<void(unsigned)> job_;
    bool stop_ = false;
};

// --- anillos de KV, uno por secuencia ---------------------------------------
struct KvRings {
    std::uint8_t* base{nullptr};
    std::size_t cap_tok{0}, per_seq{0}, stride{0};
    // per_seq = layers*2*cap_bytes

    void init(std::size_t tokens, std::size_t layers, std::size_t nkv,
              std::size_t dh, unsigned nseq) {
        cap_tok = tokens;
        stride = nkv * dh * sizeof(float);
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
    float* slot(unsigned seq, std::size_t layer, bool is_v, std::size_t t) const {
        const std::size_t cap_bytes = cap_tok * stride;
        std::uint8_t* b = base + std::size_t(seq) * per_seq
                        + layer * 2 * cap_bytes + (is_v ? cap_bytes : 0);
        return reinterpret_cast<float*>(b) + (t % cap_tok) * (stride / 4);
    }
};

struct Ctx {
    std::vector<std::uint8_t> wbin, auxbin, embed;
    std::size_t wbase{0};   // bytes de pesos; los offsets `so` del layout de dos
                            // arrays son relativos a scales.bin, no al buffer
                            // concatenado. Sin esto, las escalas se leian de
                            // los pesos -> fp16 basura -> logits NaN.
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

// --- GEMV por grupo, M vectores de entrada -----------------------------------
void gemv_group_M(Pool& pool, const std::vector<Job>& jobs, unsigned M,
                  const float* x, std::size_t K) {
    const Lin& L0 = *jobs[0].L;
    const std::size_t G = L0.G, grp = K / G;

    // La cuantizacion de x es M*K elementos: en serie costaba ~10 ms a M=4
    // (era el 10 % del paso). Se reparte entre los hilos por rango de m.
    std::vector<std::int8_t> xq(M * K), xqp(M * K);
    std::vector<std::int32_t> xg(M * G);
    std::vector<float> xs(M);
    pool.run([&](unsigned tid) {
        const unsigned T0 = pool.size();
        const unsigned a = unsigned(std::size_t(tid) * M / T0);
        const unsigned b = unsigned(std::size_t(tid + 1) * M / T0);
        for (unsigned m = a; m < b; ++m) {
            const float* xm = x + std::size_t(m) * K;
            float amax = 0.0f;
            for (std::size_t i = 0; i < K; ++i) amax = std::max(amax, std::fabs(xm[i]));
            const float sx = (amax > 0.0f ? amax / 127.0f : 1.0f);
            xs[m] = sx;
            std::int8_t* q = xq.data() + std::size_t(m) * K;
            for (std::size_t i = 0; i < K; ++i)
                q[i] = static_cast<std::int8_t>(std::clamp(std::round(xm[i] / sx), -127.0f, 127.0f));
            std::int8_t* qp = xqp.data() + std::size_t(m) * K;
            for (std::size_t bb = 0; bb < K / 64; ++bb)
                for (std::size_t i = 0; i < 64; ++i)
                    qp[bb * 64 + i] = q[bb * 64 + caudal::kPerm[i]];
            caudal::Shape s0; s0.N = 0; s0.K = K; s0.group = grp;
            caudal::compute_x_gsum(s0, q, xg.data() + std::size_t(m) * G);
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
                switch (M) {
                    case 1: caudal::gemv_int4_g32_M<1>(s, pk, sc, xg.data(), xqp.data(), xs.data(), yy + n0, j.L->N); break;
                    case 2: caudal::gemv_int4_g32_M<2>(s, pk, sc, xg.data(), xqp.data(), xs.data(), yy + n0, j.L->N); break;
                    case 3: caudal::gemv_int4_g32_M<3>(s, pk, sc, xg.data(), xqp.data(), xs.data(), yy + n0, j.L->N); break;
                    default: caudal::gemv_int4_g32_M<4>(s, pk, sc, xg.data(), xqp.data(), xs.data(), yy + n0, j.L->N); break;
                }
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

// offsets para que y[m*N + n] funcione con el kernel, que escribe y[m*N+n]
// (el kernel usa s.N como stride de M, por eso se le pasa el bloque completo).

}  // namespace

int main(int argc, char** argv) {
    if (argc < 2) {
        std::printf("uso: %s <artefacto> --n <N> [--threads T] [--ctx C] "
                    "--prompt id,id,.. [--prompt id,id,..]\n", argv[0]);
        return 1;
    }
    const std::string A = argv[1];
    int ngen = 32, ctx = 512;
    unsigned reqT = 0;
    std::vector<std::vector<int>> prompts;
    for (int i = 2; i < argc; ++i) {
        const std::string a = argv[i];
        auto next = [&]() -> std::string { return (i + 1 < argc) ? argv[++i] : std::string(); };
        if (a == "--n") ngen = std::atoi(next().c_str());
        else if (a == "--threads") reqT = std::atoi(next().c_str());
        else if (a == "--ctx") ctx = std::atoi(next().c_str());
        else if (a == "--prompt") {
            std::vector<int> v; std::stringstream ss(next()); std::string t;
            while (std::getline(ss, t, ',')) if (!t.empty()) v.push_back(std::atoi(t.c_str()));
            if (!v.empty()) prompts.push_back(v);
        }
    }
    if (prompts.empty()) { std::printf("falta --prompt\n"); return 1; }
    if (prompts.size() > kMaxM) {
        std::printf("maximo %u secuencias, pediste %zu\n", kMaxM, prompts.size());
        return 1;
    }
    // longitud comun de prompt (se trunca al minimo)
    std::size_t plen = prompts[0].size();
    for (auto& p : prompts) plen = std::min(plen, p.size());

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
    if (std::ifstream t(A + "/w.bin"); t.good()) c.wbin = slurp(A + "/w.bin");
    else { auto a1 = slurp(A + "/weights.bin"); auto a2 = slurp(A + "/scales.bin");
           c.wbase = a1.size();
           c.wbin = std::move(a1); c.wbin.insert(c.wbin.end(), a2.begin(), a2.end()); }
    c.auxbin = slurp(A + "/aux.bin");
    c.embed = slurp(A + "/embed.bf16");

    {
        std::ifstream fi(A + "/index.txt");
        std::string kind;
        while (fi >> kind) {
            if (kind == "LINEAR") {
                std::string rest; std::getline(fi, rest);
                std::istringstream is(rest);
                std::string name; is >> name;
                std::vector<std::size_t> v; std::size_t x;
                while (is >> x) v.push_back(x);
                if (v.size() < 2) continue;
                Lin L; L.N = v[0]; L.K = v[1]; L.row_bytes = L.K / 2;
                if (v.size() == 5) {                 // N K off stride G
                    L.packed = c.wbin.data() + v[2];
                    L.stride = v[3]; L.G = v[4];
                } else if (v.size() == 6) {          // N K po pb so sb
                    L.packed = c.wbin.data() + v[2];
                    L.scale = reinterpret_cast<const _Float16*>(c.wbin.data() + c.wbase + v[4]);
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

    const unsigned T = reqT ? reqT : std::max(1u, std::thread::hardware_concurrency());
    Pool pool(T);
    const unsigned NS = unsigned(prompts.size());
    KvRings kv;
    kv.init(std::size_t(ctx), cfg.NLY, cfg.NKV, cfg.DH, NS);

    std::printf("\ncaudal · generate_qwen (batching)\n");
    std::printf("modelo   : H=%zu capas=%zu heads=%zu/%zu d=%zu inter=%zu\n",
                cfg.H, cfg.NLY, cfg.NH, cfg.NKV, cfg.DH, cfg.INTER);
    std::printf("hilos    : %u\n", T);
    std::printf("secuencias: %u   prompt=%zu tokens   ctx=%d\n", NS, plen, ctx);
    std::printf("KV total : %.1f MB (%.0f MB por secuencia)\n\n",
                NS * double(kv.per_seq) / (1024 * 1024),
                double(kv.per_seq) / (1024 * 1024));

    const unsigned M = std::min(NS, kMaxM);
    std::vector<float> x(M * cfg.H), h(M * cfg.H), tmp(M * cfg.H);
    std::vector<float> q(M * cfg.NH * cfg.DH), kt(M * cfg.NKV * cfg.DH), vt(M * cfg.NKV * cfg.DH);
    std::vector<float> att(cfg.NH * std::size_t(ctx)), o(M * cfg.H);
    std::vector<float> gate(M * cfg.INTER), up(M * cfg.INTER);
    std::vector<float> logits(M * cfg.V);

    // Un paso para M secuencias en posiciones propias.
    auto step = [&](const std::vector<unsigned>& seq,
                   const std::vector<int>& toks,
                   const std::vector<std::size_t>& pos) {
        const unsigned m = unsigned(toks.size());
        const std::uint16_t* e = reinterpret_cast<const std::uint16_t*>(c.embed.data());
        for (unsigned i = 0; i < m; ++i)
            for (std::size_t j = 0; j < cfg.H; ++j)
                x[i * cfg.H + j] = bf16_to_f32(e[std::size_t(toks[i]) * cfg.H + j]);

        for (std::size_t L = 0; L < cfg.NLY; ++L) {
            const std::string p = "model.layers." + std::to_string(L) + ".";
            for (unsigned i = 0; i < m; ++i)
                rmsnorm(x.data() + i * cfg.H, c.aux_of(p + "input_layernorm.weight"),
                        h.data() + i * cfg.H, cfg.H, cfg.eps);

            const Job jq{&c.lin.at(p + "self_attn.q_proj.weight"), q.data()};
            const Job jk{&c.lin.at(p + "self_attn.k_proj.weight"), kt.data()};
            const Job jv{&c.lin.at(p + "self_attn.v_proj.weight"), vt.data()};
              gemv_group_M(pool, {jq, jk, jv}, m, h.data(), cfg.H);


            for (unsigned i = 0; i < m; ++i) {
                add_bias(q.data() + i * cfg.NH * cfg.DH, c.aux_of(p + "self_attn.q_proj.bias"), cfg.NH * cfg.DH);
                add_bias(kt.data() + i * cfg.NKV * cfg.DH, c.aux_of(p + "self_attn.k_proj.bias"), cfg.NKV * cfg.DH);
                add_bias(vt.data() + i * cfg.NKV * cfg.DH, c.aux_of(p + "self_attn.v_proj.bias"), cfg.NKV * cfg.DH);
                for (std::size_t hh = 0; hh < cfg.NH; ++hh)
                    rope_one(q.data() + (i * cfg.NH + hh) * cfg.DH, pos[i], cfg);
                for (std::size_t hh = 0; hh < cfg.NKV; ++hh)
                    rope_one(kt.data() + (i * cfg.NKV + hh) * cfg.DH, pos[i], cfg);
                std::memcpy(kv.slot(seq[i], L, false, pos[i]), kt.data() + i * cfg.NKV * cfg.DH,
                            cfg.NKV * cfg.DH * sizeof(float));
                std::memcpy(kv.slot(seq[i], L, true, pos[i]), vt.data() + i * cfg.NKV * cfg.DH,
                            cfg.NKV * cfg.DH * sizeof(float));
            }

            const std::size_t rep = cfg.NH / cfg.NKV;
            for (unsigned i = 0; i < m; ++i) {
                const std::size_t n = pos[i] + 1;
                for (std::size_t hh = 0; hh < cfg.NH; ++hh) {
                    const std::size_t kvi = hh / rep;
                    const float* kb = kv.slot(seq[i], L, false, 0);
                    const float* vb = kv.slot(seq[i], L, true, 0);
                    const float* qh = q.data() + (i * cfg.NH + hh) * cfg.DH;
                    double mx = -1e30;
                    for (std::size_t s = 0; s < n; ++s) {
                        double d = 0.0;
                        const float* ks = kb + s * (cfg.NKV * cfg.DH) + kvi * cfg.DH;
                        for (std::size_t j = 0; j < cfg.DH; ++j) d += double(qh[j]) * double(ks[j]);
                        d *= double(cfg.inv_sqrt_d);
                        att[hh * std::size_t(ctx) + s] = float(d);
                        mx = std::max(mx, d);
                    }
                    double sum = 0.0;
                    for (std::size_t s = 0; s < n; ++s) {
                        const double ex = std::exp(double(att[hh * std::size_t(ctx) + s]) - mx);
                        att[hh * std::size_t(ctx) + s] = float(ex);
                        sum += ex;
                    }
                    for (std::size_t j = 0; j < cfg.DH; ++j) {
                        double acc = 0.0;
                        for (std::size_t s = 0; s < n; ++s)
                            acc += double(att[hh * std::size_t(ctx) + s]) / sum *
                                   double(vb[s * (cfg.NKV * cfg.DH) + kvi * cfg.DH + j]);
                        o[(i * cfg.NH + hh) * cfg.DH + j] = float(acc);
                    }
                }
            }

              const Job jo{&c.lin.at(p + "self_attn.o_proj.weight"), tmp.data()};
              gemv_group_M(pool, {jo}, m, o.data(), cfg.H);

            for (std::size_t j = 0; j < m * cfg.H; ++j) x[j] += tmp[j];

            for (unsigned i = 0; i < m; ++i)
                rmsnorm(x.data() + i * cfg.H, c.aux_of(p + "post_attention_layernorm.weight"),
                        h.data() + i * cfg.H, cfg.H, cfg.eps);
            { const Job jg{&c.lin.at(p + "mlp.gate_proj.weight"), gate.data()};
              const Job ju{&c.lin.at(p + "mlp.up_proj.weight"), up.data()};
              gemv_group_M(pool, {jg, ju}, m, h.data(), cfg.H); }

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

            for (std::size_t j = 0; j < m * cfg.H; ++j) x[j] += tmp[j];
        }

        for (unsigned i = 0; i < m; ++i)
            rmsnorm(x.data() + i * cfg.H, c.aux_of("model.norm.weight"),
                    h.data() + i * cfg.H, cfg.H, cfg.eps);
          const Job jh{&c.lin.at("model.embed_tokens.weight"), logits.data()};
          gemv_group_M(pool, {jh}, m, h.data(), cfg.H);


        std::vector<int> out(m);
        for (unsigned i = 0; i < m; ++i) {
            const float* lg = logits.data() + std::size_t(i) * cfg.V;
            std::size_t best = 0;
            for (std::size_t j = 1; j < cfg.V; ++j) if (lg[j] > lg[best]) best = j;
            out[i] = int(best);
        }
        return out;
    };

    // --- prefill (una secuencia a la vez, M=1) -------------------------------
    const double tp0 = now_ms();
    for (unsigned s = 0; s < NS; ++s) {
        for (std::size_t i = 0; i + 1 < plen; ++i) (void)step({s}, {prompts[s][i]}, {i});
    }
    const double tp1 = now_ms();

    // --- decode batcheado ----------------------------------------------------
    std::vector<unsigned> seqids(NS);
    for (unsigned s = 0; s < NS; ++s) seqids[s] = s;
    std::vector<int> last(NS);
    std::vector<std::size_t> pos(NS);
    for (unsigned s = 0; s < NS; ++s) { pos[s] = plen - 1; last[s] = prompts[s][plen - 1]; }

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
    std::printf("decode   : %.1f ms para %d pasos x %u secuencias\n", td1 - td0, steps, NS);
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
