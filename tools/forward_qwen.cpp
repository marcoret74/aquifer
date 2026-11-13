// =============================================================================
//  caudal · tools/forward_qwen.cpp
//  FORWARD DE QWEN2.5-0.5B EN C++ — validado capa por capa contra PyTorch.
// =============================================================================
//  Dos modos de GEMV, para separar dos preguntas que se confunden fácil:
//
//    --exact : dequantiza int4 a float y multiplica por x float.
//              Comparado contra el oráculo con pesos dequantizados, valida la
//              MATEMÁTICA (rmsnorm, rope, GQA, swiglu) con error ~1e-5.
//
//    (default): kernel int4 real (int8 x, LUT-free, sesgo 8·Σx).
//              Comparado contra --exact, mide el costo de la cuantización de x.
//
//  Sin JSON, sin safetensors: lee index.txt con `>>`.
//
//  Uso: ./forward_qwen <artefacto> <oracle_dir> [--exact]
// =============================================================================

#include "caudal/gemv_int4.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <fstream>
#include <map>
#include <string>
#include <vector>

namespace {

constexpr std::size_t kH = 896, kNLY = 24, kNH = 14, kNKV = 2, kDH = 64;
constexpr float kEps = 1e-6f, kTheta = 1000000.0f, kInvSqrtD = 1.0f / 8.0f;

struct Lin {
    std::size_t N{0}, K{0}, G{0};
    const std::uint8_t* packed{nullptr};
    const _Float16* scale{nullptr};
    std::size_t row_bytes{0};
};

struct Ctx {
    std::vector<std::uint8_t> wbin, sbin, auxbin;
    std::vector<std::uint8_t> embed_bf16;
    std::size_t V{0}, H{0};
    std::map<std::string, Lin> lin;
    std::map<std::string, std::pair<const float*, std::size_t>> aux;

    const float* aux_of(const std::string& n) const {
        auto it = aux.find(n);
        return it == aux.end() ? nullptr : it->second.first;
    }
};

// --- utilidades --------------------------------------------------------------
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
    float f;
    std::memcpy(&f, &x, 4);
    return f;
}

void rmsnorm(const float* x, const float* w, float* out, std::size_t n) {
    double ss = 0.0;
    for (std::size_t i = 0; i < n; ++i) ss += double(x[i]) * double(x[i]);
    const float r = 1.0f / std::sqrt(float(ss / double(n)) + kEps);
    for (std::size_t i = 0; i < n; ++i) out[i] = x[i] * r * w[i];
}

// --- GEMV exacto (dequantiza a float) ---------------------------------------
void gemv_deq(const Lin& L, const float* x, float* y) {
    for (std::size_t n = 0; n < L.N; ++n) {
        const std::uint8_t* row = L.packed + n * L.row_bytes;
        double acc = 0.0;
        for (std::size_t g = 0; g < L.G; ++g) {
            const double s = double(L.scale[n * L.G + g]);
            const std::size_t k0 = g * 64;
            double part = 0.0;
            for (std::size_t j = 0; j < 64; ++j)
                part += double(caudal::code_at(row, k0 + j)) * double(x[k0 + j]);
            acc += part * s;
        }
        y[n] = float(acc);
    }
}

// --- GEMV int4 (kernel real) -------------------------------------------------
void gemv_i4(const Lin& L, const float* x, float* y) {
    caudal::Shape s;
    s.N = L.N; s.K = L.K;
    s.group = 64;              // VALORES por grupo (L.G es el NUMERO de grupos)
    std::vector<std::int8_t> xq(L.K);
    float amax = 0.0f;
    for (std::size_t i = 0; i < L.K; ++i) amax = std::max(amax, std::fabs(x[i]));
    const float sx = (amax > 0.0f ? amax / 127.0f : 1.0f);
    for (std::size_t i = 0; i < L.K; ++i) {
        float q = std::round(x[i] / sx);
        q = std::clamp(q, -127.0f, 127.0f);
        xq[i] = static_cast<std::int8_t>(q);
    }
    // El kernel espera x PRE-PERMUTADO por kPerm (el packer hace lo mismo con
    // el artefacto). La suma por grupo es invariante a la permutacion.
    std::vector<std::int8_t> xqp(L.K);
    // La permutacion es por BLOQUE de 64 valores, NO por grupo de escala:
    // con grupo 32 hay el doble de grupos y el bucle desbordaba el vector.
    for (std::size_t bb = 0; bb < L.K / 64; ++bb)
        for (std::size_t i = 0; i < 64; ++i)
            xqp[bb * 64 + i] = xq[bb * 64 + caudal::kPerm[i]];

    std::vector<std::int32_t> xg(L.G);
    caudal::compute_x_gsum(s, xq.data(), xg.data());
    const std::vector<float> xsg(s.groups(), sx);   // una escala por grupo
    caudal::gemv_int4(s, L.packed, L.scale, xg.data(), xqp.data(), xsg.data(), y);
}

// --- RoPE --------------------------------------------------------------------
void rope(float* q, float* k, std::size_t nheads, std::size_t pos) {
    const std::size_t half = kDH / 2;
    for (std::size_t h = 0; h < nheads; ++h) {
        float* v = q + h * kDH;
        for (std::size_t i = 0; i < half; ++i) {
            const float freq = 1.0f / std::pow(kTheta, float(2 * i) / float(kDH));
            const float ang = float(pos) * freq;
            const float c = std::cos(ang), s = std::sin(ang);
            const float a = v[i], b = v[i + half];
            v[i] = a * c - b * s;
            v[i + half] = b * c + a * s;
        }
    }
    (void)k;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 3) {
        std::printf("uso: %s <artefacto> <oracle_dir> [--exact]\n", argv[0]);
        return 1;
    }
    const std::string A = argv[1], O = argv[2];
    const bool exact = (argc > 3 && std::strcmp(argv[3], "--exact") == 0);

    Ctx c;
    c.wbin = slurp(A + "/weights.bin");
    c.sbin = slurp(A + "/scales.bin");
    c.auxbin = slurp(A + "/aux.bin");
    c.embed_bf16 = slurp(A + "/embed.bf16");

    // --- index ---------------------------------------------------------------
    std::ifstream fi(A + "/index.txt");
    if (!fi) { std::printf("no abrio index.txt\n"); return 1; }
    std::string kind;
    std::string last_aux;
    std::size_t aux_cursor = 0;
    while (fi >> kind) {
        if (kind == "LINEAR") {
            std::string name;
            std::size_t N, K, po, pb, so, sb;
            fi >> name >> N >> K >> po >> pb >> so >> sb;
            Lin L;
            L.N = N; L.K = K; L.row_bytes = K / 2;
            L.G = sb / (2 * N);   // derivado, no K/64
            L.packed = c.wbin.data() + po;
            L.scale = reinterpret_cast<const _Float16*>(c.sbin.data() + so);
            c.lin[name] = L;
        } else if (kind == "AUX") {
            std::string name;
            std::size_t cnt, off;
            fi >> name >> cnt >> off;
            c.aux[name] = {reinterpret_cast<const float*>(c.auxbin.data() + off * 4), cnt};
            (void)last_aux; (void)aux_cursor;
        } else if (kind == "EMBED") {
            std::string name;
            fi >> name >> c.V >> c.H;
        }
    }

    // --- entrada desde el oráculo -------------------------------------------
    std::ifstream fm(O + "/meta.json");
    std::string js((std::istreambuf_iterator<char>(fm)), std::istreambuf_iterator<char>());
    auto grab = [&](const char* key) {
        const auto p = js.find(key);
        const auto b = js.find('[', p);
        const auto e = js.find(']', b);
        return js.substr(b + 1, e - b - 1);
    };
    std::vector<int> ids;
    {
        std::string arr = grab("\"ids\"");
        std::string cur;
        for (char ch : arr) {
            if (ch == ',' || ch == ' ' || ch == '\n') {
                if (!cur.empty()) { ids.push_back(std::atoi(cur.c_str())); cur.clear(); }
            } else cur += ch;
        }
        if (!cur.empty()) ids.push_back(std::atoi(cur.c_str()));
    }
    const std::size_t T = ids.size();
    std::printf("\ncaudal · forward_qwen   (T=%zu, modo=%s)\n\n", T,
                exact ? "exacto (dequant float)" : "int4 (kernel real)");

    // --- forward -------------------------------------------------------------
    std::vector<float> x(T * kH), h(kH), tmp(kH);
    std::vector<float> q(T * kNH * kDH), k(T * kNKV * kDH), v(T * kNKV * kDH);
    std::vector<float> att(kNH * T * T), o(T * kH);
    std::vector<float> gate(T * 4864), up(T * 4864), down(T * kH);
    std::vector<std::vector<float>> ours;

    for (std::size_t t = 0; t < T; ++t) {
        const std::uint16_t* e = reinterpret_cast<const std::uint16_t*>(c.embed_bf16.data());
        for (std::size_t i = 0; i < kH; ++i)
            x[t * kH + i] = bf16_to_f32(e[std::size_t(ids[t]) * kH + i]);
    }
    ours.push_back(x);

    timespec tp0{}, tp1{};
    clock_gettime(CLOCK_MONOTONIC, &tp0);
    for (std::size_t L = 0; L < kNLY; ++L) {
        const std::string p = "model.layers." + std::to_string(L) + ".";
        const float* w_in = c.aux_of(p + "input_layernorm.weight");
        const float* w_post = c.aux_of(p + "post_attention_layernorm.weight");

        for (std::size_t t = 0; t < T; ++t) {
            rmsnorm(x.data() + t * kH, w_in, h.data(), kH);

            float* qt = q.data() + t * kNH * kDH;
            float* kt = k.data() + t * kNKV * kDH;
            float* vt = v.data() + t * kNKV * kDH;
            auto gv = [&](const std::string& nm, float* y) {
                auto it = c.lin.find(p + nm);
                if (exact) gemv_deq(it->second, h.data(), y);
                else       gemv_i4(it->second, h.data(), y);
            };
            gv("self_attn.q_proj.weight", qt);
            gv("self_attn.k_proj.weight", kt);
            gv("self_attn.v_proj.weight", vt);

            // bias de q/k/v (Qwen2 los tiene; Llama no)
            const float* bq = c.aux_of(p + "self_attn.q_proj.bias");
            const float* bk = c.aux_of(p + "self_attn.k_proj.bias");
            const float* bv = c.aux_of(p + "self_attn.v_proj.bias");
            for (std::size_t i = 0; i < kNH * kDH; ++i) qt[i] += bq[i];
            for (std::size_t i = 0; i < kNKV * kDH; ++i) { kt[i] += bk[i]; vt[i] += bv[i]; }

            rope(qt, kt, kNH, t);
            rope(kt, kt, kNKV, t);
        }

        // atención causal + GQA
        const std::size_t rep = kNH / kNKV;
        for (std::size_t hh = 0; hh < kNH; ++hh) {
            const std::size_t kv = hh / rep;
            for (std::size_t t = 0; t < T; ++t) {
                double mx = -1e30;
                for (std::size_t s = 0; s <= t; ++s) {
                    double d = 0.0;
                    for (std::size_t i = 0; i < kDH; ++i)
                        d += double(q[(t * kNH + hh) * kDH + i]) *
                             double(k[(s * kNKV + kv) * kDH + i]);
                    d *= double(kInvSqrtD);
                    att[(hh * T + t) * T + s] = float(d);
                    mx = std::max(mx, d);
                }
                double sum = 0.0;
                for (std::size_t s = 0; s <= t; ++s) {
                    const double e = std::exp(double(att[(hh * T + t) * T + s]) - mx);
                    att[(hh * T + t) * T + s] = float(e);
                    sum += e;
                }
                for (std::size_t s = 0; s <= t; ++s)
                    att[(hh * T + t) * T + s] = float(double(att[(hh * T + t) * T + s]) / sum);
                for (std::size_t i = 0; i < kDH; ++i) {
                    double acc = 0.0;
                    for (std::size_t s = 0; s <= t; ++s)
                        acc += double(att[(hh * T + t) * T + s]) *
                               double(v[(s * kNKV + kv) * kDH + i]);
                    o[(t * kNH + hh) * kDH + i] = float(acc);
                }
            }
        }

        for (std::size_t t = 0; t < T; ++t) {
            auto it = c.lin.find(p + "self_attn.o_proj.weight");
            float* ot = o.data() + t * kH;
            if (exact) gemv_deq(it->second, ot, tmp.data());
            else       gemv_i4(it->second, ot, tmp.data());
            for (std::size_t i = 0; i < kH; ++i) x[t * kH + i] += tmp[i];
        }

        // MLP
        for (std::size_t t = 0; t < T; ++t) {
            rmsnorm(x.data() + t * kH, w_post, h.data(), kH);
            float* gt = gate.data() + t * 4864;
            float* ut = up.data() + t * 4864;
            auto itg = c.lin.find(p + "mlp.gate_proj.weight");
            auto itu = c.lin.find(p + "mlp.up_proj.weight");
            if (exact) { gemv_deq(itg->second, h.data(), gt); gemv_deq(itu->second, h.data(), ut); }
            else       { gemv_i4(itg->second, h.data(), gt);  gemv_i4(itu->second, h.data(), ut); }
            for (std::size_t i = 0; i < 4864; ++i) {
                const float silu = gt[i] / (1.0f + std::exp(-gt[i]));
                gt[i] = silu * ut[i];
            }
            auto itd = c.lin.find(p + "mlp.down_proj.weight");
            float* dt = down.data() + t * kH;
            if (exact) gemv_deq(itd->second, gt, dt);
            else       gemv_i4(itd->second, gt, dt);
            for (std::size_t i = 0; i < kH; ++i) x[t * kH + i] += dt[i];
        }

        ours.push_back(x);
    }

    clock_gettime(CLOCK_MONOTONIC, &tp1);
    const double fwd_ms = (tp1.tv_sec - tp0.tv_sec) * 1e3 +
                          (tp1.tv_nsec - tp0.tv_nsec) * 1e-6;
    std::printf("forward: %.1f ms para T=%zu  ->  %.2f ms/token (sin KV cache)\n\n",
                fwd_ms, T, fwd_ms / double(T));

    // --- comparación capa por capa -------------------------------------------
    auto rel = [](const std::vector<float>& a, const std::vector<std::uint8_t>& raw) {
        const auto* b = reinterpret_cast<const float*>(raw.data());
        double num = 0.0, den = 0.0, mx = 0.0;
        for (std::size_t i = 0; i < a.size(); ++i) {
            const double d = double(a[i]) - double(b[i]);
            num += d * d;
            den += double(b[i]) * double(b[i]);
            mx = std::max(mx, std::fabs(d));
        }
        return std::pair<double, double>(std::sqrt(num / (den + 1e-30)), mx);
    };

    std::printf("%-8s %14s %14s\n", "etapa", "rel (L2)", "max|diff|");
    int bad = 0;
    for (std::size_t L = 0; L <= kNLY; ++L) {
        char fn[64];
        std::snprintf(fn, sizeof fn, "/hidden_%02zu.bin", L);
        auto raw = slurp(O + fn);
        if (raw.size() != ours[L].size() * 4) { std::printf("tamano raro en %s\n", fn); return 1; }
        auto [r, m] = rel(ours[L], raw);
        std::printf("%-8s %14.3e %14.3e%s\n", L == 0 ? "embed" : "capa", r, m,
                    (L > 0 && r > 0.05) ? "   <-- alto" : "");
        if (L > 0 && r > 0.05) ++bad;
    }

    // --- logits ---------------------------------------------------------------
    {
        std::vector<float> xn(kH);
        for (std::size_t t = T - 1; t < T; ++t) {   // sólo el último token
            const float* wf = c.aux_of("model.norm.weight");
            rmsnorm(x.data() + t * kH, wf, xn.data(), kH);
            std::vector<float> lg(c.V);
            auto it = c.lin.find("model.embed_tokens.weight");
            if (exact) gemv_deq(it->second, xn.data(), lg.data());
            else       gemv_i4(it->second, xn.data(), lg.data());
            {
                std::size_t best = 0;
                for (std::size_t i = 1; i < c.V; ++i) if (lg[i] > lg[best]) best = i;
                std::printf("\nargmax (ultimo token) = %zu\n", best);
            }
        }
    }
    std::printf("\ncapas con error alto: %d\n\n", bad);
    return 0;
}
