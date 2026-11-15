#!/usr/bin/env python3
# =============================================================================
#  caudal · tools/oracle_qwen.py
#  Oráculo de referencia en PyTorch para Qwen2.5-0.5B. SIN transformers.
# =============================================================================
#  Un oráculo no validado no sirve de nada: si el oráculo y el C++ comparten
#  un malentendido del spec (RoPE, GQA, convención de RMSNorm), coinciden y
#  ambos están mal. Por eso este script se valida generando TEXTO COHERENTE
#  antes de servir de referencia para nada.
#
#  Qué hace:
#    1. Tokenizer byte-level BPE desde tokenizer.json (sin `tokenizers`).
#    2. Forward desde el spec de Qwen2 (float32, pesos bf16 del safetensors).
#    3. Genera N tokens greedy y los decodifica -> si el texto tiene sentido,
#       el spec está bien.
#    4. Vuelca los estados ocultos de LAS 24 CAPAS a binario para que el C++
#       se compare capa por capa.
#
#  Uso:
#    python3 tools/oracle_qwen.py --model DIR --out DIR_ORACLE \
#            --prompt "¿Quién eres?" --gen 24
# =============================================================================

import argparse
import json
import os
import struct
import sys

try:
    import numpy as np
    import torch
    import torch.nn.functional as F
except ImportError:
    np = None
    torch = None
    F = None

import regex


# -----------------------------------------------------------------------------
#  safetensors
# -----------------------------------------------------------------------------
class SafeTensors:
    def __init__(self, path):
        self.f = open(path, 'rb')
        (hlen,) = struct.unpack('<Q', self.f.read(8))
        self.header = json.loads(self.f.read(hlen).decode())
        self.header.pop('__metadata__', None)
        self.base = 8 + hlen
        self._cache = {}

    def names(self):
        return self.header.keys()

    def get(self, name, dtype=None):
        if dtype is None and torch is not None:
            dtype = torch.bfloat16
        if name in self._cache:
            return self._cache[name]
        e = self.header[name]
        a, b = e['data_offsets']
        self.f.seek(self.base + a)
        buf = self.f.read(b - a)
        if e['dtype'] == 'BF16':
            u = np.frombuffer(buf, dtype=np.uint16)
            t = torch.from_numpy((u.astype(np.uint32) << 16).view(np.float32).copy())
        elif e['dtype'] == 'F32':
            t = torch.from_numpy(np.frombuffer(buf, dtype=np.float32).copy())
        elif e['dtype'] == 'F16':
            t = torch.from_numpy(np.frombuffer(buf, dtype=np.float16).copy()).float()
        else:
            raise SystemExit('dtype ' + e['dtype'])
        t = t.reshape(e['shape']).to(dtype)
        if dtype != torch.bfloat16:
            self._cache[name] = t
        return t


# -----------------------------------------------------------------------------
#  Tokenizer byte-level BPE (GPT-2 style), sin `tokenizers`
# -----------------------------------------------------------------------------
def _bytes_to_unicode():
    bs = list(range(33, 127)) + list(range(161, 173)) + list(range(174, 256))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return dict(zip(bs, [chr(c) for c in cs]))


# Patron de GPT-2: lo usan los ByteLevel con use_regex=true (Llama/SmolLM2).
GPT2_PAT = (r"'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+"
            r"|\s+(?!\S)|\s+")


def _pretok_spec(t):
    """Devuelve (modo, patron_compilado, separar_digitos).

    Tres familias de tokenizer vistas hasta ahora:
      * Qwen2   -> Sequence[Split(Regex)]: usar su regex, byte-level.
      * SmolLM2 -> Sequence[Digits, ByteLevel]: regex de GPT-2 + digitos sueltos.
      * Llama-2/TinyLlama (SentencePiece) -> pre_tokenizer null y normalizador
        que antepone U+2581 y cambia espacios por U+2581: NO hay byte-level;
        el BPE va sobre el texto normalizado y lo que falta en el vocabulario
        cae a tokens <0xXX> (byte fallback).
    """
    pt = t.get('pre_tokenizer')
    norm = json.dumps(t.get('normalizer') or {})
    if pt is None and '\\u2581' in norm:
        return 'sp', None, False
    subs = pt.get('pretokenizers', []) if pt else []
    # El Split manda: Qwen trae su propio regex y su ByteLevel NO re-corta
    # (use_regex=false). Usar el patron de GPT-2 ahi fusionaba mal los \n\n.
    split = next((x for x in subs if x.get('type') == 'Split'), None)
    dig = any(x.get('type') == 'Digits' for x in subs)
    if split is not None:
        pat = split['pattern']['Regex']
    else:
        pat = GPT2_PAT
    return 'regex', regex.compile(pat), dig


class Tokenizer:
    def __init__(self, path):
        t = json.load(open(path))
        self.vocab = dict(t['model']['vocab'])
        self.rank = {}
        for i, m in enumerate(t['model']['merges']):
            a, b = m.split(' ')
            self.rank[(a, b)] = i
        self.added = {a['content']: a['id'] for a in t.get('added_tokens', [])}
        self.mode, self.pat, self.digits_split = _pretok_spec(t)
        # post_processor: TemplateProcessing dice que el texto lleva <s> al
        # principio (TinyLlama) o nada (Qwen). Sin esto los ids no coinciden
        # con los de `tokenizers` aunque el BPE sea correcto.
        self.post = []
        pp = t.get('post_processor') or {}
        if pp.get('type') == 'TemplateProcessing':
            for el in pp.get('single', []):
                if 'SpecialToken' in el:
                    self.post.append(('S', el['SpecialToken']['id']))
                elif 'Sequence' in el:
                    self.post.append(('Q', None))
        self.b2u = _bytes_to_unicode()
        self.u2b = {v: k for k, v in self.b2u.items()}
        # id -> str, para decodificar
        self.id2s = {v: k for k, v in self.vocab.items()}
        for s, i in self.added.items():
            self.id2s[i] = s

    def _merge(self, parts):
        while len(parts) > 1:
            best, best_r = None, None
            for i in range(len(parts) - 1):
                r = self.rank.get((parts[i], parts[i + 1]))
                if r is not None and (best_r is None or r < best_r):
                    best_r, best = r, (parts[i], parts[i + 1])
            if best is None:
                break
            out, i = [], 0
            while i < len(parts):
                if i < len(parts) - 1 and (parts[i], parts[i + 1]) == best:
                    out.append(parts[i] + parts[i + 1])
                    i += 2
                else:
                    out.append(parts[i])
                    i += 1
            parts = out
        return parts

    def _bpe(self, word):
        if self.mode == 'sp':
            # SentencePiece: simbolos = caracteres; lo que no esta en el
            # vocabulario se parte en sus bytes UTF-8 (<0xXX>).
            parts = []
            for ch in word:
                if ch in self.vocab:
                    parts.append(ch)
                else:
                    parts += ['<0x%02X>' % b for b in ch.encode('utf-8')]
            return self._merge(parts)
        parts = list(word)
        while len(parts) > 1:
            best, best_r = None, None
            for i in range(len(parts) - 1):
                r = self.rank.get((parts[i], parts[i + 1]))
                if r is not None and (best_r is None or r < best_r):
                    best_r, best = r, (parts[i], parts[i + 1])
            if best is None:
                break
            out, i = [], 0
            while i < len(parts):
                if i < len(parts) - 1 and (parts[i], parts[i + 1]) == best:
                    out.append(parts[i] + parts[i + 1])
                    i += 2
                else:
                    out.append(parts[i])
                    i += 1
            parts = out
        return parts

    def _encode_plain(self, text):
        ids = []
        if self.mode == 'sp':
            s = '\u2581' + text.replace(' ', '\u2581')
            for piece in self._bpe(s):
                ids.append(self.vocab[piece])
            return ids
        for m in self.pat.finditer(text):
            chunk = m.group(0)
            # Digits(individual_digits=true): cada digito es su propio trozo.
            trozos = regex.findall(r'\d|\D+', chunk) if self.digits_split else [chunk]
            for trozo in trozos:
                s = ''.join(self.b2u[b] for b in trozo.encode('utf-8'))
                for piece in self._bpe(s):
                    ids.append(self.vocab[piece])
        return ids

    def encode(self, text, allow_special=True):
        if not allow_special or not self.added:
            return self._encode_plain(text)
        # partir por tokens especiales, preservándolos
        pat = '(' + '|'.join(regex.escape(k) for k in
                             sorted(self.added, key=len, reverse=True)) + ')'
        ids = []
        for part in regex.split(pat, text):
            if not part:
                continue
            if part in self.added:
                ids.append(self.added[part])
            else:
                ids.extend(self._encode_plain(part))
        if self.post:
            out = []
            for kind, val in self.post:
                if kind == 'S':
                    out.append(self.added.get(val, self.vocab.get(val)))
                else:
                    out.extend(ids)
            return out
        return ids

    def decode(self, ids, strip_leading_space=None):
        if self.mode == 'sp':
            trozos, pend = [], bytearray()
            for i in ids:
                s = self.id2s.get(int(i))
                if s is None:
                    continue
                if len(s) == 6 and s.startswith('<0x') and s.endswith('>'):
                    try:
                        pend.append(int(s[3:5], 16))
                        continue
                    except ValueError:
                        pass
                if pend:
                    trozos.append(pend.decode('utf-8', 'replace'))
                    pend = bytearray()
                trozos.append(s)
            if pend:
                trozos.append(pend.decode('utf-8', 'replace'))
            txt = ''.join(trozos).replace('\u2581', ' ')
            if strip_leading_space is True or (strip_leading_space is None and len(ids) > 1):
                return txt[1:] if txt.startswith(' ') else txt
            return txt
        raw = b''
        for i in ids:
            s = self.id2s.get(int(i))
            if s is None:
                continue
            if s in self.added:
                raw += s.encode('utf-8')
            else:
                raw += bytes(self.u2b.get(c, 63) for c in s)
        return raw.decode('utf-8', errors='replace')


# -----------------------------------------------------------------------------
#  Forward de Qwen2
# -----------------------------------------------------------------------------
class Qwen:
    def __init__(self, model_dir, cfg):
        self.st = SafeTensors(os.path.join(model_dir, 'model.safetensors'))
        self.cfg = cfg
        self.H = cfg['hidden_size']
        self.nL = cfg['num_hidden_layers']
        self.nH = cfg['num_attention_heads']
        self.nKV = cfg['num_key_value_heads']
        self.d = self.H // self.nH
        self.eps = cfg['rms_norm_eps']
        self.theta = cfg['rope_theta']
        inv = 1.0 / (self.theta ** (torch.arange(0, self.d, 2, dtype=torch.float32) / self.d))
        self.inv_freq = inv
        self._wc = {}

    def w(self, name):
        # lowmem=True (default): SIN cache. En un forward completo cada peso se
        # lee una sola vez, asi que cachear 3 GB no ahorra I/O y desborda la RAM
        # de 11 GB. El cache solo sirve para GENERACION token a token.
        if getattr(self, 'lowmem', True):
            return self.st.get('model.' + name).float()
        if name not in self._wc:
            self._wc[name] = self.st.get('model.' + name)
        return self._wc[name].float()

    def wq(self, name):
        """Peso de una capa LINEAL: cuantizado si --quant (igual que el C++)."""
        t = self.w(name)
        return self._qd(t) if getattr(self, 'quant', False) else t

    def _qd(self, t):
        """Cuantiza a int4/g64 y dequantiza: mismos pesos que el artefacto C++.
        Sirve para comparar la MATEMATICA sin el ruido de la cuantizacion."""
        if t.dim() != 2:
            return t
        N, K = t.shape
        qg = getattr(self, 'qgroup', 64)
        if K % qg:
            return t
        G = K // qg
        g = t.view(N, G, qg)
        # Clamp en dominio fp16: 1e-8 NO es representable (min subnormal ~6e-8),
        # y los grupos con amax==0 daban sw=0 -> division por cero -> nan.
        sw = (g.abs().amax(-1, keepdim=True) / 7.0).half()
        sw = torch.clamp(sw, min=torch.finfo(torch.float16).tiny).float()
        codes = (g / sw).round().clamp(-8, 7)
        return (codes * sw).view(N, K)

    def wb(self, name):
        """Bias opcional: Qwen2 lo tiene en q/k/v, no en o_proj."""
        full = 'model.' + name
        if full in self.st.header:
            return self.st.get(full).float()
        return None

    def rmsnorm(self, x, weight):
        v = x.float()
        v = v * torch.rsqrt(v.pow(2).mean(-1, keepdim=True) + self.eps)
        return v * weight.float()

    def rope(self, q, k, pos):
        # q: [T, nH*d], k: [T, nKV*d]  ->  rotados
        T = q.shape[0]
        freqs = torch.outer(torch.arange(pos, pos + T, dtype=torch.float32),
                            self.inv_freq)          # [T, d/2]
        emb = torch.cat([freqs, freqs], dim=-1)     # [T, d]
        cos, sin = emb.cos(), emb.sin()

        def rot(x):                                  # [T, nheads, d]
            return x * cos[:, None, :] + \
                   torch.cat([-x[..., self.d // 2:], x[..., :self.d // 2]], -1) * sin[:, None, :]

        q = q.view(T, self.nH, self.d)
        k = k.view(T, self.nKV, self.d)
        return rot(q), rot(k)

    def forward(self, ids, dump_hidden=None):
        T = len(ids)
        cfg = self.cfg
        x = self.w('embed_tokens.weight')[torch.tensor(ids)]     # [T, H]
        hidden_states = [x.clone()]

        for L in range(self.nL):
            p = f'layers.{L}.'
            h = self.rmsnorm(x, self.wq(p + 'input_layernorm.weight'))
            q = F.linear(h, self.wq(p + 'self_attn.q_proj.weight'),
                         self.wb(p + 'self_attn.q_proj.bias'))
            k = F.linear(h, self.wq(p + 'self_attn.k_proj.weight'),
                         self.wb(p + 'self_attn.k_proj.bias'))
            v = F.linear(h, self.wq(p + 'self_attn.v_proj.weight'),
                         self.wb(p + 'self_attn.v_proj.bias'))
            q, k = self.rope(q, k, 0)

            # GQA: repetir k/v nH/nKV veces
            rep = self.nH // self.nKV
            kk = k.repeat_interleave(rep, dim=1)     # [T, nH, d]
            vv = v.view(T, self.nKV, self.d).repeat_interleave(rep, dim=1)

            qq = q.transpose(0, 1)                   # [nH, T, d]
            kk = kk.transpose(0, 1)
            vv = vv.transpose(0, 1)
            att = (qq @ kk.transpose(-1, -2)) / (self.d ** 0.5)
            mask = torch.triu(torch.ones(T, T, dtype=torch.bool), diagonal=1)
            att = att.masked_fill(mask, float('-inf'))
            att = torch.softmax(att, dim=-1)
            o = (att @ vv).transpose(0, 1).reshape(T, self.H)
            x = x + F.linear(o, self.wq(p + 'self_attn.o_proj.weight'))

            h = self.rmsnorm(x, self.wq(p + 'post_attention_layernorm.weight'))
            g = F.linear(h, self.wq(p + 'mlp.gate_proj.weight'))
            u = F.linear(h, self.wq(p + 'mlp.up_proj.weight'))
            x = x + F.linear(F.silu(g) * u, self.wq(p + 'mlp.down_proj.weight'))

            hidden_states.append(x.clone())

        xn = self.rmsnorm(x, self.w('norm.weight'))
        logits = F.linear(xn, self.wq('embed_tokens.weight'))     # atado
        return xn, logits, hidden_states


# -----------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', required=True)
    ap.add_argument('--out', default=None)
    ap.add_argument('--prompt', default='¿Quién eres?')
    ap.add_argument('--gen', type=int, default=20)
    ap.add_argument('--dump', action='store_true', help='vuelca capas para el C++')
    ap.add_argument('--quant', action='store_true',
                    help='usa pesos int4-dequantizados (aisla la matematica)')
    args = ap.parse_args()

    cfg = json.load(open(os.path.join(args.model, 'config.json')))
    tok = Tokenizer(os.path.join(args.model, 'tokenizer.json'))
    m = Qwen(args.model, cfg)
    m.quant = args.quant
    if args.quant:
        print('modo: pesos int4-dequantizados (mismos que el artefacto C++)')

    # --- round-trip del tokenizer (self-check) -------------------------------
    probe = 'Hola mundo, ¿qué tal?'
    rt = tok.decode(tok.encode(probe, allow_special=False))
    print(f'tokenizer round-trip: {rt == probe}  ({probe!r} -> {len(tok.encode(probe, False))} ids)')

    # --- ChatML ---------------------------------------------------------------
    text = (f'<|im_start|>user\n{args.prompt}<|im_end|>\n'
            f'<|im_start|>assistant\n')
    ids = tok.encode(text)
    print(f'prompt: {args.prompt!r}')
    print(f'ids   : {ids}')

    generated = list(ids)
    with torch.no_grad():
        for _ in range(args.gen):
            xn, logits, _ = m.forward(generated)
            nxt = int(torch.argmax(logits[-1]))
            if nxt == cfg['eos_token_id']:
                break
            generated.append(nxt)

    out = tok.decode(generated[len(ids):])
    print(f'\n--- generado ({len(generated)-len(ids)} tokens) ---')
    print(out)
    print('--- fin ---\n')

    # --- volcado para el C++ ------------------------------------------------
    if args.dump:
        outd = args.out or 'build/oracle'
        os.makedirs(outd, exist_ok=True)
        with torch.no_grad():
            xn, logits, hs = m.forward(ids)
        for L, h in enumerate(hs):
            h.numpy().astype(np.float32).tofile(os.path.join(outd, f'hidden_{L:02d}.bin'))
        xn.numpy().astype(np.float32).tofile(os.path.join(outd, 'final_norm.bin'))
        logits.numpy().astype(np.float32).tofile(os.path.join(outd, 'logits.bin'))
        json.dump({'prompt': args.prompt, 'ids': ids, 'T': len(ids),
                   'H': m.H, 'layers': m.nL, 'vocab': logits.shape[1],
                   'argmax_last': int(torch.argmax(logits[-1])),
                   'top5_last': [[int(i), float(v)] for i, v in
                                 zip(torch.topk(logits[-1], 5).indices,
                                     torch.topk(logits[-1], 5).values)],
                   'first_gen_ids': generated[len(ids):]},
                  open(os.path.join(outd, 'meta.json'), 'w'), indent=1)
        print(f'oraculo volcado en {outd}  (T={len(ids)}, capas={m.nL})')


if __name__ == '__main__':
    sys.exit(main())
