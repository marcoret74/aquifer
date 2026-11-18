#!/usr/bin/env python3
"""gguf_vocab.py — extrae el tokenizer de un GGUF y escribe un tokenizer.json.

Permite que el tooling de Aquifer tokenice sin depender de HuggingFace:
el GGUF es la fuente de verdad (tokens, merges, especiales y chat template).

Uso:
    tools/gguf_vocab.py modelo.gguf -o tokenizer.json
    tools/gguf_vocab.py modelo.gguf --check      # valida ids conocidos
    tools/gguf_vocab.py modelo.gguf --summary    # sólo reporta

Formato de salida: tokenizer.json compatible con `tools/oracle_qwen.py`
(model.vocab, model.merges, added_tokens, pre_tokenizer, chat_template).
"""

from __future__ import annotations

import argparse
import json
import struct
import sys

# Regex de pre-tokenización de Qwen2 (la que trae su tokenizer.json).
QWEN2_REGEX = (
    r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}|"
    r" ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"
)

GPT2_REGEX = (
    r"'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"
)

_VT = {
    0: ("<B", 1), 1: ("<b", 1), 2: ("<H", 2), 3: ("<h", 2),
    4: ("<I", 4), 5: ("<i", 4), 6: ("<f", 4), 7: ("<?", 1),
    8: ("<Q", 8), 9: ("<q", 8), 10: ("<Q", 8), 11: ("<q", 8), 12: ("<d", 8),
}


class GGUF:
    """Lector mínimo de metadata GGUF (sólo lo necesario para el vocab)."""

    def __init__(self, path: str):
        self.f = open(path, "rb")
        magic = self.f.read(4)
        if magic != b"GGUF":
            raise ValueError("no es un GGUF")
        self.version = struct.unpack("<I", self.f.read(4))[0]
        self.n_tensors = struct.unpack("<Q", self.f.read(8))[0]
        self.n_kv = struct.unpack("<Q", self.f.read(8))[0]
        self.kv: dict[str, object] = {}
        for _ in range(self.n_kv):
            key = self._str()
            vtype = struct.unpack("<I", self.f.read(4))[0]
            self.kv[key] = self._value(vtype)

    def _str(self) -> str:
        n = struct.unpack("<Q", self.f.read(8))[0]
        return self.f.read(n).decode("utf-8", "replace")

    def _value(self, vtype: int):
        if vtype == 8:                       # STRING
            return self._str()
        if vtype == 9:                       # ARRAY
            etype = struct.unpack("<I", self.f.read(4))[0]
            n = struct.unpack("<Q", self.f.read(8))[0]
            return [self._value(etype) for _ in range(n)]
        if vtype in _VT:
            fmt, size = _VT[vtype]
            raw = self.f.read(size)
            if vtype == 7:                   # BOOL
                return raw != b"\x00"
            if fmt in ("<f",):
                return struct.unpack("<f", raw)[0]
            if fmt == "<d":
                return struct.unpack("<d", raw)[0]
            return int.from_bytes(raw, "little", signed=fmt in ("<b", "<h", "<i", "<q"))
        raise ValueError(f"tipo GGUF desconocido: {vtype}")


# Tipos de token GGUF
TOKEN_CONTROL, TOKEN_USER_DEFINED = 3, 4


def build_tokenizer_json(gguf: GGUF) -> dict:
    tokens: list[str] = gguf.kv["tokenizer.ggml.tokens"]        # type: ignore[assignment]
    merges: list[str] = gguf.kv.get("tokenizer.ggml.merges", [])  # type: ignore[assignment]
    types: list[int] = gguf.kv.get("tokenizer.ggml.token_type", [])  # type: ignore[assignment]

    vocab = {tok: i for i, tok in enumerate(tokens)}
    if len(vocab) != len(tokens):
        print("  aviso: hay tokens duplicados en el vocabulario", file=sys.stderr)

    added = []
    for i, tok in enumerate(tokens):
        ttype = types[i] if i < len(types) else 1
        if ttype in (TOKEN_CONTROL, TOKEN_USER_DEFINED) or (
            tok.startswith("<|") and tok.endswith("|>")
        ):
            added.append({
                "id": i, "content": tok, "single_word": False,
                "lstrip": False, "rstrip": False, "normalized": False, "special": True,
            })

    arch = str(gguf.kv.get("general.architecture", ""))
    regex = QWEN2_REGEX if arch.startswith("qwen") else GPT2_REGEX

    out = {
        "version": "1.0",
        "truncation": None,
        "padding": None,
        "added_tokens": added,
        "normalizer": None,
        "pre_tokenizer": {
            "type": "Sequence",
            "pretokenizers": [
                {"type": "Split", "pattern": {"Regex": regex},
                 "behavior": "Isolated", "invert": False},
                {"type": "ByteLevel", "add_prefix_space": False,
                 "use_regex": False, "trim_offsets": False},
            ],
        },
        "post_processor": None,
        "decoder": {"type": "ByteLevel", "add_prefix_space": False,
                    "use_regex": False, "trim_offsets": False},
        "model": {
            "type": "BPE",
            "dropout": None,
            "unk_token": None,
            "continuing_subword_prefix": None,
            "end_of_word_suffix": None,
            "fuse_unk": False,
            "byte_fallback": False,
            "vocab": vocab,
            "merges": merges,
        },
    }
    if "tokenizer.chat_template" in gguf.kv:
        out["chat_template"] = gguf.kv["tokenizer.chat_template"]
    return out


def summary(gguf: GGUF, tj: dict) -> None:
    print(f"  arquitectura : {gguf.kv.get('general.architecture')}")
    print(f"  tokens       : {len(tj['model']['vocab'])}")
    print(f"  merges       : {len(tj['model']['merges'])}")
    print(f"  especiales   : {len(tj['added_tokens'])}")
    print(f"  bos/eos      : {gguf.kv.get('tokenizer.ggml.bos_token_id')} / "
          f"{gguf.kv.get('tokenizer.ggml.eos_token_id')}")
    print(f"  chat template: {'sí' if 'chat_template' in tj else 'no'}")


EXPECTED = {
    "El cielo es": [6582, 11825, 20172, 1531],
    "Hello": [9707],
    "Hola": [68012],
    "<|im_start|>": [151644],
    "<|im_end|>": [151645],
}


def check(path: str) -> int:
    """Valida con el Tokenizer de oracle_qwen.py contra ids conocidos."""
    sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent))
    from oracle_qwen import Tokenizer  # noqa: E402
    tok = Tokenizer(path)
    fallos = 0
    for text, want in EXPECTED.items():
        got = tok.encode(text)
        ok = got == want
        fallos += 0 if ok else 1
        print(f"  {'OK ' if ok else 'MAL'} {text!r:20} -> {got}"
              f"{'' if ok else f'   (esperado {want})'}")
    return fallos


def main() -> int:
    ap = argparse.ArgumentParser(description="Extrae tokenizer.json de un GGUF")
    ap.add_argument("gguf")
    ap.add_argument("-o", "--out")
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--summary", action="store_true")
    a = ap.parse_args()

    g = GGUF(a.gguf)
    tj = build_tokenizer_json(g)
    if a.summary or not a.out:
        summary(g, tj)
    if a.out:
        with open(a.out, "w", encoding="utf-8") as f:
            json.dump(tj, f, ensure_ascii=False)
        print(f"  escrito: {a.out}")
    if a.check:
        if not a.out:
            print("  --check necesita -o", file=sys.stderr)
            return 2
        print("  verificando contra ids conocidos:")
        return 1 if check(a.out) else 0
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
