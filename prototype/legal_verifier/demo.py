#!/usr/bin/env python3
"""demo.py — Prototipo mínimo: generación + Juez Verificador simulado.

Flujo:
  1. Lee un contrato y separa sus cláusulas.
  2. Pide a Ollama una respuesta con afirmaciones y citas literales.
  3. El "Juez" comprueba que cada cita exista y sea fiel a la cláusula.
  4. Emite tarjetas de evidencia y un log auditable.

No es asesoría legal. El Juez es una simulación determinista de SG Core:
verifica coincidencia textual/lexical, no ontología formal todavía.

Uso:
  python3 prototype/legal_verifier/demo.py \
    --contract prototype/legal_verifier/sample_contract.txt \
    --question "¿Qué pasa si la arrendataria termina el contrato antes del plazo?"
"""
import argparse
import difflib
import hashlib
import json
import os
import re
import sys
import time
import unicodedata
import urllib.request

CLAUSE_RE = re.compile(r"^\s*CL[ÁA]USULA\s+(\d+)\s*[—\-:]\s*(.*)$", re.IGNORECASE)


def norm(s: str) -> str:
    s = unicodedata.normalize("NFD", s)
    s = "".join(c for c in s if unicodedata.category(c) != "Mn")
    s = s.lower()
    s = re.sub(r"[^a-z0-9ñ]+", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def load_clauses(path: str):
    clauses = []
    cur = None
    for line in open(path, encoding="utf-8"):
        m = CLAUSE_RE.match(line)
        if m:
            cur = {"id": f"C{int(m.group(1))}", "title": m.group(2).strip(), "text": ""}
            clauses.append(cur)
        elif cur is not None:
            cur["text"] += line
    for c in clauses:
        c["text"] = c["text"].strip()
        c["norm"] = norm(c["text"])
    return clauses


def ollama_chat(endpoint: str, model: str, contract: str, question: str):
    clauses = "\n\n".join(
        f"[{c['id']} — {c['title']}]\n{c['text']}" for c in load_clauses(contract))
    system = (
        "Eres un asistente legal conservador. Respondes SOLO con un objeto JSON válido, "
        "sin markdown. Cada afirmación debe citar una cláusula y copiar un fragmento "
        "LITERAL de esa cláusula. El campo claim debe describir la afirmación concreta, "
        "nunca la palabra \"afirmación\" ni un placeholder. Incluye una claim por cada "
        "condición relevante (aviso, indemnización, cálculo). Si no hay evidencia, usa "
        "clause_id null y quote vacío. No inventes hechos ni cláusulas."
    )
    user = (
        "Esquema exacto:\n"
        '{"answer":"respuesta breve","claims":[{"claim":"afirmación",'
        '"clause_id":"C6","quote":"fragmento literal de la cláusula"}]}\n\n'
        f"CONTRATO:\n{clauses}\n\n"
        f"PREGUNTA: {question}\n"
    )
    body = json.dumps({
        "model": model,
        "messages": [{"role": "system", "content": system},
                     {"role": "user", "content": user}],
        "stream": False,
        "format": "json",
        "options": {"temperature": 0, "num_ctx": 4096, "num_predict": 512,
                    "num_thread": 4, "num_gpu": 0},
    }).encode()
    req = urllib.request.Request(endpoint.rstrip("/") + "/api/chat", data=body,
                                 headers={"Content-Type": "application/json"})
    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=300) as r:
        resp = json.load(r)
    dt = time.perf_counter() - t0
    content = resp.get("message", {}).get("content", "")
    return content, resp, dt, user


def parse_json(content: str):
    s = content.strip()
    # Los modelos chicos a veces omiten la llave externa de cierre.
    for extra in range(4):
        cand = s + "}" * extra
        try:
            return json.loads(cand)
        except Exception:
            pass
        i = cand.find("{")
        if i >= 0:
            try:
                obj, _ = json.JSONDecoder().raw_decode(cand[i:])
                return obj
            except Exception:
                pass
    return None


STOP = set("""de la que el en y a los se del las un por con no una su para es al lo como mas
pero sus le ya o este si porque esta entre cuando muy sin sobre tambien me hasta hay
donde quien desde todo nos durante todos uno les ni contra otros ese eso ante ellos e
esto mi antes algunos que unos yo otro otras otra tanto esa estos mucho quienes nada
muchos cual poco ella estar estas algunas algo nosotros mis tus ellas nosotras os mio
mia mios mias tuyo tuya tuyos tuyas suyo suya suyos suyas nuestro nuestra nuestros
nuestras vuestro vuestra vuestros vuestras esos esas estoy estas esta estamos estais
estan""".split())


def terms(s: str):
    # Stemming pobre pero suficiente para el prototipo: prefijo de 6 letras.
    return {w[:6] for w in norm(s).split() if len(w) > 2 and w not in STOP}


def judge(clauses, claims, question, answer):
    by_id = {c["id"]: c for c in clauses}
    qa = terms(question) | terms(answer)
    out = []
    for raw in claims if isinstance(claims, list) else []:
        claim = str(raw.get("claim", "")).strip()
        cid = raw.get("clause_id")
        quote = str(raw.get("quote", "")).strip()
        nq = norm(quote)
        cited = by_id.get(cid)
        sim_cited = 0.0
        if cited and nq:
            if nq in cited["norm"]:
                sim_cited = 1.0
            else:
                sim_cited = difflib.SequenceMatcher(None, nq, cited["norm"]).ratio()
        best_id, best_sim = None, 0.0
        if nq:
            for c in clauses:
                if nq in c["norm"]:
                    best_id, best_sim = c["id"], 1.0
                    break
                s = difflib.SequenceMatcher(None, nq, c["norm"]).ratio()
                if s > best_sim:
                    best_id, best_sim = c["id"], s
        relevant = bool(terms(claim) & qa)
        if not relevant:
            status = "NO_RELEVANTE"
        elif not quote or len(nq) < 20 or not cited:
            status = "SIN_EVIDENCIA"
        elif sim_cited >= 0.90:
            status = "SOPORTADO"
        elif best_id and best_id != cid and best_sim >= 0.90:
            status = "CITA_ERRADA"
        elif sim_cited >= 0.65:
            status = "DÉBIL"
        else:
            status = "SIN_EVIDENCIA"
        out.append({"claim": claim, "clause": cid, "quote": quote,
                    "best_clause": best_id, "sim": round(max(sim_cited, best_sim), 3),
                    "status": status})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--contract", required=True)
    ap.add_argument("--question", required=True)
    ap.add_argument("--model", default="qwen2.5:3b")
    ap.add_argument("--endpoint", default="http://localhost:11434")
    ap.add_argument("--out", default=os.path.join(os.path.dirname(__file__), "out"))
    a = ap.parse_args()

    text = open(a.contract, encoding="utf-8").read()
    clauses = load_clauses(a.contract)
    print(f"Contrato : {a.contract} ({len(clauses)} cláusulas, {len(text)} chars)")
    print(f"Modelo   : {a.model}")
    print(f"Pregunta : {a.question}\n")

    content, resp, dt, prompt = ollama_chat(a.endpoint, a.model, a.contract, a.question)
    obj = parse_json(content)
    if not obj:
        print("El modelo no devolvió JSON válido. Respuesta cruda:\n", content[:1000])
        sys.exit(2)

    answer = str(obj.get("answer", "")).strip()
    claims = obj.get("claims", [])
    verdicts = judge(clauses, claims, a.question, answer)

    print("RESPUESTA PROPUESTA")
    print(answer or "(vacía)")
    print("\nEVIDENCIA VERIFICADA")
    print(f"{'#':<3} {'Estado':<15} {'Cláusula':<9} {'Sim':<6} Afirmación")
    for i, v in enumerate(verdicts, 1):
        print(f"{i:<3} {v['status']:<15} {str(v['clause']):<9} {v['sim']:<6} {v['claim'][:90]}")
        if v["quote"]:
            print(f"    cita: “{v['quote'][:160]}…”")
    ok = sum(1 for v in verdicts if v["status"] == "SOPORTADO")
    total = len(verdicts)
    print(f"\nRESUMEN: {ok}/{total} afirmaciones soportadas por cita literal.")
    if total == 0:
        print("No se extrajeron afirmaciones: no hay nada que verificar.")
    elif ok < total:
        print("El Juez NO valida la respuesta completa. Revisión humana requerida.")

    os.makedirs(a.out, exist_ok=True)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    audit = {
        "timestamp": stamp,
        "model": a.model,
        "contract_sha256": hashlib.sha256(text.encode()).hexdigest(),
        "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
        "question": a.question,
        "answer": answer,
        "verdicts": verdicts,
        "supported": ok,
        "total": total,
        "latency_s": round(dt, 3),
        "ollama": {k: resp.get(k) for k in
                   ("eval_count", "eval_duration", "prompt_eval_count", "prompt_eval_duration")},
    }
    path = os.path.join(a.out, f"audit-{stamp}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(audit, f, ensure_ascii=False, indent=2)
    print(f"\nLog auditable: {path}")


if __name__ == "__main__":
    main()
