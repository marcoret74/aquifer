# Prototipo: generación + Juez Verificador

Demostración mínima del flujo para una firma de abogados:

1. El contrato se separa en cláusulas.
2. Ollama genera una respuesta con afirmaciones y **citas literales**.
3. Un Juez simulado verifica cada cita contra la cláusula:
   - `SOPORTADO`: la cita aparece (o coincide ≥ 0.90) en la cláusula citada.
   - `DÉBIL`: se parece, pero no lo suficiente.
   - `CITA_ERRADA`: la cita está en otra cláusula.
   - `NO_RELEVANTE`: la afirmación no se conecta con la pregunta/respuesta.
   - `SIN_EVIDENCIA`: no hay cita válida.
4. Se emite un **log auditable** con hash del contrato, hash del prompt, modelo,
   veredictos y latencia.

El Juez es **una simulación determinista** de SG Core: compara texto y
relevancia lexical (prefijos de 6 letras), no ontología formal todavía. El objetivo es validar el flujo producto, no la
calidad legal final.

## Uso

```bash
# Requiere Ollama con qwen2.5:3b (o el modelo que elijas)
python3 prototype/legal_verifier/demo.py \
  --contract prototype/legal_verifier/sample_contract.txt \
  --question "¿Qué pasa si la arrendataria termina el contrato antes del plazo? ¿Cuánto debe pagar?"
```

Salida: tarjetas de evidencia por pantalla y `out/audit-*.json`.

## Límites

- Modelo pequeño (3B): sirve para el flujo, no para asesoría legal.
- El Juez exige cita literal; si el modelo parafrasea, marca `DÉBIL` o
  `SIN_EVIDENCIA`. Eso es deliberado: mejor falsos negativos que falsos
  positivos.
- No hay OCR, RAG, gestor documental, control de acceso ni cifrado: eso es
  producto, no prototipo.
- **No es asesoría legal.** Toda respuesta requiere revisión humana.
