# Guion de demo — 5 minutos

Objetivo: mostrar que el motor genera en local y que la orquestación ya tiene
herramientas, evidencia y métricas. No es una presentación comercial: es una
demo técnica acotada.

**Antes de empezar:** correr `./docs/pilot/demo.sh` una vez para verificar que
arranca en la máquina. Tener la terminal con fuente grande.

---

## 0:00–0:40 · Encuadre

> “Esto corre en esta máquina, sin nube. El motor está en C++ con kernels AVX2;
> la orquestación está en Python. El modelo no ve Internet: si necesita un dato,
> lo pide a una herramienta que el operador autorizó.”

Mostrar:

- `build/generate_aquifer` existe.
- El repo no tiene dependencias de llama.cpp/gguf.

---

## 0:40–1:10 · Arranque

```bash
./docs/pilot/demo.sh
```

Mientras arranca:

> “El servidor levanta el motor, carga el system prompt en KV y expone una API
> local compatible con OpenAI. El mock de clima escucha solo en 127.0.0.1.”

Señalar:

- `http://127.0.0.1:8120/health`
- `http://127.0.0.1:8120/v1/http-tools`

---

## 1:10–2:10 · Tool calling

La demo pregunta:

> “¿Qué clima hace en Rancagua? Usa consultar_clima.”

Qué mirar en la salida:

1. El modelo emite `<tool_call>{"name":"consultar_clima",...}</tool_call>`.
2. El servidor valida los argumentos contra el esquema.
3. Llama al mock HTTP **allowlisted** y recibe `{"ciudad":"Rancagua","temp_c":24,...}`.
4. Vuelve a generar y responde con el dato.
5. `tool_trace` muestra `kind: "http"`, `status: 200`, duración.

Decir:

> “La herramienta no la inventa el cliente: la define el operador en un JSON.
> El cliente solo manda argumentos. La URL tiene que estar en la allowlist.”

---

## 2:10–2:50 · Seguridad y límites

Mostrar `docs/pilot/http-tools.json` y explicar:

- Solo `http/https`, sin credenciales en la URL.
- Sin redirecciones.
- Hostnames que resuelven a IPs privadas/link-local se rechazan.
- Headers y auth los pone el operador.
- Presupuesto: `--max-tool-rounds` y `--tool-budget-ms`.
- No hay shell ni ejecución de código arbitrario: registro cerrado.

> “Si mañana querés conectar un servicio real, se agrega al JSON con su host en
> la allowlist y su token en un archivo 600.”

---

## 2:50–3:30 · Evidencia

Señalar:

- `x_aquifer.receipt`: HMAC-SHA256 del request (id, estado, tokens, latencias,
  hashes de prompt/conversación/respuesta, herramientas).
- Cabecera `x-aquifer-receipt`.
- Trace log con cadena append-only.

Mostrar la verificación:

```bash
python3 tools/bundle.py verify-traces \
  --file /tmp/aquifer-demo-traces.jsonl \
  --hmac-key <clave>
```

> “Cada request se puede verificar. Si alguien cambia una línea, la cadena y la
> firma fallan. El bundle de una sesión se firma con la misma lógica, HMAC o
> OpenSSL.”

---

## 3:30–4:10 · Operación

Mostrar:

```bash
curl -s http://127.0.0.1:8120/metrics | python3 -m json.tool
curl -s http://127.0.0.1:8120/metrics/prometheus
```

> “Hay métricas de latencia, TTFT, tokens, errores, esquema, herramientas y
> compactaciones. `docs/monitoring/` trae Prometheus, alertas y dashboard.”

Mencionar el watchdog:

> “Si el motor deja de responder, el servidor corta con STOP, drena y, si hace
> falta, reinicia el proceso. El request no queda colgado para siempre.”

---

## 4:10–4:40 · Límites honestos

Decir sin vueltas:

- El modelo es chico (0.5B en la demo); la calidad es acotada.
- Tool calling se apoya en un formato de marcador y prompts; no es un modelo
  entrenado específicamente para function calling.
- El servidor no ejecuta código arbitrario: solo herramientas del registro.
- La memoria de largo plazo y el Juez viven en el orquestador, no en el motor.
- El KV se compacta y se resume, pero no es magia: hay pérdida de contexto.

---

## 4:40–5:00 · Cierre y próximos pasos

> “Lo que está probado es el motor y la plomería: API local, herramientas
> allowlisted, recibos, bundles, métricas y watchdog. El siguiente paso es un
> piloto acotado con un caso real y un esquema de evidencia definido por
> ustedes.”

Preguntas para cerrar:

1. ¿Qué servicio o dato necesitan consultar primero?
2. ¿Quién define el esquema de evidencia y quién revisa?
3. ¿Qué métrica haría que el piloto se considere exitoso?
4. ¿Dónde correría: notebook, servidor on-prem, VM?
5. ¿Qué fecha y qué volumen de casos?

---

## Frases para evitar

- “Reemplaza al abogado.”
- “No se equivoca.”
- “Funciona con cualquier modelo sin ajustes.”
- “Es un clon de llama.cpp.”
- “Los datos se entrenan con tu información.”

## Frases seguras

- “Es un motor local; la decisión sigue siendo humana.”
- “Cada afirmación puede llevar cita y estado de verificación.”
- “El bundle permite reconstruir qué pasó y verificar que no se alteró.”
- “Es una evaluación técnica acotada, no asesoría legal.”
