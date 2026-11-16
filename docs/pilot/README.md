# Demo de piloto (5 minutos)

Demostración acotada de lo que Aquifer ya hace **en local**:

1. El motor genera en CPU, sin nube.
2. El modelo decide usar una herramienta.
3. El servidor ejecuta esa herramienta por HTTP contra un mock local allowlisted.
4. La respuesta vuelve con traza de herramienta, recibo HMAC y métricas.

No hay lógica legal ni del Juez en el repo: el mock es un servicio de clima
genérico. El Juez/orquestador viviría **fuera** del motor.

Checklist completo de despliegue:
[`../deploy/CHECKLIST-PILOTO.md`](../deploy/CHECKLIST-PILOTO.md).

## Requisitos

- `make all` hecho (binario en `build/generate_aquifer`).
- Modelo `qwen05` descargado (`python3 tools/pull.py qwen05` o el flujo habitual).
- `python3` y `curl`.
- Puertos libres: `8120` (servidor) y `9100` (mock).

## Ejecutar

```bash
./docs/pilot/demo.sh
```

Si el puerto 8120 está ocupado:

```bash
PORT=8121 ./docs/pilot/demo.sh
```

## Qué muestra

- `GET /v1/http-tools`: catálogo de herramientas HTTP (sin URL ni headers).
- Pregunta “¿Qué clima hace en Rancagua? Usa consultar_clima.”
- `tool_trace` con `kind: "http"`, argumentos, status y duración.
- Recibo HMAC en `x_aquifer.receipt` y en la cabecera `x-aquifer-receipt`.
- `/metrics`: requests, tools locales/HTTP, recibos, timeouts.
- Verificación de los recibos del trace con la clave de la demo.

## Seguridad de la demo

- El mock escucha solo en `127.0.0.1:9100`.
- La allowlist de herramientas HTTP es loopback por defecto.
- La config de herramientas la define el operador, no el cliente.
- La clave de recibos se genera en un temporal con permisos `600`.
- El servidor y el mock se matan al salir (`trap`).

## Archivos

| Archivo | Qué es |
| :--- | :--- |
| `mock_service.py` | Mock HTTP de clima (solo local) |
| `http-tools.json` | Config de la herramienta `consultar_clima` |
| `demo.sh` | Arranque, consulta, métricas y verificación |
| `GUION-5-MINUTOS.md` | Guion de la reunión |
