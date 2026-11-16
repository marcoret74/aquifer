#!/usr/bin/env python3
"""local_tools.py — registro mínimo de herramientas seguras para el servidor.

No ejecuta shell, no toca archivos y no hace red. Son funciones puras y
deterministas para demos y pilotos. El servidor solo las usa si arranca con
`--local-tools` y el cliente pide `tool_execution: "local"` o `"auto"`.

Para agregar una herramienta: definir la función, su esquema y registrarla en
`TOOLS`. Nada de código arbitrario por request.
"""

from datetime import datetime, timezone
from zoneinfo import ZoneInfo

from structured import validate


class ToolError(Exception):
    pass


def _schema(properties, required):
    return {
        "type": "object",
        "properties": properties,
        "required": required,
        "additionalProperties": False,
    }


def _num(args, key):
    value = args.get(key)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ToolError("%s debe ser un número" % key)
    return value


def _text(args, key="texto"):
    value = args.get(key)
    if not isinstance(value, str):
        raise ToolError("%s debe ser texto" % key)
    return value


def _zone(args):
    name = args.get("tz")
    if not name:
        return timezone.utc
    try:
        return ZoneInfo(str(name))
    except Exception:
        raise ToolError("zona horaria inválida: %s" % name)


def _sumar(args):
    return _num(args, "a") + _num(args, "b")


def _restar(args):
    return _num(args, "a") - _num(args, "b")


def _multiplicar(args):
    return _num(args, "a") * _num(args, "b")


def _dividir(args):
    a = _num(args, "a")
    b = _num(args, "b")
    if b == 0:
        raise ToolError("división por cero")
    return a / b


def _ahora(args):
    return datetime.now(_zone(args)).replace(microsecond=0).isoformat()


def _fecha(args):
    return datetime.now(_zone(args)).date().isoformat()


def _contar(args):
    return len(_text(args))


def _mayusculas(args):
    return _text(args).upper()


def _minusculas(args):
    return _text(args).lower()


def _eco(args):
    return _text(args)


TOOLS = {
    "sumar": {
        "description": "Suma dos números.",
        "parameters": _schema({"a": {"type": "number"}, "b": {"type": "number"}},
                              ["a", "b"]),
        "fn": _sumar,
    },
    "restar": {
        "description": "Resta b de a.",
        "parameters": _schema({"a": {"type": "number"}, "b": {"type": "number"}},
                              ["a", "b"]),
        "fn": _restar,
    },
    "multiplicar": {
        "description": "Multiplica dos números.",
        "parameters": _schema({"a": {"type": "number"}, "b": {"type": "number"}},
                              ["a", "b"]),
        "fn": _multiplicar,
    },
    "dividir": {
        "description": "Divide a entre b (b distinto de cero).",
        "parameters": _schema({"a": {"type": "number"}, "b": {"type": "number"}},
                              ["a", "b"]),
        "fn": _dividir,
    },
    "ahora": {
        "description": "Fecha y hora actual en ISO 8601.",
        "parameters": _schema({"tz": {"type": "string"}}, []),
        "fn": _ahora,
    },
    "fecha": {
        "description": "Fecha actual (YYYY-MM-DD).",
        "parameters": _schema({"tz": {"type": "string"}}, []),
        "fn": _fecha,
    },
    "contar": {
        "description": "Cuenta los caracteres de un texto.",
        "parameters": _schema({"texto": {"type": "string"}}, ["texto"]),
        "fn": _contar,
    },
    "mayusculas": {
        "description": "Devuelve el texto en mayúsculas.",
        "parameters": _schema({"texto": {"type": "string"}}, ["texto"]),
        "fn": _mayusculas,
    },
    "minusculas": {
        "description": "Devuelve el texto en minúsculas.",
        "parameters": _schema({"texto": {"type": "string"}}, ["texto"]),
        "fn": _minusculas,
    },
    "eco": {
        "description": "Devuelve el mismo texto (para pruebas).",
        "parameters": _schema({"texto": {"type": "string"}}, ["texto"]),
        "fn": _eco,
    },
}


def has(name):
    return name in TOOLS


def names():
    return sorted(TOOLS)


def catalog():
    return [{
        "type": "function",
        "function": {
            "name": name,
            "description": spec["description"],
            "parameters": spec["parameters"],
        },
    } for name, spec in sorted(TOOLS.items())]


def execute(name, args):
    """Valida y ejecuta. Solo nombres del registro; nunca código arbitrario."""
    spec = TOOLS.get(name)
    if spec is None:
        raise ToolError("herramienta local desconocida: %s" % name)
    if not isinstance(args, dict):
        raise ToolError("arguments debe ser un objeto")
    errors = validate(args, spec["parameters"])
    if errors:
        raise ToolError("; ".join(errors[:5]))
    return spec["fn"](args)
