#!/usr/bin/env python3
"""structured.py — extracción y validación de JSON para salidas verificables.

Sin dependencias. El motor genera texto; este módulo decide si ese texto sirve
como salida estructurada (por ejemplo, para que un orquestador o un Juez
Verificador pueda auditar la respuesta). No contiene lógica de dominio.

Expone:
    extract_json(texto)         -> valor JSON o SchemaError
    validate(valor, esquema)    -> lista de errores (vacía = válido)
    validate_json(texto, esquema) -> (valor, errores)

Soporta un subconjunto de JSON Schema suficiente para sobres de evidencia:
    type (string o lista), required, properties, additionalProperties,
    items, minItems, maxItems, enum, const, minLength, maxLength, pattern,
    minimum, maximum, anyOf, oneOf
"""

import json
import re

MAX_ERRORS = 50


class SchemaError(ValueError):
    """El texto no contiene un valor JSON utilizable."""


def _balanced(text, start):
    """Devuelve el primer bloque {...} o [...] balanceado desde `start`."""
    pairs = {"{": "}", "[": "]"}
    stack = []
    in_str = False
    esc = False
    for i in range(start, len(text)):
        ch = text[i]
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch in pairs:
            stack.append(pairs[ch])
        elif ch in "}]":
            if not stack or stack.pop() != ch:
                return None
            if not stack:
                return text[start:i + 1]
    return None


def extract_json(text):
    """Extrae el primer valor JSON del texto.

    Acepta JSON puro, cercas markdown ```json ... ``` y texto alrededor.
    """
    if not isinstance(text, str):
        raise SchemaError("la salida no es texto")
    s = text.strip()
    if not s:
        raise SchemaError("la salida está vacía")

    if s.startswith("```"):
        lines = s.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]
        s = "\n".join(lines).strip()
        if not s:
            raise SchemaError("bloque markdown vacío")

    try:
        return json.loads(s)
    except ValueError:
        pass

    for i, ch in enumerate(s):
        if ch in "{[":
            block = _balanced(s, i)
            if block is None:
                continue
            try:
                return json.loads(block)
            except ValueError:
                continue
    raise SchemaError("no se encontró un valor JSON válido")


def _is_number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _type_ok(value, name):
    if name == "object":
        return isinstance(value, dict)
    if name == "array":
        return isinstance(value, list)
    if name == "string":
        return isinstance(value, str)
    if name == "integer":
        return isinstance(value, int) and not isinstance(value, bool)
    if name == "number":
        return _is_number(value)
    if name == "boolean":
        return isinstance(value, bool)
    if name == "null":
        return value is None
    return True


def _add(errors, path, message):
    if len(errors) < MAX_ERRORS:
        errors.append("%s: %s" % (path, message))


def _validate(value, schema, path, errors):
    if not isinstance(schema, dict):
        return

    if "anyOf" in schema:
        branch = []
        for sub in schema["anyOf"]:
            local = []
            _validate(value, sub, path, local)
            if not local:
                branch = []
                break
            branch.append(local[0])
        else:
            _add(errors, path, "no coincide con ninguna alternativa (anyOf)")
        return

    if "oneOf" in schema:
        matches = 0
        for sub in schema["oneOf"]:
            local = []
            _validate(value, sub, path, local)
            if not local:
                matches += 1
        if matches != 1:
            _add(errors, path, "se esperaba exactamente una alternativa (oneOf), hubo %d" % matches)
        return

    expected = schema.get("type")
    if expected is not None:
        options = expected if isinstance(expected, list) else [expected]
        if not any(_type_ok(value, t) for t in options):
            _add(errors, path, "se esperaba %s, llegó %s" %
                 ("/".join(options), _kind(value)))
            return

    if "const" in schema and value != schema["const"]:
        _add(errors, path, "se esperaba el valor %r" % (schema["const"],))
    if "enum" in schema and value not in schema["enum"]:
        _add(errors, path, "valor %r fuera del enum %r" % (value, schema["enum"]))

    if isinstance(value, dict):
        required = schema.get("required", [])
        for key in required:
            if key not in value:
                _add(errors, path, "falta la clave requerida '%s'" % key)
        props = schema.get("properties", {})
        for key, sub in props.items():
            if key in value:
                _validate(value[key], sub, "%s.%s" % (path, key), errors)
        extra = schema.get("additionalProperties", True)
        if extra is False:
            for key in value:
                if key not in props:
                    _add(errors, path, "clave no permitida '%s'" % key)
        elif isinstance(extra, dict):
            for key in value:
                if key not in props:
                    _validate(value[key], extra, "%s.%s" % (path, key), errors)

    elif isinstance(value, list):
        items = schema.get("items")
        if isinstance(items, dict):
            for i, item in enumerate(value):
                _validate(item, items, "%s[%d]" % (path, i), errors)
        if "minItems" in schema and len(value) < schema["minItems"]:
            _add(errors, path, "se esperaban al menos %d elementos" % schema["minItems"])
        if "maxItems" in schema and len(value) > schema["maxItems"]:
            _add(errors, path, "se esperaban como máximo %d elementos" % schema["maxItems"])

    elif isinstance(value, str):
        if "minLength" in schema and len(value) < schema["minLength"]:
            _add(errors, path, "texto más corto que minLength=%d" % schema["minLength"])
        if "maxLength" in schema and len(value) > schema["maxLength"]:
            _add(errors, path, "texto más largo que maxLength=%d" % schema["maxLength"])
        if "pattern" in schema:
            try:
                if re.search(schema["pattern"], value) is None:
                    _add(errors, path, "no cumple el patrón %r" % schema["pattern"])
            except re.error:
                pass

    elif _is_number(value):
        if "minimum" in schema and value < schema["minimum"]:
            _add(errors, path, "menor que minimum=%s" % schema["minimum"])
        if "maximum" in schema and value > schema["maximum"]:
            _add(errors, path, "mayor que maximum=%s" % schema["maximum"])


def _kind(value):
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "boolean"
    if isinstance(value, int):
        return "integer"
    if isinstance(value, float):
        return "number"
    if isinstance(value, str):
        return "string"
    if isinstance(value, list):
        return "array"
    if isinstance(value, dict):
        return "object"
    return type(value).__name__


def validate(value, schema):
    errors = []
    _validate(value, schema, "$", errors)
    return errors


def validate_json(text, schema):
    """Extrae y valida. Devuelve (valor, errores); valor es None si no hay JSON."""
    try:
        value = extract_json(text)
    except SchemaError as err:
        return None, [str(err)]
    return value, validate(value, schema)
