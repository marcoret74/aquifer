#!/usr/bin/env python3
"""test_structured.py — pruebas del validador de salidas JSON (sin dependencias).

Uso: python3 tools/test_structured.py
"""

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from structured import SchemaError, extract_json, validate, validate_json  # noqa: E402

FAILED = []
TOTAL = 0


def check(name, cond, detail=""):
    global TOTAL
    TOTAL += 1
    if not cond:
        FAILED.append(name + ((" :: " + detail) if detail else ""))


def expect_error(name, value, schema, needle=None):
    errors = validate(value, schema)
    check(name, bool(errors), "se esperaba error y no hubo")
    if needle and errors:
        check(name + " (mensaje)", any(needle in e for e in errors), str(errors))


def expect_ok(name, value, schema):
    errors = validate(value, schema)
    check(name, not errors, str(errors))


def test_extract():
    check("json puro", extract_json('{"a": 1}') == {"a": 1})
    check("array puro", extract_json('[1, 2, 3]') == [1, 2, 3])
    check("cerca markdown", extract_json('```json\n{"a": 2}\n```') == {"a": 2})
    check("texto alrededor",
          extract_json('Claro, aquí está:\n{"a": 3}\nEspero que sirva.') == {"a": 3})
    check("llaves dentro de strings",
          extract_json('{"texto": "un { raro } y un ] }"}') ==
          {"texto": "un { raro } y un ] }"})
    check("comillas escapadas",
          extract_json(r'{"texto": "dijo \"hola\" }"}') ==
          {"texto": 'dijo "hola" }'})
    check("anidado", extract_json('x {"a": {"b": [1, {"c": 2}]}} y') ==
          {"a": {"b": [1, {"c": 2}]}})
    try:
        extract_json("no hay json aquí")
        check("sin json lanza", False)
    except SchemaError:
        check("sin json lanza", True)
    try:
        extract_json("")
        check("vacío lanza", False)
    except SchemaError:
        check("vacío lanza", True)


def test_types():
    schema = {"type": "object", "required": ["a", "b"],
              "properties": {"a": {"type": "string"}, "b": {"type": "integer"}},
              "additionalProperties": False}
    expect_ok("objeto válido", {"a": "x", "b": 3}, schema)
    expect_error("falta requerida", {"a": "x"}, schema, "requerida")
    expect_error("tipo incorrecto", {"a": 1, "b": 3}, schema, "string")
    expect_error("clave extra", {"a": "x", "b": 3, "c": 1}, schema, "no permitida")
    expect_error("bool no es integer", {"a": "x", "b": True}, schema, "integer")
    expect_error("float no es integer", {"a": "x", "b": 3.5}, schema, "integer")

    expect_ok("number acepta int", 4, {"type": "number"})
    expect_ok("number acepta float", 4.5, {"type": "number"})
    expect_error("number rechaza bool", True, {"type": "number"}, "number")
    expect_ok("type lista", 5, {"type": ["integer", "string"]})
    expect_ok("type lista str", "x", {"type": ["integer", "string"]})
    expect_error("type lista falla", 1.5, {"type": ["integer", "string"]})


def test_constraints():
    expect_ok("enum ok", "supported", {"type": "string", "enum": ["supported", "unsupported"]})
    expect_error("enum mal", "quizas", {"type": "string", "enum": ["supported"]}, "enum")
    expect_ok("const ok", "v1", {"const": "v1"})
    expect_error("const mal", "v2", {"const": "v1"}, "v1")
    expect_ok("longitud ok", "abc", {"type": "string", "minLength": 2, "maxLength": 4})
    expect_error("minLength", "a", {"type": "string", "minLength": 2}, "minLength")
    expect_error("maxLength", "abcde", {"type": "string", "maxLength": 4}, "maxLength")
    expect_ok("pattern ok", "cita-12", {"type": "string", "pattern": r"^cita-\d+$"})
    expect_error("pattern mal", "cita-x", {"type": "string", "pattern": r"^cita-\d+$"}, "patrón")
    expect_ok("rango ok", 0.5, {"type": "number", "minimum": 0, "maximum": 1})
    expect_error("minimum", -1, {"type": "number", "minimum": 0}, "minimum")
    expect_error("maximum", 2, {"type": "number", "maximum": 1}, "maximum")


def test_arrays():
    schema = {"type": "array", "items": {"type": "object", "required": ["quote"],
                                         "properties": {"quote": {"type": "string"}},
                                         "additionalProperties": False},
              "minItems": 1, "maxItems": 2}
    expect_ok("array ok", [{"quote": "a"}], schema)
    expect_error("array vacío", [], schema, "al menos")
    expect_error("array largo", [{"quote": "a"}, {"quote": "b"}, {"quote": "c"}], schema, "como máximo")
    expect_error("item malo", [{"quote": 1}], schema, "$[0].quote")


def test_any_one_of():
    schema = {"anyOf": [{"type": "string"}, {"type": "integer"}]}
    expect_ok("anyOf string", "x", schema)
    expect_ok("anyOf int", 3, schema)
    expect_error("anyOf falla", 1.5, schema, "ninguna")
    one = {"oneOf": [{"type": "string"}, {"type": "integer"}]}
    expect_ok("oneOf entero", 3, one)
    expect_error("oneOf ambiguo", 3, {"oneOf": [{"type": "number"}, {"type": "integer"}]},
                 "exactamente")


def test_envelope():
    schema = {
        "type": "object",
        "required": ["respuesta", "afirmaciones"],
        "properties": {
            "respuesta": {"type": "string", "minLength": 1},
            "afirmaciones": {
                "type": "array",
                "items": {
                    "type": "object",
                    "required": ["texto", "estado", "citas"],
                    "properties": {
                        "texto": {"type": "string"},
                        "estado": {"enum": ["SOPORTADO", "NO_SOPORTADO", "PARCIAL"]},
                        "citas": {
                            "type": "array",
                            "items": {"type": "object",
                                      "required": ["cita"],
                                      "properties": {"cita": {"type": "string"}},
                                      "additionalProperties": False},
                        },
                    },
                    "additionalProperties": False,
                },
            },
        },
        "additionalProperties": False,
    }
    value, errors = validate_json(
        '```json\n{"respuesta":"ok","afirmaciones":[{"texto":"x",'
        '"estado":"SOPORTADO","citas":[{"cita":"cláusula 3"}]}]}\n```', schema)
    check("sobre completo", not errors, str(errors))
    check("sobre valor", value["afirmaciones"][0]["estado"] == "SOPORTADO")
    _, errors = validate_json(
        '{"respuesta":"ok","afirmaciones":[{"texto":"x",'
        '"estado":"SOPORTADO","citas":[{"cita":"c","extra":1}]}]}', schema)
    check("sobre cita extra", bool(errors), str(errors))


def test_json_stream():
    from serve import JsonStreamFilter

    raw = '```json\n{"a": [1, {"b": "x}y"}]}\n```\nEspero que sirva.'
    f = JsonStreamFilter()
    out = ""
    for ch in raw:
        out += f.feed(ch)
    check("stream contenido", out == '{"a": [1, {"b": "x}y"}]}', out)
    check("stream done", f.done)

    f2 = JsonStreamFilter()
    out2 = f2.feed("Claro: ") + f2.feed('{"a":1}') + f2.feed(" texto extra")
    check("stream narración", out2 == '{"a":1}', out2)
    check("stream corta extra", f2.done)

    f3 = JsonStreamFilter(max_prefix=5)
    out3 = f3.feed("abcdefg") + f3.feed("h")
    check("stream sin json", out3 == "abcdefg", out3)
    check("stream sin json no done", not f3.done)

    f4 = JsonStreamFilter()
    out4 = "".join(f4.feed(x) for x in ['{"t": "a \\" } b"}', " tail"])
    check("stream escape", out4 == '{"t": "a \\" } b"}', out4)
    check("stream escape done", f4.done)


def main():
    test_extract()
    test_types()
    test_constraints()
    test_arrays()
    test_any_one_of()
    test_envelope()
    test_json_stream()
    if FAILED:
        print("FALLARON %d de %d pruebas:" % (len(FAILED), TOTAL))
        for name in FAILED:
            print("  - " + name)
        return 1
    print("structured: %d pruebas OK" % TOTAL)
    return 0


if __name__ == "__main__":
    sys.exit(main())
