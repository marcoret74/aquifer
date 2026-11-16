#!/usr/bin/env python3
"""test_tools.py — pruebas de function calling (sin modelo).

Uso: python3 tools/test_tools.py
"""

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from serve import ApiError, Handler  # noqa: E402

FAILED = []
TOTAL = 0


def check(name, cond, detail=""):
    global TOTAL
    TOTAL += 1
    if not cond:
        FAILED.append(name + ((" :: " + str(detail)) if detail else ""))


def handler():
    return Handler.__new__(Handler)


TOOLS = [{
    "type": "function",
    "function": {
        "name": "sumar",
        "description": "Suma dos números.",
        "parameters": {
            "type": "object",
            "required": ["a", "b"],
            "properties": {"a": {"type": "number"}, "b": {"type": "number"}},
            "additionalProperties": False,
        },
    },
}]


def expect_api_error(name, fn, fragment=""):
    try:
        fn()
    except ApiError as err:
        check(name, err.status == 400 and fragment in err.message, err.message)
        return
    check(name, False, "no lanzó ApiError")


def test_specs():
    specs, choice = Handler._tools(handler(), {"tools": TOOLS})
    check("specs nombre", specs and specs[0]["name"] == "sumar", specs)
    check("choice auto", choice == "auto", choice)

    _, choice = Handler._tools(handler(), {"tools": TOOLS,
                                           "tool_choice": "none"})
    check("choice none", choice == "none", choice)

    _, choice = Handler._tools(
        handler(), {"tools": TOOLS,
                    "tool_choice": {"type": "function",
                                    "function": {"name": "sumar"}}})
    check("choice forzado", choice == "sumar", choice)

    _, choice = Handler._tools(handler(), {})
    check("sin tools", choice == "auto", choice)

    check("signature",
          Handler._tool_signature(specs[0]) == "sumar(a: number, b: number)",
          Handler._tool_signature(specs[0]))
    check("example args",
          Handler._example_args(specs[0]["parameters"]) == {"a": 1, "b": 1})

    system = Handler._tools_system(handler(), "Eres útil.", specs, "auto")
    check("system menciona firma", "sumar(a: number, b: number)" in system)
    check("system menciona formato", "<tool_call>" in system)

    expect_api_error("tool duplicada",
                     lambda: Handler._tools(handler(), {"tools": TOOLS + TOOLS}),
                     "duplicada")
    bad = json.loads(json.dumps(TOOLS))
    bad[0]["function"]["name"] = "a b"
    expect_api_error("nombre inválido",
                     lambda: Handler._tools(handler(), {"tools": bad}),
                     "nombre")
    expect_api_error("choice inválido",
                     lambda: Handler._tools(handler(), {"tools": TOOLS,
                                                        "tool_choice": "siempre"}),
                     "tool_choice")
    expect_api_error("choice inexistente",
                     lambda: Handler._tools(
                         handler(),
                         {"tools": TOOLS,
                          "tool_choice": {"type": "function",
                                          "function": {"name": "restar"}}}),
                     "inexistente")


def parse(text, choice="auto"):
    specs, _ = Handler._tools(handler(), {"tools": TOOLS})
    return Handler._parse_tool_call(handler(), text, specs, choice)


def test_parse():
    call, errors = parse('<tool_call>{"name":"sumar","arguments":'
                         '{"a":2,"b":3}}</tool_call>')
    check("llamada válida", not errors, errors)
    check("nombre", call and call["function"]["name"] == "sumar")
    check("argumentos", json.loads(call["function"]["arguments"]) ==
          {"a": 2, "b": 3})
    check("id", call["id"].startswith("call_"))

    call, errors = parse('<tool_call>{"name":"sumar","arguments":'
                         '{"a":2,"b":3}}')
    check("sin cierre pero JSON válido", not errors and call is not None, errors)

    call, errors = parse('<tool_call>```json\n{"name":"sumar",'
                         '"arguments":{"a":2,"b":3}}\n```</tool_call>')
    check("cerca markdown", not errors and call is not None, errors)

    call, errors = parse("Hola, ¿cómo estás?")
    check("sin llamada auto", call is None and not errors, errors)

    call, errors = parse("Hola", choice="sumar")
    check("sin llamada forzada", call is None and bool(errors), errors)

    call, errors = parse('<tool_call>{"name":"restar",'
                         '"arguments":{"a":2,"b":3}}</tool_call>')
    check("herramienta desconocida", call is None and
          any("desconocida" in e for e in errors), errors)

    call, errors = parse('<tool_call>{"name":"sumar",'
                         '"arguments":{"a":2}}</tool_call>')
    check("falta argumento", call is None and
          any("b" in e for e in errors), errors)

    call, errors = parse('<tool_call>{"name":"sumar",'
                         '"arguments":{"a":"dos","b":3}}</tool_call>')
    check("tipo incorrecto", call is None and
          any("number" in e for e in errors), errors)

    call, errors = parse('<tool_call>{"name":"sumar",'
                         '"arguments":[1,2]}</tool_call>')
    check("argumentos no objeto", call is None and
          any("objeto" in e for e in errors), errors)

    call, errors = parse('<tool_call>{"name":"sumar",'
                         '"arguments":{"a":2,"b":3}}</tool_call>', choice="sumar")
    check("coincide forzado", not errors and call is not None, errors)

    call, errors = parse('<tool_call>{"name":"sumar","arguments":'
                         '{a:10,b:5}}</tool_call>')
    check("claves sin comillas", not errors and call is not None, errors)
    check("claves sin comillas args",
          call is not None and json.loads(call["function"]["arguments"]) ==
          {"a": 10, "b": 5})

    call, errors = parse("<tool_call>{'name':'sumar','arguments':"
                         "{'a':1,'b':2}}</tool_call>")
    check("comillas simples", not errors and call is not None, errors)


def main():
    test_specs()
    test_parse()
    if FAILED:
        print("FALLARON %d de %d pruebas:" % (len(FAILED), TOTAL))
        for name in FAILED:
            print("  - " + name)
        return 1
    print("tools: %d pruebas OK" % TOTAL)
    return 0


if __name__ == "__main__":
    sys.exit(main())
