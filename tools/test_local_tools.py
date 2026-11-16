#!/usr/bin/env python3
"""test_local_tools.py — pruebas del registro de herramientas locales.

Uso: python3 tools/test_local_tools.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import local_tools as lt  # noqa: E402

FAILED = []
TOTAL = 0


def check(name, cond, detail=""):
    global TOTAL
    TOTAL += 1
    if not cond:
        FAILED.append(name + ((" :: " + str(detail)) if detail else ""))


def expect_error(name, fn, fragment=""):
    try:
        fn()
    except lt.ToolError as err:
        check(name, fragment in str(err), str(err))
        return
    check(name, False, "no lanzó ToolError")


def main():
    names = lt.names()
    for tool in ("sumar", "restar", "multiplicar", "dividir", "ahora",
                 "fecha", "contar", "mayusculas", "minusculas", "eco"):
        check("registrada %s" % tool, tool in names, names)
    check("catalog", len(lt.catalog()) == len(names))
    check("catalog formato",
          all(t.get("type") == "function" and t["function"].get("name")
              for t in lt.catalog()))

    check("sumar", lt.execute("sumar", {"a": 2, "b": 3}) == 5)
    check("restar", lt.execute("restar", {"a": 10, "b": 4}) == 6)
    check("multiplicar", lt.execute("multiplicar", {"a": 3, "b": 4}) == 12)
    check("dividir", lt.execute("dividir", {"a": 9, "b": 3}) == 3)
    check("contar", lt.execute("contar", {"texto": "hola"}) == 4)
    check("mayusculas", lt.execute("mayusculas", {"texto": "Hola"}) == "HOLA")
    check("minusculas", lt.execute("minusculas", {"texto": "HOLA"}) == "hola")
    check("eco", lt.execute("eco", {"texto": "x"}) == "x")

    ahora = lt.execute("ahora", {})
    fecha = lt.execute("fecha", {})
    check("ahora ISO", isinstance(ahora, str) and "T" in ahora, ahora)
    check("fecha ISO", isinstance(fecha, str) and len(fecha) == 10, fecha)
    check("ahora tz", "T" in lt.execute("ahora", {"tz": "UTC"}))

    expect_error("dividir por cero",
                 lambda: lt.execute("dividir", {"a": 1, "b": 0}), "cero")
    expect_error("desconocida", lambda: lt.execute("rm", {}), "desconocida")
    expect_error("falta argumento",
                 lambda: lt.execute("sumar", {"a": 1}), "b")
    expect_error("tipo incorrecto",
                 lambda: lt.execute("sumar", {"a": "dos", "b": 3}), "se esperaba")
    expect_error("texto incorrecto",
                 lambda: lt.execute("contar", {"texto": 3}), "texto")
    expect_error("tz inválida",
                 lambda: lt.execute("ahora", {"tz": "Marte/Olympus"}), "zona")

    if FAILED:
        print("FALLARON %d de %d pruebas:" % (len(FAILED), TOTAL))
        for name in FAILED:
            print("  - " + name)
        return 1
    print("local_tools: %d pruebas OK" % TOTAL)
    return 0


if __name__ == "__main__":
    sys.exit(main())
