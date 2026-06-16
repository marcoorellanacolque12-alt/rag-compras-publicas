"""
citas.py
-------------------------------------------------------------------
Chequeo de FIDELIDAD de citas (determinista, barato, SIN LLM).

Regla: un marcador [N] de la respuesta es valido solo si
    1 <= N <= n_fragmentos
(los fragmentos efectivamente recuperados que se mostraron al modelo).
Un [N] fuera de rango es una cita INVENTADA: se neutraliza para no
enganar al lector, sin borrar la frase que lo contiene.

Corre en cada respuesta de produccion: cero llamadas a la red.
-------------------------------------------------------------------
"""
import re

_RE_CITA = re.compile(r"\[(\d+)\]")
MARCA_INVALIDA = "[cita no verificada]"


def verificar_citas(respuesta, n_fragmentos):
    """Valida los marcadores [N] de `respuesta` contra `n_fragmentos` recuperados.

    Devuelve un dict:
      - respuesta_limpia: la respuesta con los [N] INVALIDOS neutralizados
        (reemplazados por una marca discreta; la frase se conserva).
      - citas_validas:   list[int] ordenada de N unicos en rango [1, n_fragmentos].
      - citas_invalidas: list[int] ordenada de N unicos fuera de rango (inventados).
      - ok:              True si no se hallo ninguna cita inventada.
    """
    texto = respuesta or ""
    validas, invalidas = set(), set()

    def _sustituir(m):
        n = int(m.group(1))
        if 1 <= n <= n_fragmentos:
            validas.add(n)
            return m.group(0)          # marcador valido: se conserva tal cual
        invalidas.add(n)
        return MARCA_INVALIDA          # cita inventada: se neutraliza (no se borra la frase)

    limpia = _RE_CITA.sub(_sustituir, texto)
    return {
        "respuesta_limpia": limpia,
        "citas_validas": sorted(validas),
        "citas_invalidas": sorted(invalidas),
        "ok": not invalidas,
    }
