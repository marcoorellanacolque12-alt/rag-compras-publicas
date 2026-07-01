# -*- coding: utf-8 -*-
"""
tests/test_citas_agrupadas.py
-------------------------------------------------------------------
Blinda el arreglo de citas AGRUPADAS (marcadores "[a, b, c]"):

  (a) RAIZ  — verificar_citas reconoce CADA numero dentro de un grupo, no solo
      los sueltos. Un numero citado solo-dentro-de-un-grupo cuenta como CITADO.
  (c) BACKSTOP — la renumeracion (ordenar_y_agrupar_citas._remap) ELIMINA todo
      numero sin grupo superviviente, en vez de dejarlo pasar tal cual; asi
      ningun marcador queda huerfano ni apuntando a la fuente equivocada.

Regresion historica: antes, un numero solo-en-grupo (p.ej. Art. 100 citado en
"[6, 7, 8]") era invisible para el safeguard -> su fuente se filtraba como
"fantasma" y el marcador quedaba huerfano ([7][8] en texto plano) o —peor—
apuntando a otra fuente ([6] -> una Opinion no relacionada).
"""
import os
import sys
import re
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import responder as R  # noqa: E402
from citas import verificar_citas  # noqa: E402

_RE_MARCA = re.compile(r"\[(\d+)\]")
LEY = "6444155-ley-general-de-contrataciones-publicas-con-modificaciones-al-5-12-2025"
REG = "6444155-ref-reglamento-de-la-ley-de-contrataciones-del-estado-v2"


def _f(cat, doc=None, tipo="seccion", ref=None, parte=1, tag=""):
    return {"categoria": cat, "documento": doc or cat, "tipo_referencia": tipo,
            "referencia": ref, "parte": parte, "tag": tag}


def _reconstruir_mapa(filas, grupos):
    """viejo (1-based en filas) -> nuevo (1-based en grupos), por IDENTIDAD de la fila."""
    idx = {id(f): i + 1 for i, f in enumerate(filas)}
    mapa = {}
    for j, g in enumerate(grupos, start=1):
        for f in g:
            mapa[idx[id(f)]] = j
    return mapa


class TestVerificarCitasReconoceGrupos(unittest.TestCase):
    # (i) verificar_citas reconoce numeros DENTRO de grupos [a, b, c].

    def test_reconoce_numeros_en_grupo(self):
        ch = verificar_citas("La norma lo regula [1, 6, 8] de forma expresa.", 12)
        self.assertEqual(ch["citas_validas"], [1, 6, 8])   # los 3, aunque van en grupo
        self.assertEqual(ch["citas_invalidas"], [])
        self.assertTrue(ch["ok"])

    def test_grupo_mixto_conserva_validos_y_quita_invalidos(self):
        ch = verificar_citas("afirmacion [3, 99].", 5)
        self.assertEqual(ch["citas_validas"], [3])
        self.assertEqual(ch["citas_invalidas"], [99])
        self.assertFalse(ch["ok"])
        self.assertEqual(ch["respuesta_limpia"], "afirmacion [3].")   # 99 fuera, 3 se conserva

    def test_grupo_todos_invalidos_se_neutraliza(self):
        ch = verificar_citas("afirmacion [88, 99].", 5)
        self.assertEqual(ch["citas_validas"], [])
        self.assertEqual(ch["citas_invalidas"], [88, 99])
        self.assertEqual(ch["respuesta_limpia"], "afirmacion [cita no verificada].")

    def test_dedup_dentro_del_marcador(self):
        ch = verificar_citas("x [2, 2, 5].", 5)
        self.assertEqual(ch["citas_validas"], [2, 5])
        self.assertEqual(ch["respuesta_limpia"], "x [2, 5].")   # repetido consecutivo colapsado

    # --- regresion: el comportamiento de marcadores SUELTOS no cambia ---
    def test_suelto_valido_intacto(self):
        ch = verificar_citas("uno [3] dos [5].", 5)
        self.assertEqual(ch["citas_validas"], [3, 5])
        self.assertEqual(ch["respuesta_limpia"], "uno [3] dos [5].")

    def test_suelto_invalido_se_neutraliza(self):
        ch = verificar_citas("uno [9] dos [2].", 5)
        self.assertEqual(ch["citas_invalidas"], [9])
        self.assertEqual(ch["citas_validas"], [2])
        self.assertEqual(ch["respuesta_limpia"], "uno [cita no verificada] dos [2].")


class TestFragmentoSoloEnGrupoSobrevive(unittest.TestCase):
    # (ii) un numero citado SOLO dentro de un grupo sobrevive al filtro fantasma.

    def test_frag_solo_en_grupo_no_se_filtra(self):
        # idx2 (Reglamento Art.100) se cita SOLO dentro de "[1, 2]", nunca suelto.
        filas = [_f("leyes_y_reglamentos", LEY, "articulo", "55", tag="L55"),
                 _f("leyes_y_reglamentos", REG, "articulo", "100", tag="R100")]
        raw = "La figura [1] se desarrolla en el reglamento [1, 2]."
        ch = verificar_citas(raw, len(filas))
        self.assertEqual(ch["citas_validas"], [1, 2])            # (i) 2 reconocido en el grupo
        resp2, grupos = R.ordenar_y_agrupar_citas(
            ch["respuesta_limpia"], filas, ch["citas_validas"])
        tags = [g[0]["tag"] for g in grupos]
        self.assertIn("R100", tags)                              # su fuente NO se filtra
        self.assertEqual(len(grupos), 2)

    def test_partes_del_mismo_articulo_citadas_en_grupo_se_funden(self):
        # Art.100 en 2 partes, citadas solo dentro de un grupo -> UNA cita fundida, no huerfanos.
        filas = [_f("leyes_y_reglamentos", LEY, "articulo", "55", tag="L55"),
                 _f("leyes_y_reglamentos", REG, "articulo", "100", parte=1, tag="R100a"),
                 _f("leyes_y_reglamentos", REG, "articulo", "100", parte=2, tag="R100b")]
        raw = "Regla [1]. Desarrollo [2, 3]."
        ch = verificar_citas(raw, len(filas))
        self.assertEqual(ch["citas_validas"], [1, 2, 3])
        resp2, grupos = R.ordenar_y_agrupar_citas(
            ch["respuesta_limpia"], filas, ch["citas_validas"])
        self.assertEqual(len(grupos), 2)                         # L55 + Art.100 (p1+p2 fundidas)
        self.assertEqual([g[0]["tag"] for g in grupos], ["L55", "R100a"])
        self.assertEqual(len(grupos[1]), 2)                      # las 2 partes en una cita
        # [2, 3] -> mismo grupo, deduplicado a un solo numero
        self.assertEqual(resp2, "Regla [1]. Desarrollo [2].")


class TestSinHuerfanosNiMisPointing(unittest.TestCase):
    # (iii) tras renumerar: ningun marcador huerfano y ninguno apunta a la fuente equivocada.

    def test_invariante_cada_marcador_apunta_a_su_fuente(self):
        filas = [_f("leyes_y_reglamentos", REG, "articulo", "101", tag="R101"),   # 1
                 _f("leyes_y_reglamentos", LEY, "articulo", "55", tag="L55"),     # 2
                 _f("leyes_y_reglamentos", REG, "articulo", "100", tag="R100"),   # 3
                 _f("opiniones", "op-d017", "opinion", "D017", tag="O1"),         # 4
                 _f("opiniones", "op-d051", "opinion", "D051", tag="O2")]         # 5
        # 3 y 5 se citan SOLO dentro de grupos (reproduce la condicion del bug).
        raw = "a [2] b [2, 3] c [1] d [4, 5]"
        ch = verificar_citas(raw, len(filas))
        self.assertEqual(ch["citas_validas"], [1, 2, 3, 4, 5])
        resp2, grupos = R.ordenar_y_agrupar_citas(
            ch["respuesta_limpia"], filas, ch["citas_validas"])
        M = len(grupos)
        # (1) sin huerfanos: todo numero final cae en [1, M]
        for n in (int(x) for m in re.findall(r"\[[^\]]*\]", resp2) for x in re.findall(r"\d+", m)):
            self.assertTrue(1 <= n <= M, f"marcador [{n}] huerfano (M={M})")
        # (2) sin mis-pointing: cada fila citada esta en el grupo al que la remapea su numero
        mapa = _reconstruir_mapa(filas, grupos)
        for viejo in (1, 2, 3, 4, 5):
            self.assertIn(viejo, mapa, f"frag {viejo} citado pero su grupo se filtro")
            self.assertIn(filas[viejo - 1], grupos[mapa[viejo] - 1])
        # (3) las 5 fuentes citadas sobreviven; numeracion contigua 1..M
        self.assertEqual(M, 5)
        self.assertEqual(sorted({int(x) for m in re.findall(r"\[[^\]]*\]", resp2)
                                 for x in re.findall(r"\d+", m)}), list(range(1, M + 1)))


class TestBackstopRenumeracion(unittest.TestCase):
    # (c) el backstop elimina cualquier marcador sin grupo superviviente (nunca huerfano).

    def test_backstop_elimina_marcador_sin_grupo(self):
        # Caso artificial (aisla el mecanismo): el texto trae [2] pero citas_validas={1},
        # asi el grupo de idx2 se filtra. El backstop debe QUITAR [2], no dejarlo huerfano.
        filas = [_f("opiniones", "op-1", "opinion", "A", tag="O1"),
                 _f("opiniones", "op-2", "opinion", "B", tag="O2")]
        resp2, grupos = R.ordenar_y_agrupar_citas("a [1] b [2]", filas, {1})
        self.assertEqual(len(grupos), 1)                    # solo O1 sobrevive
        self.assertEqual([g[0]["tag"] for g in grupos], ["O1"])
        nums = [int(x) for x in _RE_MARCA.findall(resp2)]
        self.assertEqual(nums, [1])                         # [2] eliminado, no queda huerfano
        for n in nums:
            self.assertTrue(1 <= n <= len(grupos))

    def test_backstop_marcador_agrupado_descarta_solo_el_sin_grupo(self):
        # En "[1, 2]" con citas_validas={1}: 2 se descarta, 1 se conserva -> "[1]" renumerado.
        filas = [_f("opiniones", "op-1", "opinion", "A", tag="O1"),
                 _f("opiniones", "op-2", "opinion", "B", tag="O2")]
        resp2, grupos = R.ordenar_y_agrupar_citas("x [1, 2] y", filas, {1})
        self.assertEqual(len(grupos), 1)
        self.assertEqual(resp2, "x [1] y")                  # 2 descartado, 1 conservado


if __name__ == "__main__":
    unittest.main()
