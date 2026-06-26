# -*- coding: utf-8 -*-
"""
tests/test_orden_formato_citas.py
-------------------------------------------------------------------
PRESENTACION de citas (no toca recuperacion/generacion/ranking). ordenar_y_agrupar_citas:
  - FUNDE las partes de un mismo articulo en UNA cita (Art. 55 parte 1 + parte 2 -> una).
  - ORDENA por jerarquia (ORDEN_AUTORIDAD) y, dentro de leyes_y_reglamentos, la Ley antes
    que su Reglamento.
  - RENUMERA los marcadores [N] del cuerpo, incluidos los AGRUPADOS ("[2, 3]"), y deduplica
    los que caen en la misma cita.
texto_legible / unir_partes: formato del texto al mostrarlo.
"""
import os
import sys
import re
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import responder as R  # noqa: E402

_RE_MARCA = re.compile(r"\[(\d+)\]")
LEY = "6444155-ley-general-de-contrataciones-publicas-con-modificaciones-al-5-12-2025"
REG = "6444155-ref-reglamento-de-la-ley-de-contrataciones-del-estado-v2"


def _f(cat, doc=None, tipo="seccion", ref=None, parte=1, tag=""):
    return {"categoria": cat, "documento": doc or cat, "tipo_referencia": tipo,
            "referencia": ref, "parte": parte, "tag": tag}


class TestOrdenYAgrupar(unittest.TestCase):

    def test_orden_por_jerarquia_entre_categorias(self):
        filas = [_f("opiniones"), _f("leyes_y_reglamentos", LEY, "articulo", "1"),
                 _f("resoluciones_tribunal"), _f("directivas"), _f("documentos_orientacion")]
        _, grupos = R.ordenar_y_agrupar_citas("x", filas)
        cats = [g[0]["categoria"] for g in grupos]
        rank = {c: i for i, c in enumerate(R.ORDEN_AUTORIDAD)}
        self.assertEqual([rank[c] for c in cats], sorted(rank[c] for c in cats))
        self.assertEqual(cats[0], "leyes_y_reglamentos")

    def test_ley_antes_que_reglamento(self):
        # el Reglamento aparece PRIMERO en filas, pero la Ley debe quedar antes en la lista.
        filas = [_f("leyes_y_reglamentos", REG, "articulo", "101", tag="R101"),
                 _f("leyes_y_reglamentos", LEY, "articulo", "55", tag="L55"),
                 _f("leyes_y_reglamentos", REG, "articulo", "102", tag="R102")]
        _, grupos = R.ordenar_y_agrupar_citas("x", filas)
        tags = [g[0]["tag"] for g in grupos]
        self.assertEqual(tags, ["L55", "R101", "R102"])   # Ley, luego Reglamento (estable)

    def test_funde_partes_mismo_articulo(self):
        filas = [_f("leyes_y_reglamentos", LEY, "articulo", "55", parte=1, tag="p1"),
                 _f("leyes_y_reglamentos", LEY, "articulo", "55", parte=2, tag="p2")]
        _, grupos = R.ordenar_y_agrupar_citas("x", filas)
        self.assertEqual(len(grupos), 1)                  # una sola cita
        self.assertEqual([p["parte"] for p in grupos[0]], [1, 2])   # partes en orden

    def test_no_funde_articulos_distintos(self):
        filas = [_f("leyes_y_reglamentos", LEY, "articulo", "55"),
                 _f("leyes_y_reglamentos", LEY, "articulo", "56")]
        _, grupos = R.ordenar_y_agrupar_citas("x", filas)
        self.assertEqual(len(grupos), 2)

    def test_no_funde_opiniones(self):
        # opiniones NO se funden aunque compartan referencia: cada chunk es su cita.
        filas = [_f("opiniones", "op-d017", "opinion", "D017", parte=1),
                 _f("opiniones", "op-d017", "opinion", "D017", parte=2)]
        _, grupos = R.ordenar_y_agrupar_citas("x", filas)
        self.assertEqual(len(grupos), 2)

    def test_renumera_agrupados_y_dedup(self):
        # Caso completo: marcadores agrupados + fusion de partes + Ley antes que Reglamento.
        filas = [_f("leyes_y_reglamentos", REG, "articulo", "101", tag="R101"),   # pos 1
                 _f("leyes_y_reglamentos", LEY, "articulo", "55", parte=1, tag="L55a"),  # pos 2
                 _f("leyes_y_reglamentos", LEY, "articulo", "55", parte=2, tag="L55b"),  # pos 3
                 _f("leyes_y_reglamentos", REG, "articulo", "102", tag="R102"),   # pos 4
                 _f("opiniones", "op-d017", "opinion", "D017", tag="O1")]         # pos 5
        resp = "a [1] b [2, 3] c [4] d [5]"
        resp2, grupos = R.ordenar_y_agrupar_citas(resp, filas)
        # grupos esperados: L55(p1,p2) | R101 | R102 | O1
        self.assertEqual(len(grupos), 4)
        self.assertEqual(len(grupos[0]), 2)               # Art.55 fundido
        # [2,3] (las dos partes del 55) -> una sola cita [1], deduplicada
        self.assertEqual(resp2, "a [2] b [1] c [3] d [4]")

    def test_marcador_apunta_a_su_cita_tras_todo(self):
        # CHECK CLAVE: cada numero viejo -> grupo (cita) que contiene esa fila original.
        filas = [_f("leyes_y_reglamentos", REG, "articulo", "101", tag="R101"),
                 _f("leyes_y_reglamentos", LEY, "articulo", "55", parte=1, tag="L55a"),
                 _f("leyes_y_reglamentos", LEY, "articulo", "55", parte=2, tag="L55b"),
                 _f("opiniones", "op-d017", "opinion", "D017", tag="O1")]
        resp = "[1] [2] [3] [4]"
        resp2, grupos = R.ordenar_y_agrupar_citas(resp, filas)
        viejos = [int(x) for x in _RE_MARCA.findall(resp)]
        nuevos = [int(x) for x in _RE_MARCA.findall(resp2)]
        for vn, nn in zip(viejos, nuevos):
            self.assertIn(filas[vn - 1], grupos[nn - 1])   # la fila original esta en esa cita

    def test_resoluciones_en_su_posicion(self):
        filas = [_f("documentos_orientacion"), _f("resoluciones_tribunal"), _f("opiniones")]
        _, grupos = R.ordenar_y_agrupar_citas("x", filas)
        self.assertEqual([g[0]["categoria"] for g in grupos],
                         ["opiniones", "resoluciones_tribunal", "documentos_orientacion"])

    # ===== FILTRO DE CITAS FANTASMA (citas_validas) =====
    def test_filtra_no_citadas_y_conserva_citadas(self):
        # R101(1) L55p1(2) L55p2(3) R102(4) O1(5); el texto solo usa [1] y [2].
        filas = [_f("leyes_y_reglamentos", REG, "articulo", "101", tag="R101"),
                 _f("leyes_y_reglamentos", LEY, "articulo", "55", parte=1, tag="L55a"),
                 _f("leyes_y_reglamentos", LEY, "articulo", "55", parte=2, tag="L55b"),
                 _f("leyes_y_reglamentos", REG, "articulo", "102", tag="R102"),
                 _f("opiniones", "op-d017", "opinion", "D017", tag="O1")]
        resp = "a [1] b [2]"
        resp2, grupos = R.ordenar_y_agrupar_citas(resp, filas, {1, 2})
        # sobreviven solo L55 (idx 2 citado) y R101 (idx 1 citado); R102(4) y O1(5) se eliminan.
        self.assertEqual([g[0]["tag"] for g in grupos], ["L55a", "R101"])
        self.assertEqual(len(grupos[0]), 2)                # Art.55 conserva sus 2 partes
        # renumeracion contigua: orig[1]=R101->2, orig[2]=L55->1
        self.assertEqual(resp2, "a [2] b [1]")

    def test_numeracion_contigua_sin_huecos(self):
        # 4 opiniones distintas; el texto usa [1],[3],[4] (NO [2]).
        filas = [_f("opiniones", "op-1", "opinion", "A", tag="O1"),
                 _f("opiniones", "op-2", "opinion", "B", tag="O2"),
                 _f("opiniones", "op-3", "opinion", "C", tag="O3"),
                 _f("opiniones", "op-4", "opinion", "D", tag="O4")]
        resp = "[1] [3] [4]"
        resp2, grupos = R.ordenar_y_agrupar_citas(resp, filas, {1, 3, 4})
        self.assertEqual([g[0]["tag"] for g in grupos], ["O1", "O3", "O4"])   # O2 eliminada
        nums = sorted(int(x) for x in _RE_MARCA.findall(resp2))
        self.assertEqual(nums, [1, 2, 3])                  # contiguo, sin saltos ni [4]
        self.assertEqual(resp2, "[1] [2] [3]")

    def test_articulo_2_partes_citado_por_una_se_conserva(self):
        # se cita solo la parte 1 del Art.55; el grupo (p1+p2) debe conservarse completo.
        filas = [_f("leyes_y_reglamentos", LEY, "articulo", "55", parte=1, tag="p1"),
                 _f("leyes_y_reglamentos", LEY, "articulo", "55", parte=2, tag="p2")]
        resp2, grupos = R.ordenar_y_agrupar_citas("solo [1]", filas, {1})
        self.assertEqual(len(grupos), 1)
        self.assertEqual([p["parte"] for p in grupos[0]], [1, 2])
        self.assertEqual(resp2, "solo [1]")

    def test_ningun_marcador_queda_huerfano(self):
        # propiedad: todo [N] de la respuesta filtrada apunta a un grupo existente (1..len).
        filas = [_f("leyes_y_reglamentos", REG, "articulo", "101", tag="R101"),
                 _f("leyes_y_reglamentos", LEY, "articulo", "55", parte=1, tag="L55a"),
                 _f("leyes_y_reglamentos", LEY, "articulo", "55", parte=2, tag="L55b"),
                 _f("opiniones", "op-d017", "opinion", "D017", tag="O1")]
        resp = "x [1] y [2, 3]"                            # incluye un marcador AGRUPADO
        resp2, grupos = R.ordenar_y_agrupar_citas(resp, filas, {1, 2, 3})
        for n in (int(x) for x in _RE_MARCA.findall(resp2)):
            self.assertTrue(1 <= n <= len(grupos))

    def test_citas_validas_none_no_filtra(self):
        # compatibilidad: sin citas_validas se listan TODOS los grupos (comportamiento previo).
        filas = [_f("opiniones", "op-1", "opinion", "A"),
                 _f("opiniones", "op-2", "opinion", "B")]
        _, grupos = R.ordenar_y_agrupar_citas("[1]", filas)        # None -> no filtra
        self.assertEqual(len(grupos), 2)

    def test_citas_validas_vacias_lista_vacia(self):
        filas = [_f("opiniones", "op-1", "opinion", "A")]
        self.assertEqual(R.ordenar_y_agrupar_citas("sin marcadores", filas, set()),
                         ("sin marcadores", []))

    def test_lista_vacia(self):
        self.assertEqual(R.ordenar_y_agrupar_citas("sin citas", []), ("sin citas", []))

    def test_deteccion_ley_reglamento_robusta(self):
        self.assertTrue(R._es_doc_ley_general(LEY))
        self.assertFalse(R._es_doc_ley_general(REG))      # el Reglamento NO es la Ley
        self.assertTrue(R._es_doc_reglamento(REG))
        self.assertFalse(R._es_doc_reglamento(LEY))


class TestUnirPartes(unittest.TestCase):

    def test_remueve_solape(self):
        a = "El Articulo 55 regula las contrataciones sujetas a procedimiento no competitivo"
        b = "procedimiento no competitivo, conforme a las causales del presente articulo."
        unido = R.unir_partes([a, b])
        self.assertEqual(unido.count("procedimiento no competitivo"), 1)   # no se duplica el solape
        self.assertTrue(unido.endswith("del presente articulo."))

    def test_sin_solape_une_con_espacio(self):
        self.assertEqual(R.unir_partes(["Parte uno.", "Parte dos."]), "Parte uno. Parte dos.")

    def test_una_sola_parte(self):
        self.assertEqual(R.unir_partes(["Solo una."]), "Solo una.")


class TestTextoLegible(unittest.TestCase):

    def test_word_wrap_a_espacio(self):
        crudo = "La autoridad aprueba la contratación mediante procedimientos\nno competitivos."
        self.assertEqual(R.texto_legible(crudo),
                         "La autoridad aprueba la contratación mediante procedimientos no competitivos.")

    def test_conserva_parrafos(self):
        crudo = "Artículo 102. Aprobación.\n\n102.1. La autoridad aprueba\nla contratación."
        out = R.texto_legible(crudo)
        self.assertIn("\n\n", out)
        self.assertIn("La autoridad aprueba la contratación.", out)

    def test_de_hifenado(self):
        crudo = "El proceso de contrata-\nción se sujeta a las dispo-\nsiciones vigentes."
        self.assertEqual(R.texto_legible(crudo),
                         "El proceso de contratación se sujeta a las disposiciones vigentes.")

    def test_colapsa_espacios(self):
        self.assertEqual(R.texto_legible("texto   con    espacios"), "texto con espacios")

    def test_no_corrige_espacio_interno_caso_b(self):
        self.assertEqual(R.texto_legible("la contratacione s se sujeta"), "la contratacione s se sujeta")

    def test_vacio(self):
        self.assertEqual(R.texto_legible(""), "")
        self.assertIsNone(R.texto_legible(None))


if __name__ == "__main__":
    unittest.main()
