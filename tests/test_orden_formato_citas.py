# -*- coding: utf-8 -*-
"""
tests/test_orden_formato_citas.py
-------------------------------------------------------------------
PRESENTACION de citas (no toca recuperacion/generacion/ranking):
  PASO 1 reordenar_citas_por_autoridad: las fuentes citadas se listan por JERARQUIA
    (ORDEN_AUTORIDAD), no por aparicion; los marcadores [N] del cuerpo se RENUMERAN de
    forma consistente (cada [N] sigue apuntando a SU fuente).
  PASO 2 texto_legible: el texto de la cita se aplana para mostrarlo (word-wrap -> espacio,
    de-hifenado, parrafos conservados); NO inventa uniones dentro de palabra (caso b).
"""
import os
import sys
import re
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import responder as R  # noqa: E402

_RE_MARCA = re.compile(r"\[(\d+)\]")


class TestReordenPorAutoridad(unittest.TestCase):

    def _filas(self, cats):
        # cada fila lleva un 'tag' unico para rastrear su identidad tras el reordenamiento.
        return [{"categoria": c, "tag": f"f{i}"} for i, c in enumerate(cats, start=1)]

    def test_orden_sigue_ORDEN_AUTORIDAD(self):
        cats = ["opiniones", "leyes_y_reglamentos", "resoluciones_tribunal",
                "leyes_y_reglamentos", "directivas", "documentos_orientacion"]
        _, filas2 = R.reordenar_citas_por_autoridad("sin marcadores", self._filas(cats))
        rank = {c: i for i, c in enumerate(R.ORDEN_AUTORIDAD)}
        ranks = [rank[f["categoria"]] for f in filas2]
        self.assertEqual(ranks, sorted(ranks))                    # no decreciente = jerarquia
        self.assertEqual(filas2[0]["categoria"], "leyes_y_reglamentos")
        self.assertEqual(filas2[-1]["categoria"], "documentos_orientacion")

    def test_estable_dentro_de_categoria(self):
        # dos leyes_y_reglamentos: deben conservar su orden de aparicion (f2 antes que f4).
        cats = ["opiniones", "leyes_y_reglamentos", "resoluciones_tribunal", "leyes_y_reglamentos"]
        _, filas2 = R.reordenar_citas_por_autoridad("x", self._filas(cats))
        leyes = [f["tag"] for f in filas2 if f["categoria"] == "leyes_y_reglamentos"]
        self.assertEqual(leyes, ["f2", "f4"])

    def test_remapeo_marcadores_consistente(self):
        # CHECK CLAVE: cada [N] del cuerpo, tras renumerar, apunta a la MISMA fila de antes.
        cats = ["opiniones", "leyes_y_reglamentos", "resoluciones_tribunal",
                "leyes_y_reglamentos", "directivas"]
        filas = self._filas(cats)
        resp = "A [1]. B [2] y [4]. C [3]. D [5]."
        resp2, filas2 = R.reordenar_citas_por_autoridad(resp, filas)
        # mapa viejo->nuevo leido de la transformacion: el [N] viejo i-esimo en orden de
        # aparicion mapea al fila original i; verificamos via las posiciones.
        # Construimos esperado: para cada viejo n, su fila = filas[n-1]; debe estar en filas2
        # en la posicion (nuevo-1) que indica resp2.
        viejos = [int(x) for x in _RE_MARCA.findall(resp)]
        nuevos = [int(x) for x in _RE_MARCA.findall(resp2)]
        self.assertEqual(len(viejos), len(nuevos))
        for vn, nn in zip(viejos, nuevos):
            # la fila a la que apuntaba [vn] (filas[vn-1]) debe ser la misma que filas2[nn-1]
            self.assertIs(filas2[nn - 1], filas[vn - 1])

    def test_biyeccion_sin_perder_marcadores(self):
        cats = ["resoluciones_tribunal", "leyes_y_reglamentos", "opiniones"]
        filas = self._filas(cats)
        resp = "p [1] q [2] r [3]"
        resp2, _ = R.reordenar_citas_por_autoridad(resp, filas)
        self.assertEqual(sorted(_RE_MARCA.findall(resp2)), ["1", "2", "3"])   # mismos numeros, permutados

    def test_lista_vacia_no_rompe(self):
        self.assertEqual(R.reordenar_citas_por_autoridad("sin citas", []), ("sin citas", []))

    def test_resoluciones_en_su_posicion(self):
        # resoluciones (4o) van despues de opiniones (3o) y antes de doc_orientacion (5o).
        cats = ["documentos_orientacion", "resoluciones_tribunal", "opiniones"]
        _, filas2 = R.reordenar_citas_por_autoridad("x", self._filas(cats))
        self.assertEqual([f["categoria"] for f in filas2],
                         ["opiniones", "resoluciones_tribunal", "documentos_orientacion"])


class TestTextoLegible(unittest.TestCase):

    def test_word_wrap_a_espacio(self):
        crudo = "La autoridad aprueba la contratación mediante procedimientos\nno competitivos."
        self.assertEqual(R.texto_legible(crudo),
                         "La autoridad aprueba la contratación mediante procedimientos no competitivos.")

    def test_conserva_parrafos(self):
        crudo = "Artículo 102. Aprobación.\n\n102.1. La autoridad aprueba\nla contratación."
        out = R.texto_legible(crudo)
        self.assertIn("\n\n", out)                                # parrafo real conservado
        self.assertIn("La autoridad aprueba la contratación.", out)   # word-wrap unido

    def test_fin_de_oracion_es_parrafo(self):
        crudo = "Primera oración.\nSegunda oración que sigue."
        # un salto tras punto se trata como parrafo (no se pega a media frase).
        self.assertIn("Primera oración.", R.texto_legible(crudo))

    def test_de_hifenado(self):
        crudo = "El proceso de contrata-\nción se sujeta a las dispo-\nsiciones vigentes."
        self.assertEqual(R.texto_legible(crudo),
                         "El proceso de contratación se sujeta a las disposiciones vigentes.")

    def test_colapsa_espacios(self):
        self.assertEqual(R.texto_legible("texto   con    espacios"), "texto con espacios")

    def test_no_corrige_espacio_interno_caso_b(self):
        # (b): NO se debe inventar la union; el texto se conserva salvo el aplanado de saltos.
        crudo = "la contratacione s se sujeta"
        self.assertEqual(R.texto_legible(crudo), "la contratacione s se sujeta")

    def test_vacio(self):
        self.assertEqual(R.texto_legible(""), "")
        self.assertIsNone(R.texto_legible(None))


if __name__ == "__main__":
    unittest.main()
