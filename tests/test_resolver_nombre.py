# -*- coding: utf-8 -*-
"""
tests/test_resolver_nombre.py
-------------------------------------------------------------------
Capa de PRESENTACION del nombre legible por documento. Verifica la prioridad
  override (usuario) > mapa curado de singulares > derivacion por tipo > fallback legible
y la derivacion del numero desde `documento` (resoluciones/opiniones).

nombre_derivado (responder.py) es puro. resolver_nombre (app.py) antepone el override de
casos.db; se prueba con una BD TEMPORAL (no toca casos.db).

Correr:  python -m unittest discover -s tests
"""
import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import responder as R  # noqa: E402
import app             # noqa: E402

DOC_LEY = "6444155-ley-general-de-contrataciones-publicas-con-modificaciones-al-5-12-2025"
DOC_REGL = "6444155-ref-reglamento-de-la-ley-de-contrataciones-del-estado-hasta-08-ene-2026-2-v2"
DOC_27444 = "TUO_27444-PROCED_ADMINISTRA-Final"
DOC_CC = "codigo-civil-2026"
DOC_RES = "7639885-resolucion-n-8736-2025-tcp-s5"
DOC_RES_IRREG = "8147537-04813-2026-tcp-s2"
DOC_OPI = "7262499-opinion-d042-2025-oece-dtn"
DOC_DIR = "6682391-directiva-de-junta-de-prevencion-y-resolucion-de-disputas"


class TestNombreDerivado(unittest.TestCase):
    """nombre_derivado puro: curado -> derivacion -> fallback."""

    def test_curado_singulares(self):
        self.assertEqual(R.nombre_derivado(DOC_LEY, "leyes_y_reglamentos"),
                         "Ley N° 32069 – Ley General de Contrataciones Públicas")
        self.assertEqual(R.nombre_derivado(DOC_REGL, "leyes_y_reglamentos"),
                         "Reglamento de la Ley N° 32069 (DS 009-2025-EF)")
        self.assertEqual(R.nombre_derivado(DOC_27444, "leyes_y_reglamentos"),
                         "Ley N° 27444 – Ley del Procedimiento Administrativo General (TUO)")
        self.assertEqual(R.nombre_derivado(DOC_CC, "leyes_y_reglamentos"), "Código Civil")

    def test_derivacion_resolucion(self):
        self.assertEqual(R.nombre_derivado(DOC_RES, "resoluciones_tribunal"),
                         "Resolución N° 8736-2025-TCP-S5")
        # filename irregular: igual deriva el numero
        self.assertEqual(R.nombre_derivado(DOC_RES_IRREG, "resoluciones_tribunal"),
                         "Resolución N° 04813-2026-TCP-S2")

    def test_derivacion_opinion(self):
        self.assertEqual(R.nombre_derivado(DOC_OPI, "opiniones"),
                         "Opinión N° D042-2025/OECE")

    def test_fallback_directiva_sin_numero(self):
        # las directivas no traen numero en `documento` -> titulo legible (no 'Directiva N°').
        out = R.nombre_derivado(DOC_DIR, "directivas")
        self.assertNotIn("N°", out)
        self.assertIn("Directiva", out)

    def test_fallback_desconocido(self):
        self.assertEqual(R.nombre_derivado("DS217_2019EF", "leyes_y_reglamentos"),
                         R._titulo_legible("DS217_2019EF"))

    def test_doc_id_de(self):
        self.assertEqual(R.doc_id_de("mi-doc", "mi-doc__a1"), "mi-doc")          # corpus
        self.assertEqual(R.doc_id_de("x", "bib_deadbeef01__3"), "bib_deadbeef01")  # biblioteca


class TestResolverNombreOverride(unittest.TestCase):
    """resolver_nombre: el override del usuario gana sobre el derivado."""

    def setUp(self):
        fd, self.tmp = tempfile.mkstemp(suffix=".db")
        os.close(fd)
        self._orig = app.DB_PATH
        app.DB_PATH = self.tmp
        app._init_db()

    def tearDown(self):
        app.DB_PATH = self._orig
        for suf in ("", "-wal", "-shm"):
            try:
                os.remove(self.tmp + suf)
            except OSError:
                pass

    def test_sin_override_usa_derivado(self):
        self.assertEqual(app.resolver_nombre(DOC_RES, DOC_RES, "resoluciones_tribunal"),
                         "Resolución N° 8736-2025-TCP-S5")

    def test_override_gana_sobre_curado(self):
        app.fijar_nombre_override(DOC_LEY, "Mi Ley Favorita")
        self.assertEqual(app.resolver_nombre(DOC_LEY, DOC_LEY, "leyes_y_reglamentos"),
                         "Mi Ley Favorita")

    def test_revertir_override_vacio(self):
        app.fijar_nombre_override(DOC_CC, "Otro nombre")
        self.assertEqual(app.resolver_nombre(DOC_CC, DOC_CC, "leyes_y_reglamentos"), "Otro nombre")
        app.fijar_nombre_override(DOC_CC, "")          # vacio -> elimina el override
        self.assertEqual(app.resolver_nombre(DOC_CC, DOC_CC, "leyes_y_reglamentos"), "Código Civil")


if __name__ == "__main__":
    unittest.main()
