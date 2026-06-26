# -*- coding: utf-8 -*-
"""
tests/test_expansion_recuperacion.py
-------------------------------------------------------------------
Cubre los dos arreglos de RECUPERACION (no tocan generacion/citas/chunking):

  (a) _es_especifica: decide si una consulta se EXPANDE. Debe expandir consultas vagas
      legitimas (un digito suelto, una palabra-norma generica) y saltarse solo las que
      citan una referencia normativa concreta (palabra + numero, o inciso + letra).

  (b) OVER-FETCH (POOL_OVERFETCH): la busqueda vectorial trae un pool mas amplio ANTES del
      re-rank por autoridad, de modo que una norma de alta autoridad en la banda
      11..POOL_OVERFETCH por distancia ya no se pierde por el truncado a TOP_K. Se prueba
      con _buscar monkeypatcheado (sin BigQuery): determinista y offline.

Correr:  python -m unittest discover -s tests
-------------------------------------------------------------------
"""
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import responder as R  # noqa: E402


class TestEsEspecifica(unittest.TestCase):
    """(a) Afinado de _es_especifica. especifica=True -> NO se expande."""

    def _check(self, pregunta, esperado):
        self.assertEqual(R._es_especifica(pregunta), esperado,
                         msg=f"{pregunta!r}: esperaba especifica={esperado}")

    # --- DEBEN EXPANDIRSE (especifica=False): consultas vagas legitimas ---
    def test_digito_suelto_expande(self):
        # el "13" NO es una referencia de articulo -> no debe saltar la expansion
        self._check("cuáles son los 13 supuestos de contratación no competitiva", False)

    def test_palabra_norma_generica_expande(self):
        # "decreto" suelto, sin numero -> vago, debe expandir
        self._check("supuestos del decreto de contrataciones", False)
        self._check("reglamento de la ley de contrataciones", False)

    def test_consultas_vagas_cortas_expanden(self):
        self._check("contratación directa", False)
        self._check("qué es una contratación no competitiva", False)

    # --- NO DEBEN EXPANDIRSE (especifica=True): referencia normativa concreta ---
    def test_articulo_con_numero_no_expande(self):
        self._check("qué dice el artículo 55", True)
        self._check("artículo 55 ley de contrataciones", True)
        self._check("art. 55", True)
        self._check("art 55", True)

    def test_norma_con_numero_no_expande(self):
        self._check("ley 32069", True)
        self._check("DS 009-2025", True)
        self._check("numeral 55.1 de la ley", True)

    def test_inciso_por_letra_no_expande(self):
        self._check("inciso a) del artículo", True)

    def test_consulta_larga_detallada_no_expande(self):
        # regla conservada: >=6 palabras de contenido (len>3) indica consulta especifica
        self._check("requisitos documentos garantia oferta postor adjudicacion contrato vigente", True)


class TestOverFetch(unittest.TestCase):
    """(b) El pool ampliado rescata un chunk de alta autoridad en la banda 11..POOL_OVERFETCH."""

    def setUp(self):
        self._buscar_orig = R._buscar
        self._pool_orig = R.POOL_OVERFETCH
        # Candidatos sinteticos ordenados por distancia (como los devolveria VECTOR_SEARCH):
        #  - ranks 1..19: resoluciones (cercanas por distancia, pero +0.15 de penalidad)
        #  - rank 20: una norma leyes_y_reglamentos (penalidad 0) -> tras re-rank deberia ganar
        filas = []
        for i in range(19):
            filas.append({"chunk_id": f"reso_{i:02d}", "categoria": "resoluciones_tribunal",
                          "documento": f"doc_reso_{i}", "tipo_referencia": "considerando",
                          "referencia": f"Fundamento {i}", "distance": 0.20 + i * 0.004})
        filas.append({"chunk_id": "LEY_objetivo", "categoria": "leyes_y_reglamentos",
                      "documento": "ley-general-de-contrataciones", "tipo_referencia": "articulo",
                      "referencia": "55", "distance": 0.30})   # rank 20 por distancia
        for i in range(20, 40):
            filas.append({"chunk_id": f"filler_{i:02d}", "categoria": "resoluciones_tribunal",
                          "documento": f"doc_f_{i}", "tipo_referencia": "considerando",
                          "referencia": f"X{i}", "distance": 0.31 + i * 0.001})
        self._filas = filas
        # _buscar(consulta, k, filtros=None) -> primeros k (respeta el tamaño de pool pedido)
        R._buscar = lambda consulta, k, filtros=None: [dict(f) for f in self._filas[:k]]

    def tearDown(self):
        R._buscar = self._buscar_orig
        R.POOL_OVERFETCH = self._pool_orig

    def test_pool_30_rescata_chunk_rank_20(self):
        R.POOL_OVERFETCH = 30
        filas = R.recuperar(["consulta"], k=10, completar=False)
        ids = [f["chunk_id"] for f in filas]
        self.assertIn("LEY_objetivo", ids,
                      "con pool-30 el chunk de autoridad en rank 20 debe entrar al top-10")
        # y por su penalidad 0 (vs +0.15 de las resoluciones) debe quedar primero
        self.assertEqual(ids[0], "LEY_objetivo")
        self.assertEqual(len(filas), 10)

    def test_pool_10_NO_lo_rescata(self):
        # Contraprueba: sin over-fetch (pool=10) el chunk de rank 20 se pierde.
        R.POOL_OVERFETCH = 10
        filas = R.recuperar(["consulta"], k=10, completar=False)
        ids = [f["chunk_id"] for f in filas]
        self.assertNotIn("LEY_objetivo", ids,
                         "con pool-10 el chunk de rank 20 NO esta en el pool -> no se rescata")


if __name__ == "__main__":
    unittest.main()
