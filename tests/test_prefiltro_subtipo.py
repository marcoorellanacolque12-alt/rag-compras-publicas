# -*- coding: utf-8 -*-
"""
tests/test_prefiltro_subtipo.py
-------------------------------------------------------------------
Control fino de resoluciones del TCP por subtipo en _construir_prefiltro: los toggles
deciden QUE subtipos pasan el filtro. Puro (sin red): inspecciona el WHERE y los params.

  ambos OFF        -> excluye toda la categoria resoluciones_tribunal
  apelacion ON     -> resoluciones solo subtipo apelacion
  sancionadoras ON -> resoluciones solo subtipo sancionadora
  ambos ON         -> apelacion + sancionadora ; 'otra' NUNCA
"""
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import responder as R  # noqa: E402

BASE = {"categorias": [], "excluir_derogada": False, "anio": "Todos", "normas": None}


def _subtipos_param(params):
    for p in params:
        if getattr(p, "name", None) == "f_subtipos":
            return list(p.values)
    return None


class TestPrefiltroSubtipo(unittest.TestCase):
    def _where(self, **flags):
        where, params = R._construir_prefiltro({**BASE, **flags})
        return where or "", params

    def test_ambos_off_excluye_resoluciones(self):
        where, params = self._where()
        self.assertIn("!= 'resoluciones_tribunal'", where)
        self.assertIsNone(_subtipos_param(params))   # sin lista de subtipos -> ninguna entra

    def test_solo_apelacion(self):
        where, params = self._where(incluir_apelacion=True)
        self.assertIn("subtipo IN UNNEST(@f_subtipos)", where)
        self.assertEqual(_subtipos_param(params), ["apelacion"])

    def test_solo_sancionadoras(self):
        where, params = self._where(incluir_sancionadoras=True)
        self.assertEqual(_subtipos_param(params), ["sancionadora"])

    def test_ambos_on(self):
        where, params = self._where(incluir_apelacion=True, incluir_sancionadoras=True)
        self.assertEqual(set(_subtipos_param(params)), {"apelacion", "sancionadora"})

    def test_otra_nunca_pasa(self):
        # 'otra' no esta en ninguna combinacion de toggles.
        for flags in ({}, {"incluir_apelacion": True}, {"incluir_sancionadoras": True},
                      {"incluir_apelacion": True, "incluir_sancionadoras": True}):
            _, params = self._where(**flags)
            self.assertNotIn("otra", _subtipos_param(params) or [])


if __name__ == "__main__":
    unittest.main()
