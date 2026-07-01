# -*- coding: utf-8 -*-
"""
tests/test_sugerencias.py
-------------------------------------------------------------------
Tests del lado SERVIDOR de las preguntas sugeridas (Mejora 3):
  - responder._parsear_sugerencias: parseo robusto de la salida del modelo
    (arreglo JSON, cercos ```json, fallback por lineas, dedup y recorte a N).
  - responder.generar_preguntas_sugeridas: resiliencia sin red (entrada vacia -> []).

Es PURO: no llama a la red (solo se prueba el parseo y el caso de borde vacio).

Correr:  python -m unittest discover -s tests
"""
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import responder as R  # noqa: E402


class TestParsearSugerencias(unittest.TestCase):
    def test_arreglo_json_simple(self):
        salida = '["¿Que informes requiere?", "¿Quien aprueba?", "¿Cuando se prohibe?"]'
        self.assertEqual(
            R._parsear_sugerencias(salida),
            ["¿Que informes requiere?", "¿Quien aprueba?", "¿Cuando se prohibe?"],
        )

    def test_cercos_de_codigo_json(self):
        salida = '```json\n["¿A?", "¿B?", "¿C?"]\n```'
        self.assertEqual(R._parsear_sugerencias(salida), ["¿A?", "¿B?", "¿C?"])

    def test_json_con_texto_alrededor(self):
        salida = 'Claro, aqui tienes:\n["¿A?", "¿B?", "¿C?"]\nEspero que ayuden.'
        self.assertEqual(R._parsear_sugerencias(salida), ["¿A?", "¿B?", "¿C?"])

    def test_fallback_por_lineas_con_vinetas(self):
        salida = "- ¿Primera?\n- ¿Segunda?\n- ¿Tercera?"
        self.assertEqual(R._parsear_sugerencias(salida), ["¿Primera?", "¿Segunda?", "¿Tercera?"])

    def test_fallback_por_lineas_numeradas(self):
        salida = '1. ¿Primera?\n2) ¿Segunda?\n3. ¿Tercera?'
        self.assertEqual(R._parsear_sugerencias(salida), ["¿Primera?", "¿Segunda?", "¿Tercera?"])

    def test_recorta_a_max_preguntas(self):
        salida = '["¿A?", "¿B?", "¿C?", "¿D?", "¿E?"]'
        self.assertEqual(R._parsear_sugerencias(salida, max_preguntas=3), ["¿A?", "¿B?", "¿C?"])

    def test_dedup_preservando_orden(self):
        salida = '["¿A?", "¿a?", "¿B?"]'   # "¿A?" y "¿a?" colisionan (case-insensitive)
        self.assertEqual(R._parsear_sugerencias(salida), ["¿A?", "¿B?"])

    def test_vacio_o_basura(self):
        self.assertEqual(R._parsear_sugerencias(""), [])
        self.assertEqual(R._parsear_sugerencias(None), [])


class TestGenerarPreguntasResiliencia(unittest.TestCase):
    def test_respuesta_vacia_no_llama_y_devuelve_lista_vacia(self):
        # Sin respuesta no hay nada que profundizar: corta antes de tocar la red.
        self.assertEqual(R.generar_preguntas_sugeridas("una pregunta", ""), [])
        self.assertEqual(R.generar_preguntas_sugeridas("una pregunta", None), [])


if __name__ == "__main__":
    unittest.main()
