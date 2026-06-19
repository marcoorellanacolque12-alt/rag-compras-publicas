# -*- coding: utf-8 -*-
"""
tests/test_conversaciones.py
-------------------------------------------------------------------
Round-trip de la persistencia de conversaciones (SQLite), con una BD TEMPORAL
(no toca casos.db). Verifica crear/listar/leer/borrar, el meta de citas y el
borrado en CASCADA al eliminar el caso.

Correr:  python -m unittest discover -s tests
"""
import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import app  # noqa: E402


class TestConversaciones(unittest.TestCase):
    def setUp(self):
        fd, self.tmp = tempfile.mkstemp(suffix=".db")
        os.close(fd)
        self._orig = app.DB_PATH
        app.DB_PATH = self.tmp          # _conn() lee DB_PATH global -> usa la BD temporal
        app._init_db()

    def tearDown(self):
        app.DB_PATH = self._orig
        for suf in ("", "-wal", "-shm"):
            try:
                os.remove(self.tmp + suf)
            except OSError:
                pass

    def test_roundtrip(self):
        caso = app.crear_caso("Caso de prueba")
        cid = caso["id"]
        conv = app.crear_conversacion(cid, "¿Que dice el articulo 64 sobre adicionales de obra y mas?")
        # titulo acotado a 60 chars
        self.assertLessEqual(len(app.listar_conversaciones(cid)[0]["titulo"]), 60)

        fnorm = [{"numero": 1, "documento": "Ley 32069", "cita": "Ley 32069, Art. 64"}]
        fusadas = [{"id": "f1", "nombre": "contrato.pdf"}]
        app.guardar_intercambio(conv, "¿es para obras?", "Si, el Art. 64 [1] aplica a obras.", fnorm, fusadas)

        convs = app.listar_conversaciones(cid)
        self.assertEqual(len(convs), 1)
        self.assertEqual(convs[0]["id"], conv)

        msgs = app.leer_mensajes(conv)
        self.assertEqual([m["rol"] for m in msgs], ["user", "model"])
        self.assertEqual(msgs[0]["texto"], "¿es para obras?")
        self.assertIsNone(msgs[0]["meta"])
        # el meta del turno 'model' conserva las fuentes para re-pintar las citas
        self.assertEqual(msgs[1]["meta"]["fuentes_normativas"][0]["numero"], 1)
        self.assertEqual(msgs[1]["meta"]["fuentes_usadas"][0]["nombre"], "contrato.pdf")

        # aislamiento por caso
        self.assertTrue(app.conversacion_de_caso(conv, cid))
        self.assertFalse(app.conversacion_de_caso(conv, "otro-caso"))

        # borrado
        app.borrar_conversacion(conv)
        self.assertEqual(app.listar_conversaciones(cid), [])
        self.assertEqual(app.leer_mensajes(conv), [])

    def test_cascade_al_borrar_caso(self):
        caso = app.crear_caso("Caso CASCADE")
        cid = caso["id"]
        conv = app.crear_conversacion(cid)
        app.guardar_intercambio(conv, "q", "r", [], [])
        app.borrar_caso(cid)          # ON DELETE CASCADE -> conversaciones y mensajes fuera
        self.assertEqual(app.listar_conversaciones(cid), [])
        self.assertEqual(app.leer_mensajes(conv), [])


if __name__ == "__main__":
    unittest.main()
