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

    def test_general_roundtrip_y_aislamiento(self):
        """Conversacion de Consulta General (caso_id NULL): crear, listar, leer; aislada
        de las conversaciones de los casos."""
        caso = app.crear_caso("Caso X")
        cgen = app.crear_conversacion(None, "¿Que es el SEACE?")          # general
        ccaso = app.crear_conversacion(caso["id"], "Pregunta del caso")    # de caso
        app.guardar_intercambio(cgen, "¿que es?", "respuesta general", [{"numero": 1}], [])

        generales = app.listar_conversaciones(None)
        ids_gen = [c["id"] for c in generales]
        self.assertIn(cgen, ids_gen)
        self.assertNotIn(ccaso, ids_gen)                 # la de caso NO aparece en generales
        self.assertNotIn(cgen, [c["id"] for c in app.listar_conversaciones(caso["id"])])

        self.assertEqual([m["rol"] for m in app.leer_mensajes(cgen)], ["user", "model"])
        # pertenencia por ambito
        self.assertTrue(app.conversacion_de_caso(cgen, None))
        self.assertFalse(app.conversacion_de_caso(cgen, caso["id"]))
        self.assertTrue(app.conversacion_de_caso(ccaso, caso["id"]))
        self.assertFalse(app.conversacion_de_caso(ccaso, None))

    def test_hilo_reutiliza_no_forka(self):
        """La decision de backend (conversacion_de_caso) hace que un seguimiento se AGREGUE
        a la misma conversacion en vez de crear otra."""
        caso = app.crear_caso("Hilo")
        cid = caso["id"]
        conv = app.crear_conversacion(cid, "primera")
        # dos turnos en la MISMA conversacion (como hace /api/chat al recibir el conversacion_id)
        app.guardar_intercambio(conv, "p1", "r1", [], [])
        self.assertTrue(app.conversacion_de_caso(conv, cid))   # -> chat reutiliza, no crea
        app.guardar_intercambio(conv, "p2", "r2", [], [])
        self.assertEqual(len(app.leer_mensajes(conv)), 4)      # 2 turnos = 4 mensajes
        self.assertEqual(len(app.listar_conversaciones(cid)), 1)  # UNA sola conversacion


class TestMigracionCasoIdNullable(unittest.TestCase):
    """La migracion convierte conversaciones.caso_id NOT NULL -> NULLABLE preservando datos."""

    def setUp(self):
        fd, self.tmp = tempfile.mkstemp(suffix=".db")
        os.close(fd)
        self._orig = app.DB_PATH
        app.DB_PATH = self.tmp

    def tearDown(self):
        app.DB_PATH = self._orig
        for suf in ("", "-wal", "-shm"):
            try:
                os.remove(self.tmp + suf)
            except OSError:
                pass

    def test_migracion_preserva_filas(self):
        # BD "vieja": conversaciones.caso_id NOT NULL, con una fila.
        con = app._conn()
        con.execute("CREATE TABLE casos (id TEXT PRIMARY KEY, nombre TEXT NOT NULL, fecha_creacion TEXT NOT NULL)")
        con.execute("""CREATE TABLE conversaciones (
            id TEXT PRIMARY KEY, caso_id TEXT NOT NULL, titulo TEXT,
            fecha_creacion TEXT NOT NULL, fecha_actualizacion TEXT NOT NULL,
            FOREIGN KEY (caso_id) REFERENCES casos(id) ON DELETE CASCADE)""")
        con.execute("INSERT INTO casos VALUES ('c1','Caso','t')")
        con.execute("INSERT INTO conversaciones VALUES ('v1','c1','tit','t','t')")
        con.commit(); con.close()

        app._init_db()   # detecta NOT NULL y reconstruye como NULLABLE

        con = app._conn()
        try:
            col = next(c for c in con.execute("PRAGMA table_info(conversaciones)").fetchall()
                       if c["name"] == "caso_id")
            self.assertEqual(col["notnull"], 0)   # ahora NULLABLE
            self.assertIsNotNone(con.execute("SELECT 1 FROM conversaciones WHERE id='v1'").fetchone())
            # ahora SI se puede insertar una conversacion General (caso_id NULL)
            con.execute("INSERT INTO conversaciones VALUES ('g1',NULL,'gen','t','t')")
            con.commit()
            self.assertIsNotNone(con.execute("SELECT 1 FROM conversaciones WHERE id='g1' AND caso_id IS NULL").fetchone())
        finally:
            con.close()


class TestExportMarkdown(unittest.TestCase):
    """Export de conversacion a Markdown: conserva los [N] y adjunta el detalle de fuentes."""

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

    def test_markdown_incluye_fuentes_y_marcadores(self):
        caso = app.crear_caso("Caso export")
        cid = caso["id"]
        conv = app.crear_conversacion(cid, "¿el art. 64 aplica a obras?")
        fnorm = [{"numero": 1, "documento": "Ley 32069", "cita": "Ley N° 32069, Art. 64",
                  "articulo_titulo": "Adicionales de obra"}]
        fusadas = [{"id": "f1", "nombre": "contrato-obra.pdf"}]
        app.guardar_intercambio(conv, "¿es para obras?", "Sí, el Art. 64 [1] aplica a obras.",
                                fnorm, fusadas)

        md = app.construir_markdown_conversacion(app.obtener_conversacion(conv),
                                                 app.leer_mensajes(conv))
        # encabezado + nota legal
        self.assertIn("# ¿el art. 64 aplica a obras?", md)
        self.assertIn("no constituye asesoría legal formal", md)
        # la pregunta y la respuesta CON el marcador [1] intacto
        self.assertIn("**Consulta:** ¿es para obras?", md)
        self.assertIn("Sí, el Art. 64 [1] aplica a obras.", md)
        # detalle de fuentes citadas (numero + cita + titulo del articulo)
        self.assertIn("**Fuentes citadas:**", md)
        self.assertIn("**[1]** Ley N° 32069, Art. 64", md)
        self.assertIn("Adicionales de obra", md)
        # fuentes del caso usadas
        self.assertIn("**Fuentes del caso usadas:**", md)
        self.assertIn("contrato-obra.pdf", md)

    def test_markdown_conversacion_vacia_solo_encabezado(self):
        conv = app.crear_conversacion(None, "consulta general vacía")
        md = app.construir_markdown_conversacion(app.obtener_conversacion(conv),
                                                 app.leer_mensajes(conv))
        self.assertIn("# consulta general vacía", md)
        self.assertIn("no tiene mensajes", md)
        self.assertNotIn("**Consulta:**", md)


if __name__ == "__main__":
    unittest.main()
