# -*- coding: utf-8 -*-
"""
tests/test_chunker_articulo.py
-------------------------------------------------------------------
Golden tests del chunker de articulos (leyes_y_reglamentos). Blindan el bug de
citas: el badge decia "Art. 55" pero abria el Art. 102 porque RE_ARTICULO (patron
permisivo) tomaba una REFERENCIA CRUZADA del cuerpo ("...articulo 55 de la Ley...")
caida a inicio de linea por el wrap del PDF como si fuera una frontera de articulo.

El arreglo: RE_ARTICULO exige un DELIMITADOR de encabezado (. - ° º ª ) : o U+FFFD)
entre el numero y el titulo; las referencias cruzadas (numero + espacio + conector,
SIN delimitador) ya no son frontera. Puro (sin red ni GCS).

  (a) un encabezado real "Artículo NN." se detecta (ref = NN).
  (b) una referencia cruzada del cuerpo NO se vuelve el numero del chunk.
  (c) un encabezado con ruido OCR (delimitador U+FFFD o palabra de-hifenada) se detecta.
  (d) un chunk cuyo cuerpo menciona otros articulos conserva el numero de SU encabezado.
"""
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import chunking_articulos as C  # noqa: E402

FFFD = "�"   # caracter de reemplazo OCR


def _refs_articulo(texto):
    """Referencias de los bloques tipo 'articulo' que produce el troceo (en orden)."""
    bloques = C.trocear_por_articulo(C.limpiar(texto))
    return [b["referencia"] for b in bloques if b["tipo_referencia"] == "articulo"]


def _chunk_de_articulo(texto, num):
    """Primer chunk end-to-end cuyo articulo (referencia) == num."""
    chunks = C.chunkear_documento(texto, "leyes_y_reglamentos", "doc_test", "blob/doc_test")
    return next((c for c in chunks if c["tipo_referencia"] == "articulo"
                 and c["referencia"] == str(num)), None)


class TestChunkerArticulo(unittest.TestCase):

    # (a) encabezado real se detecta
    def test_encabezado_real_se_detecta(self):
        texto = ("Artículo 55.- Contrataciones directas\n"
                 "La Entidad puede contratar directamente en los supuestos siguientes.\n")
        self.assertIn("55", _refs_articulo(texto))
        # forma con grado y guion ("55°.-") y con dos puntos ("55:") tambien valen
        self.assertIn("55", _refs_articulo("Artículo 55°.- Texto del articulo.\n"))
        self.assertIn("55", _refs_articulo("Artículo 55: Texto del articulo.\n"))
        # titulo que empieza con "De/Del" se conserva (trae el delimitador antes)
        self.assertEqual(C.trocear_por_articulo(C.limpiar(
            "Artículo 244.- De la entidad encargada de las contrataciones\nCuerpo.\n"
        ))[0]["referencia"], "244")

    # (b) una referencia cruzada del cuerpo NO se vuelve el numero del chunk
    def test_referencia_cruzada_no_es_frontera(self):
        # El wrap del PDF deja "artículo 55 de la Ley" a INICIO DE LINEA dentro del 102.
        texto = ("Artículo 102.- Prestaciones adicionales\n"
                 "El titular aprueba las prestaciones conforme a lo previsto en el\n"
                 "artículo 55 de la Ley. El monto se sujeta a los limites indicados.\n")
        refs = _refs_articulo(texto)
        self.assertEqual(refs, ["102"])          # SOLO el 102 es frontera; el 55 (ref) no
        self.assertNotIn("55", refs)

    # (c) encabezado con ruido OCR se detecta (delimitador U+FFFD y/o de-hifenado)
    def test_encabezado_con_ruido_ocr_se_detecta(self):
        # OCR corrompio el ".-" a U+FFFD; igual debe reconocerse como encabezado.
        self.assertIn("55", _refs_articulo(f"Artículo 55{FFFD}- Contrataciones directas\nCuerpo.\n"))
        # palabra "Artículo" partida por guion + salto de linea: limpiar() la une.
        self.assertIn("77", _refs_articulo("Artícu-\nlo 77.- Garantías\nCuerpo del articulo.\n"))

    # (d) un chunk que menciona otros articulos conserva el numero de SU encabezado
    def test_chunk_conserva_su_propio_numero(self):
        texto = ("Artículo 102.- Prestaciones adicionales de obra\n"
                 "Para su aprobacion se observa lo dispuesto en el artículo 55 de la Ley\n"
                 "y en el artículo 34 del Reglamento, sin exceder los topes legales.\n")
        ch = _chunk_de_articulo(texto, 102)
        self.assertIsNotNone(ch, "deberia existir el chunk del Art. 102")
        self.assertEqual(ch["referencia"], "102")
        self.assertEqual(ch["articulo_num"], "102")     # espejo de transicion coherente
        # y NO debe existir un chunk-articulo rotulado 55 o 34 (eran menciones del cuerpo)
        chunks = C.chunkear_documento(texto, "leyes_y_reglamentos", "doc_test", "blob/doc_test")
        rotulos = {c["referencia"] for c in chunks if c["tipo_referencia"] == "articulo"}
        self.assertEqual(rotulos, {"102"})


if __name__ == "__main__":
    unittest.main()
