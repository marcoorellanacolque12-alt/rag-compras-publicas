# -*- coding: utf-8 -*-
"""
tests/test_prompt_golden.py
-------------------------------------------------------------------
GOLDEN del ensamblado de prompt + system_instruction de la generacion unificada
(responder.generar_respuesta via _ensamblar_prompt / _construir_contents).

Protege el comportamiento OBSERVABLE de /api/chat: para una entrada fija, el
system_instruction, el prompt y los contents deben ser EXACTAMENTE los esperados
en los 3 modos (general / chat / analisis). Es PURO: no llama a la red.

Incluye el modo "analisis" (ya no se usa desde la UI, pero se conserva en el backend):
este golden evita que se rompa silenciosamente.

Correr:  python -m unittest discover -s tests
"""
import os
import sys
import unittest

# Importar los modulos del proyecto (carpeta padre).
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import responder as R  # noqa: E402

# ===== GOLDEN: constantes verbatim (deben coincidir con responder.INSTRUCCION_*) =====
G_CITAS = (
    "FORMATO DE CITAS: tras cada afirmacion, coloca el marcador [N] (por ejemplo [1], [2]) "
    "del/los fragmento(s) de NORMAS RECUPERADAS que la respaldan. Usa solo numeros de "
    "fragmentos existentes; no inventes marcadores."
)
G_PRECISION = (
    "PRECISION Y ALCANCE: responde UNICAMENTE con la informacion del contexto. Si el usuario "
    "cita un articulo o una norma de forma imprecisa (por ejemplo, atribuye un tema al "
    "Reglamento cuando el contexto lo regula en la Ley, o viceversa) pero el contexto SI "
    "contiene la norma pertinente, RESPONDE con base en el contexto y ACLARA la referencia "
    "correcta (que norma y articulo lo regulan). Recurre a la frase de que no dispones de la "
    "informacion SOLO cuando el contexto realmente no la contenga."
)
G_ESTRUCTURA = (
    "ESTRUCTURA DE LA RESPUESTA: comienza con una apertura DIRECTA de 1-2 frases que responda "
    "la pregunta. Si hay varias reglas, supuestos o condiciones, desarrollalas despues en "
    "puntos o numeracion (una idea por punto). Cierra indicando la referencia normativa "
    "principal que sustenta la respuesta."
)
G_CRUCE = (
    "CRUCE LEY-REGLAMENTO (OBLIGATORIO): revisa TODOS los fragmentos del contexto. Si ademas "
    "del articulo de la Ley que responde la pregunta hay articulos del Reglamento (u otras "
    "normas) que desarrollan ese mismo tema, DEBES mencionarlos en la respuesta, indicando la "
    "relacion explicitamente (por ejemplo: 'regulado en el art. X de la Ley [n] y desarrollado "
    "en los arts. Y [n] y Z [n] del Reglamento') y citando cada uno con su marcador [N]. "
    "REGLA DURA: solo puedes cruzar normas PRESENTES en el contexto recuperado; NUNCA cites "
    "articulos o normas de memoria. Si el desarrollo reglamentario no esta en el contexto, "
    "no lo inventes ni lo insinues."
)


class Turno:
    def __init__(self, rol, texto):
        self.rol, self.texto = rol, texto


# Entradas fijas reutilizadas por los casos.
FILAS = [
    {"chunk_id": "d1__c1", "documento": "6444155-ley-general-de-contrataciones", "tipo_referencia": "articulo",
     "referencia": "64", "articulo_num": "64", "articulo_titulo": "Adicionales", "parte": 0,
     "texto": "Texto del articulo 64.", "distance": 0.1},
    {"chunk_id": "d2__c1", "documento": "6444155-ref-reglamento-de-la-ley-de-contrataciones", "tipo_referencia": "articulo",
     "referencia": "141", "articulo_num": "141", "articulo_titulo": "Adicionales de obra", "parte": 0,
     "texto": "Texto del 141.", "distance": 0.2},
]
PREG = "¿el articulo 64 es para obras?"
CTX_FUENTES = "[FUENTE: contrato.pdf | etiquetas: pago]\nContenido del contrato."
ETIQUETAS = ["pago", "plazo"]


class TestPromptGolden(unittest.TestCase):
    def _ctx(self):
        return R.construir_contexto(FILAS)

    def test_constantes_intactas(self):
        """Las INSTRUCCION_* del modulo coinciden con el golden (no han derivado)."""
        self.assertEqual(R.INSTRUCCION_CITAS, G_CITAS)
        self.assertEqual(R.INSTRUCCION_PRECISION, G_PRECISION)
        self.assertEqual(R.INSTRUCCION_ESTRUCTURA, G_ESTRUCTURA)
        self.assertEqual(R.INSTRUCCION_CRUCE, G_CRUCE)

    def test_modo_general(self):
        sistema, prompt = R._ensamblar_prompt(PREG, FILAS, "general", "", ())
        self.assertEqual(sistema, R.GUARDRAIL_CONSULTA_GENERAL)
        esperado = (
            R.GUARDRAIL_CONSULTA_GENERAL + "\n\n"
            "CONTEXTO NORMATIVO PROPORCIONADO (unica fuente de verdad):\n"
            + self._ctx() + "\n\n"
            + G_PRECISION + "\n\n" + G_ESTRUCTURA + "\n\n" + G_CRUCE + "\n\n" + G_CITAS + "\n\n"
            "CONSULTA DEL USUARIO:\n" + PREG
        )
        self.assertEqual(prompt, esperado)

    def test_modo_chat(self):
        sistema, prompt = R._ensamblar_prompt(PREG, FILAS, "chat", CTX_FUENTES, ETIQUETAS)
        self.assertEqual(sistema, R._sistema_chat())
        esperado = "\n\n".join([
            f"NORMAS RECUPERADAS (base vectorial):\n{self._ctx()}",
            "DOCUMENTOS DEL CASO (fuentes activas seleccionadas por el usuario):\n" + CTX_FUENTES,
            f"CONSULTA DEL USUARIO:\n{PREG}",
            G_PRECISION, G_ESTRUCTURA, G_CRUCE, G_CITAS,
        ])
        self.assertEqual(prompt, esperado)

    def test_modo_chat_sin_fuentes(self):
        sistema, prompt = R._ensamblar_prompt("", FILAS, "chat", "", ())
        self.assertEqual(sistema, R._sistema_chat())
        esperado = "\n\n".join([
            f"NORMAS RECUPERADAS (base vectorial):\n{self._ctx()}",
            "DOCUMENTOS DEL CASO: (ninguna fuente activa en este turno).",
            "CONSULTA DEL USUARIO:\n(resume y comenta las fuentes activas)",
            G_PRECISION, G_ESTRUCTURA, G_CRUCE, G_CITAS,
        ])
        self.assertEqual(prompt, esperado)

    def test_modo_analisis(self):
        """Modo conservado en backend (fuera de la UI): el golden lo blinda."""
        sistema, prompt = R._ensamblar_prompt(PREG, FILAS, "analisis", CTX_FUENTES, ETIQUETAS)
        self.assertEqual(sistema, R._sistema_auditoria(ETIQUETAS))
        esperado = "\n\n".join([
            f"NORMAS RECUPERADAS (base vectorial):\n{self._ctx()}",
            "DOCUMENTOS DEL CASO (fuentes activas seleccionadas por el usuario):\n" + CTX_FUENTES,
            "TAREA: Realiza la auditoria legal de las fuentes activas segun las "
            "instrucciones del sistema.\nFoco adicional del usuario: " + PREG,
            G_PRECISION, G_ESTRUCTURA, G_CRUCE, G_CITAS,
        ])
        self.assertEqual(prompt, esperado)

    def test_contents_historial_roles_nativos(self):
        hist = [Turno("user", "hola"), Turno("model", "buenas"), Turno("user", "sigue")]
        contents = R._construir_contents("PROMPT FINAL", hist)
        pares = [(c.role, c.parts[0].text) for c in contents]
        self.assertEqual(pares, [("user", "hola"), ("model", "buenas"),
                                 ("user", "sigue"), ("user", "PROMPT FINAL")])

    def test_contents_descarta_model_inicial(self):
        """Si el historial empieza con 'model', se descarta hasta el primer 'user'."""
        hist = [Turno("model", "intro"), Turno("user", "pregunta")]
        contents = R._construir_contents("PROMPT", hist)
        pares = [(c.role, c.parts[0].text) for c in contents]
        self.assertEqual(pares, [("user", "pregunta"), ("user", "PROMPT")])


if __name__ == "__main__":
    unittest.main()
