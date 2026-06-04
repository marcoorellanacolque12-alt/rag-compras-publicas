"""
responder.py
-------------------------------------------------------------------
RAG COMPLETO: pregunta -> recuperacion (BigQuery VECTOR_SEARCH)
              -> generacion (Gemini) con CITAS a los articulos.

Flujo:
  1. Embebe la pregunta (text-multilingual-embedding-002, RETRIEVAL_QUERY).
  2. Recupera los top-k chunks mas relevantes desde BigQuery.
  3. Construye un prompt "grounded" y pide a Gemini una respuesta
     basada UNICAMENTE en esos fragmentos, con citas (documento + articulo).

Pensado para areas usuarias, especialistas y operadores del sistema
de compras publicas.

Requisitos:
    pip install google-cloud-bigquery google-genai
    Autenticacion ADC + tabla creada con indexar_bigquery.py.

Uso:
    python responder.py "Que sancion aplica por presentar documentos falsos?"
    python responder.py "plazos de apelacion" --k 6 --modelo gemini-2.5-pro
-------------------------------------------------------------------
"""

import argparse
from google.cloud import bigquery
from google import genai
from google.genai.types import EmbedContentConfig, GenerateContentConfig

# ===================== CONFIGURACION =====================
PROJECT_ID = "project-a0134db0-3990-4ec2-bc3"
LOCATION = "us-central1"
BQ_LOCATION = "US"

DATASET = "rag_compras"
TABLA = "chunks_embeddings"

MODELO_EMB = "text-multilingual-embedding-002"
MODELO_GEN = "gemini-2.5-flash"   # alternativa de mayor calidad: gemini-2.5-pro

TOP_K = 5

INSTRUCCION_SISTEMA = (
    "Eres un asistente experto en contrataciones publicas. Respondes consultas "
    "de areas usuarias, especialistas y operadores. Reglas estrictas:\n"
    "1. Responde UNICAMENTE con base en los FRAGMENTOS NORMATIVOS proporcionados.\n"
    "2. Si la respuesta no esta en los fragmentos, dilo claramente: no inventes.\n"
    "3. Cita SIEMPRE el articulo y el documento de respaldo, ej: (Reglamento, Art. 304).\n"
    "4. Usa lenguaje claro y preciso. Si aplica, enumera plazos, montos o pasos.\n"
)
# ========================================================

_client = None


def cliente():
    global _client
    if _client is None:
        _client = genai.Client(vertexai=True, project=PROJECT_ID, location=LOCATION)
    return _client


def recuperar(pregunta, k):
    """Devuelve los top-k chunks relevantes desde BigQuery."""
    qemb = cliente().models.embed_content(
        model=MODELO_EMB, contents=[pregunta],
        config=EmbedContentConfig(task_type="RETRIEVAL_QUERY"),
    ).embeddings[0].values

    bq = bigquery.Client(project=PROJECT_ID)
    sql = f"""
    SELECT base.documento AS documento, base.articulo_num AS articulo_num,
           base.articulo_titulo AS articulo_titulo, base.texto AS texto, distance
    FROM VECTOR_SEARCH(
      TABLE `{PROJECT_ID}.{DATASET}.{TABLA}`, 'embedding',
      (SELECT @qemb AS embedding),
      top_k => @k, distance_type => 'COSINE')
    ORDER BY distance
    """
    cfg = bigquery.QueryJobConfig(query_parameters=[
        bigquery.ArrayQueryParameter("qemb", "FLOAT64", qemb),
        bigquery.ScalarQueryParameter("k", "INT64", k),
    ])
    return list(bq.query(sql, job_config=cfg, location=BQ_LOCATION).result())


def construir_contexto(filas):
    """Arma el bloque de contexto con los fragmentos numerados."""
    bloques = []
    for i, f in enumerate(filas, start=1):
        doc = "Ley" if "ley-general" in f["documento"] else "Reglamento"
        texto = " ".join(f["texto"].split())
        bloques.append(
            f"[Fragmento {i}] ({doc}, Art. {f['articulo_num']} - {f['articulo_titulo']})\n{texto}"
        )
    return "\n\n".join(bloques)


def responder(pregunta, k=TOP_K, modelo=MODELO_GEN):
    filas = recuperar(pregunta, k)
    if not filas:
        return "No se encontraron fragmentos normativos relevantes.", []

    contexto = construir_contexto(filas)
    prompt = (
        f"FRAGMENTOS NORMATIVOS:\n{contexto}\n\n"
        f"PREGUNTA DEL USUARIO:\n{pregunta}\n\n"
        f"Redacta la respuesta siguiendo las reglas, citando los articulos."
    )
    resp = cliente().models.generate_content(
        model=modelo,
        contents=prompt,
        config=GenerateContentConfig(
            system_instruction=INSTRUCCION_SISTEMA,
            temperature=0.2,
        ),
    )
    return resp.text, filas


def main():
    parser = argparse.ArgumentParser(description="RAG completo (pregunta -> respuesta con citas).")
    parser.add_argument("pregunta", help="Pregunta en lenguaje natural.")
    parser.add_argument("--k", type=int, default=TOP_K, help="Fragmentos a recuperar (default 5).")
    parser.add_argument("--modelo", default=MODELO_GEN, help="Modelo Gemini (default gemini-2.5-flash).")
    args = parser.parse_args()

    print("=" * 70)
    print(f" PREGUNTA: {args.pregunta}")
    print("=" * 70)

    texto, filas = responder(args.pregunta, args.k, args.modelo)

    print("\n--- RESPUESTA ---\n")
    print(texto)
    print("\n--- FUENTES RECUPERADAS ---")
    for i, f in enumerate(filas, start=1):
        doc = "Ley" if "ley-general" in f["documento"] else "Reglamento"
        print(f"  [{i}] {doc}, Art. {f['articulo_num']}: {f['articulo_titulo']} "
              f"(coseno {1 - f['distance']:.3f})")


if __name__ == "__main__":
    main()
