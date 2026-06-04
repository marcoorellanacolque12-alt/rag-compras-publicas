"""
buscar.py
-------------------------------------------------------------------
BUSCADOR SEMANTICO RAG (recuperacion) sobre el corpus normativo.

Dada una pregunta en lenguaje natural:
  1. Genera el embedding de la pregunta (Vertex AI, task RETRIEVAL_QUERY).
  2. Busca los chunks mas relevantes en BigQuery con VECTOR_SEARCH
     (distancia COSENO, serverless, sin costo fijo).
  3. Muestra los fragmentos con su CITA (documento + articulo).

Este es el paso de "Retrieval" del RAG. El siguiente seria pasar
estos fragmentos como contexto a un modelo Gemini para redactar la
respuesta final (Generation).

Requisitos:
    pip install google-cloud-bigquery google-genai
    Autenticacion ADC + tabla creada con indexar_bigquery.py.

Uso:
    python buscar.py "Que sancion aplica por presentar documentos falsos?"
    python buscar.py "plazos de apelacion" --k 3
-------------------------------------------------------------------
"""

import argparse
from google.cloud import bigquery
from google import genai
from google.genai.types import EmbedContentConfig

# ===================== CONFIGURACION =====================
PROJECT_ID = "project-a0134db0-3990-4ec2-bc3"
LOCATION = "us-central1"          # region para Vertex AI (embeddings)
BQ_LOCATION = "US"                # ubicacion del dataset BigQuery

DATASET = "rag_compras"
TABLA = "chunks_embeddings"
MODELO_EMB = "text-multilingual-embedding-002"
# ========================================================


def embed_consulta(texto):
    """Embedding de la pregunta con task_type RETRIEVAL_QUERY."""
    client = genai.Client(vertexai=True, project=PROJECT_ID, location=LOCATION)
    r = client.models.embed_content(
        model=MODELO_EMB,
        contents=[texto],
        config=EmbedContentConfig(task_type="RETRIEVAL_QUERY"),
    )
    return r.embeddings[0].values


def buscar(pregunta, k=5):
    qemb = embed_consulta(pregunta)
    bq = bigquery.Client(project=PROJECT_ID)

    sql = f"""
    SELECT
      base.documento        AS documento,
      base.articulo_num     AS articulo_num,
      base.articulo_titulo  AS articulo_titulo,
      base.texto            AS texto,
      distance
    FROM VECTOR_SEARCH(
      TABLE `{PROJECT_ID}.{DATASET}.{TABLA}`,
      'embedding',
      (SELECT @qemb AS embedding),
      top_k => @k,
      distance_type => 'COSINE'
    )
    ORDER BY distance
    """
    job_config = bigquery.QueryJobConfig(
        query_parameters=[
            bigquery.ArrayQueryParameter("qemb", "FLOAT64", qemb),
            bigquery.ScalarQueryParameter("k", "INT64", k),
        ]
    )
    return list(bq.query(sql, job_config=job_config, location=BQ_LOCATION).result())


def main():
    parser = argparse.ArgumentParser(description="Buscador semantico RAG sobre normativa de compras publicas.")
    parser.add_argument("pregunta", help="Pregunta en lenguaje natural.")
    parser.add_argument("--k", type=int, default=5, help="Numero de resultados (default 5).")
    args = parser.parse_args()

    print("=" * 70)
    print(f" PREGUNTA: {args.pregunta}")
    print("=" * 70)

    resultados = buscar(args.pregunta, args.k)
    if not resultados:
        print("Sin resultados.")
        return

    for i, fila in enumerate(resultados, start=1):
        similitud = 1 - fila["distance"]  # COSENO -> similitud aproximada
        print(f"\n[{i}] {fila['documento'][:50]}...")
        print(f"    Articulo {fila['articulo_num']}: {fila['articulo_titulo']}")
        print(f"    Relevancia (coseno): {similitud:.3f}")
        extracto = " ".join(fila["texto"].split())
        print(f"    {extracto[:280]}...")


if __name__ == "__main__":
    main()
