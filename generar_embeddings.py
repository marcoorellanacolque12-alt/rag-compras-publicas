"""
generar_embeddings.py
-------------------------------------------------------------------
FASE 3 del pipeline RAG: GENERACION DE EMBEDDINGS.

Lee los chunks (JSONL) del prefijo 'chunks/' en GCS, genera el
vector (embedding) de cada chunk con Vertex AI usando el SDK
moderno 'google-genai', y guarda el resultado en el prefijo
'embeddings/' con el FORMATO que espera Vertex AI Vector Search:

    {"id": <chunk_id>, "embedding": [768 floats],
     "restricts": [{"namespace": "categoria", "allow": [...]},
                   {"namespace": "documento", "allow": [...]}],
     "crowding_tag": <documento>}

Modelo: text-multilingual-embedding-002 (768 dims, fuerte en espanol).

Requisitos:
    pip install google-cloud-storage google-genai
    Autenticacion ADC + API de Vertex AI habilitada.

Uso:
    python generar_embeddings.py
    python generar_embeddings.py --solo leyes_y_reglamentos
-------------------------------------------------------------------
"""

import json
import time
import argparse
from google.cloud import storage
from google import genai
from google.genai.types import EmbedContentConfig

# ===================== CONFIGURACION =====================
PROJECT_ID = "project-a0134db0-3990-4ec2-bc3"
LOCATION = "us-central1"
BUCKET_NAME = "repositorio-compras-publicas"

PREFIJO_ENTRADA = "chunks"
PREFIJO_SALIDA = "embeddings"

MODELO = "text-multilingual-embedding-002"   # 768 dims, multilingue (espanol)
TASK_TYPE = "RETRIEVAL_DOCUMENT"             # tarea: indexar documentos para busqueda

# Tamano de lote por llamada (limites de Vertex: <=250 instancias y
# ~20.000 tokens por request). 16 es seguro para chunks de ~1000 tokens.
BATCH_SIZE = 16
MAX_REINTENTOS = 4
# ========================================================


def embed_lote(client, textos):
    """Genera embeddings para una lista de textos, con reintentos."""
    for intento in range(MAX_REINTENTOS):
        try:
            resp = client.models.embed_content(
                model=MODELO,
                contents=textos,
                config=EmbedContentConfig(task_type=TASK_TYPE),
            )
            return [e.values for e in resp.embeddings]
        except Exception as e:
            espera = 2 ** intento
            print(f"   [!] Error en lote (intento {intento+1}/{MAX_REINTENTOS}): {e}. "
                  f"Reintentando en {espera}s...")
            time.sleep(espera)
    raise RuntimeError("Lote fallido tras varios reintentos.")


def main():
    parser = argparse.ArgumentParser(description="Genera embeddings de los chunks en GCS.")
    parser.add_argument("--solo", default=None, help="Filtra por categoria (ej: leyes_y_reglamentos).")
    args = parser.parse_args()

    print("=" * 60)
    print(" FASE 3 RAG - GENERACION DE EMBEDDINGS")
    print("=" * 60)
    print(f"Modelo: {MODELO} | Bucket: gs://{BUCKET_NAME} | batch={BATCH_SIZE}\n")

    storage_client = storage.Client(project=PROJECT_ID)
    genai_client = genai.Client(vertexai=True, project=PROJECT_ID, location=LOCATION)

    prefijo = f"{PREFIJO_ENTRADA}/{args.solo}/" if args.solo else f"{PREFIJO_ENTRADA}/"

    total_vectores = 0
    total_docs = 0

    for blob in storage_client.list_blobs(BUCKET_NAME, prefix=prefijo):
        if not blob.name.endswith(".jsonl"):
            continue

        chunks = [json.loads(l) for l in blob.download_as_text().splitlines() if l.strip()]
        if not chunks:
            continue

        documento = chunks[0]["documento"]
        categoria = chunks[0]["categoria"]
        print(f"-> Embeddings: {documento} ({len(chunks)} chunks)")

        lineas_salida = []
        for i in range(0, len(chunks), BATCH_SIZE):
            lote = chunks[i:i + BATCH_SIZE]
            vectores = embed_lote(genai_client, [c["texto"] for c in lote])

            for c, vec in zip(lote, vectores):
                lineas_salida.append(json.dumps({
                    "id": c["chunk_id"],
                    "embedding": vec,
                    "restricts": [
                        {"namespace": "categoria", "allow": [c["categoria"]]},
                        {"namespace": "documento", "allow": [c["documento"]]},
                    ],
                    "crowding_tag": c["documento"],
                }))
            print(f"   ... {min(i + BATCH_SIZE, len(chunks))}/{len(chunks)}")

        salida = f"{PREFIJO_SALIDA}/{categoria}/{documento}.json"
        storage_client.bucket(BUCKET_NAME).blob(salida).upload_from_string(
            "\n".join(lineas_salida), content_type="application/json"
        )
        print(f"   [OK] {len(lineas_salida)} vectores  ->  gs://{BUCKET_NAME}/{salida}")
        total_vectores += len(lineas_salida)
        total_docs += 1

    print("\n" + "=" * 60)
    print(f" FINALIZADO. Documentos: {total_docs} | Vectores: {total_vectores}")
    print(f" Dimensiones por vector: 768 (modelo {MODELO})")
    print("=" * 60)


if __name__ == "__main__":
    main()
