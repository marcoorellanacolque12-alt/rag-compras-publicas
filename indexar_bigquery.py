"""
indexar_bigquery.py
-------------------------------------------------------------------
FASE 4 del pipeline RAG (version SIN costo fijo): INDEXACION EN BIGQUERY.

Carga los chunks (metadatos, de 'chunks/') junto con sus vectores
(de 'embeddings/') en una tabla de BigQuery, lista para busqueda
semantica con la funcion VECTOR_SEARCH.

Ventajas frente a Vertex AI Vector Search:
  - Serverless: NO hay endpoint 24/7 -> sin costo fijo.
  - Solo pagas almacenamiento (centavos) y bytes por consulta.
  - Escala a millones de chunks.

Tabla resultante: <project>.rag_compras.chunks_embeddings
  columnas: chunk_id, categoria, documento, articulo_num,
            articulo_titulo, parte, n_chars, texto,
            embedding (ARRAY<FLOAT64>)

Requisitos:
    pip install google-cloud-storage google-cloud-bigquery
    Autenticacion ADC + APIs de BigQuery habilitadas.

Uso:
    python indexar_bigquery.py
-------------------------------------------------------------------
"""

import json
from google.cloud import storage, bigquery

# ===================== CONFIGURACION =====================
PROJECT_ID = "project-a0134db0-3990-4ec2-bc3"
BUCKET_NAME = "repositorio-compras-publicas"
LOCATION = "US"

DATASET = "rag_compras"
TABLA = "chunks_embeddings"

PREFIJO_CHUNKS = "chunks"
PREFIJO_EMB = "embeddings"
# ========================================================

SCHEMA = [
    bigquery.SchemaField("chunk_id", "STRING", mode="REQUIRED"),
    bigquery.SchemaField("categoria", "STRING"),
    bigquery.SchemaField("documento", "STRING"),
    bigquery.SchemaField("articulo_num", "STRING"),
    bigquery.SchemaField("articulo_titulo", "STRING"),
    bigquery.SchemaField("parte", "INTEGER"),
    bigquery.SchemaField("n_chars", "INTEGER"),
    bigquery.SchemaField("texto", "STRING"),
    bigquery.SchemaField("embedding", "FLOAT64", mode="REPEATED"),
]


def cargar_metadatos(storage_client):
    """id -> dict de metadatos (de chunks/)."""
    meta = {}
    for blob in storage_client.list_blobs(BUCKET_NAME, prefix=f"{PREFIJO_CHUNKS}/"):
        if not blob.name.endswith(".jsonl"):
            continue
        for linea in blob.download_as_text().splitlines():
            if linea.strip():
                c = json.loads(linea)
                meta[c["chunk_id"]] = c
    return meta


def cargar_embeddings(storage_client):
    """id -> vector (de embeddings/)."""
    vects = {}
    for blob in storage_client.list_blobs(BUCKET_NAME, prefix=f"{PREFIJO_EMB}/"):
        if not blob.name.endswith(".json"):
            continue
        for linea in blob.download_as_text().splitlines():
            if linea.strip():
                o = json.loads(linea)
                vects[o["id"]] = o["embedding"]
    return vects


def main():
    print("=" * 60)
    print(" FASE 4 RAG - INDEXACION EN BIGQUERY (sin costo fijo)")
    print("=" * 60)

    storage_client = storage.Client(project=PROJECT_ID)
    bq = bigquery.Client(project=PROJECT_ID)

    # 1) Dataset (idempotente).
    ds_ref = bigquery.Dataset(f"{PROJECT_ID}.{DATASET}")
    ds_ref.location = LOCATION
    bq.create_dataset(ds_ref, exists_ok=True)
    print(f"[OK] Dataset listo: {PROJECT_ID}.{DATASET} ({LOCATION})")

    # 2) Unir metadatos + vectores por id.
    print("-> Cargando chunks y embeddings desde GCS...")
    meta = cargar_metadatos(storage_client)
    vects = cargar_embeddings(storage_client)
    print(f"   chunks: {len(meta)} | vectores: {len(vects)}")

    filas = []
    sin_vector = 0
    for cid, c in meta.items():
        if cid not in vects:
            sin_vector += 1
            continue
        filas.append({
            "chunk_id": cid,
            "categoria": c.get("categoria"),
            "documento": c.get("documento"),
            "articulo_num": c.get("articulo_num"),
            "articulo_titulo": c.get("articulo_titulo"),
            "parte": c.get("parte"),
            "n_chars": c.get("n_chars"),
            "texto": c.get("texto"),
            "embedding": vects[cid],
        })
    if sin_vector:
        print(f"   [!] {sin_vector} chunks sin vector (omitidos).")

    # 3) Cargar a BigQuery (reemplaza la tabla).
    tabla_id = f"{PROJECT_ID}.{DATASET}.{TABLA}"
    job_config = bigquery.LoadJobConfig(
        schema=SCHEMA,
        write_disposition=bigquery.WriteDisposition.WRITE_TRUNCATE,
    )
    print(f"-> Cargando {len(filas)} filas en {tabla_id} ...")
    job = bq.load_table_from_json(filas, tabla_id, job_config=job_config)
    job.result()  # espera

    tabla = bq.get_table(tabla_id)
    print("\n" + "=" * 60)
    print(f" [OK] Tabla cargada: {tabla_id}")
    print(f"      Filas: {tabla.num_rows} | Tamano: {tabla.num_bytes/1024/1024:.2f} MB")
    print("      Lista para consultas con VECTOR_SEARCH (ver buscar.py).")
    print("=" * 60)


if __name__ == "__main__":
    main()
