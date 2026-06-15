"""
indexar_bigquery.py
-------------------------------------------------------------------
FASE 4 del pipeline RAG (version SIN costo fijo): INDEXACION EN BIGQUERY.

Carga los chunks (metadatos, de 'chunks/') junto con sus vectores
(de 'embeddings/') en una tabla de BigQuery, lista para busqueda
semantica con la funcion VECTOR_SEARCH.

Carga INCREMENTAL e IDEMPOTENTE (WRITE_APPEND): antes de insertar borra los
chunks previos del corpus para los documentos del lote (clave = `documento`),
sin tocar las directivas de la Biblioteca web (chunk_id 'bib_*'). Permite cargar
por lotes (--solo <categoria>) y re-cargar un documento sin duplicar.

Ventajas frente a Vertex AI Vector Search:
  - Serverless: NO hay endpoint 24/7 -> sin costo fijo.
  - Solo pagas almacenamiento (centavos) y bytes por consulta.
  - Escala a millones de chunks.

Tabla resultante: <project>.rag_compras.chunks_embeddings
  columnas: chunk_id, categoria, documento, tipo_referencia, referencia,
            articulo_num, articulo_titulo, parte, n_chars, texto,
            fase, emisor, anio, vigente, embedding (ARRAY<FLOAT64>)

Requisitos:
    pip install google-cloud-storage google-cloud-bigquery
    Autenticacion ADC + APIs de BigQuery habilitadas.

Uso:
    python indexar_bigquery.py
-------------------------------------------------------------------
"""

import json
import argparse
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
    # Referencia GENERAL (troceo consciente del tipo): reemplaza el viejo "Art. 0".
    bigquery.SchemaField("tipo_referencia", "STRING"),   # articulo|numeral|seccion|opinion|considerando|anexo|preambulo
    bigquery.SchemaField("referencia", "STRING"),        # "45" | "5.2" | "D013-2025-OECE-DTN" | "Fundamento 7" ...
    # Espejo de transicion (compatibilidad con el esquema actual).
    bigquery.SchemaField("articulo_num", "STRING"),
    bigquery.SchemaField("articulo_titulo", "STRING"),
    bigquery.SchemaField("parte", "INTEGER"),
    bigquery.SchemaField("n_chars", "INTEGER"),
    bigquery.SchemaField("texto", "STRING"),
    # Metadatos para BUSQUEDA HIBRIDA (pre-filtering) y cita.
    bigquery.SchemaField("fase", "STRING"),        # actuaciones_preparatorias|seleccion|ejecucion_contractual|transversal
    bigquery.SchemaField("emisor", "STRING"),      # dato de cita (no filtro)
    bigquery.SchemaField("anio", "INTEGER"),       # ano de emision del documento
    bigquery.SchemaField("vigente", "BOOLEAN"),    # True = vigente; False = derogada
    bigquery.SchemaField("embedding", "FLOAT64", mode="REPEATED"),
]


def _lineas_jsonl(blob):
    """Itera registros de un JSONL. Divide SOLO por '\\n' (no splitlines()): el texto de
    los chunks puede contener separadores Unicode (\\u2028/\\u2029/\\x85, comunes en PDFs)
    que json.dumps(ensure_ascii=False) deja literales y que splitlines() partiria, rompiendo
    el registro. Una linea ilegible se SALTA con aviso (no aborta la carga)."""
    for linea in blob.download_as_text().split("\n"):
        if not linea.strip():
            continue
        try:
            yield json.loads(linea)
        except json.JSONDecodeError as e:
            print(f"   [!] linea ilegible en {blob.name} (saltada): {e}")


def cargar_metadatos(storage_client, solo=None):
    """id -> dict de metadatos (de chunks/, opcionalmente filtrado por categoria)."""
    prefijo = f"{PREFIJO_CHUNKS}/{solo}/" if solo else f"{PREFIJO_CHUNKS}/"
    meta = {}
    for blob in storage_client.list_blobs(BUCKET_NAME, prefix=prefijo):
        if not blob.name.endswith(".jsonl"):
            continue
        for c in _lineas_jsonl(blob):
            meta[c["chunk_id"]] = c
    return meta


def cargar_embeddings(storage_client, solo=None):
    """id -> vector (de embeddings/, opcionalmente filtrado por categoria)."""
    prefijo = f"{PREFIJO_EMB}/{solo}/" if solo else f"{PREFIJO_EMB}/"
    vects = {}
    for blob in storage_client.list_blobs(BUCKET_NAME, prefix=prefijo):
        if not blob.name.endswith(".json"):
            continue
        for o in _lineas_jsonl(blob):
            vects[o["id"]] = o["embedding"]
    return vects


def main():
    parser = argparse.ArgumentParser(description="Indexa chunks+embeddings en BigQuery (incremental).")
    parser.add_argument("--solo", default=None, help="Filtra por categoria (ej: directivas).")
    args = parser.parse_args()

    print("=" * 60)
    print(" FASE 4 RAG - INDEXACION EN BIGQUERY (sin costo fijo)")
    print("=" * 60)
    if args.solo:
        print(f"Filtro: solo '{args.solo}/'")

    storage_client = storage.Client(project=PROJECT_ID)
    bq = bigquery.Client(project=PROJECT_ID)

    # 1) Dataset (idempotente).
    ds_ref = bigquery.Dataset(f"{PROJECT_ID}.{DATASET}")
    ds_ref.location = LOCATION
    bq.create_dataset(ds_ref, exists_ok=True)
    print(f"[OK] Dataset listo: {PROJECT_ID}.{DATASET} ({LOCATION})")

    # 2) Unir metadatos + vectores por id (acotado por --solo si se indica).
    print("-> Cargando chunks y embeddings desde GCS...")
    meta = cargar_metadatos(storage_client, args.solo)
    vects = cargar_embeddings(storage_client, args.solo)
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
            "tipo_referencia": c.get("tipo_referencia"),   # troceo consciente del tipo
            "referencia": c.get("referencia"),
            "articulo_num": c.get("articulo_num"),          # espejo de transicion
            "articulo_titulo": c.get("articulo_titulo"),
            "parte": c.get("parte"),
            "n_chars": c.get("n_chars"),
            "texto": c.get("texto"),
            "fase": c.get("fase"),                 # pre-filtering por fase
            "emisor": c.get("emisor"),             # dato de cita
            "anio": c.get("anio"),                 # metadato para pre-filtering
            "vigente": c.get("vigente", True),     # por defecto: vigente
            "embedding": vects[cid],
        })
    if sin_vector:
        print(f"   [!] {sin_vector} chunks sin vector (omitidos).")

    if not filas:
        print("   [!] No hay chunks para cargar. Nada que hacer.")
        return

    tabla_id = f"{PROJECT_ID}.{DATASET}.{TABLA}"

    # 3a) DEDUP idempotente: borra los chunks PREVIOS del corpus para los documentos
    #     de este lote (clave = documento), SIN tocar las directivas de la Biblioteca web
    #     (chunk_id 'bib_*'). Asi re-cargar un documento lo reemplaza limpio, sin duplicar.
    docs_lote = sorted({f["documento"] for f in filas if f.get("documento")})
    if docs_lote:
        del_sql = (f"DELETE FROM `{tabla_id}` "
                   f"WHERE documento IN UNNEST(@docs) AND NOT STARTS_WITH(chunk_id, 'bib_')")
        del_cfg = bigquery.QueryJobConfig(query_parameters=[
            bigquery.ArrayQueryParameter("docs", "STRING", docs_lote)])
        print(f"-> Dedup: borrando chunks previos de {len(docs_lote)} documento(s) del corpus...")
        bq.query(del_sql, job_config=del_cfg, location=LOCATION).result()

    # 3b) Cargar (WRITE_APPEND -> carga incremental por lotes, no borra lo anterior).
    job_config = bigquery.LoadJobConfig(
        schema=SCHEMA,
        write_disposition=bigquery.WriteDisposition.WRITE_APPEND,
    )
    print(f"-> Cargando {len(filas)} filas (APPEND) en {tabla_id} ...")
    job = bq.load_table_from_json(filas, tabla_id, job_config=job_config)
    job.result()  # espera

    tabla = bq.get_table(tabla_id)
    print("\n" + "=" * 60)
    print(f" [OK] Lote cargado: {len(filas)} filas | {len(docs_lote)} documento(s) reemplazado(s).")
    print(f"      Tabla {tabla_id}: {tabla.num_rows} filas | {tabla.num_bytes/1024/1024:.2f} MB")
    print("      Carga INCREMENTAL idempotente (WRITE_APPEND + dedup por documento).")
    print("=" * 60)


if __name__ == "__main__":
    main()
