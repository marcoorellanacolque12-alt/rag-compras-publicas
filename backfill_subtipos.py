# -*- coding: utf-8 -*-
"""
backfill_subtipos.py
-------------------------------------------------------------------
BACKFILL (una sola vez para las ~9.940 resoluciones ya indexadas) del metadato
`subtipo` (apelacion|sancionadora|otra) clasificando por el VISTO. La LOGICA de
clasificacion es compartida con la ingesta (chunking_articulos.clasificar_subtipo);
las resoluciones FUTURAS se etiquetan solas al chunkear.

NO re-embebe, NO toca vectores ni recuperacion. Solo escribe la columna `subtipo`.
-------------------------------------------------------------------
"""
import collections
from google.cloud import bigquery
from chunking_articulos import clasificar_subtipo, _region_visto

PROJECT_ID = "project-a0134db0-3990-4ec2-bc3"
DATASET = "rag_compras"
TABLA = "chunks_embeddings"
LOC = "US"
T = f"`{PROJECT_ID}.{DATASET}.{TABLA}`"


def main():
    bq = bigquery.Client(project=PROJECT_ID)

    # 1) Asegurar la columna (idempotente).
    bq.query(f"ALTER TABLE {T} ADD COLUMN IF NOT EXISTS subtipo STRING", location=LOC).result()

    # 2) VISTO por resolucion: primeros 2 chunks (por chunk_id) concatenados.
    sql = f"""
    SELECT documento, STRING_AGG(texto, '\\n' ORDER BY chunk_id) AS inicio
    FROM (
      SELECT documento, chunk_id, texto,
             ROW_NUMBER() OVER (PARTITION BY documento ORDER BY chunk_id) AS rn
      FROM {T} WHERE categoria = 'resoluciones_tribunal'
    ) WHERE rn <= 2
    GROUP BY documento
    """
    filas = list(bq.query(sql, location=LOC).result())
    print(f"resoluciones a clasificar: {len(filas)}")

    mapa, motivos, muestras = {}, collections.Counter(), collections.defaultdict(list)
    for r in filas:
        subtipo, motivo = clasificar_subtipo(r["inicio"])
        mapa[r["documento"]] = subtipo
        motivos[(subtipo, motivo)] += 1
        if len(muestras[subtipo]) < 6:
            frag = " ".join((_region_visto(r["inicio"]) or "(VISTO no localizable)").split())[:170]
            muestras[subtipo].append((r["documento"], frag))

    # 3) Escribir subtipo: un UPDATE por subtipo (array de documentos).
    porsub = collections.defaultdict(list)
    for doc, st in mapa.items():
        porsub[st].append(doc)
    for st, docs in porsub.items():
        bq.query(
            f"UPDATE {T} SET subtipo=@s WHERE documento IN UNNEST(@docs) AND categoria='resoluciones_tribunal'",
            job_config=bigquery.QueryJobConfig(query_parameters=[
                bigquery.ScalarQueryParameter("s", "STRING", st),
                bigquery.ArrayQueryParameter("docs", "STRING", docs)]),
            location=LOC).result()

    # 4) Reporte.
    print("\n=== (1) DISTRIBUCION ===")
    dist = collections.Counter(mapa.values())
    for st in ("apelacion", "sancionadora", "otra"):
        print(f"  {st}: {dist.get(st, 0)}")
    print("\n=== (3) DESGLOSE de 'otra' ===")
    print(f"  VISTO no localizable (no_visto): {motivos[('otra','no_visto')]}")
    print(f"  VISTO sin ninguna de las dos frases (sin_frase): {motivos[('otra','sin_frase')]}")
    print("\n=== (2) SPOT-CHECK (5 por tipo: documento + fragmento del VISTO) ===")
    for st in ("apelacion", "sancionadora", "otra"):
        print(f"  --- {st} ---")
        for doc, frag in muestras[st][:5]:
            print(f"    {doc[:46]}: {frag.encode('ascii','replace').decode()}")


if __name__ == "__main__":
    main()
