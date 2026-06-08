"""
reset_corpus.py
-------------------------------------------------------------------
COMANDO DE RESET del vector store (para recargar el corpus desde cero).

Hace, EN ESTE ORDEN y solo tras CONFIRMACION explicita:
  1) RESPALDO:
       - copia casos.db            -> backups/casos-reset-<ts>.db
       - copia la tabla BigQuery   -> rag_compras.chunks_embeddings_bak_<ts>
  2) VACIA la tabla BigQuery chunks_embeddings (TRUNCATE).
  3) LIMPIA el registro `biblioteca` en SQLite.
  NO toca `casos` ni `fuentes` (los casos del usuario quedan intactos).

NO corre solo: pide escribir exactamente  BORRAR CORPUS  para continuar.

Uso (manual):
    python reset_corpus.py
-------------------------------------------------------------------
"""

import os
import sqlite3
import shutil
from datetime import datetime, timezone
from google.cloud import bigquery

# ===================== CONFIGURACION =====================
PROJECT_ID = "project-a0134db0-3990-4ec2-bc3"
DATASET = "rag_compras"
TABLA = "chunks_embeddings"
BQ_LOCATION = "US"

DB_PATH = "casos.db"
BACKUP_DIR = "backups"
FRASE_CONFIRMACION = "BORRAR CORPUS"
# ========================================================


def _ts():
    return datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")


def _conteos(bq):
    tabla_id = f"{PROJECT_ID}.{DATASET}.{TABLA}"
    n_bq = list(bq.query(f"SELECT COUNT(*) n FROM `{tabla_id}`", location=BQ_LOCATION).result())[0].n
    n_bib = n_casos = n_fuentes = "n/a"
    if os.path.exists(DB_PATH):
        con = sqlite3.connect(DB_PATH)
        try:
            def cnt(t):
                try:
                    return con.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0]
                except sqlite3.Error:
                    return "n/a"
            n_bib, n_casos, n_fuentes = cnt("biblioteca"), cnt("casos"), cnt("fuentes")
        finally:
            con.close()
    return n_bq, n_bib, n_casos, n_fuentes


def respaldar_casos_db(ts):
    if not os.path.exists(DB_PATH):
        print("   [i] casos.db no existe; nada que respaldar.")
        return None
    os.makedirs(BACKUP_DIR, exist_ok=True)
    destino = os.path.join(BACKUP_DIR, f"casos-reset-{ts}.db")
    shutil.copy2(DB_PATH, destino)
    print(f"   [OK] casos.db -> {destino}")
    return destino


def respaldar_bq(bq, ts):
    origen = f"{PROJECT_ID}.{DATASET}.{TABLA}"
    bak = f"{PROJECT_ID}.{DATASET}.{TABLA}_bak_{ts.replace('-', '_')}"
    bq.query(f"CREATE TABLE `{bak}` AS SELECT * FROM `{origen}`", location=BQ_LOCATION).result()
    print(f"   [OK] BigQuery {origen} -> {bak}")
    return bak


def vaciar_bq(bq):
    tabla_id = f"{PROJECT_ID}.{DATASET}.{TABLA}"
    bq.query(f"TRUNCATE TABLE `{tabla_id}`", location=BQ_LOCATION).result()
    print(f"   [OK] Vaciada {tabla_id}")


def limpiar_biblioteca_sqlite():
    if not os.path.exists(DB_PATH):
        print("   [i] casos.db no existe; no hay registro biblioteca que limpiar.")
        return
    con = sqlite3.connect(DB_PATH)
    try:
        con.execute("DELETE FROM biblioteca")    # NO toca casos ni fuentes
        con.commit()
        print("   [OK] Registro `biblioteca` limpiado (casos y fuentes intactos).")
    except sqlite3.Error as e:
        print(f"   [!] No se pudo limpiar biblioteca: {e}")
    finally:
        con.close()


def main():
    bq = bigquery.Client(project=PROJECT_ID)
    n_bq, n_bib, n_casos, n_fuentes = _conteos(bq)

    print("=" * 64)
    print(" RESET DEL CORPUS (vector store)")
    print("=" * 64)
    print("Estado actual:")
    print(f"  - BigQuery chunks_embeddings : {n_bq} filas  -> se VACIARA")
    print(f"  - SQLite  biblioteca         : {n_bib} filas  -> se LIMPIARA")
    print(f"  - SQLite  casos              : {n_casos} filas  -> INTACTO")
    print(f"  - SQLite  fuentes            : {n_fuentes} filas  -> INTACTO")
    print("\nAntes de borrar se hara un RESPALDO de casos.db y de la tabla BigQuery.")
    print(f"\nEscribe exactamente  {FRASE_CONFIRMACION}  para continuar (cualquier otra cosa cancela).")
    try:
        resp = input("> ").strip()
    except (EOFError, KeyboardInterrupt):
        print("\nCancelado (sin entrada).")
        return
    if resp != FRASE_CONFIRMACION:
        print("Cancelado. No se modifico nada.")
        return

    ts = _ts()
    print("\n[1/3] Respaldo...")
    respaldar_casos_db(ts)
    respaldar_bq(bq, ts)
    print("[2/3] Vaciando BigQuery...")
    vaciar_bq(bq)
    print("[3/3] Limpiando registro biblioteca (SQLite)...")
    limpiar_biblioteca_sqlite()

    print("\n" + "=" * 64)
    print(" RESET COMPLETO. Vector store vacio y listo para la recarga incremental")
    print(" (python indexar_bigquery.py [--solo <categoria>]).")
    print(" Respaldos en backups/ y en la tabla *_bak_* de BigQuery.")
    print("=" * 64)


if __name__ == "__main__":
    main()
