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

INCREMENTAL: salta los documentos ya embebidos (mismo conjunto de chunk_id en
'embeddings/'), para cargar por categoria y retomar lotes interrumpidos sin
re-pagar ni duplicar. Re-embebe solo lo nuevo o lo re-chunkeado. Con --rehacer
fuerza re-embeber todo.

Uso:
    python generar_embeddings.py
    python generar_embeddings.py --solo leyes_y_reglamentos
    python generar_embeddings.py --solo directivas --rehacer
-------------------------------------------------------------------
"""

import json
import time
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from google.cloud import storage
from google import genai
from google.genai.types import EmbedContentConfig
# Config CENTRAL de embeddings (mismo modelo/dims que la CONSULTA en responder.py).
from config_emb import MODELO_EMB, OUTPUT_DIM, recortar, normalizar

# ===================== CONFIGURACION =====================
PROJECT_ID = "project-a0134db0-3990-4ec2-bc3"
LOCATION = "us-central1"
BUCKET_NAME = "repositorio-compras-publicas"

PREFIJO_ENTRADA = "chunks"
PREFIJO_SALIDA = "embeddings"

MODELO = MODELO_EMB                           # gemini-embedding-001 (fuente unica)
TASK_TYPE = "RETRIEVAL_DOCUMENT"             # tarea: indexar documentos para busqueda

# gemini-embedding-001: 1 texto por request (batch efectivo = 1).
BATCH_SIZE = 1
MAX_REINTENTOS = 4
# ========================================================


def _excede_tokens(e):
    """True si el error es por exceder el limite de tokens por request (no transitorio)."""
    s = str(e).lower()
    return ("token count" in s or "reduce the input" in s
            or ("invalid_argument" in s and "token" in s))


def embed_lote(client, textos):
    """Genera embeddings para una lista de textos. Si el lote excede el limite de tokens
    por request, lo PARTE recursivamente en mitades; otros errores -> reintentos."""
    for intento in range(MAX_REINTENTOS):
        try:
            resp = client.models.embed_content(
                model=MODELO,
                contents=[recortar(t) for t in textos],
                config=EmbedContentConfig(task_type=TASK_TYPE, output_dimensionality=OUTPUT_DIM),
            )
            return [normalizar(e.values) for e in resp.embeddings]
        except Exception as e:
            # Lote demasiado grande en tokens -> dividir y reintentar por mitades.
            if len(textos) > 1 and _excede_tokens(e):
                mid = len(textos) // 2
                print(f"   [i] Lote de {len(textos)} excede el limite de tokens; "
                      f"dividiendo en {mid}+{len(textos)-mid}...")
                return embed_lote(client, textos[:mid]) + embed_lote(client, textos[mid:])
            espera = 2 ** intento
            print(f"   [!] Error en lote (intento {intento+1}/{MAX_REINTENTOS}): {e}. "
                  f"Reintentando en {espera}s...")
            time.sleep(espera)
    raise RuntimeError("Lote fallido tras varios reintentos.")


def _procesar_blob(storage_client, genai_client, blob_name, existentes, rehacer):
    """Procesa UN documento (descarga chunks, salto incremental, embebe, sube).
    Thread-safe: los clientes de storage/genai admiten uso concurrente; embed_lote
    conserva el backoff. Devuelve ('skip'|'ok', n_vectores)."""
    bucket = storage_client.bucket(BUCKET_NAME)
    chunks = [json.loads(l) for l in bucket.blob(blob_name).download_as_text().splitlines()
              if l.strip()]
    if not chunks:
        return ("skip", 0)

    documento = chunks[0]["documento"]
    categoria = chunks[0]["categoria"]
    salida = f"{PREFIJO_SALIDA}/{categoria}/{documento}.json"

    # Salto incremental: si ya existe el embedding con EXACTAMENTE los mismos chunk_id,
    # esta completo -> no re-procesar. Si difieren (re-chunkeo) o falta -> (re)generar.
    if not rehacer and salida in existentes:
        ids_in = {c["chunk_id"] for c in chunks}
        try:
            ids_out = {json.loads(l)["id"]
                       for l in existentes[salida].download_as_text().splitlines() if l.strip()}
        except Exception:
            ids_out = set()
        if ids_out == ids_in:
            print(f"-> SKIP (ya embebido, {len(chunks)} chunks): {documento}", flush=True)
            return ("skip", 0)
        print(f"-> RE-EMBED (cambiaron los chunks): {documento} ({len(chunks)} chunks)", flush=True)
    else:
        print(f"-> Embeddings: {documento} ({len(chunks)} chunks)", flush=True)

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

    bucket.blob(salida).upload_from_string(
        "\n".join(lineas_salida), content_type="application/json")
    print(f"   [OK] {len(lineas_salida)} vectores  ->  {documento}", flush=True)
    return ("ok", len(lineas_salida))


def main():
    parser = argparse.ArgumentParser(description="Genera embeddings de los chunks en GCS.")
    parser.add_argument("--solo", default=None, help="Filtra por categoria (ej: leyes_y_reglamentos).")
    parser.add_argument("--rehacer", action="store_true",
                        help="Re-embebe TODO aunque ya exista (ignora el salto incremental).")
    parser.add_argument("--workers", type=int, default=1,
                        help="Documentos en paralelo (default 1 = secuencial). El backoff "
                             "absorbe 429 si se toca el techo de cuota.")
    args = parser.parse_args()

    print("=" * 60)
    print(" FASE 3 RAG - GENERACION DE EMBEDDINGS")
    print("=" * 60)
    print(f"Modelo: {MODELO} | Bucket: gs://{BUCKET_NAME} | batch={BATCH_SIZE} | workers={args.workers}\n", flush=True)

    storage_client = storage.Client(project=PROJECT_ID)
    genai_client = genai.Client(vertexai=True, project=PROJECT_ID, location=LOCATION)

    sub = f"{args.solo}/" if args.solo else ""
    prefijo = f"{PREFIJO_ENTRADA}/{sub}"

    # INCREMENTAL: pre-listar los embeddings ya generados (1 sola llamada) para saltar
    # los documentos que ya estan completos y NO re-pagar.
    existentes = {b.name: b for b in storage_client.list_blobs(BUCKET_NAME, prefix=f"{PREFIJO_SALIDA}/{sub}")
                  if b.name.endswith(".json")}
    if args.rehacer:
        print("[modo --rehacer] se re-embebera todo, ignorando lo existente.\n", flush=True)

    nombres = [b.name for b in storage_client.list_blobs(BUCKET_NAME, prefix=prefijo)
               if b.name.endswith(".jsonl")]
    print(f"Documentos a revisar: {len(nombres)}\n", flush=True)

    total_vectores = total_docs = omitidos = errores = 0
    hechos = 0
    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as pool:
        futs = {pool.submit(_procesar_blob, storage_client, genai_client, n,
                            existentes, args.rehacer): n for n in nombres}
        for fut in as_completed(futs):
            hechos += 1
            try:
                estado, nvec = fut.result()
            except Exception as e:
                errores += 1
                print(f"   [ERROR] {futs[fut]}: {e}", flush=True)
                continue
            if estado == "ok":
                total_docs += 1
                total_vectores += nvec
            else:
                omitidos += 1
            if hechos % 200 == 0:
                print(f"   == progreso: {hechos}/{len(nombres)} docs "
                      f"(ok={total_docs} skip={omitidos} err={errores}) ==", flush=True)
    if errores:
        print(f"\n[!] {errores} documento(s) con error: relanzar el script los reintenta "
              f"(incremental).", flush=True)

    print("\n" + "=" * 60)
    print(f" FINALIZADO. Embebidos: {total_docs} doc(s) | Omitidos (ya listos): {omitidos} | "
          f"Vectores nuevos: {total_vectores}")
    print(f" Dimensiones por vector: {OUTPUT_DIM} (modelo {MODELO})")
    print("=" * 60)


if __name__ == "__main__":
    main()
