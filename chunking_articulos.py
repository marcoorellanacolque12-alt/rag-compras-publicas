"""
chunking_articulos.py
-------------------------------------------------------------------
FASE 2 del pipeline RAG: CHUNKING POR ARTICULO.

Lee los textos planos del prefijo 'texto_extraido/' en GCS, los
divide en fragmentos (chunks) usando como frontera natural cada
"Articulo N", y guarda el resultado como JSONL en el prefijo
'chunks/'.

- Cada articulo = 1 chunk semanticamente completo y citable.
- Si un articulo excede MAX_CHARS, se sub-divide en partes con
  solapamiento (overlap) para no perder contexto en los bordes.
- El texto previo al "Articulo 1" se guarda como 'preambulo'.

Cada linea del JSONL es un chunk con metadatos:
    chunk_id, categoria, documento, source_blob,
    articulo_num, articulo_titulo, parte, total_partes,
    n_chars, texto

Requisitos:
    pip install google-cloud-storage
    Autenticacion ADC ya configurada.

Uso:
    python chunking_articulos.py
    python chunking_articulos.py --solo leyes_y_reglamentos
-------------------------------------------------------------------
"""

import re
import json
import argparse
from google.cloud import storage

# ===================== CONFIGURACION =====================
PROJECT_ID = "project-a0134db0-3990-4ec2-bc3"
BUCKET_NAME = "repositorio-compras-publicas"

PREFIJO_ENTRADA = "texto_extraido"
PREFIJO_SALIDA = "chunks"

# Tamano maximo de un chunk (en caracteres). ~4000 chars ~= 1000 tokens,
# comodo para los modelos de embedding (limite ~2048 tokens) y para
# mantener buena precision de recuperacion.
MAX_CHARS = 4000
# Solapamiento entre sub-chunks de un mismo articulo largo.
OVERLAP_CHARS = 300

# Encabezado de articulo: captura (numero, resto_de_linea=titulo).
RE_ARTICULO = re.compile(
    r'(?im)^[ \t]*art[ií]culo[ \t]+(\d+)[ \t]*[\.\-°ºª)]*[ \t]*(.*)$'
)
# ========================================================


def limpiar(texto):
    """Normaliza espacios/saltos excesivos sin alterar el contenido."""
    texto = texto.replace("\r\n", "\n")
    texto = re.sub(r'[ \t]+', ' ', texto)
    texto = re.sub(r'\n{3,}', '\n\n', texto)
    return texto.strip()


def subdividir(texto, max_chars=MAX_CHARS, overlap=OVERLAP_CHARS):
    """
    Divide un texto largo en partes <= max_chars, intentando cortar en
    un fin de oracion/parrafo cercano y aplicando solapamiento.
    """
    if len(texto) <= max_chars:
        return [texto]

    partes = []
    inicio = 0
    n = len(texto)
    while inicio < n:
        fin = min(inicio + max_chars, n)
        if fin < n:
            # Busca un corte limpio (salto de parrafo, punto) hacia atras.
            ventana = texto[inicio:fin]
            corte = max(ventana.rfind("\n\n"), ventana.rfind(". "), ventana.rfind("\n"))
            if corte > max_chars * 0.5:  # solo si el corte no es demasiado prematuro
                fin = inicio + corte + 1
        partes.append(texto[inicio:fin].strip())
        if fin >= n:
            break
        inicio = max(fin - overlap, inicio + 1)
    return [p for p in partes if p]


def chunkear_documento(texto, categoria, documento, source_blob):
    """Devuelve lista de dicts (chunks) para un documento."""
    texto = limpiar(texto)
    chunks = []

    matches = list(RE_ARTICULO.finditer(texto))

    # Preambulo: todo lo anterior al primer articulo.
    if matches:
        preambulo = texto[:matches[0].start()].strip()
    else:
        preambulo = texto  # documento sin articulos detectados

    if preambulo:
        for i, parte in enumerate(subdividir(preambulo), start=1):
            chunks.append(_mk_chunk(categoria, documento, source_blob,
                                    "0", "Preambulo / encabezado",
                                    parte, i, None))

    # Un bloque por articulo (desde su encabezado hasta el siguiente).
    for idx, m in enumerate(matches):
        num = m.group(1)
        titulo = m.group(2).strip()
        ini = m.start()
        fin = matches[idx + 1].start() if idx + 1 < len(matches) else len(texto)
        cuerpo = texto[ini:fin].strip()

        sub = subdividir(cuerpo)
        total = len(sub)
        for i, parte in enumerate(sub, start=1):
            chunks.append(_mk_chunk(categoria, documento, source_blob,
                                    num, titulo, parte, i, total))

    # Asigna un chunk_id GLOBALMENTE UNICO por documento (indice secuencial).
    # Evita colisiones cuando un mismo numero de articulo aparece varias veces
    # (p.ej. en el indice/tabla de contenido y luego en el articulado real).
    for seq, ch in enumerate(chunks, start=1):
        ch["chunk_id"] = f"{documento}__c{seq:04d}__art{ch['articulo_num']}_p{ch['parte']}"

    return chunks


def _mk_chunk(categoria, documento, source_blob, art_num, art_titulo,
              texto, parte, total_partes):
    chunk_id = f"{documento}__art{art_num}__p{parte}"
    return {
        "chunk_id": chunk_id,
        "categoria": categoria,
        "documento": documento,
        "source_blob": source_blob,
        "articulo_num": art_num,
        "articulo_titulo": art_titulo[:200],
        "parte": parte,
        "total_partes": total_partes,
        "n_chars": len(texto),
        "texto": texto,
    }


def nombre_documento(blob_name):
    """Extrae un nombre de documento legible desde la ruta del blob."""
    base = blob_name.split("/")[-1]
    for suf in (".pdf.txt", ".docx.txt", ".txt"):
        if base.endswith(suf):
            base = base[: -len(suf)]
            break
    return base


def main():
    parser = argparse.ArgumentParser(description="Chunking por articulo de textos legales en GCS.")
    parser.add_argument("--solo", default=None, help="Filtra por categoria (ej: leyes_y_reglamentos).")
    args = parser.parse_args()

    print("=" * 60)
    print(" FASE 2 RAG - CHUNKING POR ARTICULO")
    print("=" * 60)
    print(f"Bucket: gs://{BUCKET_NAME} | MAX_CHARS={MAX_CHARS} overlap={OVERLAP_CHARS}\n")

    client = storage.Client(project=PROJECT_ID)
    bucket = client.bucket(BUCKET_NAME)

    # Prefijo de entrada (opcionalmente filtrado por categoria).
    prefijo = f"{PREFIJO_ENTRADA}/{args.solo}/" if args.solo else f"{PREFIJO_ENTRADA}/"

    total_chunks = 0
    total_docs = 0

    for blob in client.list_blobs(BUCKET_NAME, prefix=prefijo):
        if blob.name.endswith("/") or not blob.name.endswith(".txt"):
            continue

        # categoria = primer segmento despues de 'texto_extraido/'
        partes_ruta = blob.name.split("/")
        categoria = partes_ruta[1] if len(partes_ruta) > 2 else "sin_categoria"
        documento = nombre_documento(blob.name)

        print(f"-> Chunking: {documento}  (categoria: {categoria})")
        texto = blob.download_as_text()
        chunks = chunkear_documento(texto, categoria, documento, blob.name)

        # Escribe un JSONL por documento en chunks/<categoria>/<documento>.jsonl
        salida = f"{PREFIJO_SALIDA}/{categoria}/{documento}.jsonl"
        contenido = "\n".join(json.dumps(c, ensure_ascii=False) for c in chunks)
        bucket.blob(salida).upload_from_string(
            contenido, content_type="application/x-ndjson; charset=utf-8"
        )
        print(f"   [OK] {len(chunks)} chunks  ->  gs://{BUCKET_NAME}/{salida}")
        total_chunks += len(chunks)
        total_docs += 1

    print("\n" + "=" * 60)
    print(f" FINALIZADO. Documentos: {total_docs} | Chunks generados: {total_chunks}")
    print("=" * 60)


if __name__ == "__main__":
    main()
