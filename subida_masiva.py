"""
subida_masiva.py
-------------------------------------------------------------------
Script de CARGA MASIVA multi-formato hacia el Data Lake en GCS.

Escanea carpetas locales de tu computadora y sube OBLIGATORIAMENTE
archivos .pdf y .docx (Word) a las 5 rutas normativas del bucket.

  >> NO se ejecuta automaticamente. Ejecutalo cuando ya tengas
     tus documentos organizados localmente.

Requisitos:
    pip install google-cloud-storage
    Autenticacion previa (ADC):
        gcloud auth application-default login

COMO USARLO
-----------
1) Edita la variable BUCKET_NAME con el nombre EXACTO que devolvio
   'crear_datalake.py' (puede tener un sufijo numerico).

2) Edita el diccionario MAPEO_CARPETAS para que cada ruta normativa
   apunte a la carpeta local donde tienes esos documentos.
   - La CLAVE   = prefijo (carpeta) dentro del bucket.
   - El VALOR   = ruta de la carpeta en tu PC.

3) Ejecuta:
       python subida_masiva.py
   (agrega --dry-run para simular sin subir nada)

OCR MASIVO (opcional, con costo):
   Por defecto el OCR esta DESACTIVADO para proteger el presupuesto.
   Para procesar PDFs escaneados con Cloud Vision al subir, usa la bandera:
       python subida_masiva.py --habilitar-ocr-masivo
   (requiere ademas: pip install pymupdf google-cloud-vision + Vision API)

El escaneo es RECURSIVO: tambien revisa subcarpetas.
-------------------------------------------------------------------
"""

import os
import sys
import argparse
from google.cloud import storage

# ===================== CONFIGURACION =====================
PROJECT_ID = "project-a0134db0-3990-4ec2-bc3"

# IMPORTANTE: reemplaza por el nombre EXACTO devuelto por crear_datalake.py
# (si el nombre base estaba tomado, tendra un sufijo, ej: repositorio-compras-publicas-1)
BUCKET_NAME = "repositorio-compras-publicas"

# Formatos OBLIGATORIOS a detectar y subir.
EXTENSIONES_PERMITIDAS = (".pdf", ".docx")

# Mapeo: prefijo en el bucket -> carpeta local en tu PC.
# Ajusta las rutas locales a tu estructura real.
MAPEO_CARPETAS = {
    "leyes_y_reglamentos":     r"C:\Users\Usuario\Desktop\LGCP\leyes_y_reglamentos",
    "directivas":              r"C:\Users\Usuario\documentos\directivas",
    "opiniones":               r"C:\Users\Usuario\documentos\opiniones",
    "resoluciones_tribunal":   r"C:\Users\Usuario\documentos\resoluciones_tribunal",
    "documentos_orientacion":  r"C:\Users\Usuario\documentos\documentos_orientacion",
}
# ========================================================


def es_archivo_valido(nombre_archivo):
    """True si el archivo es .pdf o .docx (case-insensitive)."""
    return nombre_archivo.lower().endswith(EXTENSIONES_PERMITIDAS)


def subir_carpeta(bucket, prefijo, carpeta_local, dry_run=False, habilitar_ocr=False):
    """
    Escanea recursivamente 'carpeta_local' y sube los .pdf/.docx
    al prefijo correspondiente del bucket. Conserva subcarpetas.

    Si habilitar_ocr=True, los PDFs ESCANEADOS se procesan con OCR
    (Cloud Vision) al momento de subir y su texto se guarda en
    'texto_extraido/'. Por defecto (False) NO se aplica OCR (sin costo).

    Devuelve (subidos, omitidos, ocr_aplicados).
    """
    subidos = 0
    omitidos = 0
    ocr_aplicados = 0

    if not os.path.isdir(carpeta_local):
        print(f"   [AVISO] Carpeta local no encontrada, se omite: {carpeta_local}")
        return subidos, omitidos, ocr_aplicados

    for raiz, _dirs, archivos in os.walk(carpeta_local):
        for archivo in archivos:
            ruta_local = os.path.join(raiz, archivo)

            if not es_archivo_valido(archivo):
                omitidos += 1
                continue

            # Ruta relativa para preservar la estructura de subcarpetas dentro del prefijo.
            rel = os.path.relpath(ruta_local, carpeta_local).replace("\\", "/")
            blob_name = f"{prefijo}/{rel}"

            if dry_run:
                print(f"   [DRY-RUN] subiria: {ruta_local}  ->  gs://{bucket.name}/{blob_name}")
                subidos += 1
                continue

            blob = bucket.blob(blob_name)
            blob.upload_from_filename(ruta_local)
            print(f"   [OK] {archivo}  ->  gs://{bucket.name}/{blob_name}")
            subidos += 1

            # OCR opcional para PDFs escaneados (solo si la bandera esta activa).
            if habilitar_ocr and archivo.lower().endswith(".pdf"):
                ocr_aplicados += _ocr_si_escaneado(bucket, ruta_local, blob_name)

    return subidos, omitidos, ocr_aplicados


def _ocr_si_escaneado(bucket, ruta_local, blob_name):
    """
    Si el PDF esta escaneado, lo OCR-ea (Cloud Vision) y guarda el texto
    en 'texto_extraido/<blob_name>.txt'. Devuelve 1 si aplico OCR, 0 si no.
    Import perezoso: solo carga pymupdf/vision cuando el OCR esta activo.
    """
    from extraccion_texto import extraer_pdf_inteligente
    try:
        with open(ruta_local, "rb") as fh:
            data = fh.read()
        texto, _paginas, metodo = extraer_pdf_inteligente(data)
        if metodo == "ocr":
            destino = f"texto_extraido/{blob_name}.txt"
            bucket.blob(destino).upload_from_string(
                texto, content_type="text/plain; charset=utf-8")
            print(f"      [OCR] escaneo detectado -> texto guardado en gs://{bucket.name}/{destino}")
            return 1
    except Exception as e:
        print(f"      [X] OCR fallido para {blob_name}: {e}")
    return 0


def main():
    parser = argparse.ArgumentParser(description="Carga masiva multi-formato (.pdf/.docx) a GCS.")
    parser.add_argument("--dry-run", action="store_true",
                        help="Simula la subida sin transferir archivos.")
    parser.add_argument("--habilitar-ocr-masivo", dest="habilitar_ocr", action="store_true",
                        help="ACTIVA el OCR (Cloud Vision) para PDFs escaneados durante la carga. "
                             "DESACTIVADO por defecto para proteger el presupuesto: el OCR tiene costo "
                             "(~US$1.50 por 1000 paginas). Sin esta bandera solo se suben los archivos.")
    args = parser.parse_args()

    print("=" * 60)
    print(" CARGA MASIVA MULTI-FORMATO (.pdf / .docx)")
    print("=" * 60)
    print(f"Proyecto : {PROJECT_ID}")
    print(f"Bucket   : gs://{BUCKET_NAME}")
    if args.dry_run:
        print("MODO     : DRY-RUN (no se subira nada)")
    # Estado del OCR masivo (por defecto DESACTIVADO -> proteccion de presupuesto).
    if args.habilitar_ocr:
        print("OCR masivo: ACTIVADO (PDFs escaneados se procesaran con Cloud Vision; genera costo).")
    else:
        print("OCR masivo: DESACTIVADO (por defecto). Solo se suben archivos, sin OCR.")
        print("            Para activarlo: python subida_masiva.py --habilitar-ocr-masivo")
    print()

    client = storage.Client(project=PROJECT_ID)
    bucket = client.bucket(BUCKET_NAME)

    # Verificacion temprana de que el bucket existe / es accesible.
    if not args.dry_run and not bucket.exists():
        print(f"[ERROR] El bucket '{BUCKET_NAME}' no existe o no tienes acceso.")
        print("        Ejecuta primero crear_datalake.py y verifica BUCKET_NAME.")
        sys.exit(1)

    total_subidos = 0
    total_omitidos = 0
    total_ocr = 0

    for prefijo, carpeta_local in MAPEO_CARPETAS.items():
        print(f"-> Procesando '{prefijo}' desde: {carpeta_local}")
        s, o, oc = subir_carpeta(bucket, prefijo, carpeta_local,
                                 dry_run=args.dry_run, habilitar_ocr=args.habilitar_ocr)
        total_subidos += s
        total_omitidos += o
        total_ocr += oc
        print(f"   Resumen: {s} subidos / {o} omitidos (formato no permitido)\n")

    print("=" * 60)
    print(f" PROCESO FINALIZADO. Total subidos: {total_subidos} | omitidos: {total_omitidos}")
    if args.habilitar_ocr:
        print(f" PDFs escaneados procesados con OCR: {total_ocr}")
    print("=" * 60)


if __name__ == "__main__":
    main()
