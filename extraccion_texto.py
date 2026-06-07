"""
extraccion_texto.py
-------------------------------------------------------------------
FASE 1 del pipeline RAG: EXTRACCION DE TEXTO.

Recorre los documentos (.pdf / .docx) ya almacenados en el bucket
de GCS, extrae su texto plano y lo guarda de vuelta en el mismo
bucket bajo un prefijo paralelo 'texto_extraido/', espejando la
estructura original.

  origen :  leyes_y_reglamentos/ley.pdf
  destino:  texto_extraido/leyes_y_reglamentos/ley.pdf.txt

Si un PDF esta ESCANEADO (texto nativo casi nulo), aplica un
FALLBACK de OCR con Google Cloud Vision: renderiza cada pagina a
imagen con PyMuPDF (sin ejecutables externos) y extrae el texto.

Requisitos:
    pip install google-cloud-storage pypdf python-docx pymupdf google-cloud-vision
    Autenticacion ADC ya configurada + Vision API habilitada.

Uso:
    python extraccion_texto.py
    python extraccion_texto.py --solo leyes_y_reglamentos   # filtra un prefijo
-------------------------------------------------------------------
"""

import io
import argparse
from google.cloud import storage
from pypdf import PdfReader
import docx

# ===================== CONFIGURACION =====================
PROJECT_ID = "project-a0134db0-3990-4ec2-bc3"
BUCKET_NAME = "repositorio-compras-publicas"

# Prefijo donde se guarda el texto extraido (no se reprocesa a si mismo).
PREFIJO_SALIDA = "texto_extraido"

# Umbral: si un PDF arroja menos de estos caracteres por pagina (promedio),
# probablemente es un escaneo y se activa el fallback de OCR.
MIN_CHARS_POR_PAGINA = 50

# Resolucion (DPI) para rasterizar paginas antes del OCR. Mas alto = mas
# preciso pero mas pesado; 200-300 es un buen rango para documentos.
OCR_DPI = 220
# Idiomas sugeridos a Vision (mejora la precision del OCR).
OCR_IDIOMAS = ["es"]
# Maximo de imagenes por request a Vision (limite del API = 16).
OCR_LOTE = 16
# ========================================================


def extraer_pdf(data_bytes):
    """Devuelve (texto, num_paginas) de un PDF en memoria."""
    reader = PdfReader(io.BytesIO(data_bytes))
    partes = []
    for pagina in reader.pages:
        txt = pagina.extract_text() or ""
        partes.append(txt)
    return "\n".join(partes), len(reader.pages)


def extraer_docx(data_bytes):
    """Devuelve (texto, num_parrafos) de un .docx en memoria."""
    documento = docx.Document(io.BytesIO(data_bytes))
    parrafos = [p.text for p in documento.paragraphs if p.text.strip()]
    # Tambien extrae texto de tablas (frecuente en directivas/resoluciones).
    for tabla in documento.tables:
        for fila in tabla.rows:
            celdas = [c.text.strip() for c in fila.cells if c.text.strip()]
            if celdas:
                parrafos.append(" | ".join(celdas))
    return "\n".join(parrafos), len(parrafos)


def ocr_pdf_vision(data_bytes):
    """
    FALLBACK OCR: rasteriza cada pagina del PDF con PyMuPDF y aplica
    Google Cloud Vision (DOCUMENT_TEXT_DETECTION). Todo EN MEMORIA, sin
    ejecutables externos ni archivos temporales en disco.
    Devuelve (texto, num_paginas).
    """
    import fitz  # PyMuPDF
    from google.cloud import vision

    doc = fitz.open(stream=data_bytes, filetype="pdf")
    # Rasterizar todas las paginas a PNG.
    paginas_png = []
    for pagina in doc:
        pix = pagina.get_pixmap(dpi=OCR_DPI)
        paginas_png.append(pix.tobytes("png"))
    doc.close()

    client = vision.ImageAnnotatorClient()
    feature = vision.Feature(type_=vision.Feature.Type.DOCUMENT_TEXT_DETECTION)
    contexto = vision.ImageContext(language_hints=OCR_IDIOMAS)

    textos = []
    for i in range(0, len(paginas_png), OCR_LOTE):
        lote = paginas_png[i:i + OCR_LOTE]
        peticiones = [
            vision.AnnotateImageRequest(
                image=vision.Image(content=png), features=[feature], image_context=contexto
            )
            for png in lote
        ]
        respuesta = client.batch_annotate_images(requests=peticiones)
        for r in respuesta.responses:
            if r.error.message:
                textos.append("")  # pagina con error: se omite su texto
            else:
                textos.append(r.full_text_annotation.text or "")

    return "\n".join(textos), len(paginas_png)


def extraer_pdf_inteligente(data_bytes):
    """
    Extrae texto de un PDF con FALLBACK automatico a OCR.
      1. Intenta extraccion nativa (texto digital, rapida y gratuita).
      2. Si el resultado es vacio o demasiado corto (PDF escaneado),
         aplica OCR con Cloud Vision.
    Devuelve (texto, num_paginas, metodo) con metodo in {"nativo", "ocr"}.
    """
    texto, n_pag = extraer_pdf(data_bytes)
    suficiente = n_pag > 0 and (len(texto.strip()) / n_pag) >= MIN_CHARS_POR_PAGINA
    if suficiente:
        return texto, n_pag, "nativo"

    # Fallback OCR.
    texto_ocr, n_ocr = ocr_pdf_vision(data_bytes)
    return texto_ocr, n_ocr, "ocr"


# ============================================================================
#  MOTORES DE EXTRACCION ADICIONALES (uso del modulo web app.py)
#  Imagenes -> OCR Vision | Excel/CSV -> Tabla Markdown (mejor para el LLM)
# ============================================================================

MAX_FILAS_TABLA = 2000   # tope de filas por hoja para no inflar el contexto del LLM


def ocr_imagen_vision(data_bytes):
    """OCR de una imagen (png/jpg/jpeg) con Google Cloud Vision
    (DOCUMENT_TEXT_DETECTION). Extrae texto impreso, numeros de serie, marcas o
    notas tecnicas. Devuelve texto plano ('' si la imagen no contiene texto)."""
    from google.cloud import vision
    client = vision.ImageAnnotatorClient()
    feature = vision.Feature(type_=vision.Feature.Type.DOCUMENT_TEXT_DETECTION)
    contexto = vision.ImageContext(language_hints=OCR_IDIOMAS)
    req = vision.AnnotateImageRequest(
        image=vision.Image(content=data_bytes), features=[feature], image_context=contexto)
    resp = client.batch_annotate_images(requests=[req]).responses[0]
    if resp.error.message:
        raise RuntimeError(f"Cloud Vision: {resp.error.message}")
    return resp.full_text_annotation.text or ""


def _matriz_a_markdown(filas, titulo=None):
    """Convierte una matriz (lista de filas; cada fila lista de celdas) en una
    TABLA MARKDOWN. Los LLM procesan vectores de celdas con mucha mayor precision
    bajo esta estructura. La 1a fila no vacia se usa como encabezado."""
    filas = [f for f in filas if any((c is not None and str(c).strip() != "") for c in f)]
    if not filas:
        return ""
    truncado = len(filas) > MAX_FILAS_TABLA
    if truncado:
        filas = filas[:MAX_FILAS_TABLA]
    ancho = max(len(f) for f in filas)

    def celda(v):
        s = "" if v is None else str(v)
        return s.replace("|", "\\|").replace("\r", " ").replace("\n", " ").strip()

    norm = [[celda(f[i]) if i < len(f) else "" for i in range(ancho)] for f in filas]
    lineas = []
    if titulo:
        lineas.append(f"### {titulo}")
    lineas.append("| " + " | ".join(norm[0]) + " |")
    lineas.append("| " + " | ".join(["---"] * ancho) + " |")
    for r in norm[1:]:
        lineas.append("| " + " | ".join(r) + " |")
    if truncado:
        lineas.append(f"_(tabla truncada a {MAX_FILAS_TABLA} filas)_")
    return "\n".join(lineas)


def extraer_xlsx(data_bytes):
    """Lee un .xlsx (openpyxl) y devuelve (texto Markdown de todas las hojas, n_filas)."""
    import openpyxl
    wb = openpyxl.load_workbook(io.BytesIO(data_bytes), read_only=True, data_only=True)
    bloques, total = [], 0
    try:
        for ws in wb.worksheets:
            filas = [list(fila) for fila in ws.iter_rows(values_only=True)]
            md = _matriz_a_markdown(filas, titulo=f"Hoja: {ws.title}")
            if md:
                bloques.append(md)
                total += len(filas)
    finally:
        wb.close()
    return "\n\n".join(bloques), total


def extraer_xls(data_bytes):
    """Lee un .xls (Excel 97-2003, xlrd) y devuelve (texto Markdown, n_filas)."""
    import xlrd
    libro = xlrd.open_workbook(file_contents=data_bytes)
    bloques, total = [], 0
    for hoja in libro.sheets():
        filas = [hoja.row_values(r) for r in range(hoja.nrows)]
        md = _matriz_a_markdown(filas, titulo=f"Hoja: {hoja.name}")
        if md:
            bloques.append(md)
            total += hoja.nrows
    return "\n\n".join(bloques), total


def extraer_csv(data_bytes):
    """Lee un .csv (stdlib csv, sin pandas) y devuelve (texto Markdown, n_filas).
    Detecta codificacion (utf-8/latin-1) y delimitador (coma, ; , tab, |)."""
    import csv
    try:
        texto = data_bytes.decode("utf-8-sig")
    except UnicodeDecodeError:
        texto = data_bytes.decode("latin-1", errors="replace")
    try:
        dialecto = csv.Sniffer().sniff(texto[:4096], delimiters=",;\t|")
    except csv.Error:
        dialecto = csv.excel
    filas = list(csv.reader(io.StringIO(texto), dialecto))
    return _matriz_a_markdown(filas, titulo=None), len(filas)


def main():
    parser = argparse.ArgumentParser(description="Extraccion de texto de PDFs/DOCX en GCS.")
    parser.add_argument("--solo", default=None,
                        help="Procesa solo los documentos bajo este prefijo (ej: leyes_y_reglamentos).")
    parser.add_argument("--habilitar-ocr-masivo", dest="habilitar_ocr", action="store_true",
                        help="ACTIVA el OCR (Cloud Vision) para PDFs escaneados durante la carga masiva. "
                             "DESACTIVADO por defecto para proteger el presupuesto: el OCR tiene costo "
                             "(~US$1.50 por 1000 paginas). Sin esta bandera, los escaneos solo se reportan.")
    args = parser.parse_args()

    print("=" * 60)
    print(" FASE 1 RAG - EXTRACCION DE TEXTO")
    print("=" * 60)
    print(f"Bucket: gs://{BUCKET_NAME}")
    if args.solo:
        print(f"Filtro: solo '{args.solo}/'")
    # Estado del OCR masivo (por defecto DESACTIVADO -> proteccion de presupuesto).
    if args.habilitar_ocr:
        print("OCR masivo: ACTIVADO (se aplicara Cloud Vision a PDFs escaneados; genera costo).")
    else:
        print("OCR masivo: DESACTIVADO (por defecto). Los PDFs escaneados solo se reportaran.")
        print("            Para activarlo: python extraccion_texto.py --habilitar-ocr-masivo")
    print()

    client = storage.Client(project=PROJECT_ID)
    bucket = client.bucket(BUCKET_NAME)

    prefijo_filtro = f"{args.solo}/" if args.solo else None
    procesados = 0
    saltados = 0
    candidatos_ocr = []

    for blob in client.list_blobs(BUCKET_NAME, prefix=prefijo_filtro):
        nombre = blob.name

        # Ignora marcadores de carpeta y la propia salida de texto.
        if nombre.endswith("/") or nombre.startswith(f"{PREFIJO_SALIDA}/"):
            continue

        lower = nombre.lower()
        if not (lower.endswith(".pdf") or lower.endswith(".docx")):
            saltados += 1
            continue

        print(f"-> Extrayendo: {nombre}")
        data = blob.download_as_bytes()

        try:
            if lower.endswith(".pdf"):
                tipo, etiqueta = "pdf", "paginas"
                if args.habilitar_ocr:
                    # OCR habilitado: extraccion nativa con fallback automatico a Cloud Vision.
                    texto, unidades, metodo = extraer_pdf_inteligente(data)
                    if metodo == "ocr":
                        print(f"   [OCR] PDF escaneado: texto recuperado con Cloud Vision.")
                else:
                    # OCR DESACTIVADO (default): solo extraccion nativa, sin costo.
                    texto, unidades = extraer_pdf(data)
            else:
                texto, unidades = extraer_docx(data)
                tipo, etiqueta = "docx", "parrafos"
        except Exception as e:
            print(f"   [X] Error al extraer: {e}")
            saltados += 1
            continue

        n_chars = len(texto)

        # Heuristica de PDF escaneado: si el OCR esta DESACTIVADO, solo se reporta
        # como candidato (no se incurre en costo). Si estuviera activado, ya se OCR-eo arriba.
        if (not args.habilitar_ocr) and tipo == "pdf" and unidades > 0 \
                and (n_chars / unidades) < MIN_CHARS_POR_PAGINA:
            print(f"   [!] Texto casi nulo ({n_chars} chars / {unidades} pag). "
                  f"Probable ESCANEO -> reejecuta con --habilitar-ocr-masivo para aplicar OCR.")
            candidatos_ocr.append(nombre)

        # Sube el texto extraido al prefijo paralelo.
        destino = f"{PREFIJO_SALIDA}/{nombre}.txt"
        bucket.blob(destino).upload_from_string(texto, content_type="text/plain; charset=utf-8")
        print(f"   [OK] {n_chars:,} chars ({unidades} {etiqueta})  ->  gs://{BUCKET_NAME}/{destino}")
        procesados += 1

    print("\n" + "=" * 60)
    print(f" FINALIZADO. Procesados: {procesados} | Saltados: {saltados}")
    if candidatos_ocr:
        print(f" Candidatos a OCR ({len(candidatos_ocr)}) [OCR desactivado, no procesados]:")
        for c in candidatos_ocr:
            print(f"   - {c}")
        print(" Para procesarlos con OCR: python extraccion_texto.py --habilitar-ocr-masivo")
    print("=" * 60)


if __name__ == "__main__":
    main()
