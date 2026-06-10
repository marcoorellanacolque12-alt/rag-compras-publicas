"""
chunking_articulos.py
-------------------------------------------------------------------
FASE 2 del pipeline RAG: CHUNKING CONSCIENTE DEL TIPO.

Lee los textos planos de 'texto_extraido/<categoria>/' en GCS y los
trocea SEGUN LA CATEGORIA del documento (la carpeta), generando una
referencia natural por tipo (en vez del antiguo "Art. 0"):

  leyes_y_reglamentos   -> por Articulo        (tipo_referencia=articulo)
  directivas            -> por numeral/disposicion (numeral)
  documentos_orientacion-> por seccion / tamaño (seccion)
  opiniones             -> por tamaño, ref = N° de opinion (opinion)
  resoluciones_tribunal -> por considerando    (considerando)
  (fallback)            -> por tamaño con solape, referencia nula

Cada chunk lleva metadatos: categoria, fase, emisor, vigente,
tipo_referencia, referencia (+ espejo de transicion articulo_num/
articulo_titulo para compatibilidad), parte, total_partes, n_chars,
texto, source_blob, chunk_id.

Anexos/formatos: se identifican y se CONSERVAN como chunks propios.

Requisitos:
    pip install google-cloud-storage
    Autenticacion ADC.

Uso:
    python chunking_articulos.py
    python chunking_articulos.py --solo directivas
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

MAX_CHARS = 4000          # tope por chunk (~1000 tokens)
OVERLAP_CHARS = 300       # solape entre sub-partes de un bloque largo

CATEGORIAS_VALIDAS = {
    "leyes_y_reglamentos", "directivas", "documentos_orientacion",
    "opiniones", "resoluciones_tribunal",
}

# ---- Fronteras de troceo por tipo ----
RE_ARTICULO = re.compile(r'(?im)^[ \t]*art[ií]culo[ \t]+(\d+)[ \t]*[\.\-°ºª)]*[ \t]*(.*)$')
# Numeral de directiva: "5", "5.2", "5.2.1" al inicio de linea seguido de un titulo.
RE_NUMERAL = re.compile(r'(?im)^[ \t]*(\d{1,2}(?:\.\d{1,2}){0,3})[ \t]*[\.\)]?[ \t]+([A-ZÁÉÍÓÚÑ].{2,150})$')
RE_DISPOSICION = re.compile(r'(?im)^[ \t]*(DISPOSICI[OÓ]N[ \t]+[A-ZÁÉÍÓÚÑ].{0,120})$')
# Considerandos / parte resolutiva de una resolucion.
_ORDINALES = ("primero","segundo","tercero","cuarto","quinto","sexto","s[eé]ptimo","octavo",
              "noveno","d[eé]cimo","und[eé]cimo","duod[eé]cimo","decimotercero","decimocuarto",
              "decimoquinto","decimosexto","decimos[eé]ptimo","decimoctavo","decimonoveno","vig[eé]simo")
RE_CONSIDERANDO = re.compile(r'(?im)^[ \t]*(' + "|".join(_ORDINALES) + r')[ \t]*[\.\-:]')
RE_RESUELVE = re.compile(r'(?im)^[ \t]*(SE\s+RESUELVE|RESUELVE|SE\s+ACUERDA)[ \t]*[:\.]?[ \t]*$')
# Resoluciones del TCP: secciones romanas + puntos numerados.
RE_SEC_RES = re.compile(r'(?im)^[ \t]*(?:[IVX]+\.[ \t]*)?(ANTECEDENTES|FUNDAMENTACI[OÓ]N|AN[AÁ]LISIS|CONSIDERANDO|SE\s+RESUELVE|RESUELVE|POR\s+ESTOS\s+FUNDAMENTOS|LA\s+SALA\s+RESUELVE)\b')
RE_PUNTO_NUM = re.compile(r'(?im)^[ \t]*(\d{1,3})\.[ \t]+\S')
_ABREV_RES = {
    "ANTECEDENTES": "Antecedente", "FUNDAMENTACION": "Fundamento",
    "ANALISIS": "Análisis", "CONSIDERANDO": "Considerando",
    "RESUELVE": "Resuelve", "SE RESUELVE": "Resuelve",
    "POR ESTOS FUNDAMENTOS": "Resuelve", "LA SALA RESUELVE": "Resuelve",
}
# Secciones de guias/bases.
RE_SECCION = re.compile(r'(?im)^[ \t]*(t[ií]tulo|cap[ií]tulo|secci[oó]n)[ \t]+[\wIVXÁÉÍÓÚÑ].{0,140}$')
# Anexos / formatos.
RE_ANEXO = re.compile(r'(?im)^[ \t]*(anexo|formato)[ \t]*(?:n[°º]?\s*)?([\w\.\-]+)?[ \t]*[:\-]?[ \t]*(.*)$')
# N° de opinion (ej. D000013-2025-OECE-DTN) o "Opinion N° ...".
RE_OPINION = re.compile(r'(?i)\b([A-Z]?\d{2,}-\d{4}-OECE-DTN)\b')
RE_OPINION_ALT = re.compile(r'(?i)opini[oó]n[ \t]+n[°º]?[ \t]*([A-Za-z0-9\-\/]+)')
# Titulos del Reglamento que nombran la fase (mas robustos que los cortes numericos).
RE_FASE_TITULO = [
    (re.compile(r'(?im)^.{0,40}actuaciones?\s+preparatorias'), "actuaciones_preparatorias"),
    (re.compile(r'(?im)^.{0,40}(procedimientos?\s+de\s+selecci[oó]n|m[eé]todos?\s+de\s+contrataci[oó]n)'), "seleccion"),
    (re.compile(r'(?im)^.{0,40}ejecuci[oó]n\s+(contractual|del\s+contrato)'), "ejecucion_contractual"),
    (re.compile(r'(?im)^.{0,40}disposiciones\s+(generales|preliminares|complementarias)'), "transversal"),
]

# ---- Mapa de fase del documento entero (subcadena del nombre -> fase) ----
FASE_POR_DOC = {
    "pac": "actuaciones_preparatorias",
    "ficha": "actuaciones_preparatorias",
    "actuaciones-preparatorias": "actuaciones_preparatorias",
    "bases-estandar": "seleccion",
    "bases_estandar": "seleccion",
    "directiva-005": "seleccion",
    "pladicop": "seleccion",
    "arbitraje": "ejecucion_contractual",
    "jprd": "ejecucion_contractual",
    "junta-de-prevencion": "ejecucion_contractual",
    "prevencion-y-resolucion-de-disputas": "ejecucion_contractual",
}
FASE_DEFAULT_POR_CAT = {
    "resoluciones_tribunal": "seleccion",
}
# ========================================================


# ============================ UTILIDADES ============================
def limpiar(texto):
    texto = texto.replace("\r\n", "\n")
    texto = re.sub(r'[ \t]+', ' ', texto)
    texto = re.sub(r'\n{3,}', '\n\n', texto)
    return texto.strip()


def subdividir(texto, max_chars=MAX_CHARS, overlap=OVERLAP_CHARS):
    """Divide un bloque largo en partes <= max_chars con corte limpio + solape."""
    if len(texto) <= max_chars:
        return [texto] if texto.strip() else []
    partes, inicio, n = [], 0, len(texto)
    while inicio < n:
        fin = min(inicio + max_chars, n)
        if fin < n:
            ventana = texto[inicio:fin]
            corte = max(ventana.rfind("\n\n"), ventana.rfind(". "), ventana.rfind("\n"))
            if corte > max_chars * 0.5:
                fin = inicio + corte + 1
        parte = texto[inicio:fin].strip()
        if parte:
            partes.append(parte)
        if fin >= n:
            break
        inicio = max(fin - overlap, inicio + 1)
    return partes


def nombre_documento(blob_name):
    base = blob_name.split("/")[-1]
    for suf in (".pdf.txt", ".docx.txt", ".txt"):
        if base.endswith(suf):
            base = base[:-len(suf)]
            break
    return base


# ============================ EMISOR Y FASE ============================
def detectar_emisor(documento, texto, categoria=None):
    """Emisor (solo dato de cita). DIRIGIDO POR CATEGORIA (la carpeta es la señal mas
    fiable) y luego por CODIGO/PRIMERA LINEA. No se escanea el cuerpo: una norma que
    MENCIONA a OECE/TCP/otra Ley causaba falsos positivos."""
    dn = documento.lower()
    ini = texto[:120].upper()                       # primera linea = emisor real
    up = (documento + " " + texto[:600]).upper()    # encabezado: para codigos especificos

    if categoria == "leyes_y_reglamentos":
        # Textos del marco general: distinguir su emisor real (solo dato de cita).
        if "constitucion" in dn or "CONSTITUCIÓN POLÍTICA" in up or "CONSTITUCION POLITICA" in up:
            return "Congreso Constituyente"
        if "codigo-civil" in dn or "codigo civil" in dn or "CÓDIGO CIVIL" in ini or "CODIGO CIVIL" in ini:
            return "Poder Ejecutivo (Decreto Legislativo)"
        if dn.startswith("tuo") or "TEXTO ÚNICO ORDENADO" in up or "TEXTO UNICO ORDENADO" in up:
            return "Poder Ejecutivo (TUO)"
        if re.match(r'^ds[ _\-]?\d', dn) or "-ef" in dn or "_ef" in dn or ini.lstrip().startswith("DECRETO SUPREMO"):
            return "MEF/Ejecutivo"                  # Decreto Supremo (por nombre/1a linea: no escanear cuerpo)
        if re.match(r'^dl[ _\-]?\d', dn) or ini.lstrip().startswith("DECRETO LEGISLATIVO"):
            return "Poder Ejecutivo (Decreto Legislativo)"
        if "ley-general" in dn or (ini.lstrip().startswith("LEY ") and "REGLAMENTO" not in ini[:40]):
            return "Congreso de la República"
        return "MEF/Ejecutivo"                      # Reglamento de contrataciones / Decreto Supremo
    if categoria == "resoluciones_tribunal":
        return "TCP"
    if categoria == "opiniones":
        return "OECE-DTN" if "OECE-DTN" in up else "OECE"

    # directivas / documentos_orientacion / desconocida -> por codigo o primera linea
    if "EF/54.01" in up:
        return "DGA-MEF"
    if "OECE-DTN" in up:
        return "OECE-DTN"
    if "OECE-CD" in up or "OECE" in ini:
        return "OECE"
    if "PERU COMPRAS" in ini or "PERÚ COMPRAS" in ini:
        return "Perú Compras"
    if "SALA PLENA" in ini or "TRIBUNAL DE CONTRATAC" in ini:
        return "TCP"
    return "(emisor no identificado)"


def es_reglamento(documento, categoria=None):
    """True solo para el Reglamento de contrataciones. Restringido a la categoria
    'leyes_y_reglamentos': asi una directiva con 'reglamento' en el nombre (p.ej.
    'reglamento-interno') NUNCA se trata como el Reglamento (fase por articulo)."""
    if categoria is not None and categoria != "leyes_y_reglamentos":
        return False
    d = documento.lower()
    return "reglamento" in d and "ley-general" not in d


def fase_documento(categoria, documento):
    """Fase del documento entero (para todo lo que NO es el Reglamento)."""
    dn = documento.lower()
    for clave, fase in FASE_POR_DOC.items():
        if clave in dn:
            return fase
    if categoria in FASE_DEFAULT_POR_CAT:
        return FASE_DEFAULT_POR_CAT[categoria]
    return "transversal"


def fase_por_articulo(num):
    """Cortes por numero de articulo del Reglamento."""
    try:
        n = int(num)
    except (TypeError, ValueError):
        return "transversal"
    if n < 41:
        return "transversal"
    if n <= 61:
        return "actuaciones_preparatorias"
    if n <= 103:
        return "seleccion"
    return "ejecucion_contractual"


def escanear_titulos_fase(texto):
    """Posiciones de los titulos del Reglamento que nombran una fase."""
    marcas = []
    for rx, fase in RE_FASE_TITULO:
        for m in rx.finditer(texto):
            marcas.append((m.start(), fase))
    marcas.sort()
    return marcas


def fase_articulo_con_titulos(num, pos, titulos):
    """Prefiere el ultimo titulo de fase anterior al articulo; si no hay, usa los cortes."""
    fase = None
    for p, f in titulos:
        if p <= pos:
            fase = f
        else:
            break
    return fase or fase_por_articulo(num)


# ============================ SEPARACION DE ANEXOS ============================
def separar_anexos(texto):
    """Separa el cuerpo principal de los anexos/formatos. Devuelve (cuerpo, [bloques_anexo])."""
    matches = list(RE_ANEXO.finditer(texto))
    # Solo consideramos anexos en la segunda mitad del documento (evita falsos positivos).
    matches = [m for m in matches if m.start() > len(texto) * 0.4]
    if not matches:
        return texto, []
    primer = matches[0].start()
    cuerpo = texto[:primer].strip()
    anexos = []
    for idx, m in enumerate(matches):
        ini = m.start()
        fin = matches[idx + 1].start() if idx + 1 < len(matches) else len(texto)
        etiqueta = (m.group(1) or "Anexo").strip().capitalize()
        num = (m.group(2) or "").strip()
        ref = f"{etiqueta} {num}".strip()
        anexos.append({
            "tipo_referencia": "anexo",
            "referencia": ref or None,
            "titulo": ref,
            "cuerpo": texto[ini:fin].strip(),
        })
    return cuerpo, anexos


# ============================ TROCEADORES POR TIPO ============================
def _bloques_por_frontera(texto, matches, ref_de_match, titulo_de_match, tipo, etiqueta_pre):
    """Helper: arma bloques desde encabezados (matches) + preambulo previo al 1ro."""
    bloques = []
    pre = texto[:matches[0].start()].strip() if matches else texto.strip()
    if pre:
        bloques.append({"tipo_referencia": "preambulo", "referencia": None,
                        "titulo": etiqueta_pre, "cuerpo": pre, "_pos": 0})
    for idx, m in enumerate(matches):
        ini = m.start()
        fin = matches[idx + 1].start() if idx + 1 < len(matches) else len(texto)
        bloques.append({
            "tipo_referencia": tipo,
            "referencia": ref_de_match(m),
            "titulo": titulo_de_match(m),
            "cuerpo": texto[ini:fin].strip(),
            "_pos": ini,
        })
    return bloques


def trocear_por_articulo(texto):
    ms = list(RE_ARTICULO.finditer(texto))
    if not ms:
        return trocear_por_tamano(texto)
    return _bloques_por_frontera(
        texto, ms,
        ref_de_match=lambda m: m.group(1),
        titulo_de_match=lambda m: (m.group(2) or "").strip()[:200],
        tipo="articulo", etiqueta_pre="Preámbulo / encabezado")


def trocear_por_numeral(texto):
    ms = list(RE_NUMERAL.finditer(texto)) + list(RE_DISPOSICION.finditer(texto))
    ms.sort(key=lambda m: m.start())
    if not ms:
        return trocear_por_tamano(texto)

    def ref(m):
        g1 = m.group(1) or ""
        return g1 if re.match(r'^\d', g1) else None   # disposicion -> sin numero

    def tit(m):
        try:
            return (m.group(2) or "").strip()[:200]
        except IndexError:
            return (m.group(1) or "").strip()[:200]
    return _bloques_por_frontera(texto, ms, ref, tit, "numeral", "Preámbulo / objeto")


def trocear_por_considerando(texto):
    """Resoluciones del TCP: secciones (Antecedentes/Fundamentación/Resuelve) + puntos
    numerados; tambien ordinales (Primero.-). Referencia consciente de la seccion."""
    fronteras = []
    for m in RE_SEC_RES.finditer(texto):
        fronteras.append((m.start(), "sec", m.group(1).upper()))
    for m in RE_PUNTO_NUM.finditer(texto):
        fronteras.append((m.start(), "num", m.group(1)))
    for m in RE_CONSIDERANDO.finditer(texto):           # ordinales (Primero.-, Segundo.-)
        fronteras.append((m.start(), "ord", m.group(1)))
    fronteras.sort(key=lambda x: x[0])
    dedup = []
    for f in fronteras:
        if not (dedup and f[0] == dedup[-1][0]):
            dedup.append(f)
    fronteras = dedup
    if not fronteras:
        return trocear_por_tamano(texto)

    bloques = []
    pre = texto[:fronteras[0][0]].strip()
    if pre:
        bloques.append({"tipo_referencia": "preambulo", "referencia": None,
                        "titulo": "Vistos / encabezado", "cuerpo": pre})
    seccion = None
    for i, (pos, kind, val) in enumerate(fronteras):
        fin = fronteras[i + 1][0] if i + 1 < len(fronteras) else len(texto)
        cuerpo = texto[pos:fin].strip()
        if kind == "sec":
            seccion = val
            if len(cuerpo) <= 120:          # solo encabezado: fija la seccion, no emite chunk
                continue
            ref = _ABREV_RES.get(val.replace("Ó", "O").replace("Á", "A"), val.title())
            bloques.append({"tipo_referencia": "considerando", "referencia": ref, "titulo": ref, "cuerpo": cuerpo})
        elif kind == "ord":
            bloques.append({"tipo_referencia": "considerando", "referencia": val.capitalize(),
                            "titulo": f"Considerando {val.lower()}", "cuerpo": cuerpo})
        else:  # num
            etq = _ABREV_RES.get((seccion or "").replace("Ó", "O").replace("Á", "A"), "Considerando")
            ref = f"{etq} {val}"
            bloques.append({"tipo_referencia": "considerando", "referencia": ref, "titulo": ref, "cuerpo": cuerpo})
    return bloques or trocear_por_tamano(texto)


def trocear_opinion(texto, documento):
    """Una opinion es un dictamen: trocea por tamaño y aplica su N° de opinion a todo.
    El numero se toma del NOMBRE DE ARCHIVO PRIMERO (identifica al documento propio y es
    fiable); NO del cuerpo, que puede CITAR otras opiniones (causaba mis-atribucion, p.ej.
    una opinion etiquetada con el N° de otra que mencionaba). Se normaliza a MAYUSCULAS
    para un formato uniforme."""
    m = RE_OPINION.search(documento) or RE_OPINION.search(texto)
    if m:
        num = m.group(1).upper()
    else:
        ma = RE_OPINION_ALT.search(texto)
        num = ma.group(1) if ma else None
    bloques = []
    for parte in subdividir(texto):
        bloques.append({"tipo_referencia": "opinion", "referencia": num,
                        "titulo": f"Opinión {num}" if num else "Opinión", "cuerpo": parte})
    return bloques or [{"tipo_referencia": "opinion", "referencia": num,
                        "titulo": "Opinión", "cuerpo": texto}]


def trocear_por_seccion(texto):
    ms = list(RE_SECCION.finditer(texto))
    if not ms:
        # sin estructura clara -> por tamaño con solape, referencia nula
        return trocear_por_tamano(texto)
    return _bloques_por_frontera(
        texto, ms,
        ref_de_match=lambda m: None,
        titulo_de_match=lambda m: m.group(0).strip()[:200],
        tipo="seccion", etiqueta_pre="Introducción")


def trocear_por_tamano(texto):
    """Fallback: por tamaño con solape, SIN referencia (nunca 'Art. 0')."""
    return [{"tipo_referencia": "seccion", "referencia": None, "titulo": "", "cuerpo": parte}
            for parte in subdividir(texto)]


# ============================ ORQUESTADOR ============================
def _titulo_por_tipo(tipo, ref):
    if tipo == "articulo":
        return f"Artículo {ref}" if ref else "Artículo"
    if tipo == "numeral":
        return f"Numeral {ref}" if ref else "Disposición"
    if tipo == "opinion":
        return f"Opinión {ref}" if ref else "Opinión"
    if tipo == "considerando":
        return f"Considerando {ref}" if ref else "Parte resolutiva"
    if tipo == "anexo":
        return ref or "Anexo"
    if tipo == "preambulo":
        return "Preámbulo"
    return "Sección"


def _mk_chunk(categoria, documento, source_blob, emisor, fase, vigente,
              tipo_referencia, referencia, titulo, texto, parte, total_partes):
    ref = referencia if referencia not in (None, "") else None
    titulo_final = (titulo or _titulo_por_tipo(tipo_referencia, ref))[:200]
    return {
        "chunk_id": None,                       # se asigna al final
        "categoria": categoria,
        "documento": documento,
        "source_blob": source_blob,
        "fase": fase,
        "emisor": emisor,
        "vigente": bool(vigente),
        "tipo_referencia": tipo_referencia,
        "referencia": ref,
        # --- espejo de transicion (compatibilidad con el esquema actual) ---
        "articulo_num": ref if ref is not None else "",
        "articulo_titulo": titulo_final,
        # -------------------------------------------------------------------
        "parte": parte,
        "total_partes": total_partes,
        "n_chars": len(texto),
        "texto": texto,
    }


def _emitir(bloques, categoria, documento, source_blob, emisor, fase_doc, vigente):
    chunks = []
    for b in bloques:
        sub = subdividir(b["cuerpo"])
        total = len(sub)
        fase = b.get("fase") or fase_doc
        for i, parte in enumerate(sub, start=1):
            chunks.append(_mk_chunk(categoria, documento, source_blob, emisor, fase, vigente,
                                    b["tipo_referencia"], b.get("referencia"), b.get("titulo", ""),
                                    parte, i, total))
    return chunks


def chunkear_documento(texto, categoria, documento, source_blob, vigente=True):
    """Trocea un documento segun su categoria y devuelve la lista de chunks con metadata."""
    texto = limpiar(texto)
    emisor = detectar_emisor(documento, texto, categoria)
    fase_doc = fase_documento(categoria, documento)

    cuerpo, anexos = separar_anexos(texto)

    if categoria == "leyes_y_reglamentos":
        bloques = trocear_por_articulo(cuerpo)
        if es_reglamento(documento, categoria):
            # Fase por CORTES NUMERICOS de articulo (deterministas). Se evito la
            # "preferencia por titulo" porque el indice/TOC lista todos los titulos al
            # inicio y contamina la fase de los articulos tempranos.
            for b in bloques:
                b["fase"] = fase_por_articulo(b.get("referencia")) if b["tipo_referencia"] == "articulo" else "transversal"
    elif categoria == "directivas":
        bloques = trocear_por_numeral(cuerpo)
    elif categoria == "resoluciones_tribunal":
        bloques = trocear_por_considerando(cuerpo)
    elif categoria == "opiniones":
        bloques = trocear_opinion(cuerpo, documento)
    elif categoria == "documentos_orientacion":
        bloques = trocear_por_seccion(cuerpo)
    else:
        bloques = trocear_por_tamano(cuerpo)

    chunks = _emitir(bloques, categoria, documento, source_blob, emisor, fase_doc, vigente)
    chunks += _emitir(anexos, categoria, documento, source_blob, emisor, fase_doc, vigente)

    # chunk_id GLOBALMENTE UNICO por documento (indice secuencial + tipo/referencia).
    for seq, ch in enumerate(chunks, start=1):
        tag = re.sub(r'[^0-9A-Za-z.]+', '', str(ch["referencia"] or ch["tipo_referencia"]))[:24]
        ch["chunk_id"] = f"{documento}__c{seq:04d}__{ch['tipo_referencia']}_{tag}_p{ch['parte']}"
    return chunks


# ============================ MAIN (GCS) ============================
def main():
    parser = argparse.ArgumentParser(description="Chunking consciente del tipo (GCS).")
    parser.add_argument("--solo", default=None, help="Filtra por categoria (ej: directivas).")
    args = parser.parse_args()

    print("=" * 60)
    print(" FASE 2 RAG - CHUNKING CONSCIENTE DEL TIPO")
    print("=" * 60)
    print(f"Bucket: gs://{BUCKET_NAME} | MAX_CHARS={MAX_CHARS} overlap={OVERLAP_CHARS}\n")

    client = storage.Client(project=PROJECT_ID)
    bucket = client.bucket(BUCKET_NAME)
    prefijo = f"{PREFIJO_ENTRADA}/{args.solo}/" if args.solo else f"{PREFIJO_ENTRADA}/"

    total_chunks = total_docs = 0
    for blob in client.list_blobs(BUCKET_NAME, prefix=prefijo):
        if blob.name.endswith("/") or not blob.name.endswith(".txt"):
            continue
        partes_ruta = blob.name.split("/")
        categoria = partes_ruta[1] if len(partes_ruta) > 2 else "sin_categoria"
        documento = nombre_documento(blob.name)

        print(f"-> Chunking: {documento}  (categoria: {categoria})")
        texto = blob.download_as_text()
        chunks = chunkear_documento(texto, categoria, documento, blob.name)

        salida = f"{PREFIJO_SALIDA}/{categoria}/{documento}.jsonl"
        contenido = "\n".join(json.dumps(c, ensure_ascii=False) for c in chunks)
        bucket.blob(salida).upload_from_string(
            contenido, content_type="application/x-ndjson; charset=utf-8")
        print(f"   [OK] {len(chunks)} chunks  ->  gs://{BUCKET_NAME}/{salida}")
        total_chunks += len(chunks)
        total_docs += 1

    print("\n" + "=" * 60)
    print(f" FINALIZADO. Documentos: {total_docs} | Chunks: {total_chunks}")
    print("=" * 60)


if __name__ == "__main__":
    main()
