"""
config_emb.py
-------------------------------------------------------------------
Configuracion CENTRAL de embeddings (FUENTE UNICA DE VERDAD).

El embedding de CONSULTA (responder._embed) y el de DOCUMENTOS
(generar_embeddings.embed_lote y responder._embed_indexar) DEBEN usar el
MISMO modelo y las MISMAS dimensiones; si no coinciden, la busqueda
vectorial (coseno) deja de funcionar. Por eso viven aqui y todos importan
de este modulo.

gemini-embedding-001 (Vertex AI):
  - Maximo 2.048 tokens por texto.
  - 1 texto por request (batch efectivo = 1).
  - Dimensiones: 3072 (default/maxima) | 1536 (balance) | 768 (liviano).
  - NO normaliza los vectores cuando output_dimensionality != 3072 ->
    hay que normalizarlos (L2) manualmente (lo hace `normalizar`).
-------------------------------------------------------------------
"""
import math

MODELO_EMB = "gemini-embedding-001"   # actual de Vertex AI (mejor multilingue/legal)
OUTPUT_DIM = 1536                     # 3072=max calidad | 1536=balance | 768=liviano
MAX_TOKENS_EMB = 2048                 # tope de tokens por texto del modelo
MAX_CHARS_EMB = 7000                  # guardia defensiva (~<2048 tokens en espanol)


def recortar(texto):
    """Acota un texto al limite seguro de caracteres antes de embeber (evita exceder
    el tope de tokens del modelo en chunks densos)."""
    t = texto or ""
    return t if len(t) <= MAX_CHARS_EMB else t[:MAX_CHARS_EMB]


def normalizar(vec):
    """Normaliza L2 un vector. gemini-embedding-001 NO normaliza los embeddings cuando
    output_dimensionality != 3072, asi que lo hacemos aqui para que el coseno (busqueda)
    sea consistente entre consulta y documentos."""
    s = math.sqrt(sum(x * x for x in vec))
    return [x / s for x in vec] if s else list(vec)
