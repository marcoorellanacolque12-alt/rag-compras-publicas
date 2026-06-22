"""
eval/run_eval.py
-------------------------------------------------------------------
Harness de evaluacion REPRODUCIBLE del RAG legal de contrataciones.
Carga eval/casos.jsonl y corre el pipeline REAL segun el tipo de fallo:

  recall               -> recuperar(consultas_busqueda(pregunta), k); recall@k de las
                          refs esperadas (match flexible: doc_label contiene el doc Y la
                          referencia coincide con referencia o articulo_num).
  fidelidad_cita       -> responder() + verificar_citas(); tasa de respuestas SIN citas
                          inventadas (marcadores [N] fuera de rango).
  frontera_no_dispongo -> responder(); detecta la frase de declinacion del guardrail y
                          la compara con lo esperado (matriz de confusion).

Reproducibilidad: la generacion del eval usa temperatura 0 por defecto (--temp), SIN
tocar el default de produccion (0.2) que vive en responder.py.

Uso:
    python eval/run_eval.py
    python eval/run_eval.py --solo recall --k 10
    python eval/run_eval.py --temp 0 --juez
-------------------------------------------------------------------
"""
import os
import sys
import json
import argparse
from datetime import datetime, timezone

# Importar los modulos del proyecto (carpeta padre) sin duplicar constantes.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from responder import (recuperar, consultas_busqueda, generar_respuesta, doc_label,  # noqa: E402
                       cliente, MODELO_GEN)
from google.genai.types import GenerateContentConfig  # noqa: E402

DIR = os.path.dirname(os.path.abspath(__file__))
CASOS = os.path.join(DIR, "casos.jsonl")
REPORTES = os.path.join(DIR, "reportes")

# Fragmento ESTABLE de la frase de declinacion del guardrail (deteccion robusta a acentos).
FRASE_DECLINA = "no dispongo de la informaci"


def _declino(texto):
    return FRASE_DECLINA in (texto or "").lower()


# Filtros por DEFECTO de produccion (resoluciones del TCP apagadas salvo opt-in).
FILTROS_DEFAULT = {"categorias": [], "excluir_derogada": True, "anio": "Todos",
                   "normas": None, "incluir_apelacion": False, "incluir_sancionadoras": False}


def _ref_encontrada(filas, esperada):
    """True si alguna fila satisface la ref esperada: doc_label contiene el `doc`; si la
    `referencia` esperada esta presente, ademas debe coincidir (referencia o articulo_num).
    Sin `referencia` -> match a nivel DOCUMENTO (recall de que la norma aparezca)."""
    doc_e = (esperada.get("doc") or "").lower()
    ref_e = str(esperada.get("referencia") or "").strip()
    for f in filas:
        etiqueta = (doc_label(f.get("documento"), f.get("chunk_id")) or "").lower()
        if doc_e and doc_e not in etiqueta:
            continue
        if not ref_e:
            return True                       # recall a nivel documento
        if ref_e in {str(f.get("referencia") or ""), str(f.get("articulo_num") or "")}:
            return True
    return False


# ---------------------------- evaluadores por tipo ----------------------------
def ev_recall(caso, k, temp, juez):
    # Mide con los filtros por defecto de produccion (resoluciones apagadas).
    filas = recuperar(consultas_busqueda(caso["pregunta"]), k=k, filtros=FILTROS_DEFAULT)
    esperadas = caso.get("refs_esperadas", [])
    hits = [e for e in esperadas if _ref_encontrada(filas, e)]
    recall = len(hits) / len(esperadas) if esperadas else 0.0
    return {"recall": round(recall, 3), "encontradas": len(hits), "esperadas": len(esperadas),
            "faltantes": [e for e in esperadas if e not in hits],
            "n_filas": len(filas), "pass": recall == 1.0}


def ev_fidelidad(caso, k, temp, juez):
    # MISMA ruta que produccion: recupera en modo general y genera con generar_respuesta.
    filas = recuperar(consultas_busqueda(caso["pregunta"]), k=k, filtros=FILTROS_DEFAULT)
    r = generar_respuesta(caso["pregunta"], filas, modo="general", temperatura=temp)
    ok = not r["citas_invalidas"]   # citas calculadas sobre la salida CRUDA del modelo
    res = {"ok": ok, "citas_validas": r["citas_validas"],
           "citas_invalidas": r["citas_invalidas"], "n_filas": len(filas), "pass": ok}
    if juez:
        res["juez"] = _juez_fidelidad(caso["pregunta"], r["respuesta"], filas, r["citas_validas"])
    return res


def ev_frontera(caso, k, temp, juez):
    filas = recuperar(consultas_busqueda(caso["pregunta"]), k=k, filtros=FILTROS_DEFAULT)
    r = generar_respuesta(caso["pregunta"], filas, modo="general", temperatura=temp)
    declino = _declino(r["respuesta"])
    esperado = caso.get("esperado")
    if esperado == "debe_declinar":
        clase = "declinacion_correcta" if declino else "alucinacion"
    else:  # debe_responder
        clase = "rechazo_falso" if declino else "respuesta_correcta"
    return {"declino": declino, "esperado": esperado, "clase": clase,
            "n_filas": len(filas), "pass": clase in ("declinacion_correcta", "respuesta_correcta")}


def _juez_fidelidad(pregunta, respuesta, filas, citas_validas):
    """Juez semantico opcional (modelo flash): ¿la respuesta esta respaldada por los
    fragmentos citados? Cuesta tokens; apagado salvo --juez."""
    frags = "\n\n".join(f"[{i}] {' '.join((f.get('texto') or '').split())[:800]}"
                        for i, f in enumerate(filas, 1) if i in citas_validas)
    if not frags:
        return {"grounded": None, "motivo": "respuesta sin citas validas"}
    prompt = (
        "Eres un verificador estricto. ¿La RESPUESTA esta ENTERAMENTE respaldada por los "
        "FRAGMENTOS citados (sin afirmaciones no sustentadas)? Responde SOLO un JSON: "
        "{\"grounded\": true|false, \"motivo\": \"breve\"}.\n\n"
        f"PREGUNTA: {pregunta}\n\nRESPUESTA:\n{respuesta}\n\nFRAGMENTOS CITADOS:\n{frags}")
    try:
        r = cliente().models.generate_content(
            model=MODELO_GEN, contents=prompt, config=GenerateContentConfig(temperature=0))
        txt = (r.text or "").strip().strip("`")
        if txt.lower().startswith("json"):
            txt = txt[4:].strip()
        return json.loads(txt)
    except Exception as e:
        return {"grounded": None, "motivo": f"juez fallo: {e}"}


EVALUADORES = {"recall": ev_recall, "fidelidad_cita": ev_fidelidad,
               "frontera_no_dispongo": ev_frontera}


def _cargar_casos():
    casos = []
    with open(CASOS, encoding="utf-8") as fh:
        for linea in fh:
            if linea.strip():
                casos.append(json.loads(linea))
    return casos


def main():
    ap = argparse.ArgumentParser(description="Harness de evaluacion del RAG legal.")
    ap.add_argument("--solo", choices=list(EVALUADORES), default=None, help="Un solo tipo de fallo.")
    ap.add_argument("--k", type=int, default=10, help="Fragmentos a recuperar (default 10).")
    ap.add_argument("--temp", type=float, default=0.0, help="Temperatura de generacion del eval (default 0).")
    ap.add_argument("--juez", action="store_true", help="Juez semantico de fidelidad (cuesta tokens).")
    args = ap.parse_args()

    casos = [c for c in _cargar_casos() if not args.solo or c.get("tipo_fallo") == args.solo]
    print("=" * 60)
    print(f" EVAL RAG — casos: {len(casos)} | k={args.k} | temp={args.temp} | juez={args.juez}")
    print("=" * 60)

    detalle = []
    for c in casos:
        tipo = c.get("tipo_fallo")
        ev = EVALUADORES.get(tipo)
        if not ev:
            print(f"  [skip] {c.get('id')}: tipo desconocido '{tipo}'"); continue
        try:
            r = ev(c, args.k, args.temp, args.juez)
        except Exception as e:
            r = {"pass": False, "error": str(e)}
        marca = "OK " if r.get("pass") else "FALLA"
        print(f"  [{marca}] {c['id']} ({tipo}): "
              + json.dumps({x: r[x] for x in r if x not in ('faltantes',)}, ensure_ascii=False))
        detalle.append({"id": c["id"], "tipo_fallo": tipo, "pregunta": c["pregunta"], "resultado": r})

    # ---------------- resumen por categoria + global ----------------
    print("\n" + "=" * 60)
    print(" RESUMEN POR CATEGORIA")
    print("=" * 60)
    por_tipo = {}
    for d in detalle:
        por_tipo.setdefault(d["tipo_fallo"], []).append(d["resultado"])

    if "recall" in por_tipo:
        rs = por_tipo["recall"]
        prom = sum(x.get("recall", 0) for x in rs) / len(rs)
        print(f" recall            : recall@{args.k} promedio = {prom:.2f} | "
              f"casos perfectos {sum(1 for x in rs if x.get('pass'))}/{len(rs)}")
    if "fidelidad_cita" in por_tipo:
        fs = por_tipo["fidelidad_cita"]
        oks = sum(1 for x in fs if x.get("ok"))
        print(f" fidelidad_cita    : tasa de fidelidad = {oks}/{len(fs)} "
              f"({100*oks/len(fs):.0f}%) sin citas inventadas")
    if "frontera_no_dispongo" in por_tipo:
        fr = por_tipo["frontera_no_dispongo"]
        clases = {}
        for x in fr:
            clases[x.get("clase")] = clases.get(x.get("clase"), 0) + 1
        print(f" frontera_no_dispongo: "
              f"declinaciones correctas={clases.get('declinacion_correcta',0)} | "
              f"respuestas correctas={clases.get('respuesta_correcta',0)} | "
              f"RECHAZOS FALSOS={clases.get('rechazo_falso',0)} | "
              f"ALUCINACIONES={clases.get('alucinacion',0)}")

    total = len(detalle)
    passed = sum(1 for d in detalle if d["resultado"].get("pass"))
    print("\n" + "=" * 60)
    print(f" GLOBAL: {passed}/{total} casos PASS")
    print("=" * 60)

    # ---------------- volcado JSON ----------------
    os.makedirs(REPORTES, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    ruta = os.path.join(REPORTES, f"eval-{ts}.json")
    with open(ruta, "w", encoding="utf-8") as fh:
        json.dump({"ts": ts, "k": args.k, "temp": args.temp, "juez": args.juez,
                   "total": total, "pass": passed, "detalle": detalle},
                  fh, ensure_ascii=False, indent=2)
    print(f"Detalle por caso -> {ruta}")


if __name__ == "__main__":
    main()
