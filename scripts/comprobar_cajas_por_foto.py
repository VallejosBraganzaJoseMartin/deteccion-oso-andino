#!/usr/bin/env python3
"""
mAP de COCO con las k cajas de mayor confianza por foto (pregunta 33, ficha D26).

Qué responde. RT-DETRv3 entrega 300 cajas por foto y YOLO26 unas 2. En el mAP de
COCO, las cajas de confianza baja pueden sumar aciertos al final de la curva y casi
no restan, así que un modelo que entrega muchas cajas puede salir favorecido. Este
script recalcula el mAP de cada modelo quedándose solo con las k cajas de mayor
confianza de cada foto (k = 1, 3, 10 y 100) y aplica la lectura que se fijó el 7 de
octubre de 2026, antes de calcular nada (ficha D26 de registro_decisiones.md):

    Con k = 3, si la media de E6 y E6b (RT-DETRv3) en mAP50-95 global sigue por
    encima de la media de E4 y E4b (YOLO26m) por más de 0.015, la ventaja de
    RT-DETRv3 no viene del número de cajas. Si queda a 0.015 o menos, la comparación
    de mAP entre arquitecturas se reporta con el mismo k para todos y se declara.
    E7 se informa aparte.

Orden. Primero dos controles, y solo si pasan se calcula y se lee lo demás:
  1. La corrida del cuaderno repite la del 7 de octubre: el map_comun.csv de <dir>
     frente al de la evaluación anterior (--anterior).
  2. Con k = 100, este script repite el map_comun.csv de <dir>.

Por qué hacen falta las detecciones completas. El cuaderno de evaluación calcula el
mAP con todas las cajas de confianza >= 0.001, pero hasta la versión 3 solo guardaba
las de confianza >= 0.01 (umbral_detecciones.csv.gz). Medido el 15 de noviembre de
2026 con la evaluación del 7 de octubre: con ese archivo, el mAP50-95 global de
YOLO26 sale entre 0.006 y 0.010 por debajo del cuaderno (y el mAP50, hasta 0.020),
porque cerca de la mitad de sus cajas tienen confianza menor que 0.01 y algunas son
aciertos. El control no pasaba. Por eso el cuaderno versión 4 guarda además
detecciones_completas.csv.gz y este script exige ese archivo.

Cómo se calcula. Con pycocotools 2.0.11, el mismo evaluador del cuaderno (D26), y con
las mismas conversiones del cuaderno para las cajas reales (gt_coco) y para las
detecciones (dets_coco). El corte en k cajas lo hace el propio pycocotools con su
parámetro maxDets. Verificado en su código (cocoeval.py):
  - ordena las detecciones de cada foto de mayor a menor confianza, con un orden
    estable, y se queda con las primeras maxDets (computeIoU y evaluateImg);
  - empareja en ese orden, así que lo que pasa con las k primeras no depende de las
    que vienen después, y accumulate() toma las k primeras de cada foto;
  - summarize() solo sirve con los parámetros por defecto (lo dice su comentario):
    stats[0] busca siempre maxDets = 100. Por eso aquí el mAP de cada k se lee de la
    matriz de precisión de accumulate() con la misma operación que usa summarize().
Dos comprobaciones internas: con k = 100 esa lectura tiene que dar exactamente lo
mismo que summarize(), y con k = 3 tiene que dar lo mismo que quitar a mano las cajas
sobrantes de cada foto y evaluar con los parámetros por defecto.

Entradas (todas de validación; no usa GPU ni imágenes):
  - <dir>/detecciones_completas.csv.gz: todas las detecciones de los modelos de
    Ultralytics con confianza >= 0.001, sin redondear (cuaderno versión 4).
  - <rtdetr>/<RUN_ID>_detecciones_val.csv.gz (o .csv): detecciones completas de
    RT-DETRv3, las mismas que lee el cuaderno. Si no están en esa carpeta, se buscan
    en sus subcarpetas y en <dir>.
  - <dir>/umbral_imagenes.csv: ancho, alto y fuente de cada foto, en el orden del
    cuaderno.
  - <labels>: las etiquetas de validación (data/processed/labels/val).
  - <dir>/map_comun.csv y el de la evaluación anterior, para los controles.
  - <dir>/umbral_detecciones.csv.gz: solo para la variante de abajo.

Lectura principal y variante, fijadas antes de ver ningún resultado por k:
  - Principal: todos los modelos con sus detecciones completas (confianza >= 0.001).
  - Variante: todos con solo las cajas de confianza >= 0.01 (umbral_detecciones.csv.gz).
    Si las dos no dan la misma lectura, se informa y no se concluye.

Uso (desde la raíz del proyecto, con el Python de venv_paddle):
    $PY = "C:\\Users\\Martin\\venv_paddle\\Scripts\\python.exe"
    & $PY scripts\\comprobar_cajas_por_foto.py

Salidas: <dir>/cajas_por_foto.csv (una fila por modelo, entrada, partición y k) y
<dir>/salida_cajas_por_foto.txt (lo mismo que imprime). No las sobrescribe sin
--sobrescribir.

Versión 1 (15 nov 2026). Probado en Linux con Python 3.12.3, numpy 1.26.4 y
pycocotools 2.0.11: los controles, con los datos reales de RT-DETRv3 (repite el
cuaderno exacto); el cálculo por k y la lectura, con datos sintéticos.

Versión 1.1 (15 nov 2026). La versión 1 solo buscaba los archivos de RT-DETRv3
directamente en <rtdetr> y, si no estaban, avisaba sin decir qué faltaba ni dónde
había buscado. Ahora los busca también en las subcarpetas de <rtdetr> y en <dir>,
muestra la ruta del que usa y, si no los encuentra, dice cuál falta y dónde ponerlo.
No cambia ningún cálculo.
"""

import argparse
import contextlib
import csv
import gzip
import io
import sys
from importlib.metadata import version
from pathlib import Path

import numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

# ---------------------------------------------------------------------------------------
# Parámetros de la lectura, fijados el 7 de octubre de 2026 (ficha D26). No se tocan.
# ---------------------------------------------------------------------------------------
K_LISTA = [1, 3, 10, 100]
K_LECTURA = 3
MARGEN = 0.015
GRUPO_RTDETR = ["e6_rtdetrv3r18_s0", "e6b_rtdetrv3r18_s1"]
GRUPO_YOLO26M = ["e4_m_deg0_s0", "e4b_m_deg0_s1"]
APARTE = ["e7_m_deg0_s0"]          # YOLO26m con 200 épocas: se informa aparte

# ---------------------------------------------------------------------------------------
# Tolerancias de los controles, fijadas el 15 de noviembre de 2026, antes de ver ningún
# resultado por k. La ficha decía "unas milésimas" sin dar una cifra.
# ---------------------------------------------------------------------------------------
TOL_REPETICION = 0.0005   # control 1: la corrida nueva del cuaderno frente a la anterior ("al cuarto decimal")
TOL_COMPLETO = 0.0005     # control 2: con las detecciones completas son las mismas del cuaderno: tiene que repetir

CONF_MIN = 0.001                   # el mismo corte del cuaderno (CONF_MIN)
CONF_MIN_VARIANTE = 0.01           # el corte con que el cuaderno guarda umbral_detecciones.csv.gz
CONJUNTO = "val"                   # el conjunto de prueba no se usa para decidir nada
PARTICIONES = ["global", "ena24", "inat"]
TITULOS = {"global": "Global", "ena24": "ENA24", "inat": "iNaturalist"}
ESPERADO = {"imagenes": 598, "positivas": 326, "cajas": 351, "ena24": 338, "inat": 260}  # informe de preparación

INFORME = []


def decir(texto=""):
    print(texto)
    INFORME.append(texto)


def silencio():
    """pycocotools imprime mucho; se descarta lo que escribe en pantalla."""
    return contextlib.redirect_stdout(io.StringIO())


def abrir_texto(ruta):
    ruta = Path(ruta)
    if ruta.suffix == ".gz":
        return gzip.open(ruta, "rt", encoding="utf-8", newline="")
    return open(ruta, "r", encoding="utf-8", newline="")


# ---------------------------------------------------------------------------------------
# Lectura de las entradas
# ---------------------------------------------------------------------------------------
def leer_imagenes(ruta):
    """Fotos de validación en el orden del cuaderno (el id de cada foto es su posición, desde 1)."""
    with abrir_texto(ruta) as f:
        filas = [r for r in csv.DictReader(f) if r["conjunto"] == CONJUNTO]
    return [{"archivo": r["archivo"], "fuente": r["fuente"], "n_cajas": int(r["n_cajas"]),
             "ancho": int(r["ancho"]), "alto": int(r["alto"])} for r in filas]


def leer_cajas(ruta_txt):
    """Cajas YOLO (cx, cy, w, h normalizadas) convertidas a esquinas (x1, y1, x2, y2). Igual que el cuaderno."""
    filas = [l.split() for l in ruta_txt.read_text(encoding="utf-8").splitlines() if l.strip()]
    if not filas:
        return np.zeros((0, 4))
    a = np.array([[float(v) for v in f[1:5]] for f in filas])
    return np.stack([a[:, 0] - a[:, 2] / 2, a[:, 1] - a[:, 3] / 2,
                     a[:, 0] + a[:, 2] / 2, a[:, 1] + a[:, 3] / 2], axis=1)


def gt_coco(imgs, cajas):
    """Cajas reales en formato COCO (píxeles). Copia de gt_coco del cuaderno de evaluación."""
    imagenes, anotaciones, recortadas, descartadas = [], [], 0, 0
    for img_id, r in enumerate(imgs, start=1):
        W, H = r["ancho"], r["alto"]
        imagenes.append({"id": img_id, "file_name": r["archivo"], "width": W, "height": H})
        for x1, y1, x2, y2 in cajas[r["archivo"]]:
            c = (float(x1 * W), float(y1 * H), float(x2 * W), float(y2 * H))
            rc = (max(0.0, c[0]), max(0.0, c[1]), min(float(W), c[2]), min(float(H), c[3]))
            recortadas += rc != c
            if rc[2] - rc[0] <= 0 or rc[3] - rc[1] <= 0:
                descartadas += 1
                continue
            anotaciones.append({"id": len(anotaciones) + 1, "image_id": img_id, "category_id": 1, "iscrowd": 0,
                                "bbox": [rc[0], rc[1], rc[2] - rc[0], rc[3] - rc[1]],
                                "area": (rc[2] - rc[0]) * (rc[3] - rc[1])})
    gt = COCO()
    gt.dataset = {"images": imagenes, "annotations": anotaciones, "categories": [{"id": 1, "name": "oso"}]}
    with silencio():
        gt.createIndex()
    return gt, recortadas, descartadas


def deteccion(r, nombre, archivos):
    """Una fila de detecciones -> (archivo, conf, x1, y1, x2, y2), con las comprobaciones de leer_externo del cuaderno."""
    d = (r["archivo"], float(r["conf"]), float(r["x1"]), float(r["y1"]), float(r["x2"]), float(r["y2"]))
    if d[0] not in archivos:
        raise SystemExit(f"{nombre}: '{d[0]}' no es una foto de '{CONJUNTO}'")
    if not 0.0 <= d[1] <= 1.0:
        raise SystemExit(f"{nombre}: confianza fuera de [0, 1] en {d[0]}")
    if min(d[2:]) < -1e-6 or max(d[2:]) > 1 + 1e-6 or d[4] < d[2] or d[5] < d[3]:
        raise SystemExit(f"{nombre}: caja con coordenadas imposibles en {d[0]}")
    return d


def leer_por_modelo(ruta, archivos, conf_min):
    """Archivo con varias corridas (run_id, conjunto, archivo, conf, x1, y1, x2, y2) ->
    {run_id: [detecciones]} en el orden del archivo, solo de validación y desde conf_min."""
    por_modelo = {}
    with abrir_texto(ruta) as f:
        lector = csv.DictReader(f)
        faltan = {"run_id", "conjunto", "archivo", "conf", "x1", "y1", "x2", "y2"} - set(lector.fieldnames or [])
        if faltan:
            raise SystemExit(f"{ruta.name}: faltan las columnas {sorted(faltan)}")
        for r in lector:
            if r["conjunto"] != CONJUNTO:
                continue
            d = deteccion(r, ruta.name, archivos)
            if d[1] >= conf_min:
                por_modelo.setdefault(r["run_id"], []).append(d)
    return por_modelo


def leer_externo(ruta, archivos):
    """Detecciones exportadas de RT-DETRv3 (archivo, conf, x1, y1, x2, y2), como leer_externo del cuaderno."""
    dets = []
    with abrir_texto(ruta) as f:
        lector = csv.DictReader(f)
        faltan = {"archivo", "conf", "x1", "y1", "x2", "y2"} - set(lector.fieldnames or [])
        if faltan:
            raise SystemExit(f"{ruta.name}: faltan las columnas {sorted(faltan)}")
        for r in lector:
            d = deteccion(r, ruta.name, archivos)
            if d[1] >= CONF_MIN:
                dets.append(d)
    return dets


def nombres_externo(rid):
    return (f"{rid}_detecciones_{CONJUNTO}.csv.gz", f"{rid}_detecciones_{CONJUNTO}.csv")


def buscar_externo(carpetas, rid):
    """Copias de <RUN_ID>_detecciones_val.csv.gz (o .csv). Por orden: en cada carpeta y, si ahí no está, en sus
    subcarpetas. Devuelve la lista de la primera carpeta donde aparece (vacía si no está en ninguna)."""
    for carpeta in carpetas:
        directos = [carpeta / n for n in nombres_externo(rid) if (carpeta / n).is_file()]
        if directos:
            return directos
        dentro = sorted(p for n in nombres_externo(rid) for p in carpeta.rglob(n) if p.is_file())
        if dentro:
            return dentro
    return []


def leer_map_comun(ruta):
    """{(run_id, particion): (mAP50-95, mAP50)} de pycocotools según el cuaderno, el tipo de cada modelo y su orden."""
    ref, tipo, orden = {}, {}, []
    with abrir_texto(ruta) as f:
        for r in csv.DictReader(f):
            if r["conjunto"] != CONJUNTO:
                continue
            ref[(r["run_id"], r["particion"])] = (float(r["mAP50_95_coco"]), float(r["mAP50_coco"]))
            tipo[r["run_id"]] = r["tipo"]
            if r["run_id"] not in orden:
                orden.append(r["run_id"])
    return ref, tipo, orden


# ---------------------------------------------------------------------------------------
# Evaluación con pycocotools
# ---------------------------------------------------------------------------------------
def dets_coco(dets, tam, id_imagen):
    """Detecciones normalizadas -> lista COCO (x, y, ancho, alto en píxeles). Como dets_coco del cuaderno."""
    lista = []
    for archivo, s, a, b, c, e in dets:
        w, h = tam[archivo]
        lista.append({"image_id": id_imagen[archivo], "category_id": 1, "score": s,
                      "bbox": [a * w, b * h, (c - a) * w, (e - b) * h]})
    return lista


def cargar(gt, dets, tam, id_imagen):
    with silencio():
        return gt.loadRes(dets_coco(dets, tam, id_imagen))


def evaluar_estandar(gt, dt, img_ids):
    """Como la sección 5b del cuaderno: parámetros por defecto y summarize(). Devuelve (mAP50-95, mAP50)."""
    with silencio():
        ev = COCOeval(gt, dt, "bbox")
        ev.params.imgIds = list(img_ids)
        ev.evaluate()
        ev.accumulate()
        ev.summarize()
    return float(ev.stats[0]), float(ev.stats[1])


def evaluar_por_k(gt, dt, img_ids):
    """{k: (mAP50-95, mAP50)} con las k cajas de mayor confianza por foto (maxDets = k)."""
    with silencio():
        ev = COCOeval(gt, dt, "bbox")
        ev.params.imgIds = list(img_ids)
        ev.params.maxDets = list(K_LISTA)
        ev.evaluate()
        ev.accumulate()
    p = ev.params
    precision = ev.eval["precision"]                   # dimensiones [T, R, K, A, M], como en summarize()
    a = p.areaRngLbl.index("all")
    t50 = int(np.where(0.5 == p.iouThrs)[0][0])

    def media(s):
        return float(np.mean(s[s > -1])) if (s > -1).any() else float("nan")

    return {k: (media(precision[:, :, :, a, m]), media(precision[t50, :, :, a, m]))
            for m, k in enumerate(p.maxDets)}


def top_k(dets, k):
    """Las k detecciones de mayor confianza de cada foto, con orden estable (como pycocotools)."""
    por_foto = {}
    for i, d in enumerate(dets):
        por_foto.setdefault(d[0], []).append(i)
    quedan = set()
    for idx in por_foto.values():
        quedan.update(sorted(idx, key=lambda i: -dets[i][1])[:k])
    return [d for i, d in enumerate(dets) if i in quedan]


def cajas_por_foto(dets, n_fotos):
    cuenta = {}
    for d in dets:
        cuenta[d[0]] = cuenta.get(d[0], 0) + 1
    return {"total": len(dets), "media": len(dets) / n_fotos, "maximo": max(cuenta.values()) if cuenta else 0,
            "quedan": {k: sum(min(n, k) for n in cuenta.values()) for k in K_LISTA}}


# ---------------------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="mAP de COCO con las k cajas de mayor confianza por foto (pregunta 33).")
    ap.add_argument("--dir", default="docs/modelado/eval_p33",
                    help="Carpeta con las salidas del cuaderno de evaluación versión 4")
    ap.add_argument("--anterior", default="docs/modelado/eval_e7/map_comun.csv",
                    help="map_comun.csv de la evaluación anterior, que la corrida nueva tiene que repetir")
    ap.add_argument("--rtdetr", default="docs/modelado/rtdetrv3",
                    help="Carpeta con <RUN_ID>_detecciones_val.csv.gz de RT-DETRv3")
    ap.add_argument("--labels", default="data/processed/labels/val", help="Carpeta con las etiquetas de validación")
    ap.add_argument("--out", default=None, help="CSV de salida (por defecto <dir>/cajas_por_foto.csv)")
    ap.add_argument("--solo-control", action="store_true",
                    help="Detenerse después de los controles, sin calcular el mAP por k ni escribir las salidas")
    ap.add_argument("--sobrescribir", action="store_true", help="Reemplazar las salidas si ya existen")
    args = ap.parse_args()
    with contextlib.suppress(Exception):               # que una consola rara no tumbe el script por un carácter
        sys.stdout.reconfigure(errors="replace")

    base, rtdetr, labels, anterior = Path(args.dir), Path(args.rtdetr), Path(args.labels), Path(args.anterior)
    out_csv = Path(args.out) if args.out else base / "cajas_por_foto.csv"
    out_txt = out_csv.with_name("salida_" + out_csv.stem + ".txt")
    ruta_comp, ruta_filt = base / "detecciones_completas.csv.gz", base / "umbral_detecciones.csv.gz"
    if base.is_dir() and not ruta_comp.exists():
        raise SystemExit(f"No está {ruta_comp}. Lo deja el cuaderno de evaluación versión 4 (evaluar_umbral_yolo26.ipynb).\n"
                         "Con umbral_detecciones.csv.gz solo (confianza >= 0.01) el control no pasa: ver el encabezado del script.")
    for ruta in (ruta_comp, ruta_filt, base / "umbral_imagenes.csv", base / "map_comun.csv", anterior, rtdetr, labels):
        if not ruta.exists():
            raise SystemExit(f"No existe {ruta}. Correr desde la raíz del proyecto o indicar --dir, --anterior, --rtdetr y --labels.")
    for ruta in (out_csv, out_txt):
        if ruta.exists() and not args.sobrescribir and not args.solo_control:
            raise SystemExit(f"Ya existe {ruta}. Para reemplazarlo, añadir --sobrescribir.")

    # ---------------------------------------------------------------- 1. entradas
    decir("== 1. Entradas ==")
    v_coco = version("pycocotools")
    decir(f"Python {sys.version.split()[0]} | numpy {np.__version__} | pycocotools {v_coco}")
    if v_coco != "2.0.11":
        decir("AVISO: el cuaderno usa pycocotools 2.0.11 (D26). Con otra versión las cifras pueden cambiar.")

    imgs = leer_imagenes(base / "umbral_imagenes.csv")
    archivos = {r["archivo"] for r in imgs}
    cajas, problemas = {}, []
    for r in imgs:
        txt = labels / (Path(r["archivo"]).stem + ".txt")
        if not txt.exists():
            problemas.append(f"falta {txt.name}")
            continue
        cajas[r["archivo"]] = leer_cajas(txt)
        if len(cajas[r["archivo"]]) != r["n_cajas"]:
            problemas.append(f"{txt.name}: {len(cajas[r['archivo']])} cajas, y umbral_imagenes.csv dice {r['n_cajas']}")
    if problemas:
        raise SystemExit(f"Las etiquetas de {labels} no corresponden a umbral_imagenes.csv "
                         f"({len(problemas)} problemas; el primero: {problemas[0]}).")
    obtenido = {"imagenes": len(imgs), "positivas": sum(r["n_cajas"] > 0 for r in imgs),
                "cajas": sum(r["n_cajas"] for r in imgs),
                "ena24": sum(r["fuente"] == "ena24" for r in imgs), "inat": sum(r["fuente"] == "inat" for r in imgs)}
    gt, recortadas, descartadas = gt_coco(imgs, cajas)
    n_anot = len(gt.dataset["annotations"])
    bien = obtenido == ESPERADO and n_anot == ESPERADO["cajas"] and descartadas == 0
    decir(f"Fotos de '{CONJUNTO}': {obtenido['imagenes']} (ENA24 {obtenido['ena24']}, iNaturalist {obtenido['inat']}), "
          f"{obtenido['positivas']} con oso; cajas reales {obtenido['cajas']}; para pycocotools {n_anot} "
          f"({recortadas} recortadas al borde, {descartadas} descartadas) -> "
          f"{'coincide con el informe de preparación' if bien else 'NO COINCIDE'}")
    if not bien:
        raise SystemExit(f"Se esperaba {ESPERADO}. No se sigue hasta explicarlo.")

    id_imagen = {r["archivo"]: i for i, r in enumerate(imgs, start=1)}
    tam = {r["archivo"]: (float(r["ancho"]), float(r["alto"])) for r in imgs}
    ids = {"global": [id_imagen[r["archivo"]] for r in imgs],
           "ena24": [id_imagen[r["archivo"]] for r in imgs if r["fuente"] == "ena24"],
           "inat": [id_imagen[r["archivo"]] for r in imgs if r["fuente"] == "inat"]}

    ref, tipo, modelos = leer_map_comun(base / "map_comun.csv")
    ref_ant, _, _ = leer_map_comun(anterior)
    completas = leer_por_modelo(ruta_comp, archivos, CONF_MIN)
    filtradas = leer_por_modelo(ruta_filt, archivos, CONF_MIN_VARIANTE)

    ENTRADAS, VARIANTE = {}, {}
    decir(f"Detecciones completas de cada modelo (confianza >= {CONF_MIN}):")
    for rid in modelos:
        if tipo[rid] == "externo":
            copias = buscar_externo([rtdetr, base], rid)
            if not copias:
                decir(f"   {rid:<20} NO ENCONTRADO: {nombres_externo(rid)[0]} (ni .csv) no está en {rtdetr}, "
                      f"ni en sus subcarpetas, ni en {base}")
                continue
            dets, origen = leer_externo(copias[0], archivos), str(copias[0])
            if len(copias) > 1:
                origen += f" (hay {len(copias)} copias; se usa esta)"
        else:
            dets, origen = completas.get(rid), ruta_comp.name
            if not dets:
                decir(f"   {rid:<20} NO ENCONTRADO en {ruta_comp.name}: ¿se evaluó con el cuaderno versión 4?")
                continue
        ENTRADAS[rid] = {"dets": dets, "origen": "completas", "conf_min": CONF_MIN,
                         "cuenta": cajas_por_foto(dets, len(imgs))}
        c = ENTRADAS[rid]["cuenta"]
        decir(f"   {rid:<20} {c['total']:>7} cajas, {c['media']:>6.1f} por foto de media, máximo {c['maximo']:>3} | {origen}")
        if rid in filtradas:
            VARIANTE[rid] = {"dets": filtradas[rid], "origen": "umbral_detecciones", "conf_min": CONF_MIN_VARIANTE,
                             "cuenta": cajas_por_foto(filtradas[rid], len(imgs))}
    modelos = [m for m in modelos if m in ENTRADAS]
    falta = []
    for m in GRUPO_RTDETR + GRUPO_YOLO26M:
        if m not in tipo:
            falta.append(f"   {m}: no está en {base / 'map_comun.csv'}. Evaluarlo con el cuaderno (INCLUIR, celda 0).")
        elif m not in ENTRADAS and tipo[m] == "externo":
            falta.append(f"   {m}: copiar {nombres_externo(m)[0]} a {rtdetr} y volver a correr.")
        elif m not in ENTRADAS:
            falta.append(f"   {m}: no está en {ruta_comp}.")
        elif m not in VARIANTE:
            falta.append(f"   {m}: no está en {ruta_filt}.")
    if falta:
        raise SystemExit("No se puede seguir: faltan detecciones de modelos de la lectura.\n" + "\n".join(falta))

    DT = {rid: cargar(gt, e["dets"], tam, id_imagen) for rid, e in ENTRADAS.items()}

    # ---------------------------------------------------------------- 2. controles
    decir()
    decir("== 2. Controles ==")
    decir(f"Control 1: la corrida del cuaderno repite la anterior ({anterior}); se acepta hasta {TOL_REPETICION}.")
    comunes = sorted(set(ref) & set(ref_ant))
    if not comunes:
        raise SystemExit("Los dos map_comun.csv no tienen ningún modelo en común: no se puede hacer el control 1.")
    peor1 = max(max(abs(ref[c][i] - ref_ant[c][i]) for i in (0, 1)) for c in comunes)
    sin_ant = [m for m in GRUPO_RTDETR + GRUPO_YOLO26M if (m, "global") not in ref_ant]
    control1 = peor1 <= TOL_REPETICION and not sin_ant
    decir(f"   {len(comunes)} cifras comparadas (modelo y partición); mayor diferencia en mAP50-95 o mAP50: {peor1:.6f} -> "
          f"{'repite' if peor1 <= TOL_REPETICION else 'NO REPITE'}")
    if sin_ant:
        decir(f"   La evaluación anterior no tiene los modelos de la lectura {sin_ant}.")
    if peor1 > TOL_REPETICION:
        for c in comunes:
            d = max(abs(ref[c][i] - ref_ant[c][i]) for i in (0, 1))
            if d > TOL_REPETICION:
                decir(f"      {c[0]:<20} {c[1]:<12} mAP50-95 {ref[c][0]:.4f} frente a {ref_ant[c][0]:.4f}; "
                      f"mAP50 {ref[c][1]:.4f} frente a {ref_ant[c][1]:.4f}")

    decir(f"Control 2: con k = 100 este script repite el map_comun.csv de {base}; se acepta hasta {TOL_COMPLETO}.")
    decir(f"   {'modelo':<20} {'partición':<12} {'mAP50-95':>9} {'cuaderno':>9} {'dif.':>9}   {'mAP50':>7} {'cuaderno':>9} {'dif.':>9}")
    ESTANDAR, control2, peor2 = {}, True, 0.0
    for rid in modelos:
        for part in PARTICIONES:
            ESTANDAR[(rid, part)] = m5095, m50 = evaluar_estandar(gt, DT[rid], ids[part])
            r5095, r50 = ref[(rid, part)]
            d1, d2 = m5095 - r5095, m50 - r50
            ok = abs(d1) <= TOL_COMPLETO and abs(d2) <= TOL_COMPLETO
            control2 &= ok
            peor2 = max(peor2, abs(d1), abs(d2))
            decir(f"   {rid:<20} {TITULOS[part]:<12} {m5095:>9.4f} {r5095:>9.4f} {d1:>+9.5f}   {m50:>7.4f} {r50:>9.4f} {d2:>+9.5f}"
                  f"{'' if ok else '   <-- FUERA DE TOLERANCIA'}")
    decir(f"   Mayor diferencia: {peor2:.6f} -> {'repite' if control2 else 'NO REPITE'}")
    if not (control1 and control2):
        decir("ALGÚN CONTROL NO PASA. No se calcula ni se lee nada más hasta explicar la diferencia.")
        raise SystemExit(1)
    decir("Los dos controles pasan.")
    if args.solo_control:
        decir("Se pidió solo el control: no se calcula el mAP por k ni se escriben las salidas.")
        return

    # ---------------------------------------------------------------- 3. mAP por k
    POR_K, POR_K_VAR = {}, {}
    for rid in modelos:
        for part in PARTICIONES:
            POR_K[(rid, part)] = evaluar_por_k(gt, DT[rid], ids[part])
            dif = max(abs(POR_K[(rid, part)][100][i] - ESTANDAR[(rid, part)][i]) for i in (0, 1))
            if dif > 1e-9:
                raise SystemExit(f"Error interno: con k = 100, {rid} en {part} no repite summarize() (diferencia {dif:.2e}).")
    for rid, e in VARIANTE.items():
        dt = cargar(gt, e["dets"], tam, id_imagen)
        for part in PARTICIONES:
            POR_K_VAR[(rid, part)] = evaluar_por_k(gt, dt, ids[part])
    peor_topk = 0.0
    for rid in GRUPO_RTDETR + GRUPO_YOLO26M + [m for m in APARTE if m in ENTRADAS]:
        a_mano = evaluar_estandar(gt, cargar(gt, top_k(ENTRADAS[rid]["dets"], K_LECTURA), tam, id_imagen), ids["global"])
        peor_topk = max(peor_topk, max(abs(a_mano[i] - POR_K[(rid, "global")][K_LECTURA][i]) for i in (0, 1)))
    if peor_topk > 1e-9:
        raise SystemExit(f"Error interno: quitar a mano las cajas sobrantes no da lo mismo que maxDets = {K_LECTURA} "
                         f"(diferencia {peor_topk:.2e}).")

    decir()
    decir("== 3. mAP con las k cajas de mayor confianza por foto (pycocotools, detecciones completas) ==")
    decir("Comprobaciones internas: con k = 100 repite summarize(), y con k = "
          f"{K_LECTURA} da lo mismo que quitar a mano las cajas sobrantes de cada foto.")
    cab = "".join(f"{'k = ' + str(k):>10}" for k in K_LISTA)
    for i, nombre in ((0, "mAP50-95"), (1, "mAP50")):
        for part in PARTICIONES:
            decir()
            decir(f"=== {nombre} | {TITULOS[part]} ===")
            decir(f"   {'modelo':<20}{cab}   {'cuaderno':>9}")
            for rid in modelos:
                fila = "".join(f"{POR_K[(rid, part)][k][i]:>10.4f}" for k in K_LISTA)
                decir(f"   {rid:<20}{fila}   {ref[(rid, part)][i]:>9.4f}")
    decir()
    decir("Cajas que quedan en total con cada k (detecciones completas):")
    decir(f"   {'modelo':<20}{cab}   {'todas':>9}")
    for rid in modelos:
        c = ENTRADAS[rid]["cuenta"]
        decir(f"   {rid:<20}" + "".join(f"{c['quedan'][k]:>10}" for k in K_LISTA) + f"   {c['total']:>9}")

    # ---------------------------------------------------------------- 4. diferencias y lectura
    def media(grupo, k, i=0, tabla=POR_K):
        return float(np.mean([tabla[(rid, "global")][k][i] for rid in grupo]))

    decir()
    decir("== 4. RT-DETRv3 (media de E6 y E6b) frente a YOLO26m (media de E4 y E4b), global ==")
    for titulo, tabla in ((f"Detecciones completas (confianza >= {CONF_MIN})", POR_K),
                          (f"Variante: solo las cajas de confianza >= {CONF_MIN_VARIANTE}, para todos", POR_K_VAR)):
        decir(titulo)
        decir(f"   {'':<24}{cab}")
        for i, nombre in ((0, "mAP50-95"), (1, "mAP50")):
            decir(f"   {nombre + ', RT-DETRv3':<24}" + "".join(f"{media(GRUPO_RTDETR, k, i, tabla):>10.4f}" for k in K_LISTA))
            decir(f"   {nombre + ', YOLO26m':<24}" + "".join(f"{media(GRUPO_YOLO26M, k, i, tabla):>10.4f}" for k in K_LISTA))
            decir(f"   {nombre + ', diferencia':<24}"
                  + "".join(f"{media(GRUPO_RTDETR, k, i, tabla) - media(GRUPO_YOLO26M, k, i, tabla):>+10.4f}" for k in K_LISTA))

    dif = media(GRUPO_RTDETR, K_LECTURA) - media(GRUPO_YOLO26M, K_LECTURA)
    dif_var = media(GRUPO_RTDETR, K_LECTURA, 0, POR_K_VAR) - media(GRUPO_YOLO26M, K_LECTURA, 0, POR_K_VAR)
    dif_100 = media(GRUPO_RTDETR, 100) - media(GRUPO_YOLO26M, 100)
    decir()
    decir("== 5. Lectura (regla de la ficha D26, fijada el 7 de octubre de 2026, antes de calcular) ==")
    decir(f"Con k = {K_LECTURA}, mAP50-95 global:")
    for rid in GRUPO_RTDETR + GRUPO_YOLO26M:
        decir(f"   {rid:<20} {POR_K[(rid, 'global')][K_LECTURA][0]:.4f}")
    decir(f"Media de RT-DETRv3 {media(GRUPO_RTDETR, K_LECTURA):.4f} - media de YOLO26m {media(GRUPO_YOLO26M, K_LECTURA):.4f} "
          f"= {dif:+.4f} (hace falta más de {MARGEN}). Con k = 100 es {dif_100:+.4f}.")
    supera, supera_var = dif > MARGEN, dif_var > MARGEN
    if supera:
        decir("Resultado: RT-DETRv3 sigue por encima por más del margen -> su ventaja en mAP50-95 no viene del número "
              "de cajas por foto.")
    else:
        decir("Resultado: la diferencia queda dentro del margen -> la comparación de mAP entre arquitecturas se reporta "
              "con el mismo k para todos y se declara.")
    decir(f"Variante con solo las cajas de confianza >= {CONF_MIN_VARIANTE}, para todos: {dif_var:+.4f} -> "
          f"{'la misma lectura' if supera_var == supera else 'LECTURA DISTINTA: informar las dos y no concluir'}.")
    for rid in APARTE:
        if rid in ENTRADAS:
            v = POR_K[(rid, "global")][K_LECTURA][0]
            decir(f"Aparte, {rid} (YOLO26m con 200 épocas, una semilla): {v:.4f} con k = {K_LECTURA}; "
                  f"la media de RT-DETRv3 queda {media(GRUPO_RTDETR, K_LECTURA) - v:+.4f}.")
    decir("Esto no cambia la regla B ni D19: la regla B solo usa las cajas por encima del umbral de cada modelo.")

    # ---------------------------------------------------------------- 5. salidas
    with open(out_csv, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["run_id", "tipo", "entrada", "conf_min", "conjunto", "particion", "k", "mAP50_95", "mAP50",
                    "mAP50_95_cuaderno", "mAP50_cuaderno", "cajas_que_quedan", "cajas_total"])
        for tabla, fuente in ((POR_K, ENTRADAS), (POR_K_VAR, VARIANTE)):
            for (rid, part), por_k in tabla.items():
                e = fuente[rid]
                for k in K_LISTA:
                    es_ref = e["origen"] == "completas" and k == 100
                    w.writerow([rid, tipo[rid], e["origen"], e["conf_min"], CONJUNTO, part, k,
                                repr(por_k[k][0]), repr(por_k[k][1]),
                                repr(ref[(rid, part)][0]) if es_ref else "", repr(ref[(rid, part)][1]) if es_ref else "",
                                e["cuenta"]["quedan"][k], e["cuenta"]["total"]])
    decir()
    decir(f"Resultados guardados en {out_csv} y en {out_txt}")
    out_txt.write_text("\n".join(INFORME) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
