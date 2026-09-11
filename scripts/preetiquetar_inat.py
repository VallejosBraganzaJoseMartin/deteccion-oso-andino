#!/usr/bin/env python3
"""
Fase CRISP-DM: Preparacion de los datos (etiquetado, iNaturalist).

Toma los positivos de iNaturalist segun inat_estado.csv, copia cada imagen a
<out>/images/ y genera <out>/labels/<nombre>.txt con las cajas de la clase
'bear' que propone YOLO26n preentrenado en COCO, en formato YOLO:
    0 cx cy w h        (clase 0 = oso; coordenadas normalizadas a [0, 1])
Las imagenes en las que el modelo no propone ninguna caja reciben un .txt
vacio y quedan marcadas en el CSV de salida para dibujarlas a mano.
Tambien escribe <out>/labels/classes.txt con la unica clase ('oso').

Estas cajas son solo un punto de partida: TODAS se revisan y corrigen a mano
en la herramienta de anotacion.

Uso (desde la raiz del proyecto; tarda unos 10-15 min en CPU):
    python scripts/preetiquetar_inat.py --estado docs/comprension_datos/inat_estado.csv \
        --dirs data/raw/inaturalist data/revisar/inat_sin_oso/oso data/revisar/inat_conf_baja/oso \
        --out data/etiquetado/inat --csv docs/comprension_datos/preetiquetado_inat.csv

NO PROBADO con las imagenes reales. La API de Ultralytics usada (YOLO, predict
con stream=True, r.boxes.cls, r.boxes.conf, r.boxes.xywhn, r.path, model.names)
esta verificada en el codigo fuente de la version 8.4.143; xywhn devuelve
[cx, cy, w, h] normalizados respecto a la imagen original.
"""

import argparse
import csv
import os
import shutil
from collections import Counter

import pandas as pd

EXT_IMG = {".jpg", ".jpeg", ".png"}


def indexar(carpetas):
    """Devuelve {nombre_de_archivo: ruta}; si un nombre se repite, gana la primera carpeta."""
    indice = {}
    for carpeta in carpetas:
        if not os.path.isdir(carpeta):
            print(f"AVISO: no existe la carpeta {carpeta}")
            continue
        for raiz, _, archivos in os.walk(carpeta):
            for a in archivos:
                if os.path.splitext(a)[1].lower() in EXT_IMG and a not in indice:
                    indice[a] = os.path.join(raiz, a)
    return indice


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--estado", required=True, help="inat_estado.csv de consolidar_inat.py")
    ap.add_argument("--dirs", nargs="+", required=True,
                    help="Carpetas donde buscar las imagenes (raw primero, luego las de fotogramas)")
    ap.add_argument("--out", required=True, help="Carpeta de salida (crea images/ y labels/)")
    ap.add_argument("--csv", required=True, help="CSV con el resultado del pre-etiquetado por imagen")
    ap.add_argument("--pesos", default="yolo26n.pt")
    ap.add_argument("--conf", type=float, default=0.2,
                    help="Umbral bajo a proposito: borrar una caja de mas es mas rapido que dibujarla")
    ap.add_argument("--imgsz", type=int, default=640)
    args = ap.parse_args()

    estado = pd.read_csv(args.estado)
    pos = estado[estado["estado"].astype(str).str.startswith("positivo")].copy()
    print(f"Positivos en {args.estado}: {len(pos)}")

    indice = indexar(args.dirs)
    faltan = [a for a in pos["archivo"] if a not in indice]
    if faltan:
        print(f"ATENCION: {len(faltan)} positivos no se encontraron en las carpetas dadas. Primeros:")
        for a in faltan[:15]:
            print(f"  {a}")
        print("Se continua con los que si se encontraron.")
    pos = pos[pos["archivo"].isin(indice)]

    dir_img = os.path.join(args.out, "images")
    dir_lbl = os.path.join(args.out, "labels")
    os.makedirs(dir_img, exist_ok=True)
    os.makedirs(dir_lbl, exist_ok=True)
    with open(os.path.join(dir_lbl, "classes.txt"), "w", encoding="utf-8") as f:
        f.write("oso\n")

    rutas = []
    copiadas = 0
    for a in pos["archivo"]:
        destino = os.path.join(dir_img, a)
        if not os.path.exists(destino):
            shutil.copy2(indice[a], destino)
            copiadas += 1
        rutas.append(destino)
    print(f"Imagenes en {dir_img}: {len(rutas)} ({copiadas} copiadas ahora)")

    from ultralytics import YOLO  # import tardio para que --help funcione sin la libreria

    modelo = YOLO(args.pesos)
    nombres = modelo.names
    ids_oso = [i for i, n in nombres.items() if n == "bear"]
    if not ids_oso:
        raise SystemExit(f"El modelo {args.pesos} no tiene clase 'bear'; clases: {nombres}")
    id_oso = ids_oso[0]

    estado_por_archivo = dict(zip(pos["archivo"], pos["estado"]))
    resumen = Counter()
    total_cajas = 0
    os.makedirs(os.path.dirname(os.path.abspath(args.csv)), exist_ok=True)
    with open(args.csv, "w", newline="", encoding="utf-8") as f:
        wr = csv.writer(f)
        wr.writerow(["archivo", "estado", "n_cajas", "conf_max"])
        resultados = modelo.predict(source=rutas, stream=True, conf=args.conf,
                                    imgsz=args.imgsz, verbose=False)
        for i, r in enumerate(resultados, 1):
            nombre = os.path.basename(r.path)
            lineas, confs = [], []
            if r.boxes is not None and len(r.boxes):
                clases = r.boxes.cls.tolist()
                cf = r.boxes.conf.tolist()
                xywhn = r.boxes.xywhn.tolist()
                for k, c, (cx, cy, w, h) in zip(clases, cf, xywhn):
                    if int(k) == id_oso:
                        lineas.append(f"0 {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")
                        confs.append(c)
            with open(os.path.join(dir_lbl, os.path.splitext(nombre)[0] + ".txt"),
                      "w", encoding="utf-8") as fl:
                fl.write("\n".join(lineas) + ("\n" if lineas else ""))
            wr.writerow([nombre, estado_por_archivo.get(nombre, ""), len(lineas),
                         round(max(confs), 3) if confs else 0.0])
            total_cajas += len(lineas)
            resumen["con_cajas" if lineas else "sin_cajas"] += 1
            if len(lineas) > 1:
                resumen["mas_de_una"] += 1
            if not lineas:
                resumen[f"sin_cajas_{estado_por_archivo.get(nombre, '')}"] += 1
            if i % 200 == 0:
                print(f"{i}/{len(rutas)}")

    print("\nResumen del pre-etiquetado:")
    print(f"  Imagenes con al menos una caja: {resumen['con_cajas']}")
    print(f"  Imagenes con mas de una caja: {resumen['mas_de_una']}")
    print(f"  Imagenes sin caja (dibujar a mano): {resumen['sin_cajas']}")
    for k, v in sorted(resumen.items()):
        if k.startswith("sin_cajas_"):
            print(f"    de estado {k[len('sin_cajas_'):]}: {v}")
    print(f"  Cajas propuestas en total: {total_cajas}")
    print(f"Etiquetas en {dir_lbl}; detalle por imagen en {args.csv}")


if __name__ == "__main__":
    main()
