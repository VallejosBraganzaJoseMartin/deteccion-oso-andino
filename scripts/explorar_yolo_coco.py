#!/usr/bin/env python3
"""
Fase CRISP-DM: Comprension de los datos (explorar).

Corre un detector YOLO preentrenado en COCO (COCO incluye la clase 'bear')
sobre una carpeta de imagenes y registra, por imagen, la confianza maxima
con la que ve un oso, cuantas cajas de oso encuentra y que otras clases detecta.

Sirve para dos cosas:
  1. Priorizar la revision manual: las imagenes donde NO ve oso son las
     candidatas a ser huellas, excrementos, restos o casos dificiles.
  2. Tener un baseline zero-shot (sin entrenar nada) que documentar.

Uso:
    pip install ultralytics
    python explorar_yolo_coco.py --dir data/raw/inaturalist --out docs/comprension_datos/yolo_inat.csv
    python explorar_yolo_coco.py --dir data/raw/ena24/images --out docs/comprension_datos/yolo_ena24.csv \
        --copiar-sin-oso data/revisar/ena24_sin_oso

NO PROBADO contra la libreria real desde este entorno. Verificar:
  - El nombre del archivo de pesos (--pesos). Ultralytics nombra sus modelos
    como yolo11n.pt, yolo26n.pt, etc.; confirmar en docs.ultralytics.com.
  - Que model.names, results.boxes.cls / .conf y results.path existan con
    esos nombres en la version instalada.
"""

import argparse
import csv
import os
import shutil
from collections import Counter

EXT_IMG = {".jpg", ".jpeg", ".png"}


def listar_imagenes(carpeta):
    rutas = []
    for raiz, _, archivos in os.walk(carpeta):
        for a in archivos:
            if os.path.splitext(a)[1].lower() in EXT_IMG:
                rutas.append(os.path.join(raiz, a))
    return sorted(rutas)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True, help="Carpeta con imagenes")
    ap.add_argument("--out", required=True, help="CSV de salida")
    ap.add_argument("--pesos", default="yolo26n.pt")
    ap.add_argument("--conf", type=float, default=0.05,
                    help="Umbral bajo a proposito: queremos ver hasta las detecciones dudosas")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--copiar-sin-oso", default=None,
                    help="Carpeta a la que copiar las imagenes sin ninguna deteccion de oso")
    args = ap.parse_args()

    from ultralytics import YOLO  # import tardio para que --help funcione sin la libreria

    modelo = YOLO(args.pesos)
    nombres = modelo.names  # dict {id: nombre_de_clase}
    ids_oso = [i for i, n in nombres.items() if n == "bear"]
    if not ids_oso:
        raise SystemExit(f"El modelo {args.pesos} no tiene una clase 'bear'; clases: {nombres}")
    id_oso = ids_oso[0]

    rutas = listar_imagenes(args.dir)
    print(f"{len(rutas)} imagenes en {args.dir}")
    if args.copiar_sin_oso:
        os.makedirs(args.copiar_sin_oso, exist_ok=True)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)

    resumen = Counter()
    with open(args.out, "w", newline="", encoding="utf-8") as f:
        wr = csv.writer(f)
        wr.writerow(["archivo", "ruta", "n_cajas_oso", "conf_max_oso", "otras_clases"])
        resultados = modelo.predict(source=rutas, stream=True, conf=args.conf,
                                    imgsz=args.imgsz, verbose=False)
        for i, r in enumerate(resultados, 1):
            clases = [int(c) for c in r.boxes.cls.tolist()] if r.boxes is not None else []
            confs = [float(c) for c in r.boxes.conf.tolist()] if r.boxes is not None else []
            conf_oso = [c for k, c in zip(clases, confs) if k == id_oso]
            otras = sorted({nombres[k] for k in clases if k != id_oso})
            n_oso = len(conf_oso)
            conf_max = max(conf_oso) if conf_oso else 0.0
            wr.writerow([os.path.basename(r.path), r.path, n_oso, round(conf_max, 3), ";".join(otras)])
            if n_oso == 0:
                resumen["sin_oso"] += 1
                if args.copiar_sin_oso:
                    shutil.copy2(r.path, os.path.join(args.copiar_sin_oso, os.path.basename(r.path)))
            elif conf_max >= 0.5:
                resumen["oso_conf_alta"] += 1
            else:
                resumen["oso_conf_baja"] += 1
            if i % 500 == 0:
                print(f"{i}/{len(rutas)}")

    total = sum(resumen.values()) or 1
    print("\nResumen (umbral de 'alta' = 0.5):")
    for k in ("oso_conf_alta", "oso_conf_baja", "sin_oso"):
        print(f"  {k}: {resumen[k]} ({100 * resumen[k] / total:.1f} %)")
    print(f"CSV escrito en {args.out}")


if __name__ == "__main__":
    main()
