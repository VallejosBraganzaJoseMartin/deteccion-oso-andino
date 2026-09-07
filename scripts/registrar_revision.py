#!/usr/bin/env python3
"""
Registra en un CSV la revision manual hecha moviendo imagenes a subcarpetas.

Estructura esperada:
    CARPETA_REVISION/
        oso/            fotos con oso visible
        oso_cautivo/    oso en zoologico o encierro evidente
        rastro/         huellas, excrementos, restos
        sin_oso/        nada de lo anterior
        duda/           no se pudo decidir
        (archivos sueltos en la raiz = aun sin clasificar)

Uso:
    python registrar_revision.py --dir data/revisar/inat_sin_oso --lote sin_oso_yolo \
        --out docs/comprension_datos/revision_manual_inat.csv

Se puede correr varias veces con lotes distintos: cada corrida agrega filas
al CSV y reemplaza las de ese mismo lote si ya existian.
"""

import argparse
import os
from collections import Counter

import pandas as pd

EXT_IMG = {".jpg", ".jpeg", ".png"}
CATEGORIAS = ["oso", "oso_cautivo", "rastro", "sin_oso", "duda"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True, help="Carpeta de revision con las subcarpetas")
    ap.add_argument("--lote", required=True, help="Nombre del lote revisado (p. ej. sin_oso_yolo)")
    ap.add_argument("--out", required=True, help="CSV acumulado de revision manual")
    args = ap.parse_args()

    filas = []
    for cat in CATEGORIAS:
        carpeta = os.path.join(args.dir, cat)
        if not os.path.isdir(carpeta):
            continue
        for a in sorted(os.listdir(carpeta)):
            if os.path.splitext(a)[1].lower() in EXT_IMG:
                filas.append({"archivo": a, "categoria_manual": cat, "lote": args.lote})

    sueltos = [a for a in os.listdir(args.dir)
               if os.path.splitext(a)[1].lower() in EXT_IMG
               and os.path.isfile(os.path.join(args.dir, a))]
    otras_carpetas = [d for d in os.listdir(args.dir)
                      if os.path.isdir(os.path.join(args.dir, d)) and d not in CATEGORIAS]

    nuevo = pd.DataFrame(filas)
    if os.path.exists(args.out):
        previo = pd.read_csv(args.out)
        previo = previo[previo["lote"] != args.lote]
        total = pd.concat([previo, nuevo], ignore_index=True)
    else:
        total = nuevo
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    total.to_csv(args.out, index=False)

    print(f"Lote '{args.lote}': {len(nuevo)} imagenes clasificadas")
    for cat, n in Counter(nuevo["categoria_manual"]).most_common() if len(nuevo) else []:
        print(f"  {cat}: {n}")
    if sueltos:
        print(f"ATENCION: {len(sueltos)} imagenes siguen sin clasificar en la raiz de {args.dir}")
    if otras_carpetas:
        print(f"ATENCION: subcarpetas con nombre no reconocido (se ignoraron): {otras_carpetas}")
    print(f"CSV acumulado: {args.out} ({len(total)} filas en total)")


if __name__ == "__main__":
    main()
