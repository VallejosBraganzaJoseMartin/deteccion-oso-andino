#!/usr/bin/env python3
"""
Genera las particiones por fuente para la evaluacion separada (decision D15).

A partir de division.csv escribe, por cada conjunto (val y test por defecto) y
cada fuente (inat, ena24):
  - una lista de rutas, una imagen por linea, en la RAIZ del dataset:
        <dataset>/<conjunto>_<fuente>.txt
  - un dataset_<conjunto>_<fuente>.yaml en <out>/ que apunta a esa lista

Asi se puede correr la validacion de Ultralytics tres veces por modelo
(conjunto completo, solo ENA24, solo iNaturalist) sin mover ni duplicar
imagenes: las listas apuntan a los archivos donde ya estan.

Por que las listas van en la raiz y no en una subcarpeta (verificado el 15 de
septiembre de 2026 en ultralytics/data/base.py, metodo get_img_files, rama
main, version 8.4.153): al leer un .txt, Ultralytics sustituye el prefijo
'./' de cada linea por la carpeta donde esta el .txt, no por el campo 'path'
del yaml. Con la lista en la raiz, './images/test/x.jpg' resuelve a
<dataset>/images/test/x.jpg, que es lo que queremos. Es la misma convencion
de las particiones de COCO (val2017.txt en la raiz del dataset). La etiqueta
se localiza sustituyendo el ultimo '/images/' por '/labels/', asi que
funciona igual que con carpetas.

Los yaml llevan en 'path' la ruta absoluta del dataset en este equipo. En
Kaggle hay que cambiar ese campo (o regenerar con --base); las listas no
cambian porque son relativas a la raiz.

Uso tipico (desde la raiz del proyecto):
    python scripts/particiones_por_fuente.py
    python scripts/particiones_por_fuente.py --base /kaggle/input/oso-proxy-v1
"""

import argparse
import os

import pandas as pd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--division", default="docs/preparacion_datos/division.csv")
    ap.add_argument("--dataset", default="data/processed",
                    help="Raiz del dataset; las listas se escriben aqui")
    ap.add_argument("--out", default="data/processed/particiones",
                    help="Carpeta donde se escriben los yaml")
    ap.add_argument("--base", default=None,
                    help="Raiz que se escribira en el campo 'path' de los yaml. "
                         "Por defecto, la ruta absoluta de --dataset")
    ap.add_argument("--conjuntos", nargs="+", default=["val", "test"])
    args = ap.parse_args()

    df = pd.read_csv(args.division)
    faltan = {"archivo_final", "fuente", "conjunto"} - set(df.columns)
    if faltan:
        raise SystemExit(f"A {args.division} le faltan columnas: {sorted(faltan)}")

    base = args.base or os.path.abspath(args.dataset).replace("\\", "/")
    os.makedirs(args.out, exist_ok=True)

    generados = []
    faltantes = 0
    for conjunto in args.conjuntos:
        sub_conj = df[df["conjunto"] == conjunto]
        if sub_conj.empty:
            print(f"AVISO: no hay imagenes del conjunto '{conjunto}' en la division")
            continue
        for fuente in sorted(sub_conj["fuente"].unique()):
            sub = sub_conj[sub_conj["fuente"] == fuente]
            nombre = f"{conjunto}_{fuente}"
            ruta_lista = os.path.join(args.dataset, nombre + ".txt")
            with open(ruta_lista, "w", encoding="utf-8") as f:
                for archivo in sorted(sub["archivo_final"]):
                    f.write(f"./images/{conjunto}/{archivo}\n")
                    if not os.path.isfile(os.path.join(args.dataset, "images", conjunto, archivo)):
                        faltantes += 1

            ruta_yaml = os.path.join(args.out, f"dataset_{nombre}.yaml")
            with open(ruta_yaml, "w", encoding="utf-8") as f:
                f.write(f"# Particion de evaluacion: conjunto '{conjunto}', fuente '{fuente}'.\n"
                        f"# Generado por particiones_por_fuente.py a partir de division.csv.\n"
                        f"# En Kaggle, regenerar con --base o cambiar 'path' a mano.\n"
                        f"path: {base}\n"
                        f"train: images/train\n"
                        f"val: {nombre}.txt\n"
                        f"test: {nombre}.txt\n"
                        f"nc: 1\n"
                        f"names: ['oso']\n")
            n_cajas = int(sub["n_cajas"].sum()) if "n_cajas" in sub.columns else -1
            con_oso = int((sub["n_cajas"] > 0).sum()) if "n_cajas" in sub.columns else -1
            generados.append((nombre, len(sub), con_oso, n_cajas, ruta_lista, ruta_yaml))

    print(f"Raiz declarada en los yaml (campo 'path'): {base}\n")
    print(f"{'particion':<14}{'imagenes':>10}{'con oso':>9}{'cajas':>8}  lista")
    for nombre, n, con_oso, cajas, lista, _ in generados:
        print(f"{nombre:<14}{n:>10}{con_oso:>9}{cajas:>8}  {lista}")
    print(f"\nYaml en {args.out}")
    if faltantes:
        print(f"AVISO: {faltantes} rutas de las listas no existen en disco. Revisar division.csv.")
    else:
        print("Todas las rutas de las listas existen en disco.")
    print("Para evaluar, pasar el yaml correspondiente al argumento 'data' de la "
          "validacion de Ultralytics.")


if __name__ == "__main__":
    main()
