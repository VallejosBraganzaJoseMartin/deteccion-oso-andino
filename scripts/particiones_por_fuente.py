#!/usr/bin/env python3
"""
Genera las particiones por fuente para la evaluacion separada (decision D15).

A partir de division.csv escribe, por cada conjunto (val y test por defecto) y
cada fuente (inat, ena24):
  - una lista de rutas, un archivo por linea, en <out>/listas/
  - un dataset_<conjunto>_<fuente>.yaml en <out>/ que apunta a esa lista

Asi se puede correr la validacion de Ultralytics tres veces por modelo
(conjunto completo, solo ENA24, solo iNaturalist) sin mover ni duplicar
imagenes: las listas apuntan a los archivos donde ya estan.

Las rutas de las listas se escriben relativas a la raiz del dataset (el campo
'path' del yaml), que es la forma en que estan escritas las particiones de
COCO. Con --base se puede fijar otra raiz, por ejemplo la que tendra el
dataset dentro de Kaggle:

    python scripts/particiones_por_fuente.py
    python scripts/particiones_por_fuente.py --base /kaggle/input/oso-proxy-v1

Uso tipico (desde la raiz del proyecto):
    python scripts/particiones_por_fuente.py

NO PROBADO con Ultralytics. Que la libreria acepte un archivo de texto con una
ruta por linea en lugar de una carpeta hay que confirmarlo en la documentacion
de la version instalada antes de depender de ello. Si no lo aceptara, la
alternativa es evaluar sobre el conjunto completo y separar las metricas por
fuente a partir de las predicciones guardadas.
"""

import argparse
import os

import pandas as pd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--division", default="docs/preparacion_datos/division.csv")
    ap.add_argument("--dataset", default="data/processed",
                    help="Carpeta del dataset, de donde se lee el yaml original")
    ap.add_argument("--out", default="data/processed/particiones")
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
    dir_listas = os.path.join(args.out, "listas")
    os.makedirs(dir_listas, exist_ok=True)

    generados = []
    for conjunto in args.conjuntos:
        sub_conj = df[df["conjunto"] == conjunto]
        if sub_conj.empty:
            print(f"AVISO: no hay imagenes del conjunto '{conjunto}' en la division")
            continue
        for fuente in sorted(sub_conj["fuente"].unique()):
            sub = sub_conj[sub_conj["fuente"] == fuente]
            nombre = f"{conjunto}_{fuente}"
            ruta_lista = os.path.join(dir_listas, nombre + ".txt")
            with open(ruta_lista, "w", encoding="utf-8") as f:
                for archivo in sorted(sub["archivo_final"]):
                    f.write(f"./images/{conjunto}/{archivo}\n")

            # La lista se referencia relativa a 'path', igual que las carpetas
            rel = os.path.relpath(ruta_lista, os.path.abspath(args.dataset)).replace("\\", "/")
            ruta_yaml = os.path.join(args.out, f"dataset_{nombre}.yaml")
            with open(ruta_yaml, "w", encoding="utf-8") as f:
                f.write(f"# Particion de evaluacion: conjunto '{conjunto}', fuente '{fuente}'.\n"
                        f"# Generado por particiones_por_fuente.py a partir de division.csv.\n"
                        f"# En Kaggle, regenerar con --base o cambiar 'path' a mano.\n"
                        f"path: {base}\n"
                        f"train: images/train\n"
                        f"val: {rel}\n"
                        f"test: {rel}\n"
                        f"nc: 1\n"
                        f"names: ['oso']\n")
            n_cajas = int(sub["n_cajas"].sum()) if "n_cajas" in sub.columns else -1
            con_oso = int((sub["n_cajas"] > 0).sum()) if "n_cajas" in sub.columns else -1
            generados.append((nombre, len(sub), con_oso, n_cajas, ruta_yaml))

    print(f"Raiz declarada en los yaml (campo 'path'): {base}\n")
    print(f"{'particion':<14}{'imagenes':>10}{'con oso':>9}{'cajas':>8}  yaml")
    for nombre, n, con_oso, cajas, ruta in generados:
        print(f"{nombre:<14}{n:>10}{con_oso:>9}{cajas:>8}  {ruta}")
    print(f"\nListas en {dir_listas}")
    print("Para evaluar, pasar el yaml correspondiente al argumento 'data' de la "
          "validacion de Ultralytics.")


if __name__ == "__main__":
    main()
