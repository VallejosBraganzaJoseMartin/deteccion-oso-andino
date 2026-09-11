#!/usr/bin/env python3
"""
Fase CRISP-DM: Preparacion de los datos (seleccion de imagenes, ENA24).

Selecciona de ENA24 los positivos (todas las imagenes con caja de oso negro
americano) y una muestra de negativos (imagenes de otras especies, sin ninguna
caja de oso) que en el dataset final llevaran archivo de etiquetas vacio, es
decir, se entrenan como fondo.

Criterios de la muestra de negativos:
  - Solo imagenes que existen en disco y no estan corruptas (ena24_metadatos.csv).
  - Cuota parecida por cada categoria de --prioridad (subcadenas del nombre de
    categoria del JSON, en orden). Si una categoria no alcanza su cuota, el
    faltante se reparte entre las demas al final.
  - Dentro de cada categoria, mitad nocturnas (escala de grises) y mitad
    diurnas, en la medida en que haya.
  - Maximo --max-por-grupo imagenes de una misma rafaga (mismo phash), para
    que la muestra no se llene de fotos casi identicas.
  - Semilla fija: la misma corrida da siempre la misma seleccion.

Uso (desde la raiz del proyecto):
    python scripts/seleccionar_ena24.py --json data/raw/ena24/ena24.json \
        --meta docs/comprension_datos/ena24_metadatos.csv \
        --n-negativos 1200 --out docs/comprension_datos/ena24_seleccion.csv

NO PROBADO con el JSON real de ENA24 (solo con datos sinteticos con la misma
estructura). Revisar la lista de categorias que imprime al inicio y ajustar
--prioridad si los nombres no coinciden con lo esperado.
"""

import argparse
import json
import os
import random
from collections import Counter, defaultdict

import pandas as pd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", required=True, help="JSON COCO Camera Traps de ENA24")
    ap.add_argument("--meta", required=True, help="ena24_metadatos.csv de describir_dataset.py")
    ap.add_argument("--out", required=True, help="CSV de salida con la seleccion")
    ap.add_argument("--n-negativos", type=int, default=1200)
    ap.add_argument("--prioridad", default="dog,horse,coyote,bobcat,deer,domestic cat",
                    help="Subcadenas de categoria en orden de prioridad, separadas por comas")
    ap.add_argument("--max-por-grupo", type=int, default=2,
                    help="Maximo de negativos tomados de una misma rafaga (mismo phash)")
    ap.add_argument("--semilla", type=int, default=42)
    args = ap.parse_args()
    rng = random.Random(args.semilla)

    # ---- JSON: categorias por imagen y cajas de oso ----
    with open(args.json, encoding="utf-8") as f:
        coco = json.load(f)
    cats = {c["id"]: c["name"] for c in coco["categories"]}
    imgs = {im["id"]: im for im in coco["images"]}
    cats_img = defaultdict(set)
    cajas_oso = Counter()
    for a in coco["annotations"]:
        nombre = cats[a["category_id"]]
        cats_img[a["image_id"]].add(nombre)
        if "bear" in nombre.lower():
            cajas_oso[a["image_id"]] += 1

    # ---- Metadatos: existencia en disco, corrupcion, nocturna, grupo ----
    meta = pd.read_csv(args.meta)
    meta = meta[meta["corrupta"].astype(str).str.lower() != "true"].copy()
    meta["escala_grises"] = meta["escala_grises"].astype(str).str.lower() == "true"
    meta_d = meta.set_index("archivo").to_dict("index")

    filas, sin_disco = [], 0
    for iid, im in imgs.items():
        archivo = os.path.basename(im["file_name"])
        m = meta_d.get(archivo)
        if m is None:
            sin_disco += 1
            continue
        filas.append({
            "archivo": archivo, "image_id": iid,
            "categorias": sorted(cats_img.get(iid, [])),
            "n_cajas_oso": cajas_oso.get(iid, 0),
            "grupo": m["phash"], "nocturna": bool(m["escala_grises"]),
            "brillo_medio": m.get("brillo_medio"),
            "img_w": im.get("width"), "img_h": im.get("height"),
        })

    print(f"Imagenes en el JSON: {len(imgs)}; en disco y legibles: {len(filas)}; "
          f"sin archivo en disco (p. ej. Human): {sin_disco}")
    print("\nCategorias del JSON (imagenes en disco por categoria):")
    cont_cat = Counter(c for f in filas for c in f["categorias"])
    for nombre, n in sorted(cont_cat.items(), key=lambda x: -x[1]):
        print(f"  {nombre}: {n}")

    # ---- Positivos: toda imagen con al menos una caja de oso ----
    positivos = [f for f in filas if f["n_cajas_oso"] > 0]
    candidatos = [f for f in filas if f["n_cajas_oso"] == 0]

    # ---- Negativos: reparto por categoria prioritaria ----
    prioridad = [p.strip().lower() for p in args.prioridad.split(",") if p.strip()]

    def indice_prioridad(nombres):
        for i, p in enumerate(prioridad):
            if any(p in n.lower() for n in nombres):
                return i
        return None

    por_cat, resto = defaultdict(list), []
    for f in candidatos:
        i = indice_prioridad(f["categorias"])
        f["cat_prioritaria"] = prioridad[i] if i is not None else "otras"
        (por_cat[i] if i is not None else resto).append(f)
    for lst in por_cat.values():
        rng.shuffle(lst)
    rng.shuffle(resto)

    print("\nCandidatos a negativo por categoria prioritaria (nocturnas / diurnas):")
    for i, p in enumerate(prioridad):
        pool = por_cat.get(i, [])
        print(f"  {p}: {len(pool)} ({sum(f['nocturna'] for f in pool)} / "
              f"{sum(not f['nocturna'] for f in pool)})")
    print(f"  otras: {len(resto)}")

    tomados_por_grupo = Counter()
    elegidos = set()
    negativos = []

    def tomar(pool, cuota):
        n = 0
        for f in pool:
            if n >= cuota:
                break
            if f["archivo"] in elegidos or tomados_por_grupo[f["grupo"]] >= args.max_por_grupo:
                continue
            elegidos.add(f["archivo"])
            tomados_por_grupo[f["grupo"]] += 1
            negativos.append(f)
            n += 1
        return n

    cuota_cat = args.n_negativos // max(len(prioridad), 1)
    for i in range(len(prioridad)):
        pool = por_cat.get(i, [])
        noche = [f for f in pool if f["nocturna"]]
        dia = [f for f in pool if not f["nocturna"]]
        n1 = tomar(noche, cuota_cat // 2)
        n2 = tomar(dia, cuota_cat - n1)
        if n1 + n2 < cuota_cat:
            tomar(noche, cuota_cat - n1 - n2)

    # Relleno por turnos entre categorias (y "otras" al final) si falta cuota
    faltan = args.n_negativos - len(negativos)
    pools = [por_cat.get(i, []) for i in range(len(prioridad))] + [resto]
    idx = [0] * len(pools)
    while faltan > 0 and any(idx[k] < len(pools[k]) for k in range(len(pools))):
        for k, pool in enumerate(pools):
            if faltan <= 0:
                break
            while idx[k] < len(pool):
                f = pool[idx[k]]
                idx[k] += 1
                if tomar([f], 1):
                    faltan -= 1
                    break

    # ---- Salida ----
    filas_out = []
    for f in positivos:
        filas_out.append({**f, "rol": "positivo", "cat_prioritaria": ""})
    for f in negativos:
        filas_out.append({**f, "rol": "negativo"})
    df = pd.DataFrame(filas_out)
    df["categorias"] = df["categorias"].apply(";".join)
    cols = ["archivo", "image_id", "rol", "categorias", "cat_prioritaria", "n_cajas_oso",
            "grupo", "nocturna", "brillo_medio", "img_w", "img_h"]
    df = df[cols].sort_values(["rol", "archivo"])
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    df.to_csv(args.out, index=False)

    print(f"\nPositivos: {len(positivos)} imagenes, {sum(f['n_cajas_oso'] for f in positivos)} cajas de oso; "
          f"nocturnas {sum(f['nocturna'] for f in positivos)}, grupos distintos "
          f"{len({f['grupo'] for f in positivos})}")
    print(f"Negativos: {len(negativos)} de {args.n_negativos} pedidos; "
          f"nocturnas {sum(f['nocturna'] for f in negativos)}, grupos distintos "
          f"{len({f['grupo'] for f in negativos})}, maximo por grupo "
          f"{max(tomados_por_grupo.values()) if tomados_por_grupo else 0}")
    print("Negativos por categoria prioritaria:")
    for p, n in Counter(f["cat_prioritaria"] for f in negativos).most_common():
        print(f"  {p}: {n}")
    print(f"\nSeleccion escrita en {args.out} ({len(df)} filas)")


if __name__ == "__main__":
    main()
