#!/usr/bin/env python3
"""
Fase CRISP-DM: Preparacion de los datos (division train/val/test y dataset final).

Construye data/processed/ en el formato que espera Ultralytics:
    images/{train,val,test}/   labels/{train,val,test}/   dataset.yaml
a partir de:
  - iNaturalist: imagenes de data/etiquetado/inat/images con las cajas
    revisadas de labels_rev (segun inat_etiquetado.csv, solo 'etiquetada').
  - ENA24: imagenes de data/raw/ena24 segun ena24_seleccion.csv; las cajas de
    oso se convierten de pixeles (x, y, w, h) a YOLO normalizado; los
    negativos reciben un .txt vacio (Ultralytics los trata como fondo).

Division por GRUPOS, no por imagen, para que fotos casi identicas no queden
repartidas entre conjuntos:
  - iNaturalist: mismo observacion_id o mismo phash => mismo grupo (union).
  - ENA24: mismo phash (rafaga) => mismo grupo.
Estratificada por (fuente, rol, dia/noche) y con semilla fija. Los archivos
de ENA24 se renombran con prefijo 'ena24_' para conservar la procedencia.

Ademas dibuja las cajas finales sobre una muestra de imagenes de cada fuente
en docs/preparacion_datos/muestra_cajas/ para comprobar a ojo la conversion.

Uso (desde la raiz del proyecto; los valores por defecto siguen la
estructura del proyecto, asi que normalmente basta):
    python scripts/dividir_dataset.py

NO PROBADO con los datos reales (solo con datos sinteticos con la misma
estructura). Si data/processed ya existe y no esta vacia, el script se
detiene: borrarla a mano antes de volver a correr.
"""

import argparse
import os
import random
import shutil
from collections import Counter, defaultdict

import pandas as pd
from PIL import Image, ImageDraw

CONJUNTOS = ("train", "val", "test")


def uf_find(padre, x):
    while padre[x] != x:
        padre[x] = padre[padre[x]]
        x = padre[x]
    return x


def uf_union(padre, a, b):
    ra, rb = uf_find(padre, a), uf_find(padre, b)
    if ra != rb:
        padre[rb] = ra


def asignar_grupos(tamanos, fracciones, rng):
    """Reparte grupos entre conjuntos acercandose a las fracciones (por numero de imagenes)."""
    orden = list(tamanos)
    rng.shuffle(orden)
    conteo = {c: 0 for c in CONJUNTOS}
    asignacion, total = {}, 0
    for g in orden:
        total += tamanos[g]
        c = max(CONJUNTOS, key=lambda k: fracciones[k] * total - conteo[k])
        asignacion[g] = c
        conteo[c] += tamanos[g]
    return asignacion


def leer_lineas(ruta):
    with open(ruta, encoding="utf-8") as f:
        return [l.strip() for l in f if l.strip()]


def dibujar(ruta_img, lineas, destino):
    with Image.open(ruta_img) as im:
        im = im.convert("RGB")
        W, H = im.size
        d = ImageDraw.Draw(im)
        for l in lineas:
            _, cx, cy, w, h = map(float, l.split())
            x1, y1 = (cx - w / 2) * W, (cy - h / 2) * H
            x2, y2 = (cx + w / 2) * W, (cy + h / 2) * H
            d.rectangle([x1, y1, x2, y2], outline=(255, 0, 0), width=max(2, W // 300))
        im.thumbnail((1024, 1024))
        im.save(destino, quality=85)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inat", default="docs/preparacion_datos/inat_etiquetado.csv")
    ap.add_argument("--inat-img", default="data/etiquetado/inat/images")
    ap.add_argument("--inat-lbl", default="data/etiquetado/inat/labels_rev")
    ap.add_argument("--inat-meta", default="docs/comprension_datos/inat_metadatos.csv")
    ap.add_argument("--ena", default="docs/comprension_datos/ena24_seleccion.csv")
    ap.add_argument("--ena-img", default="data/raw/ena24")
    ap.add_argument("--ena-cajas", default="docs/comprension_datos/ena24_cajas_oso.csv")
    ap.add_argument("--ena-meta", default="docs/comprension_datos/ena24_metadatos.csv")
    ap.add_argument("--out", default="data/processed")
    ap.add_argument("--division", default="docs/preparacion_datos/division.csv")
    ap.add_argument("--muestra", default="docs/preparacion_datos/muestra_cajas")
    ap.add_argument("--n-muestra", type=int, default=12, help="Imagenes por fuente para la muestra visual")
    ap.add_argument("--frac", nargs=3, type=float, default=[0.70, 0.15, 0.15], metavar=("TRAIN", "VAL", "TEST"))
    ap.add_argument("--semilla", type=int, default=42)
    args = ap.parse_args()
    rng = random.Random(args.semilla)
    fracciones = dict(zip(CONJUNTOS, args.frac))

    if os.path.isdir(args.out) and os.listdir(args.out):
        raise SystemExit(f"{args.out} ya existe y no esta vacia. Borrala a mano si quieres regenerarla.")

    registros = []  # una entrada por imagen final

    # ---------------- iNaturalist ----------------
    inat = pd.read_csv(args.inat)
    inat = inat[inat["estado_final"] == "etiquetada"].copy()
    inat["observacion_id"] = inat["observacion_id"].astype("Int64").astype(str)
    padre = {a: a for a in inat["archivo"]}
    por_obs = defaultdict(list)
    for a, o in zip(inat["archivo"], inat["observacion_id"]):
        por_obs[o].append(a)
    for lst in por_obs.values():
        for a in lst[1:]:
            uf_union(padre, lst[0], a)
    uniones_hash = 0
    if os.path.exists(args.inat_meta):
        meta = pd.read_csv(args.inat_meta, dtype={"phash": str})
        meta = meta[meta["archivo"].isin(padre)]
        for h, g in meta.groupby("phash"):
            archivos = list(g["archivo"])
            for a in archivos[1:]:
                if uf_find(padre, archivos[0]) != uf_find(padre, a):
                    uniones_hash += 1
                uf_union(padre, archivos[0], a)
    faltan_inat = []
    for _, r in inat.iterrows():
        base = os.path.splitext(r["archivo"])[0]
        src_img = os.path.join(args.inat_img, r["archivo"])
        src_lbl = os.path.join(args.inat_lbl, base + ".txt")
        if not (os.path.exists(src_img) and os.path.exists(src_lbl)):
            faltan_inat.append(r["archivo"])
            continue
        lineas = leer_lineas(src_lbl)
        registros.append({"archivo_final": r["archivo"], "archivo_origen": src_img, "fuente": "inat",
                          "rol": "positivo", "nocturna": False,
                          "grupo": "inat_" + os.path.splitext(uf_find(padre, r["archivo"]))[0].replace("inat_", "", 1),
                          "estrato": "inat_pos", "lineas": lineas, "n_cajas": len(lineas)})
    print(f"iNaturalist: {len(registros)} imagenes etiquetadas; grupos por observacion {len(por_obs)}, "
          f"uniones extra por phash {uniones_hash}; faltantes {len(faltan_inat)}")
    for a in faltan_inat[:10]:
        print(f"  falta imagen o etiqueta: {a}")

    # ---------------- ENA24 ----------------
    ena = pd.read_csv(args.ena, dtype={"grupo": str, "archivo": str})
    cajas = pd.read_csv(args.ena_cajas)
    cajas["archivo"] = cajas["file_name"].astype(str).map(os.path.basename)
    cajas_por_img = {a: g for a, g in cajas.groupby("archivo")}
    dims_disco = {}
    if os.path.exists(args.ena_meta):
        m = pd.read_csv(args.ena_meta)
        dims_disco = {a: (w, h) for a, w, h in zip(m["archivo"], m["ancho"], m["alto"])}
    n_ena, faltan_ena, dims_distintas, cajas_recortadas = 0, [], 0, 0
    for _, r in ena.iterrows():
        src_img = os.path.join(args.ena_img, r["archivo"])
        if not os.path.exists(src_img):
            alt = os.path.join(args.ena_img, "images", r["archivo"])
            if os.path.exists(alt):
                src_img = alt
            else:
                faltan_ena.append(r["archivo"])
                continue
        lineas = []
        if r["rol"] == "positivo":
            g = cajas_por_img.get(r["archivo"])
            if g is None:
                faltan_ena.append(r["archivo"] + " (sin cajas en el CSV)")
                continue
            for _, c in g.iterrows():
                W, H = float(c["img_w"]), float(c["img_h"])
                if r["archivo"] in dims_disco and (int(W), int(H)) != tuple(int(v) for v in dims_disco[r["archivo"]]):
                    dims_distintas += 1
                x, y, w, h = float(c["x"]), float(c["y"]), float(c["w"]), float(c["h"])
                x1, y1 = max(0.0, x), max(0.0, y)
                x2, y2 = min(W, x + w), min(H, y + h)
                if (x1, y1, x2, y2) != (x, y, x + w, y + h):
                    cajas_recortadas += 1
                if x2 <= x1 or y2 <= y1:
                    continue
                cx, cy = (x1 + x2) / 2 / W, (y1 + y2) / 2 / H
                lineas.append(f"0 {cx:.6f} {cy:.6f} {(x2 - x1) / W:.6f} {(y2 - y1) / H:.6f}")
        noct = str(r["nocturna"]).lower() == "true"
        estrato = f"ena24_{'pos' if r['rol'] == 'positivo' else 'neg'}_{'noche' if noct else 'dia'}"
        registros.append({"archivo_final": "ena24_" + r["archivo"], "archivo_origen": src_img,
                          "fuente": "ena24", "rol": r["rol"], "nocturna": noct,
                          "grupo": "ena24_" + str(r["grupo"]), "estrato": estrato,
                          "lineas": lineas, "n_cajas": len(lineas)})
        n_ena += 1
    print(f"ENA24: {n_ena} imagenes ({int((ena['rol'] == 'positivo').sum())} positivas en el CSV); "
          f"faltantes {len(faltan_ena)}; cajas con dimensiones JSON != disco: {dims_distintas}; "
          f"cajas recortadas al borde: {cajas_recortadas}")
    for a in faltan_ena[:10]:
        print(f"  falta: {a}")

    # ---------------- Division por grupos y estratos ----------------
    df = pd.DataFrame(registros)
    df["conjunto"] = ""
    for estrato, sub in df.groupby("estrato"):
        tamanos = Counter(sub["grupo"])
        asign = asignar_grupos(tamanos, fracciones, rng)
        df.loc[sub.index, "conjunto"] = sub["grupo"].map(asign)
    # Un grupo podria aparecer en dos estratos (rafaga de ENA24 con y sin oso); se fuerza un solo conjunto
    conj_por_grupo = df.groupby("grupo")["conjunto"].agg(lambda s: s.iloc[0])
    df["conjunto"] = df["grupo"].map(conj_por_grupo)
    assert df.groupby("grupo")["conjunto"].nunique().max() == 1

    # ---------------- Escritura ----------------
    for c in CONJUNTOS:
        os.makedirs(os.path.join(args.out, "images", c), exist_ok=True)
        os.makedirs(os.path.join(args.out, "labels", c), exist_ok=True)
    for i, r in enumerate(df.itertuples(index=False), 1):
        shutil.copy2(r.archivo_origen, os.path.join(args.out, "images", r.conjunto, r.archivo_final))
        base = os.path.splitext(r.archivo_final)[0]
        with open(os.path.join(args.out, "labels", r.conjunto, base + ".txt"), "w", encoding="utf-8") as f:
            f.write("\n".join(r.lineas) + ("\n" if r.lineas else ""))
        if i % 500 == 0:
            print(f"  copiadas {i}/{len(df)}")

    ruta_abs = os.path.abspath(args.out).replace("\\", "/")
    with open(os.path.join(args.out, "dataset.yaml"), "w", encoding="utf-8") as f:
        f.write(f"# Dataset proxy iteracion 1 (iNaturalist + ENA24). Generado por dividir_dataset.py\n"
                f"# En Kaggle cambiar 'path' por la ruta donde quede descomprimida la carpeta.\n"
                f"path: {ruta_abs}\ntrain: images/train\nval: images/val\ntest: images/test\n"
                f"nc: 1\nnames: ['oso']\n")

    os.makedirs(os.path.dirname(os.path.abspath(args.division)), exist_ok=True)
    df.drop(columns=["lineas"]).to_csv(args.division, index=False)

    # ---------------- Muestra visual ----------------
    os.makedirs(args.muestra, exist_ok=True)
    for fuente in ("inat", "ena24"):
        sub = df[(df["fuente"] == fuente) & (df["n_cajas"] > 0)]
        for r in sub.sample(n=min(args.n_muestra, len(sub)), random_state=args.semilla).itertuples():
            dibujar(os.path.join(args.out, "images", r.conjunto, r.archivo_final), r.lineas,
                    os.path.join(args.muestra, f"{fuente}_{r.archivo_final}"))

    # ---------------- Resumen ----------------
    print(f"\nTotal: {len(df)} imagenes, {int(df['n_cajas'].sum())} cajas, "
          f"{df['grupo'].nunique()} grupos")
    print(f"{'conjunto':<8}{'imagenes':>10}{'positivas':>11}{'negativas':>11}{'cajas':>8}"
          f"{'inat':>7}{'ena24':>7}{'noche':>7}")
    for c in CONJUNTOS:
        s = df[df["conjunto"] == c]
        print(f"{c:<8}{len(s):>10}{int((s['rol'] == 'positivo').sum()):>11}"
              f"{int((s['rol'] == 'negativo').sum()):>11}{int(s['n_cajas'].sum()):>8}"
              f"{int((s['fuente'] == 'inat').sum()):>7}{int((s['fuente'] == 'ena24').sum()):>7}"
              f"{int(s['nocturna'].sum()):>7}")
    print("\nPor estrato y conjunto:")
    print(pd.crosstab(df["estrato"], df["conjunto"]).to_string())
    print(f"\nDataset en {args.out} (dataset.yaml con path absoluto)")
    print(f"Division en {args.division}; muestra con cajas dibujadas en {args.muestra}")


if __name__ == "__main__":
    main()
