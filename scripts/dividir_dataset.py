#!/usr/bin/env python3
"""
Fase CRISP-DM: Preparacion de los datos (division train/val/test y dataset final).

VERSION 2: reemplaza a la primera version. Cambios respecto de aquella:
  - Incorpora negativos de iNaturalist (rastros, paisajes y fotos de otras
    especies), que antes no existian.
  - Agrupa ademas por distancia de Hamming entre hashes perceptuales, no solo
    por coincidencia exacta, para que las rafagas casi identicas no se
    repartan entre conjuntos.
  - Nuevo estrato inat_neg en la estratificacion.

Construye data/processed/ en el formato que espera Ultralytics:
    images/{train,val,test}/   labels/{train,val,test}/   dataset.yaml

Fuentes:
  - iNaturalist positivos: data/etiquetado/inat/images + labels_rev, segun
    inat_etiquetado.csv (solo estado_final == 'etiquetada').
  - iNaturalist negativos: las carpetas de --inat-neg-dirs, excluyendo los
    archivos listados en --inat-neg-descartes. Reciben .txt vacio.
  - ENA24: segun ena24_seleccion.csv; las cajas de oso se convierten de
    pixeles a YOLO normalizado y los negativos reciben .txt vacio.

Grupos (unidad de la division, para no repartir imagenes casi identicas):
  - iNaturalist: misma observacion (prefijo del nombre de archivo) o hashes a
    distancia <= --hamming.
  - ENA24: mismo phash o a distancia <= --hamming.
Estratificado por (fuente, rol, dia/noche) con semilla fija.

Uso (desde la raiz del proyecto):
    python scripts/dividir_dataset.py

NO PROBADO con los datos reales (solo con datos sinteticos con la misma
estructura). Si data/processed ya existe y no esta vacia, el script se
detiene: borrarla a mano antes de volver a correr.
"""

import argparse
import itertools
import os
import random
import re
import shutil
from collections import Counter, defaultdict

import pandas as pd
from PIL import Image, ImageDraw

CONJUNTOS = ("train", "val", "test")
EXT_IMG = {".jpg", ".jpeg", ".png"}
PATRON_OBS = re.compile(r"^inat(?:neg)?_(\d+)_")

DIRS_NEG_INAT = [
    "data/revisar/inat_sin_oso/rastro",
    "data/revisar/inat_conf_baja/rastro",
    "data/revisar/inat_sin_oso/sin_oso/paisaje",
    "data/revisar/inat_conf_baja/sin_oso/paisaje",
    "data/raw/inaturalist_negativos",
    "data/frames_gif/inat_negativas_elegidos",
]


def uf_find(padre, x):
    while padre[x] != x:
        padre[x] = padre[padre[x]]
        x = padre[x]
    return x


def uf_union(padre, a, b):
    ra, rb = uf_find(padre, a), uf_find(padre, b)
    if ra != rb:
        padre[rb] = ra


def unir_por_hamming(padre, hashes, umbral):
    """hashes: {archivo: int}. Une los pares a distancia <= umbral. Devuelve las uniones nuevas."""
    if umbral <= 0:
        return 0
    items = [(a, h) for a, h in hashes.items() if h is not None]
    nuevas = 0
    for (a, ha), (b, hb) in itertools.combinations(items, 2):
        if bin(ha ^ hb).count("1") <= umbral:
            if uf_find(padre, a) != uf_find(padre, b):
                nuevas += 1
            uf_union(padre, a, b)
    return nuevas


def phash_de(ruta):
    import imagehash
    try:
        with Image.open(ruta) as im:
            return int(str(imagehash.phash(im.convert("RGB"))), 16)
    except Exception:
        return None


def obs_de(nombre):
    m = PATRON_OBS.match(nombre)
    return m.group(1) if m else None


def asignar_grupos(tamanos, fracciones, rng):
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
            d.rectangle([(cx - w / 2) * W, (cy - h / 2) * H, (cx + w / 2) * W, (cy + h / 2) * H],
                        outline=(255, 0, 0), width=max(2, W // 300))
        im.thumbnail((1024, 1024))
        im.save(destino, quality=85)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inat", default="docs/preparacion_datos/inat_etiquetado.csv")
    ap.add_argument("--inat-img", default="data/etiquetado/inat/images")
    ap.add_argument("--inat-lbl", default="data/etiquetado/inat/labels_rev")
    ap.add_argument("--inat-meta", default="docs/comprension_datos/inat_metadatos.csv")
    ap.add_argument("--inat-neg-dirs", nargs="*", default=DIRS_NEG_INAT)
    ap.add_argument("--inat-neg-descartes", default="docs/preparacion_datos/descartes_inat_negativos.txt")
    ap.add_argument("--ena", default="docs/comprension_datos/ena24_seleccion.csv")
    ap.add_argument("--ena-img", default="data/raw/ena24")
    ap.add_argument("--ena-cajas", default="docs/comprension_datos/ena24_cajas_oso.csv")
    ap.add_argument("--out", default="data/processed")
    ap.add_argument("--division", default="docs/preparacion_datos/division.csv")
    ap.add_argument("--muestra", default="docs/preparacion_datos/muestra_cajas")
    ap.add_argument("--n-muestra", type=int, default=40)
    ap.add_argument("--hamming", type=int, default=8, help="0 desactiva la union por distancia")
    ap.add_argument("--frac", nargs=3, type=float, default=[0.70, 0.15, 0.15],
                    metavar=("TRAIN", "VAL", "TEST"))
    ap.add_argument("--semilla", type=int, default=42)
    args = ap.parse_args()
    rng = random.Random(args.semilla)
    fracciones = dict(zip(CONJUNTOS, args.frac))

    if os.path.isdir(args.out) and os.listdir(args.out):
        raise SystemExit(f"{args.out} ya existe y no esta vacia. Borrala a mano si quieres regenerarla.")

    registros = []
    hashes = {"inat": {}, "ena24": {}}

    # ---------------- iNaturalist: positivos ----------------
    inat = pd.read_csv(args.inat)
    inat = inat[inat["estado_final"] == "etiquetada"].copy()
    meta_hash = {}
    if os.path.exists(args.inat_meta):
        m = pd.read_csv(args.inat_meta, dtype={"phash": str})
        meta_hash = {a: h for a, h in zip(m["archivo"], m["phash"]) if isinstance(h, str)}
    faltan_inat = []
    for _, r in inat.iterrows():
        base = os.path.splitext(r["archivo"])[0]
        src_img = os.path.join(args.inat_img, r["archivo"])
        src_lbl = os.path.join(args.inat_lbl, base + ".txt")
        if not (os.path.exists(src_img) and os.path.exists(src_lbl)):
            faltan_inat.append(r["archivo"])
            continue
        h = meta_hash.get(r["archivo"])
        hashes["inat"][r["archivo"]] = int(h, 16) if h else phash_de(src_img)
        registros.append({"archivo_final": r["archivo"], "archivo_origen": src_img, "fuente": "inat",
                          "rol": "positivo", "nocturna": False, "estrato": "inat_pos",
                          "lineas": leer_lineas(src_lbl), "n_cajas": len(leer_lineas(src_lbl))})
    print(f"iNaturalist positivos: {len(registros)} (faltantes: {len(faltan_inat)})")
    for a in faltan_inat[:10]:
        print(f"  falta imagen o etiqueta: {a}")

    # ---------------- iNaturalist: negativos ----------------
    descartados = set()
    if os.path.exists(args.inat_neg_descartes):
        for l in open(args.inat_neg_descartes, encoding="utf-8"):
            if ";" in l:
                descartados.add(l.split(";", 1)[0].strip())
    else:
        print(f"  AVISO: no se encontro {args.inat_neg_descartes}; no se excluye nada")
    vistos = {r["archivo_final"] for r in registros}
    n_neg, por_dir, saltados = 0, Counter(), 0
    for d in args.inat_neg_dirs:
        if not os.path.isdir(d):
            print(f"  AVISO: no existe la carpeta de negativos {d}")
            continue
        for raiz, _, archivos in os.walk(d):
            for a in sorted(archivos):
                if os.path.splitext(a)[1].lower() not in EXT_IMG:
                    continue
                if a in descartados or a in vistos:
                    saltados += 1
                    continue
                ruta = os.path.join(raiz, a)
                vistos.add(a)
                hashes["inat"][a] = phash_de(ruta)
                registros.append({"archivo_final": a, "archivo_origen": ruta, "fuente": "inat",
                                  "rol": "negativo", "nocturna": False, "estrato": "inat_neg",
                                  "lineas": [], "n_cajas": 0})
                n_neg += 1
                por_dir[d] += 1
    print(f"iNaturalist negativos: {n_neg} (descartados o repetidos: {saltados})")
    for d, n in por_dir.items():
        print(f"  {d}: {n}")

    # ---------------- ENA24 ----------------
    ena = pd.read_csv(args.ena, dtype={"grupo": str, "archivo": str})
    cajas = pd.read_csv(args.ena_cajas)
    cajas["archivo"] = cajas["file_name"].astype(str).map(os.path.basename)
    cajas_por_img = {a: g for a, g in cajas.groupby("archivo")}
    n_ena, faltan_ena, cajas_recortadas = 0, [], 0
    for _, r in ena.iterrows():
        src_img = os.path.join(args.ena_img, r["archivo"])
        if not os.path.exists(src_img):
            alt = os.path.join(args.ena_img, "images", r["archivo"])
            if not os.path.exists(alt):
                faltan_ena.append(r["archivo"])
                continue
            src_img = alt
        lineas = []
        if r["rol"] == "positivo":
            g = cajas_por_img.get(r["archivo"])
            if g is None:
                faltan_ena.append(r["archivo"] + " (sin cajas en el CSV)")
                continue
            for _, c in g.iterrows():
                W, H = float(c["img_w"]), float(c["img_h"])
                x, y, w, h = float(c["x"]), float(c["y"]), float(c["w"]), float(c["h"])
                x1, y1, x2, y2 = max(0.0, x), max(0.0, y), min(W, x + w), min(H, y + h)
                if (x1, y1, x2, y2) != (x, y, x + w, y + h):
                    cajas_recortadas += 1
                if x2 <= x1 or y2 <= y1:
                    continue
                lineas.append(f"0 {(x1 + x2) / 2 / W:.6f} {(y1 + y2) / 2 / H:.6f} "
                              f"{(x2 - x1) / W:.6f} {(y2 - y1) / H:.6f}")
        noct = str(r["nocturna"]).lower() == "true"
        nombre = "ena24_" + r["archivo"]
        grupo_hex = str(r["grupo"])
        hashes["ena24"][nombre] = int(grupo_hex, 16) if re.fullmatch(r"[0-9a-fA-F]+", grupo_hex) else None
        registros.append({"archivo_final": nombre, "archivo_origen": src_img, "fuente": "ena24",
                          "rol": r["rol"], "nocturna": noct,
                          "estrato": f"ena24_{'pos' if r['rol'] == 'positivo' else 'neg'}"
                                     f"_{'noche' if noct else 'dia'}",
                          "lineas": lineas, "n_cajas": len(lineas)})
        n_ena += 1
    print(f"ENA24: {n_ena} imagenes; faltantes {len(faltan_ena)}; "
          f"cajas recortadas al borde: {cajas_recortadas}")
    for a in faltan_ena[:10]:
        print(f"  falta: {a}")

    # ---------------- Grupos ----------------
    df = pd.DataFrame(registros)
    padre = {a: a for a in df["archivo_final"]}
    por_obs = defaultdict(list)
    for a in df.loc[df["fuente"] == "inat", "archivo_final"]:
        o = obs_de(a)
        if o:
            por_obs[o].append(a)
    for lst in por_obs.values():
        for a in lst[1:]:
            uf_union(padre, lst[0], a)
    print(f"\nGrupos: iNaturalist por observacion {len(por_obs)}")
    for fuente in ("inat", "ena24"):
        n = unir_por_hamming(padre, hashes[fuente], args.hamming)
        sin_hash = sum(1 for v in hashes[fuente].values() if v is None)
        print(f"  {fuente}: uniones extra por distancia <= {args.hamming}: {n}"
              f"{f' (sin hash: {sin_hash})' if sin_hash else ''}")
    df["grupo"] = df["archivo_final"].map(lambda a: uf_find(padre, a))

    # ---------------- Division ----------------
    df["conjunto"] = ""
    for _, sub in df.groupby("estrato"):
        asign = asignar_grupos(Counter(sub["grupo"]), fracciones, rng)
        df.loc[sub.index, "conjunto"] = sub["grupo"].map(asign)
    df["conjunto"] = df["grupo"].map(df.groupby("grupo")["conjunto"].agg(lambda s: s.iloc[0]))
    assert df.groupby("grupo")["conjunto"].nunique().max() == 1

    # ---------------- Escritura ----------------
    for c in CONJUNTOS:
        os.makedirs(os.path.join(args.out, "images", c), exist_ok=True)
        os.makedirs(os.path.join(args.out, "labels", c), exist_ok=True)
    for i, r in enumerate(df.itertuples(index=False), 1):
        shutil.copy2(r.archivo_origen, os.path.join(args.out, "images", r.conjunto, r.archivo_final))
        with open(os.path.join(args.out, "labels", r.conjunto,
                               os.path.splitext(r.archivo_final)[0] + ".txt"), "w", encoding="utf-8") as f:
            f.write("\n".join(r.lineas) + ("\n" if r.lineas else ""))
        if i % 500 == 0:
            print(f"  copiadas {i}/{len(df)}")

    ruta_abs = os.path.abspath(args.out).replace("\\", "/")
    with open(os.path.join(args.out, "dataset.yaml"), "w", encoding="utf-8") as f:
        f.write("# Dataset proxy iteracion 1 (iNaturalist + ENA24). Generado por dividir_dataset.py\n"
                "# En Kaggle cambiar 'path' por la ruta donde quede descomprimida la carpeta.\n"
                f"path: {ruta_abs}\ntrain: images/train\nval: images/val\ntest: images/test\n"
                "nc: 1\nnames: ['oso']\n")

    os.makedirs(os.path.dirname(os.path.abspath(args.division)), exist_ok=True)
    df.drop(columns=["lineas"]).to_csv(args.division, index=False)

    os.makedirs(args.muestra, exist_ok=True)
    for fuente in ("inat", "ena24"):
        sub = df[(df["fuente"] == fuente) & (df["n_cajas"] > 0)]
        for r in sub.sample(n=min(args.n_muestra, len(sub)), random_state=args.semilla).itertuples():
            dibujar(os.path.join(args.out, "images", r.conjunto, r.archivo_final), r.lineas,
                    os.path.join(args.muestra, f"{fuente}_{r.archivo_final}"))

    # ---------------- Resumen ----------------
    print(f"\nTotal: {len(df)} imagenes, {int(df['n_cajas'].sum())} cajas, {df['grupo'].nunique()} grupos")
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
    print(f"\nDataset en {args.out}; division en {args.division}; muestra en {args.muestra}")


if __name__ == "__main__":
    main()
