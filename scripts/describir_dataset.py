#!/usr/bin/env python3
"""
Fase CRISP-DM: Comprension de los datos (describir, explorar, verificar calidad).

Recorre las imagenes de iNaturalist y de ENA24, calcula metadatos basicos por
imagen, detecta duplicados y archivos corruptos, resume el JSON COCO de ENA24
y escribe un informe en Markdown junto con CSVs de metadatos.

Uso:
    pip install pillow imagehash pandas
    python describir_dataset.py \
        --inat-dir data/raw/inaturalist \
        --inat-csv data/raw/inaturalist/registro_licencias.csv \
        --ena-dir data/raw/ena24/images \
        --ena-json data/raw/ena24/ena24.json \
        --out docs/comprension_datos

Salidas en --out:
    inat_metadatos.csv, ena24_metadatos.csv, duplicados.csv,
    ena24_cajas_oso.csv, informe_descripcion_datos.md
"""

import argparse
import json
import os
from collections import Counter, defaultdict

import imagehash
import pandas as pd
from PIL import Image, ImageChops, ImageStat

EXT_IMG = {".jpg", ".jpeg", ".png"}

# Caja aproximada de Ecuador continental para contrastar el campo "lugar"
ECU_LAT = (-5.1, 1.7)
ECU_LON = (-81.2, -75.1)


def conteo(serie):
    """value_counts como dict de enteros nativos (legible en el informe)."""
    return {str(k): int(v) for k, v in serie.value_counts().items()}


def listar_imagenes(carpeta):
    rutas = []
    for raiz, _, archivos in os.walk(carpeta):
        for a in archivos:
            if os.path.splitext(a)[1].lower() in EXT_IMG:
                rutas.append(os.path.join(raiz, a))
    return sorted(rutas)


def metadatos_imagen(ruta):
    """Devuelve un dict con metadatos basicos; 'corrupta' = True si no se pudo leer."""
    fila = {"archivo": os.path.basename(ruta), "ruta": ruta,
            "bytes": os.path.getsize(ruta), "corrupta": False}
    try:
        with Image.open(ruta) as img:
            fila["ancho"], fila["alto"] = img.size
            fila["modo"] = img.mode
            # draft acelera mucho la decodificacion de JPEG grandes
            try:
                img.draft("RGB", (400, 400))
            except Exception:
                pass
            rgb = img.convert("RGB")
            peq = rgb.resize((64, 64))
            r, g, b = peq.split()
            gris = ImageStat.Stat(peq.convert("L")).mean[0]
            # Diferencia media entre canales: cercana a 0 => imagen en escala
            # de grises (tipico de disparos infrarrojos nocturnos)
            dif_rg = ImageStat.Stat(ImageChops.difference(r, g)).mean[0]
            dif_gb = ImageStat.Stat(ImageChops.difference(g, b)).mean[0]
            fila["brillo_medio"] = round(gris, 1)
            fila["dif_canales"] = round((dif_rg + dif_gb) / 2, 2)
            fila["escala_grises"] = fila["dif_canales"] < 2.0
            fila["phash"] = str(imagehash.phash(rgb))
    except Exception as e:
        fila["corrupta"] = True
        fila["error"] = str(e)[:120]
        return fila
    fila["orientacion"] = ("horizontal" if fila["ancho"] > fila["alto"]
                           else "vertical" if fila["alto"] > fila["ancho"] else "cuadrada")
    fila["megapixeles"] = round(fila["ancho"] * fila["alto"] / 1e6, 2)
    return fila


def describir_carpeta(carpeta, etiqueta):
    rutas = listar_imagenes(carpeta)
    print(f"[{etiqueta}] {len(rutas)} archivos de imagen en {carpeta}")
    filas = []
    for i, ruta in enumerate(rutas, 1):
        filas.append(metadatos_imagen(ruta))
        if i % 500 == 0:
            print(f"[{etiqueta}] {i}/{len(rutas)}")
    df = pd.DataFrame(filas)
    df.insert(0, "fuente", etiqueta)
    return df


def resumen_basico(df, etiqueta):
    ok = df[~df["corrupta"]].copy()
    for col in ("ancho", "alto"):
        if col in ok:
            ok[col] = ok[col].astype(int)
    lineas = [f"### {etiqueta}", "",
              f"- Archivos de imagen encontrados: {len(df)}",
              f"- Archivos corruptos o ilegibles: {int(df['corrupta'].sum())}"]
    if len(ok) == 0:
        return lineas + [""]
    lineas += [
        f"- Resolucion (megapixeles): mediana {ok['megapixeles'].median():.2f}, "
        f"minimo {ok['megapixeles'].min():.2f}, maximo {ok['megapixeles'].max():.2f}",
        f"- Ancho x alto mas frecuente: "
        f"{(ok['ancho'].astype(str) + 'x' + ok['alto'].astype(str)).mode().iloc[0]}",
        f"- Orientacion: {conteo(ok['orientacion'])}",
        f"- Imagenes en escala de grises (posible infrarrojo): "
        f"{int(ok['escala_grises'].sum())} ({100 * ok['escala_grises'].mean():.1f} %)",
        f"- Brillo medio (0-255): mediana {ok['brillo_medio'].median():.0f}; "
        f"imagenes oscuras (< 60): {int((ok['brillo_medio'] < 60).sum())}",
        f"- Tamano en disco: {ok['bytes'].sum() / 1e9:.2f} GB", "",
    ]
    return lineas


def duplicados(df):
    ok = df[~df["corrupta"]]
    grupos = ok.groupby("phash")
    filas = []
    for h, g in grupos:
        if len(g) > 1:
            for _, r in g.iterrows():
                filas.append({"phash": h, "fuente": r["fuente"], "archivo": r["archivo"],
                              "n_en_grupo": len(g)})
    return pd.DataFrame(filas)


def resumen_inat(df_meta, ruta_csv):
    lineas = ["### iNaturalist: licencias y procedencia", ""]
    if not ruta_csv or not os.path.exists(ruta_csv):
        return lineas + ["- No se encontro el CSV de licencias.", ""]
    reg = pd.read_csv(ruta_csv)
    reg = reg.drop_duplicates(subset=["archivo"])
    lineas.append(f"- Registros en el CSV de licencias: {len(reg)} "
                  f"(observaciones distintas: {reg['observacion_id'].nunique()})")
    lineas.append(f"- Licencias: {conteo(reg['licencia'])}")
    if "lugar" in reg:
        por_texto = reg["lugar"].fillna("").str.contains("Ecuador", case=False).sum()
        lineas.append(f"- Con 'Ecuador' en el campo lugar (texto libre): {int(por_texto)}")
    if {"latitud", "longitud"} <= set(reg.columns):
        lat = pd.to_numeric(reg["latitud"], errors="coerce")
        lon = pd.to_numeric(reg["longitud"], errors="coerce")
        en_caja = ((lat.between(*ECU_LAT)) & (lon.between(*ECU_LON))).sum()
        lineas.append(f"- Con coordenadas dentro de la caja aproximada de Ecuador: {int(en_caja)} "
                      f"(sin coordenadas: {int(lat.isna().sum())})")
    if "cautivo" in reg:
        lineas.append(f"- Marcadas como cautivas: {conteo(reg['cautivo'].astype(str))}")
    fotos_por_obs = reg.groupby("observacion_id").size()
    lineas.append(f"- Fotos por observacion: media {fotos_por_obs.mean():.2f}, maximo {fotos_por_obs.max()}; "
                  f"observaciones con mas de una foto: {int((fotos_por_obs > 1).sum())}")
    en_disco = set(df_meta["archivo"])
    sin_archivo = reg[~reg["archivo"].isin(en_disco)]
    lineas.append(f"- Registros del CSV sin archivo en disco: {len(sin_archivo)}")
    lineas.append("")
    return lineas


def resumen_ena24(df_meta, ruta_json, salida):
    lineas = ["### ENA24: anotaciones (JSON COCO Camera Traps)", ""]
    if not ruta_json or not os.path.exists(ruta_json):
        return lineas + ["- No se encontro el JSON de anotaciones.", ""]
    with open(ruta_json, encoding="utf-8") as f:
        coco = json.load(f)
    imgs = {im["id"]: im for im in coco.get("images", [])}
    cats = {c["id"]: c["name"] for c in coco.get("categories", [])}
    anns = coco.get("annotations", [])
    lineas.append(f"- Imagenes en el JSON: {len(imgs)}; anotaciones: {len(anns)}; categorias: {len(cats)}")
    campos_img = sorted(set().union(*(im.keys() for im in imgs.values()))) if imgs else []
    lineas.append(f"- Campos disponibles por imagen: {campos_img}")
    lineas.append("  (si hay 'location', 'seq_id' o 'datetime', usalos para dividir "
                  "train/val/test sin fugas)")

    # Conteo de imagenes y anotaciones por categoria
    img_por_cat = defaultdict(set)
    ann_por_cat = Counter()
    for a in anns:
        img_por_cat[a["category_id"]].add(a["image_id"])
        ann_por_cat[a["category_id"]] += 1
    lineas.append("- Imagenes (y anotaciones) por categoria:")
    for cid, n in sorted(img_por_cat.items(), key=lambda x: -len(x[1])):
        lineas.append(f"  - {cats.get(cid, cid)}: {len(n)} imagenes, {ann_por_cat[cid]} cajas")
    sin_ann = set(imgs) - set().union(*img_por_cat.values()) if img_por_cat else set(imgs)
    lineas.append(f"- Imagenes sin ninguna anotacion (posibles vacias): {len(sin_ann)}")

    # Cajas de oso: tamano relativo
    cats_oso = [cid for cid, nombre in cats.items() if "bear" in nombre.lower()]
    lineas.append(f"- Categorias que contienen 'bear': {[cats[c] for c in cats_oso]}")
    filas = []
    for a in anns:
        if a["category_id"] in cats_oso and "bbox" in a:
            im = imgs.get(a["image_id"], {})
            w, h = im.get("width"), im.get("height")
            x, y, bw, bh = a["bbox"]
            # Si todos los valores <= 1, la caja esta normalizada
            normalizada = max(x, y, bw, bh) <= 1.0
            if not normalizada and w and h:
                area_rel = (bw * bh) / (w * h)
            elif normalizada:
                area_rel = bw * bh
            else:
                area_rel = None
            filas.append({"image_id": a["image_id"], "file_name": im.get("file_name"),
                          "categoria": cats[a["category_id"]], "x": x, "y": y, "w": bw, "h": bh,
                          "img_w": w, "img_h": h, "area_relativa": area_rel,
                          "bbox_normalizada": normalizada})
    df_oso = pd.DataFrame(filas)
    if len(df_oso):
        df_oso.to_csv(os.path.join(salida, "ena24_cajas_oso.csv"), index=False)
        ar = df_oso["area_relativa"].dropna()
        lineas.append(f"- Cajas de oso: {len(df_oso)} en {df_oso['image_id'].nunique()} imagenes; "
                      f"formato de bbox {'normalizado' if df_oso['bbox_normalizada'].all() else 'en pixeles'}")
        if len(ar):
            lineas.append(f"- Area relativa de la caja de oso (fraccion de la imagen): mediana {ar.median():.3f}; "
                          f"cajas pequenas (< 1 % de la imagen): {int((ar < 0.01).sum())}; "
                          f"cajas grandes (> 25 %): {int((ar > 0.25).sum())}")
    # Cruce JSON vs disco
    en_disco = set(df_meta["archivo"])
    nombres_json = {os.path.basename(im.get("file_name", "")) for im in imgs.values()}
    lineas.append(f"- Imagenes del JSON sin archivo en disco: {len(nombres_json - en_disco)}; "
                  f"archivos en disco que no aparecen en el JSON: {len(en_disco - nombres_json)}")
    lineas.append("")
    return lineas


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inat-dir", required=True)
    ap.add_argument("--inat-csv", default=None)
    ap.add_argument("--ena-dir", required=True)
    ap.add_argument("--ena-json", default=None)
    ap.add_argument("--out", default="docs/comprension_datos")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    df_inat = describir_carpeta(args.inat_dir, "inaturalist")
    df_ena = describir_carpeta(args.ena_dir, "ena24")
    df_inat.to_csv(os.path.join(args.out, "inat_metadatos.csv"), index=False)
    df_ena.to_csv(os.path.join(args.out, "ena24_metadatos.csv"), index=False)

    df_dup = duplicados(pd.concat([df_inat, df_ena], ignore_index=True))
    df_dup.to_csv(os.path.join(args.out, "duplicados.csv"), index=False)

    lineas = ["# Informe de descripcion y calidad de los datos (iteracion 1, dataset proxy)", "",
              "Generado automaticamente por describir_dataset.py. Revisar y completar a mano.", ""]
    lineas += resumen_basico(df_inat, "iNaturalist (Tremarctos ornatus)")
    lineas += resumen_inat(df_inat, args.inat_csv)
    lineas += resumen_basico(df_ena, "ENA24 (LILA BC)")
    lineas += resumen_ena24(df_ena, args.ena_json, args.out)
    lineas += ["### Duplicados (mismo hash perceptual)", "",
               f"- Grupos de duplicados: {df_dup['phash'].nunique() if len(df_dup) else 0}; "
               f"archivos implicados: {len(df_dup)} (detalle en duplicados.csv)", "",
               "### Pendiente de revision manual", "",
               "- iNaturalist: separar huellas, excrementos, restos y fotos de cautiverio evidentes.",
               "- Definir criterio de division train/val/test: por observacion_id (iNaturalist) "
               "y por location/seq_id (ENA24).", ""]
    ruta_md = os.path.join(args.out, "informe_descripcion_datos.md")
    with open(ruta_md, "w", encoding="utf-8") as f:
        f.write("\n".join(lineas))
    print("\n".join(lineas))
    print(f"\nInforme escrito en {ruta_md}")


if __name__ == "__main__":
    main()
