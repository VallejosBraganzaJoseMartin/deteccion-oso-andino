"""
explorar_wcs.py

Explora los metadatos del dataset WCS Camera Traps (LILA BC) para las imágenes etiquetadas como oso andino
(Tremarctos ornatus). No descarga imágenes y no toca el dataset del proyecto.

Responde, solo con los metadatos:
  - cuántas imágenes tienen la etiqueta, en cuántas secuencias y en cuántas ubicaciones (cámaras),
  - si la etiqueta se puso por secuencia (en ese caso algunas fotos pueden no mostrar el oso),
  - cuántas tienen caja en el archivo de cajas y de qué tamaño son esas cajas,
  - fechas, horas y tamaño de las fotos, y qué otras especies aparecen en las mismas ubicaciones.

Uso, desde la raíz del proyecto:
    python scripts\\explorar_wcs.py

Entradas (en --dir-wcs, por defecto data/raw/wcs), tal como se bajaron de lila.science:
  wcs_camera_traps.json.zip              etiquetas por imagen o por secuencia (formato COCO Camera Traps)
  wcs_20220205_bboxes_with_classes.zip   cajas, con las mismas clases
Se leen sin descomprimir.

Salidas (en --salida, por defecto docs/wcs):
  wcs_oso_andino.csv   una fila por imagen con la etiqueta de oso andino, con su URL de descarga y todos los
                       campos que trae el JSON (país, identificador de WCS, número de individuos...)
  wcs_secuencias.csv   una fila por secuencia: fotos etiquetadas, con caja, vacías según el archivo de cajas
  wcs_resumen.txt      el mismo resumen que se imprime
No sobrescribe nada salvo con --sobrescribir.

Versión 2 (2 oct 2026): añade los campos propios de WCS a la tabla, sus conteos al resumen y la tabla por
secuencia. La versión 1 se ejecutó con los archivos reales el 2 de octubre; la 2 solo con datos sintéticos.
"""

import argparse
import collections
import gc
import json
import re
import sys
import zipfile
from pathlib import Path

import pandas as pd

URL_BASE = "https://storage.googleapis.com/public-datasets-lila/wcs-unzipped/"  # según lila.science, 2 oct 2026
ZIP_META = "wcs_camera_traps.json.zip"
ZIP_CAJAS = "wcs_20220205_bboxes_with_classes.zip"
PATRON = "tremarctos"
PARECIDOS = ("ursid", "ursus", "ursinae", "bear")  # solo para avisar si hay otras categorías de oso
VACIAS = {"empty", "blank"}
TRAMOS = [0, 16, 32, 64, float("inf")]
ETQ_TRAMOS = ["< 16", "16-32", "32-64", ">= 64"]
RE_FECHA = re.compile(r"(\d{4})[-:/](\d{1,2})[-:/](\d{1,2})")
RE_HORA = re.compile(r"[ T](\d{1,2}):(\d{2})")

LINEAS = []


def p(*partes):
    texto = " ".join(str(x) for x in partes)
    print(texto)
    LINEAS.append(texto)


def es_oso(nombre):
    return PATRON in str(nombre).lower()


def ram():
    try:
        import psutil
        mi = psutil.Process().memory_info()
        pico = getattr(mi, "peak_wset", None)  # solo existe en Windows
        return f"RAM del proceso {mi.rss / 1e9:.1f} GB" + (f", pico {pico / 1e9:.1f} GB" if pico else "")
    except Exception:
        return "RAM: sin dato (falta psutil)"


def cargar_json(ruta_zip):
    """Lee el único .json que hay dentro del zip."""
    with zipfile.ZipFile(ruta_zip) as zf:
        miembros = [m for m in zf.namelist() if m.lower().endswith(".json")]
        if len(miembros) != 1:
            sys.exit(f"ERROR: esperaba un solo .json dentro de {ruta_zip}; hay {miembros}")
        info = zf.getinfo(miembros[0])
        p(f"  {ruta_zip.name}: {ruta_zip.stat().st_size / 1e6:.1f} MB; dentro, {miembros[0]} "
          f"({info.file_size / 1e6:.1f} MB sin comprimir)")
        with zf.open(miembros[0]) as f:
            datos = json.load(f)
    p(f"  Leído. {ram()}")
    return datos


def fecha(dt):
    m = RE_FECHA.search(str(dt)) if dt is not None else None
    return f"{int(m.group(1)):04d}-{int(m.group(2)):02d}-{int(m.group(3)):02d}" if m else None


def hora(dt):
    m = RE_HORA.search(str(dt)) if dt is not None else None
    return int(m.group(1)) if m else None


def num(v):
    """Número positivo o None (también para NaN, que en Python cuenta como verdadero)."""
    try:
        v = float(v)
    except (TypeError, ValueError):
        return None
    return v if v == v and v > 0 else None


def caja_relativa(a, W, H):
    """[x, y, w, h] relativos (0 a 1) desde bbox_relative o desde bbox en píxeles."""
    if a.get("bbox_relative") is not None:
        return [float(v) for v in a["bbox_relative"]]
    if a.get("bbox") is not None and W and H:
        x, y, w, h = (float(v) for v in a["bbox"])
        return [x / W, y / H, w / W, h / H]
    return None


def contar(valores, orden=None):
    c = collections.Counter(valores)
    claves = orden if orden is not None else sorted(c, key=lambda k: (str(type(k)), k))
    return ", ".join(f"{k}: {c[k]}" for k in claves if c[k])


def main():
    ap = argparse.ArgumentParser(description="Metadatos de WCS Camera Traps para Tremarctos ornatus.")
    ap.add_argument("--dir-wcs", type=Path, default=Path("data") / "raw" / "wcs")
    ap.add_argument("--salida", type=Path, default=Path("docs") / "wcs")
    ap.add_argument("--sobrescribir", action="store_true")
    args = ap.parse_args()

    csv_salida, txt_salida = args.salida / "wcs_oso_andino.csv", args.salida / "wcs_resumen.txt"
    csv_seq = args.salida / "wcs_secuencias.csv"
    if not args.sobrescribir and (csv_salida.exists() or txt_salida.exists() or csv_seq.exists()):
        sys.exit(f"ERROR: ya existen salidas en {args.salida}. Usa --sobrescribir para rehacerlas.")
    for nombre in (ZIP_META, ZIP_CAJAS):
        if not (args.dir_wcs / nombre).exists():
            sys.exit(f"ERROR: no encuentro {args.dir_wcs / nombre}. Ejecuta el script desde la raíz del proyecto "
                     f"y deja ahí el archivo tal como se bajó de lila.science.")

    # ------------------------------------------------------------------ 1. Etiquetas
    p("=== 1. Etiquetas por imagen o por secuencia ===")
    datos = cargar_json(args.dir_wcs / ZIP_META)
    p("  info:", json.dumps(datos.get("info", {}), ensure_ascii=False)[:500])
    imgs, anot, cats = datos.get("images", []), datos.get("annotations", []), datos.get("categories", [])
    nombre_cat = {c["id"]: str(c["name"]) for c in cats}
    p(f"  Totales del dataset: {len(imgs)} imágenes, {len(anot)} anotaciones, {len(cats)} categorías")
    ids_oso = {i for i, n in nombre_cat.items() if es_oso(n)}
    parecidas = {i: n for i, n in nombre_cat.items()
                 if i not in ids_oso and any(t in n.lower() for t in PARECIDOS)}
    p("  Categorías de oso andino:", {i: nombre_cat[i] for i in sorted(ids_oso)} or "NINGUNA")
    if parecidas:
        p("  Otras categorías con nombre de oso (no se usan):", parecidas)
    if not ids_oso:
        sys.exit("ERROR: no hay ninguna categoría con 'tremarctos' en el nombre.")

    anot_oso = [a for a in anot if a.get("category_id") in ids_oso]
    ids_img_oso = list(dict.fromkeys(str(a["image_id"]) for a in anot_oso))  # sin repetir, en orden
    img_por_id = {str(im["id"]): im for im in imgs}
    faltan = [k for k in ids_img_oso if k not in img_por_id]
    p(f"  Anotaciones de oso andino: {len(anot_oso)}; imágenes distintas: {len(ids_img_oso)}; "
      f"anotaciones que apuntan a imágenes ausentes de 'images': {len(faltan)}")
    ids_img_oso = [k for k in ids_img_oso if k in img_por_id]
    reg = {k: img_por_id[k] for k in ids_img_oso}

    p("  Campos de esas imágenes (cuántas los tienen):",
      dict(collections.Counter(c for im in reg.values() for c in im)))
    p("  Campos de esas anotaciones:", dict(collections.Counter(c for a in anot_oso for c in a)))
    nivel = collections.defaultdict(set)
    for a in anot_oso:
        nivel[str(a["image_id"])].add(str(a.get("sequence_level_annotation", "sin dato")))
    p("  sequence_level_annotation de esas anotaciones:",
      dict(collections.Counter(str(a.get("sequence_level_annotation", "sin dato")) for a in anot_oso)))
    if ids_img_oso:
        p("  Ejemplo de imagen:", json.dumps(reg[ids_img_oso[0]], ensure_ascii=False))
        p("  Ejemplo de anotación:", json.dumps(anot_oso[0], ensure_ascii=False))
    corruptas = sum(1 for im in reg.values() if im.get("corrupt"))
    if corruptas:
        p(f"  Marcadas como corruptas: {corruptas}")

    # Etiquetas de todas las fotos de las secuencias con oso (y de las fotos de oso sin secuencia)
    seqs_oso = {im.get("seq_id") for im in reg.values()} - {None}
    imgs_seq = {str(im["id"]): im for im in imgs if im.get("seq_id") in seqs_oso}
    objetivo = set(imgs_seq) | set(ids_img_oso)
    etq = collections.defaultdict(list)
    for a in anot:
        k = str(a.get("image_id"))
        if k in objetivo:
            etq[k].append(nombre_cat.get(a.get("category_id"), "?"))

    otras_img = collections.Counter(n for k in ids_img_oso for n in set(etq[k]) if not es_oso(n))
    n_con_otras = sum(1 for k in ids_img_oso if any(not es_oso(n) for n in etq[k]))
    p(f"  Fotos de oso con otra etiqueta además: {n_con_otras}" + (f" ({contar(otras_img.elements())})"
                                                                   if n_con_otras else ""))

    p("\n  Secuencias:")
    sin_seq = sum(1 for im in reg.values() if im.get("seq_id") is None)
    if sin_seq:
        p(f"    fotos de oso sin seq_id: {sin_seq}")
    por_seq = collections.defaultdict(list)
    for k, im in imgs_seq.items():
        por_seq[im["seq_id"]].append(k)
    n_seq_meta, n_seq_oso, otras_seq, distinto_declarado = {}, {}, collections.Counter(), 0
    for s, ks in por_seq.items():
        n_seq_meta[s] = len(ks)
        n_seq_oso[s] = sum(1 for k in ks if any(es_oso(n) for n in etq[k]))
        otras_seq.update(n for k in ks if not any(es_oso(x) for x in etq[k]) for n in etq[k])
        declarado = {imgs_seq[k].get("seq_num_frames") for k in ks} - {None}
        if declarado and max(declarado) != len(ks):
            distinto_declarado += 1
    completas = sum(1 for s in por_seq if n_seq_oso[s] == n_seq_meta[s])
    p(f"    secuencias con alguna foto de oso andino: {len(por_seq)}")
    p(f"    fotos por secuencia en los metadatos: {contar(n_seq_meta.values())}")
    p(f"    secuencias con todas sus fotos etiquetadas como oso: {completas}; con solo una parte: "
      f"{len(por_seq) - completas}")
    if otras_seq:
        p(f"    etiquetas de las fotos sin oso de esas secuencias: {contar(otras_seq.elements())}")
    if distinto_declarado:
        p(f"    secuencias cuyo seq_num_frames no coincide con las fotos que hay en los metadatos: "
          f"{distinto_declarado} (pueden faltar fotos, por ejemplo con personas, que LILA no publica)")

    # Ubicaciones
    loc_de = {k: str(im.get("location")) for k, im in reg.items()}
    locs = collections.Counter(loc_de.values())
    img_loc, total_loc = {}, collections.Counter()
    for im in imgs:
        loc = str(im.get("location"))
        if loc in locs:
            img_loc[str(im["id"])] = loc
            total_loc[loc] += 1
    esp_loc = collections.defaultdict(collections.Counter)
    for a in anot:
        k = str(a.get("image_id"))
        if k in img_loc:
            esp_loc[img_loc[k]][nombre_cat.get(a.get("category_id"), "?")] += 1

    # Campos propios del JSON que no son del formato estándar (en WCS: wcs_id, country_code, status...)
    fijos_img = {"id", "file_name", "location", "datetime", "seq_id", "seq_num_frames", "frame_num", "width",
                 "height"}
    extras_img = sorted({c for im in reg.values() for c in im} - fijos_img)
    fijos_anot = {"id", "image_id", "category_id", "bbox", "bbox_relative", "sequence_level_annotation"}
    primera_anot = {}
    for a in anot_oso:
        primera_anot.setdefault(str(a["image_id"]), a)
    extras_anot = sorted({c for a in anot_oso for c in a} - fijos_anot)

    filas = []
    for k in ids_img_oso:
        im = reg[k]
        extra = {c: im.get(c) for c in extras_img}
        extra.update({"anot_" + c: primera_anot[k].get(c) for c in extras_anot})
        filas.append({
            "image_id": k,
            "file_name": im.get("file_name"),
            "url": URL_BASE + str(im.get("file_name", "")).lstrip("/"),
            "location": loc_de[k],
            "datetime": im.get("datetime"),
            "fecha": fecha(im.get("datetime")),
            "hora": hora(im.get("datetime")),
            "seq_id": im.get("seq_id"),
            "seq_num_frames": im.get("seq_num_frames"),
            "frame_num": im.get("frame_num"),
            "fotos_seq_metadatos": n_seq_meta.get(im.get("seq_id")),
            "fotos_seq_con_oso": n_seq_oso.get(im.get("seq_id")),
            "width": im.get("width"),
            "height": im.get("height"),
            "sequence_level_annotation": "|".join(sorted(nivel[k])),
            "otras_etiquetas": "|".join(sorted({n for n in etq[k] if not es_oso(n)})),
        } | extra)
    df = pd.DataFrame(filas)
    for c in ("hora", "seq_num_frames", "frame_num", "fotos_seq_metadatos", "fotos_seq_con_oso", "width", "height"):
        df[c] = pd.to_numeric(df[c], errors="coerce").round().astype("Int64")  # enteros que admiten vacíos

    p("\n  Campos propios de WCS en las fotos de oso (valores y cuántas fotos):")
    for c in extras_img + ["anot_" + c for c in extras_anot]:
        distintos = df[c].astype(str).nunique()
        if distintos <= 25:
            p(f"    {c}: {contar(df[c].astype(str))}")
        else:
            p(f"    {c}: {distintos} valores distintos; ejemplos: {', '.join(df[c].astype(str).head(3))}")
    p(f"    carpeta de file_name: {contar(df.file_name.astype(str).str.rsplit('/', n=1).str[0])}")

    p("\n  Ubicaciones (cámaras):")
    p(f"    ubicaciones con oso andino: {len(locs)}")
    tabla = df.groupby("location").agg(fotos_oso=("image_id", "size"), secuencias=("seq_id", "nunique"),
                                       desde=("fecha", "min"), hasta=("fecha", "max"))
    tabla["fotos_totales_ubicacion"] = [total_loc[loc] for loc in tabla.index]
    tabla["pct_vacias_ubicacion"] = [round(100 * sum(esp_loc[loc][n] for n in esp_loc[loc]
                                                     if n.strip().lower() in VACIAS) / max(1, total_loc[loc]))
                                     for loc in tabla.index]
    p("    " + tabla.sort_values("fotos_oso", ascending=False).to_string().replace("\n", "\n    "))
    co = collections.Counter()
    for loc in locs:
        co.update({n: c for n, c in esp_loc[loc].items() if not es_oso(n) and n.strip().lower() not in VACIAS})
    p("    otras especies en esas ubicaciones (anotaciones), las 25 más frecuentes:")
    p("    " + ", ".join(f"{n} {c}" for n, c in co.most_common(25)))

    p("\n  Fechas, horas y tamaño de las fotos de oso:")
    p(f"    fechas: de {df.fecha.min()} a {df.fecha.max()}; sin fecha legible: {int(df.fecha.isna().sum())}")
    p(f"    por año: {contar(df.fecha.dropna().str[:4])}")
    p(f"    por hora del reloj de la cámara: {contar(df.hora.dropna().astype(int), orden=range(24))}")
    dia = df.hora.between(6, 17).sum()
    p(f"    de 06:00 a 17:59: {int(dia)}; resto: {int(df.hora.notna().sum() - dia)} "
      f"(aproximado: depende del reloj de cada cámara)")
    p(f"    ancho x alto: {contar(df.width.astype(str) + 'x' + df.height.astype(str))}")

    del datos, imgs, anot, img_por_id, imgs_seq, etq, img_loc
    gc.collect()

    # ------------------------------------------------------------------ 2. Cajas
    p("\n=== 2. Cajas ===")
    cajas = cargar_json(args.dir_wcs / ZIP_CAJAS)
    p("  info:", json.dumps(cajas.get("info", {}), ensure_ascii=False)[:500])
    imgs_b, anot_b, cats_b = cajas.get("images", []), cajas.get("annotations", []), cajas.get("categories", [])
    nombre_b = {c["id"]: str(c["name"]) for c in cats_b}
    p(f"  Totales del archivo: {len(imgs_b)} imágenes, {len(anot_b)} anotaciones "
      f"({sum(1 for a in anot_b if 'bbox' in a or 'bbox_relative' in a)} con caja), {len(cats_b)} categorías")
    p("  Campos de las imágenes:", dict(collections.Counter(c for im in imgs_b[:50000] for c in im)),
      "(en las primeras 50 000)")
    p("  Campos de las anotaciones:", dict(collections.Counter(c for a in anot_b[:50000] for c in a)),
      "(en las primeras 50 000)")

    por_id_b = {str(im["id"]): im for im in imgs_b}
    enlace = {k: por_id_b[k] for k in df.image_id if k in por_id_b}
    criterio = "image_id"
    if not enlace:
        por_nombre_b = {str(im.get("file_name")): im for im in imgs_b}
        enlace = {k: por_nombre_b[f] for k, f in zip(df.image_id, df.file_name.astype(str)) if f in por_nombre_b}
        criterio = "file_name"
    p(f"  Fotos de oso encontradas en el archivo de cajas: {len(enlace)} de {len(df)} (enlazadas por {criterio})")
    id_b = {k: str(im["id"]) for k, im in enlace.items()}
    buscados = set(id_b.values())
    anot_por_b = collections.defaultdict(list)
    for a in anot_b:
        if str(a.get("image_id")) in buscados:
            anot_por_b[str(a["image_id"])].append(a)
    if enlace:
        k0 = next(iter(enlace))
        p("  Ejemplo:", json.dumps({"imagen": enlace[k0], "anotaciones": anot_por_b[id_b[k0]][:3]},
                                   ensure_ascii=False))

    col = collections.defaultdict(list)
    areas, clases_cajas, clases_sin_caja = [], collections.Counter(), collections.Counter()
    for r in df.itertuples():
        im_b = enlace.get(r.image_id)
        lista = anot_por_b[id_b[r.image_id]] if im_b else []
        W = num((im_b or {}).get("width")) or num(r.width)
        H = num((im_b or {}).get("height")) or num(r.height)
        con_caja = [a for a in lista if a.get("bbox") is not None or a.get("bbox_relative") is not None]
        sin_caja = [a for a in lista if a not in con_caja]
        rel_oso = [caja_relativa(a, W, H) for a in con_caja if es_oso(nombre_b.get(a.get("category_id"), ""))]
        rel_oso = [c for c in rel_oso if c is not None]
        clases_cajas.update(nombre_b.get(a.get("category_id"), "?") for a in con_caja)
        clases_sin_caja.update(nombre_b.get(a.get("category_id"), "?") for a in sin_caja)
        areas += [w * h * 100 for _, _, w, h in rel_oso]
        lados = [min(w * W, h * H) * 640 / max(W, H) for _, _, w, h in rel_oso] if W and H else []
        col["en_archivo_cajas"].append(im_b is not None)
        col["n_cajas"].append(len(con_caja))
        col["n_cajas_oso"].append(len([a for a in con_caja if es_oso(nombre_b.get(a.get("category_id"), ""))]))
        col["clases_cajas"].append("|".join(sorted({nombre_b.get(a.get("category_id"), "?") for a in con_caja})))
        col["etiquetas_sin_caja"].append("|".join(sorted({nombre_b.get(a.get("category_id"), "?")
                                                           for a in sin_caja})))
        col["cajas_oso_rel"].append(json.dumps([[round(v, 4) for v in c] for c in rel_oso]) if rel_oso else "")
        col["area_max_pct"].append(round(max(w * h * 100 for _, _, w, h in rel_oso), 2) if rel_oso else None)
        col["lado_min_px640"].append(round(min(lados), 1) if lados else None)
        col["sla_archivo_cajas"].append("|".join(sorted({str(a.get("sequence_level_annotation", "sin dato"))
                                                          for a in lista})))
    for nombre, valores in col.items():
        df[nombre] = valores
    df["estado_lila"] = ["caja de oso" if n > 0 else "caja de otra clase" if t > 0 else
                         "vacía según cajas" if e else "sin revisar"
                         for n, t, e in zip(df.n_cajas_oso, df.n_cajas, df.en_archivo_cajas)]

    en = df[df.en_archivo_cajas]
    p(f"\n  De las {len(df)} fotos con etiqueta de oso andino:")
    p(f"    no están en el archivo de cajas: {int((~df.en_archivo_cajas).sum())}")
    p(f"    están y tienen al menos una caja de oso andino: {int((en.n_cajas_oso > 0).sum())}")
    p(f"    están y tienen cajas, pero ninguna de oso andino: {int(((en.n_cajas > 0) & (en.n_cajas_oso == 0)).sum())}")
    p(f"    están sin ninguna caja: {int((en.n_cajas == 0).sum())}")
    p(f"    clases de las cajas en esas fotos: {contar(clases_cajas.elements()) or 'ninguna'}")
    if clases_sin_caja:
        p(f"    anotaciones sin caja en esas fotos (por clase): {contar(clases_sin_caja.elements())}")
    con = df[df.n_cajas_oso > 0]
    if len(con):
        p(f"    cajas de oso andino: {int(con.n_cajas_oso.sum())}; cajas por foto: {contar(con.n_cajas_oso)}")
        s = pd.Series(areas)
        p(f"    área relativa de la caja: mediana {s.median():.1f} %; < 1 %: {int((s < 1).sum())}; "
          f"> 25 %: {int((s > 25).sum())}")
        tr = pd.cut(con.lado_min_px640, TRAMOS, labels=ETQ_TRAMOS, right=False).value_counts()
        p(f"    lado menor de la caja más pequeña, en px a 640: "
          f"{', '.join(f'{e}: {int(tr.get(e, 0))}' for e in ETQ_TRAMOS)}")
    seq_con_caja = df[df.n_cajas_oso > 0].seq_id.nunique()
    p(f"    secuencias con al menos una caja de oso andino: {seq_con_caja} de {df.seq_id.nunique()}")
    p("    por ubicación (fotos con caja de oso / fotos de oso): " + ", ".join(
        f"{loc} {int((g.n_cajas_oso > 0).sum())}/{len(g)}" for loc, g in df.groupby("location")))
    p("    sequence_level_annotation en el archivo de cajas, según el estado: " + "; ".join(
        f"{e}: {contar(g.sla_archivo_cajas)}" for e, g in df[df.en_archivo_cajas].groupby("estado_lila")))

    # Tabla por secuencia
    orden = df.assign(seq_id=df.seq_id.fillna("(sin secuencia)")).sort_values(
        ["location", "fecha", "hora", "seq_id", "frame_num"])
    seq = orden.groupby("seq_id", sort=False).agg(
        location=("location", "first"), fecha=("fecha", "first"), hora=("hora", "first"),
        fotos_seq=("fotos_seq_metadatos", "first"), etiquetadas_oso=("image_id", "size"),
        con_caja_oso=("estado_lila", lambda s: int((s == "caja de oso").sum())),
        vacias_segun_cajas=("estado_lila", lambda s: int((s == "vacía según cajas").sum())),
        sin_revisar=("estado_lila", lambda s: int((s == "sin revisar").sum())),
    )
    if (df.estado_lila == "caja de otra clase").any():
        seq["caja_otra_clase"] = orden.groupby("seq_id", sort=False).estado_lila.apply(
            lambda s: int((s == "caja de otra clase").sum()))
    frames = {s_id: " ".join(str(f) for f in g[g.estado_lila == "caja de oso"].frame_num)
              for s_id, g in orden.groupby("seq_id", sort=False)}
    seq["frames_con_caja"] = seq.index.map(frames)
    p("\n  Por secuencia (frames_con_caja: números de foto dentro de la secuencia que tienen caja de oso):")
    p("    " + seq.to_string().replace("\n", "\n    "))

    args.salida.mkdir(parents=True, exist_ok=True)
    df.to_csv(csv_salida, index=False)
    seq.to_csv(csv_seq)
    p(f"\nTablas: {csv_salida} ({len(df)} filas) y {csv_seq} ({len(seq)} filas)")
    p(f"Resumen: {txt_salida}")
    txt_salida.write_text("\n".join(LINEAS) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
