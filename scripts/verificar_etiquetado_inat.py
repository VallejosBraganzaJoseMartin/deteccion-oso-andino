#!/usr/bin/env python3
"""
Fase CRISP-DM: Preparacion de los datos (cierre del etiquetado, iNaturalist).

1. Lee la lista de descartes (archivo;motivo) y mueve esas imagenes, con su
   JSON de X-AnyLabeling y sus .txt, a <dir>/descartadas/.
2. Verifica que cada imagen que queda tenga su .txt en labels_rev y viceversa,
   y que cada linea sea '0 cx cy w h' con valores dentro de [0, 1].
3. Compara las cajas revisadas (labels_rev) con las propuestas por el modelo
   (labels_pre) para contar cuantas quedaron igual, se ajustaron, se borraron
   o se dibujaron nuevas.
4. Escribe una tabla por imagen con el resultado, que sera la entrada de la
   division train/val/test.

Uso (desde la raiz del proyecto):
    python scripts/verificar_etiquetado_inat.py --dir data/etiquetado/inat \
        --descartes docs/comprension_datos/descartes_etiquetado_inat.txt \
        --estado docs/comprension_datos/inat_estado.csv \
        --out docs/preparacion_datos/inat_etiquetado.csv

Si la exportacion de X-AnyLabeling quedo en otra carpeta, indicarla con
--labels-rev. Se puede volver a correr: los descartes ya movidos se detectan.

NO PROBADO con la exportacion real de X-AnyLabeling (solo con .txt sinteticos).
"""

import argparse
import os
import shutil
from collections import Counter

import pandas as pd

EXT_IMG = {".jpg", ".jpeg", ".png"}
IOU_SIN_CAMBIO = 0.95
IOU_AJUSTADA = 0.5


def leer_cajas(ruta):
    """Devuelve (lista de (cls, cx, cy, w, h), lista de errores)."""
    cajas, errores = [], []
    if not os.path.exists(ruta):
        return cajas, errores
    with open(ruta, encoding="utf-8") as f:
        for n, linea in enumerate(f, 1):
            partes = linea.split()
            if not partes:
                continue
            if len(partes) != 5:
                errores.append(f"linea {n}: {len(partes)} campos")
                continue
            try:
                c = int(float(partes[0]))
                cx, cy, w, h = map(float, partes[1:])
            except ValueError:
                errores.append(f"linea {n}: valores no numericos")
                continue
            if c != 0:
                errores.append(f"linea {n}: clase {c} (se esperaba 0)")
            if not all(0.0 <= v <= 1.0 for v in (cx, cy, w, h)) or w <= 0 or h <= 0:
                errores.append(f"linea {n}: coordenadas fuera de [0, 1] o caja vacia")
            cajas.append((c, cx, cy, w, h))
    return cajas, errores


def iou(a, b):
    ax1, ay1, ax2, ay2 = a[1] - a[3] / 2, a[2] - a[4] / 2, a[1] + a[3] / 2, a[2] + a[4] / 2
    bx1, by1, bx2, by2 = b[1] - b[3] / 2, b[2] - b[4] / 2, b[1] + b[3] / 2, b[2] + b[4] / 2
    iw = max(0.0, min(ax2, bx2) - max(ax1, bx1))
    ih = max(0.0, min(ay2, by2) - max(ay1, by1))
    inter = iw * ih
    union = a[3] * a[4] + b[3] * b[4] - inter
    return inter / union if union > 0 else 0.0


def comparar(pre, rev):
    """Empareja cajas por IoU (voraz) y cuenta sin_cambio, ajustadas, eliminadas, nuevas."""
    pares = sorted(((iou(p, r), i, j) for i, p in enumerate(pre) for j, r in enumerate(rev)),
                   reverse=True)
    usados_p, usados_r = set(), set()
    sin_cambio = ajustadas = 0
    for v, i, j in pares:
        if v < IOU_AJUSTADA or i in usados_p or j in usados_r:
            continue
        usados_p.add(i)
        usados_r.add(j)
        if v >= IOU_SIN_CAMBIO:
            sin_cambio += 1
        else:
            ajustadas += 1
    return sin_cambio, ajustadas, len(pre) - len(usados_p), len(rev) - len(usados_r)


def mover(origen, destino_dir):
    if os.path.exists(origen):
        os.makedirs(destino_dir, exist_ok=True)
        shutil.move(origen, os.path.join(destino_dir, os.path.basename(origen)))
        return True
    return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True, help="Carpeta con images/, labels_pre/ y labels_rev/")
    ap.add_argument("--labels-rev", default=None, help="Carpeta exportada por X-AnyLabeling (por defecto <dir>/labels_rev)")
    ap.add_argument("--descartes", required=True, help="Archivo de texto: nombre;motivo por linea")
    ap.add_argument("--estado", required=True, help="inat_estado.csv de consolidar_inat.py")
    ap.add_argument("--out", required=True, help="CSV de salida por imagen")
    args = ap.parse_args()

    dir_img = os.path.join(args.dir, "images")
    dir_pre = os.path.join(args.dir, "labels_pre")
    dir_rev = args.labels_rev or os.path.join(args.dir, "labels_rev")
    dir_desc = os.path.join(args.dir, "descartadas")
    for d in (dir_img, dir_rev):
        if not os.path.isdir(d):
            raise SystemExit(f"No existe la carpeta {d}")

    # ---- 1. Descartes ----
    descartes, repetidas, malformadas = {}, [], []
    with open(args.descartes, encoding="utf-8") as f:
        for linea in f:
            linea = linea.strip()
            if not linea:
                continue
            if ";" not in linea:
                malformadas.append(linea)
                continue
            nombre, motivo = (x.strip() for x in linea.split(";", 1))
            if nombre in descartes:
                repetidas.append(nombre)
            descartes[nombre] = motivo
    print(f"Descartes en la lista: {len(descartes)} distintos "
          f"(lineas repetidas: {len(repetidas)}, lineas sin ';': {len(malformadas)})")
    for m in malformadas:
        print(f"  ATENCION linea sin motivo: {m}")

    movidas, ya_movidas, no_encontradas = 0, 0, []
    for nombre, motivo in descartes.items():
        base = os.path.splitext(nombre)[0]
        if os.path.exists(os.path.join(dir_img, nombre)):
            mover(os.path.join(dir_img, nombre), os.path.join(dir_desc, "images"))
            mover(os.path.join(dir_img, base + ".json"), os.path.join(dir_desc, "images"))
            mover(os.path.join(dir_pre, base + ".txt"), os.path.join(dir_desc, "labels_pre"))
            mover(os.path.join(dir_rev, base + ".txt"), os.path.join(dir_desc, "labels_rev"))
            movidas += 1
        elif os.path.exists(os.path.join(dir_desc, "images", nombre)):
            ya_movidas += 1
        else:
            no_encontradas.append(nombre)
    print(f"Movidas a {dir_desc}: {movidas} (ya estaban: {ya_movidas})")
    for n in no_encontradas:
        print(f"  ATENCION no encontrada: {n}")
    print(f"Motivos: {dict(Counter(descartes.values()))}")

    # ---- 2. Correspondencia imagen <-> etiqueta y validez ----
    imagenes = sorted(a for a in os.listdir(dir_img) if os.path.splitext(a)[1].lower() in EXT_IMG)
    bases_img = {os.path.splitext(a)[0]: a for a in imagenes}
    bases_lbl = {os.path.splitext(a)[0] for a in os.listdir(dir_rev)
                 if a.lower().endswith(".txt") and a != "classes.txt"}
    sin_txt = sorted(set(bases_img) - bases_lbl)
    txt_sin_img = sorted(bases_lbl - set(bases_img))
    print(f"\nImagenes que quedan: {len(imagenes)}; etiquetas en {dir_rev}: {len(bases_lbl)}")
    print(f"Imagenes sin .txt: {len(sin_txt)}")
    for b in sin_txt[:20]:
        print(f"  {bases_img[b]}")
    print(f".txt sin imagen: {len(txt_sin_img)}")
    for b in txt_sin_img[:20]:
        print(f"  {b}.txt")

    # ---- 3. Cajas y comparacion ----
    estado = pd.read_csv(args.estado).drop_duplicates("archivo").set_index("archivo")
    filas, invalidas, sin_cajas = [], [], []
    tot = Counter()
    areas = []
    for base, archivo in bases_img.items():
        rev, err = leer_cajas(os.path.join(dir_rev, base + ".txt"))
        pre, _ = leer_cajas(os.path.join(dir_pre, base + ".txt"))
        if err:
            invalidas.append((archivo, err))
        if not rev:
            sin_cajas.append(archivo)
        sc, aj, el, nu = comparar(pre, rev)
        tot.update({"sin_cambio": sc, "ajustadas": aj, "eliminadas": el, "nuevas": nu,
                    "cajas_pre": len(pre), "cajas_rev": len(rev)})
        if len(rev) > 1:
            tot["img_mas_de_una"] += 1
        areas.extend(w * h for _, _, _, w, h in rev)
        e = estado.loc[archivo] if archivo in estado.index else {}
        filas.append({"archivo": archivo, "observacion_id": e.get("observacion_id"),
                      "foto_id": e.get("foto_id"), "licencia": e.get("licencia"),
                      "estado_previo": e.get("estado"), "estado_final": "etiquetada",
                      "motivo_descarte": "", "n_cajas_pre": len(pre), "n_cajas": len(rev),
                      "cajas_sin_cambio": sc, "cajas_ajustadas": aj,
                      "cajas_eliminadas": el, "cajas_nuevas": nu})
    for nombre, motivo in descartes.items():
        if nombre in no_encontradas:
            continue
        e = estado.loc[nombre] if nombre in estado.index else {}
        filas.append({"archivo": nombre, "observacion_id": e.get("observacion_id"),
                      "foto_id": e.get("foto_id"), "licencia": e.get("licencia"),
                      "estado_previo": e.get("estado"), "estado_final": "descartada",
                      "motivo_descarte": motivo, "n_cajas_pre": None, "n_cajas": 0,
                      "cajas_sin_cambio": None, "cajas_ajustadas": None,
                      "cajas_eliminadas": None, "cajas_nuevas": None})

    df = pd.DataFrame(filas)
    for col in ("n_cajas_pre", "cajas_sin_cambio", "cajas_ajustadas", "cajas_eliminadas", "cajas_nuevas"):
        df[col] = df[col].astype("Int64")
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    df.to_csv(args.out, index=False)

    print(f"\nEtiquetas invalidas: {len(invalidas)}")
    for archivo, err in invalidas[:20]:
        print(f"  {archivo}: {'; '.join(err)}")
    print(f"Imagenes etiquetadas sin ninguna caja (revisar: olvido o no se exporto): {len(sin_cajas)}")
    for a in sin_cajas[:20]:
        print(f"  {a}")

    print(f"\nCajas finales: {tot['cajas_rev']} en {len(imagenes)} imagenes; "
          f"con mas de una caja: {tot['img_mas_de_una']}")
    if areas:
        s = pd.Series(areas)
        print(f"Area relativa de la caja: mediana {s.median():.3f}; "
              f"< 1 %: {int((s < 0.01).sum())}; > 25 %: {int((s > 0.25).sum())}")
    print(f"\nComparacion con las cajas propuestas por el modelo ({tot['cajas_pre']} cajas):")
    print(f"  Sin cambio apreciable (IoU >= {IOU_SIN_CAMBIO}): {tot['sin_cambio']}")
    print(f"  Ajustadas ({IOU_AJUSTADA} <= IoU < {IOU_SIN_CAMBIO}): {tot['ajustadas']}")
    print(f"  Eliminadas: {tot['eliminadas']}")
    print(f"  Nuevas (dibujadas a mano): {tot['nuevas']}")
    desc = df[df["estado_final"] == "descartada"]
    if len(desc):
        print(f"\nDescartes por estado previo: "
              f"{ {str(k): int(v) for k, v in desc['estado_previo'].value_counts().items()} }")
    print(f"\nTabla escrita en {args.out} ({len(df)} filas)")


if __name__ == "__main__":
    main()
