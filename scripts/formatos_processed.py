#!/usr/bin/env python3
"""
Inventaria el formato interno real de las imagenes de data/processed y, si se
pide, convierte a JPEG las que no lo sean.

Motivo (15 de septiembre de 2026): al validar con Ultralytics 8.4.138 sobre la
particion test_inat, cinco archivos .jpg resultaron ser GIF por dentro y la
libreria los descarto ("ignoring corrupt image/label: Invalid image format
GIF"). Ultralytics comprueba el formato con PIL (im.format) en
ultralytics/data/utils.py::check_image y solo acepta bmp, dng, jpeg, jpg,
mpo, png, tif, tiff, webp, pfm, heic, heif, avif, jp2. Un GIF de un solo
fotograma abre bien con PIL y con OpenCV, por eso paso las etapas anteriores.

Sin --convertir: solo informa y escribe docs/preparacion_datos/formatos_no_jpeg.txt
con una linea por archivo no JPEG: nombre;formato;fotogramas;conjunto.

Con --convertir: reescribe cada archivo no JPEG como JPEG (primer fotograma,
RGB, calidad 95) con el MISMO nombre y las MISMAS dimensiones, asi las
etiquetas normalizadas siguen siendo validas. No toca data/raw.

Uso (desde la raiz del proyecto):
    python scripts/formatos_processed.py
    python scripts/formatos_processed.py --convertir

Probado solo con imagenes sinteticas, no con el dataset real.
"""

import argparse
import os
from collections import Counter

from PIL import Image

CONJUNTOS = ["train", "val", "test"]


def fuente_de(nombre):
    if nombre.startswith("inatneg_"):
        return "inat_neg"
    if nombre.startswith("inat_"):
        return "inat"
    if nombre.startswith("ena24_"):
        return "ena24"
    return "otra"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="data/processed")
    ap.add_argument("--registro", default="docs/preparacion_datos/formatos_no_jpeg.txt")
    ap.add_argument("--convertir", action="store_true",
                    help="Reescribe como JPEG los archivos que no lo sean")
    args = ap.parse_args()

    por_conjunto = {}
    no_jpeg = []
    for conjunto in CONJUNTOS:
        carpeta = os.path.join(args.dataset, "images", conjunto)
        if not os.path.isdir(carpeta):
            print(f"AVISO: no existe {carpeta}")
            continue
        formatos = Counter()
        for nombre in sorted(os.listdir(carpeta)):
            ruta = os.path.join(carpeta, nombre)
            if not os.path.isfile(ruta):
                continue
            try:
                with Image.open(ruta) as im:
                    fmt = im.format or "DESCONOCIDO"
                    frames = getattr(im, "n_frames", 1)
                    tam = im.size
            except Exception as e:
                fmt, frames, tam = f"ERROR:{type(e).__name__}", 0, None
            formatos[fmt] += 1
            if fmt != "JPEG":
                no_jpeg.append((nombre, fmt, frames, conjunto, tam, ruta))
        por_conjunto[conjunto] = formatos

    print("Formato interno por conjunto:")
    for conjunto, formatos in por_conjunto.items():
        print(f"  {conjunto:<6}" + "  ".join(f"{k}={v}" for k, v in sorted(formatos.items())))

    print(f"\nArchivos que no son JPEG: {len(no_jpeg)}")
    if no_jpeg:
        resumen = Counter((fuente_de(n), c) for n, _, _, c, _, _ in no_jpeg)
        for (fuente, conjunto), k in sorted(resumen.items()):
            print(f"  {fuente:<9}{conjunto:<6}{k}")
        animados = [n for n, _, fr, _, _, _ in no_jpeg if fr > 1]
        print(f"  con mas de un fotograma: {len(animados)}")

        os.makedirs(os.path.dirname(args.registro), exist_ok=True)
        with open(args.registro, "w", encoding="utf-8") as f:
            for nombre, fmt, frames, conjunto, _, _ in no_jpeg:
                f.write(f"{nombre};{fmt};{frames};{conjunto}\n")
        print(f"  registro escrito en {args.registro}")

    if not args.convertir:
        if no_jpeg:
            print("\nNo se ha modificado nada. Para convertir a JPEG: --convertir")
        return

    print("\nConvirtiendo a JPEG...")
    ok, fallos = 0, []
    for nombre, fmt, frames, conjunto, tam, ruta in no_jpeg:
        if fmt.startswith("ERROR"):
            fallos.append((nombre, fmt))
            continue
        try:
            with Image.open(ruta) as im:
                im.seek(0)  # primer fotograma
                rgb = im.convert("RGB")
            rgb.save(ruta, format="JPEG", quality=95)
            with Image.open(ruta) as im2:
                if im2.format != "JPEG" or im2.size != tam:
                    fallos.append((nombre, f"verificacion: {im2.format} {im2.size} != {tam}"))
                    continue
            ok += 1
        except Exception as e:
            fallos.append((nombre, f"{type(e).__name__}: {e}"))

    print(f"Convertidos: {ok}   Fallos: {len(fallos)}")
    for nombre, motivo in fallos:
        print(f"  FALLO {nombre}: {motivo}")


if __name__ == "__main__":
    main()
