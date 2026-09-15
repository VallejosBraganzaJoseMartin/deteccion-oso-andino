#!/usr/bin/env python3
"""
Diagnostico de fugas: pares de imagenes casi identicas (distancia de Hamming
baja entre hashes perceptuales) que quedaron en conjuntos distintos
(train/val, train/test, val/test).

Los hashes se toman de inat_metadatos.csv (positivos de iNaturalist) y de la
columna 'grupo' de ena24_seleccion.csv (que es el phash de cada imagen de
ENA24); los que falten (negativos de iNaturalist, fotogramas de GIF) se
calculan sobre la imagen copiada en data/processed.

Imprime, para varios umbrales, cuantos pares cruzados hay, cuantas imagenes
de val/test estan implicadas y algunos ejemplos para mirarlos a ojo.

Uso (desde la raiz del proyecto):
    python scripts/fugas_hamming.py
    python scripts/fugas_hamming.py --ejemplos 10

Referencia: dos hashes iguales tienen distancia 0; la misma escena con
cambios pequenos suele quedar por debajo de 10 (64 bits en total).
"""

import argparse
import itertools
import os
from collections import Counter, defaultdict

import pandas as pd

UMBRALES = (0, 2, 4, 8, 12, 16)


def phash_de(ruta):
    from PIL import Image
    import imagehash
    try:
        with Image.open(ruta) as im:
            return int(str(imagehash.phash(im.convert("RGB"))), 16)
    except Exception:
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--division", default="docs/preparacion_datos/division.csv")
    ap.add_argument("--inat-meta", default="docs/comprension_datos/inat_metadatos.csv")
    ap.add_argument("--ena", default="docs/comprension_datos/ena24_seleccion.csv")
    ap.add_argument("--processed", default="data/processed")
    ap.add_argument("--ejemplos", type=int, default=6)
    args = ap.parse_args()

    df = pd.read_csv(args.division)
    hashes = {}
    if os.path.exists(args.inat_meta):
        m = pd.read_csv(args.inat_meta, dtype={"phash": str})
        for a, h in zip(m["archivo"], m["phash"]):
            if isinstance(h, str):
                hashes[a] = int(h, 16)
    if os.path.exists(args.ena):
        e = pd.read_csv(args.ena, dtype={"grupo": str, "archivo": str})
        for a, h in zip(e["archivo"], e["grupo"]):
            try:
                hashes["ena24_" + a] = int(h, 16)
            except (TypeError, ValueError):
                pass

    filas, calculados, sin_hash = [], 0, 0
    for r in df.itertuples(index=False):
        h = hashes.get(r.archivo_final)
        if h is None:
            ruta = os.path.join(args.processed, "images", r.conjunto, r.archivo_final)
            h = phash_de(ruta)
            calculados += 1
        if h is None:
            sin_hash += 1
            continue
        filas.append((r.archivo_final, r.fuente, r.conjunto, h))
    print(f"Imagenes con hash: {len(filas)} (calculados ahora: {calculados}); sin hash: {sin_hash}")

    por_conj = defaultdict(list)
    for f in filas:
        por_conj[f[2]].append(f)
    pares = []
    for c1, c2 in itertools.combinations(sorted(por_conj), 2):
        for a in por_conj[c1]:
            for b in por_conj[c2]:
                if a[1] != b[1]:
                    continue
                d = bin(a[3] ^ b[3]).count("1")
                if d <= max(UMBRALES):
                    pares.append((d, a[1], a[0], a[2], b[0], b[2]))
    pares.sort()

    print("\nPares casi identicos en conjuntos distintos, por umbral de distancia:")
    print(f"{'umbral':>7}{'pares':>8}{'imgs val/test':>15}{'ena24':>8}{'inat':>7}")
    for u in UMBRALES:
        sel = [p for p in pares if p[0] <= u]
        imgs = {p[2] for p in sel if p[3] != "train"} | {p[4] for p in sel if p[5] != "train"}
        cf = Counter(p[1] for p in sel)
        print(f"{'<= ' + str(u):>7}{len(sel):>8}{len(imgs):>15}{cf['ena24']:>8}{cf['inat']:>7}")

    total_vt = int((df["conjunto"] != "train").sum())
    print(f"\n(val + test tienen {total_vt} imagenes en total)")
    if pares:
        print("\nEjemplos (los mas parecidos primero), para abrirlos y comparar:")
        for d, fuente, a, ca, b, cb in pares[:args.ejemplos]:
            print(f"  d={d:>2} {fuente:<6} {a} [{ca}]  <->  {b} [{cb}]")
    else:
        print("\nNo hay pares a distancia <= 16 entre conjuntos distintos.")


if __name__ == "__main__":
    main()
