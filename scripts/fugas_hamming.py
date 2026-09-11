#!/usr/bin/env python3
"""
Diagnostico de fugas: pares de imagenes casi identicas (distancia de Hamming
baja entre hashes perceptuales) que quedaron en conjuntos distintos
(train/val, train/test, val/test).

Lee division.csv y los hashes (el grupo de ENA24 ya es el phash; para
iNaturalist se toman de inat_metadatos.csv). Imprime, para varios umbrales,
cuantos pares cruzados hay y cuantas imagenes de val/test estan implicadas,
mas algunos ejemplos para mirarlos a ojo.

Uso (desde la raiz del proyecto):
    python scripts/fugas_hamming.py
    python scripts/fugas_hamming.py --division docs/preparacion_datos/division.csv --ejemplos 8

Referencia: dos hashes iguales tienen distancia 0; la misma escena con
pequenos cambios suele quedar por debajo de 10 (64 bits en total).
"""

import argparse
import itertools
import os
from collections import Counter, defaultdict

import pandas as pd

UMBRALES = (0, 4, 8, 12, 16)


def hamming(a, b):
    return bin(a ^ b).count("1")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--division", default="docs/preparacion_datos/division.csv")
    ap.add_argument("--inat-meta", default="docs/comprension_datos/inat_metadatos.csv")
    ap.add_argument("--ejemplos", type=int, default=6)
    args = ap.parse_args()

    df = pd.read_csv(args.division, dtype={"grupo": str})
    hashes = {}
    if os.path.exists(args.inat_meta):
        meta = pd.read_csv(args.inat_meta, dtype={"phash": str})
        hashes.update({a: h for a, h in zip(meta["archivo"], meta["phash"]) if isinstance(h, str)})
    filas, sin_hash = [], 0
    for r in df.itertuples(index=False):
        if r.fuente == "ena24":
            h = r.grupo.split("_", 1)[1]
        else:
            h = hashes.get(r.archivo_final)
        if not h:
            sin_hash += 1
            continue
        filas.append((r.archivo_final, r.fuente, r.conjunto, int(h, 16)))
    print(f"Imagenes con hash: {len(filas)}; sin hash (se omiten, p. ej. fotogramas de GIF): {sin_hash}")

    por_conj = defaultdict(list)
    for f in filas:
        por_conj[f[2]].append(f)
    pares = []
    for c1, c2 in itertools.combinations(sorted(por_conj), 2):
        for a in por_conj[c1]:
            for b in por_conj[c2]:
                if a[1] != b[1]:
                    continue
                d = hamming(a[3], b[3])
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

    if pares:
        print(f"\nEjemplos (los mas parecidos primero), para abrirlos y comparar:")
        for d, fuente, a, ca, b, cb in pares[:args.ejemplos]:
            print(f"  d={d:>2} {fuente:<6} {a} [{ca}]  <->  {b} [{cb}]")
    else:
        print("\nNo hay pares a distancia <= 16 entre conjuntos distintos.")


if __name__ == "__main__":
    main()
