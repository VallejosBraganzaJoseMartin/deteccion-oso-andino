#!/usr/bin/env python3
"""
Umbral de confianza con la regla B (revisión del 30 de septiembre de 2026 de
criterios_exito_provisional.md, apartado 1.1).

Regla B: se elige, sobre ENA24 de validación, el umbral más alto que cumpla a la
vez las dos metas de recall de los criterios:
  - recall por imagen >= 0.95 (fotos con oso en las que hay al menos una
    detección por encima del umbral), y
  - recall de cajas >= 0.90 (la meta "no negociable"; es el umbral de la regla
    anterior, que ya calculó evaluar_umbral_yolo26.ipynb).
Restricción: con ese umbral, las falsas alarmas (fotos sin oso con alguna
detección por encima del umbral) deben quedar por debajo del 10 % en los
negativos de ENA24 y en los de iNaturalist, por separado. Si no se cumple, se
sube el umbral hasta el valor más bajo que la cumpla y se informa que la meta
de recall no se alcanza con ese modelo.

No usa GPU ni imágenes: trabaja sobre los tres archivos que deja la celda 7 del
cuaderno evaluar_umbral_yolo26.ipynb.

Uso (desde la raíz del proyecto):
    python scripts/umbral_opcion_b.py
    python scripts/umbral_opcion_b.py --dir docs/modelado

Probado solo con datos sintéticos de la misma estructura, no con los archivos
reales.
"""

import argparse
import math
from pathlib import Path

import numpy as np
import pandas as pd

PARTICIONES = ["ena24", "inat", "global", "ena24_dia", "ena24_noche"]
TITULOS = {"ena24": "ENA24 (resultado principal)", "inat": "iNaturalist", "global": "Global",
           "ena24_dia": "ENA24 de día", "ena24_noche": "ENA24 de noche"}


def wilson(k, n, z=1.96):
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    den = 1 + z * z / n
    centro = (p + z * z / (2 * n)) / den
    margen = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return (max(0.0, centro - margen), min(1.0, centro + margen))


def mascara(df, particion):
    if particion == "global":
        return pd.Series(True, index=df.index)
    if particion in ("ena24", "inat"):
        return df.fuente == particion
    if particion == "ena24_dia":
        return (df.fuente == "ena24") & ~df.nocturna
    if particion == "ena24_noche":
        return (df.fuente == "ena24") & df.nocturna
    raise ValueError(particion)


def umbral_recall_imagen(maxconf_pos, objetivo):
    """Umbral más alto con el que al menos `objetivo` de las fotos con oso tienen alguna detección."""
    n = len(maxconf_pos)
    k = math.ceil(round(objetivo * n, 9))
    v = np.sort(maxconf_pos)[::-1]
    if n == 0 or k > n or v[k - 1] <= 0:
        return float("nan")  # inalcanzable: hay fotos con oso sin ninguna detección
    return float(v[k - 1])


def umbral_min_falsas(maxconf_neg, maximo, paso=1e-4):
    """Umbral más bajo con el que las falsas alarmas quedan estrictamente por debajo de `maximo`.
    Las confianzas vienen redondeadas a 4 decimales, así que se sube un paso por encima del
    valor que habría que excluir."""
    n = len(maxconf_neg)
    if n == 0:
        return 0.0
    permitidas = math.ceil(round(maximo * n, 9)) - 1
    v = np.sort(maxconf_neg)[::-1]
    if permitidas >= n:
        return 0.0
    return float(v[permitidas]) + paso


def por_particion(img, det, rid, t):
    filas = []
    col = "maxconf_" + rid
    for part in PARTICIONES:
        sub = img[mascara(img, part)]
        alarma = sub[col] >= t
        pos = sub.positiva
        k_pos, n_pos = int((alarma & pos).sum()), int(pos.sum())
        k_neg, n_neg = int((alarma & ~pos).sum()), int((~pos).sum())
        fila = {"run_id": rid, "particion": part, "umbral": t,
                "n_pos": n_pos, "pos_detectadas": k_pos,
                "recall_imagen": k_pos / n_pos if n_pos else float("nan"),
                "n_neg": n_neg, "neg_con_alarma": k_neg,
                "falsas_alarmas": k_neg / n_neg if n_neg else float("nan")}
        fila["ri_ic_inf"], fila["ri_ic_sup"] = wilson(k_pos, n_pos)
        fila["fa_ic_inf"], fila["fa_ic_sup"] = wilson(k_neg, n_neg)
        n_gt = int(sub.n_cajas.sum())
        if det is not None and t >= 0.01:
            d = det[(det.run_id == rid) & det.archivo.isin(set(sub.archivo)) & (det.conf >= t)]
            n_tp = int(d.tp.sum())
            fila["P_caja"] = n_tp / len(d) if len(d) else float("nan")
            fila["R_caja"] = n_tp / n_gt if n_gt else float("nan")
        else:
            fila["P_caja"] = fila["R_caja"] = float("nan")
        filas.append(fila)
    return filas


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="docs/modelado", help="Carpeta con los archivos de la celda 7")
    ap.add_argument("--conjunto", default="val")
    ap.add_argument("--recall-imagen", type=float, default=0.95)
    ap.add_argument("--max-falsas", type=float, default=0.10)
    ap.add_argument("--out", default=None, help="CSV de salida (por defecto <dir>/umbral_opcion_b.csv)")
    args = ap.parse_args()
    base = Path(args.dir)

    img = pd.read_csv(base / "umbral_imagenes.csv")
    img = img[img.conjunto == args.conjunto].copy()
    for c in ("positiva", "nocturna"):
        img[c] = img[c].astype(str).str.lower().isin(["true", "1"])
    met = pd.read_csv(base / "umbral_metricas.csv")
    ruta_det = base / "umbral_detecciones.csv.gz"
    det = None
    if ruta_det.exists():
        det = pd.read_csv(ruta_det)
        det = det[det.conjunto == args.conjunto]
        det["tp"] = det["tp"].astype(str).str.lower().isin(["true", "1"])
    else:
        print(f"AVISO: no se encontró {ruta_det}; no se calcularán precisión y recall de cajas.")

    sobre = set(met.umbral_sobre.unique())
    if sobre != {"ena24"}:
        raise SystemExit(f"umbral_metricas.csv se generó con umbral_sobre={sobre}; esta regla supone 'ena24'.")

    modelos = [c[len("maxconf_"):] for c in img.columns if c.startswith("maxconf_")]
    e_pos = img[(img.fuente == "ena24") & img.positiva]
    negativos = {"ena24": img[(img.fuente == "ena24") & ~img.positiva],
                 "inat": img[(img.fuente == "inat") & ~img.positiva]}
    print(f"Conjunto '{args.conjunto}': {len(img)} imágenes; ENA24 con oso {len(e_pos)}; "
          f"negativos ENA24 {len(negativos['ena24'])}, iNaturalist {len(negativos['inat'])}")
    print(f"Regla B: recall por imagen >= {args.recall_imagen} y recall de cajas >= 0.90 en ENA24, "
          f"falsas alarmas < {args.max_falsas} por fuente\n")

    resumen, filas = [], []
    for rid in modelos:
        col = "maxconf_" + rid
        fila_a = met[(met.run_id == rid) & (met.conjunto == args.conjunto) & (met.particion == "ena24")]
        t_a = float(fila_a.umbral.iloc[0])
        t_img = umbral_recall_imagen(e_pos[col].values, args.recall_imagen)
        t_rec = min(t_a, t_img) if not math.isnan(t_img) else float("nan")
        t_fa = max(umbral_min_falsas(g[col].values, args.max_falsas) for g in negativos.values())
        if math.isnan(t_rec):
            t_final, estado = t_fa, "recall por imagen inalcanzable: hay fotos con oso sin ninguna detección"
        elif t_rec >= t_fa:
            t_final, estado = t_rec, "cumple las dos metas de recall con falsas alarmas < 10 %"
        else:
            t_final, estado = t_fa, "no cumple: para bajar de 10 % de falsas alarmas hay que perder recall"
        resumen.append({"run_id": rid, "umbral_regla_A": t_a, "umbral_recall_imagen": t_img,
                        "umbral_min_falsas": t_fa, "umbral_regla_B": t_final, "estado": estado})
        filas += por_particion(img, det, rid, t_final)

    res = pd.DataFrame(resumen)
    print("Umbrales (regla A = recall de cajas 0.90; regla B = la de esta revisión):")
    print(res.round(4).to_string(index=False))

    tabla = pd.DataFrame(filas)
    def frac(k, n, lo, hi):
        return f"{k}/{n} = {k / n:.2f} [{lo:.2f}-{hi:.2f}]" if n else "-"
    for part in PARTICIONES:
        sub = tabla[tabla.particion == part]
        print(f"\n=== {args.conjunto} | {TITULOS[part]} | regla B ===")
        print(pd.DataFrame({
            "modelo": sub.run_id, "umbral": sub.umbral.round(4),
            "P caja": sub.P_caja.round(3), "R caja": sub.R_caja.round(3),
            "recall por imagen": [frac(r.pos_detectadas, r.n_pos, r.ri_ic_inf, r.ri_ic_sup) for r in sub.itertuples()],
            "falsas alarmas": [frac(r.neg_con_alarma, r.n_neg, r.fa_ic_inf, r.fa_ic_sup) for r in sub.itertuples()],
        }).to_string(index=False))

    out = Path(args.out) if args.out else base / "umbral_opcion_b.csv"
    tabla.merge(res, on="run_id").to_csv(out, index=False)
    print(f"\nResultados guardados en {out}")


if __name__ == "__main__":
    main()
