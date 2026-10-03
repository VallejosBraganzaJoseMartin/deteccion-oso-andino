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

Si un modelo no alcanza alguna de las dos metas de recall ni con la confianza
mínima, se informa cuál falla y se reporta con el umbral más bajo que deja las
falsas alarmas por debajo del 10 %: es el mayor recall que ese modelo da sin
pasarse de falsas alarmas.

No usa GPU ni imágenes: trabaja sobre los tres archivos que deja la celda 7 del
cuaderno evaluar_umbral_yolo26.ipynb.

Uso (desde la raíz del proyecto):
    python scripts\\calcular_umbral.py
    python scripts\\calcular_umbral.py --dir docs\\modelado\\eval_coco_ablacion

Versión 2 (2 oct 2026): distingue si lo inalcanzable es el recall de cajas, el
recall por imagen o los dos (antes, cuando fallaba el de cajas, el mensaje
culpaba al recall por imagen), y redondea hacia abajo el umbral de la regla A a
los 4 decimales con que el cuaderno guarda las confianzas, para que la detección
que lo define no se pierda por el redondeo. Las cifras de la regla B del 30 de
septiembre no cambian: en los seis modelos el umbral B quedó por debajo del A y
salió de las confianzas ya redondeadas. Solo la columna umbral_regla_A sale
ahora con 4 decimales. El texto del estado usa los valores de --max-falsas y
--recall-imagen. Probado con datos sintéticos.
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


def decidir(t_a, t_img, t_fa, max_falsas=0.10, recall_imagen=0.95):
    """Umbral final y estado, según qué metas de recall se pueden alcanzar."""
    fa, ri = f"{100 * max_falsas:g} %", f"{recall_imagen:g}"
    falla_cajas, falla_imagen = math.isnan(t_a), math.isnan(t_img)
    if falla_cajas and falla_imagen:
        return t_fa, (f"inalcanzables el recall de cajas de 0.90 y el recall por imagen de {ri} en ENA24; "
                      f"se informa con el umbral mínimo que deja las falsas alarmas < {fa}")
    if falla_cajas:
        return t_fa, ("inalcanzable el recall de cajas de 0.90 en ENA24 (ni con la confianza mínima); "
                      f"se informa con el umbral mínimo que deja las falsas alarmas < {fa}")
    if falla_imagen:
        return t_fa, (f"inalcanzable el recall por imagen de {ri} en ENA24: hay fotos con oso sin ninguna "
                      f"detección; se informa con el umbral mínimo que deja las falsas alarmas < {fa}")
    t_rec = min(t_a, t_img)
    if t_rec >= t_fa:
        return t_rec, f"cumple las dos metas de recall con falsas alarmas < {fa}"
    return t_fa, f"no cumple: para bajar de {fa} de falsas alarmas hay que perder recall"


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
        if not math.isnan(t_a):
            # Las confianzas de umbral_imagenes.csv y umbral_detecciones.csv.gz vienen con 4 decimales y el
            # umbral A con todos los suyos: sin esto, la detección que define el umbral (p. ej. 0.376121,
            # guardada como 0.3761) quedaría por debajo de él y se perdería una foto o una caja.
            t_a = math.floor(t_a * 1e4 + 1e-9) / 1e4
        t_img = umbral_recall_imagen(e_pos[col].values, args.recall_imagen)
        t_fa = max(umbral_min_falsas(g[col].values, args.max_falsas) for g in negativos.values())
        t_final, estado = decidir(t_a, t_img, t_fa, args.max_falsas, args.recall_imagen)
        resumen.append({"run_id": rid, "umbral_regla_A": t_a, "umbral_recall_imagen": t_img,
                        "umbral_min_falsas": t_fa, "umbral_regla_B": t_final, "estado": estado})
        filas += por_particion(img, det, rid, t_final)

    res = pd.DataFrame(resumen)
    print("Umbrales (regla A = recall de cajas 0.90; regla B = la de esta revisión; NaN = inalcanzable):")
    with pd.option_context("display.max_colwidth", 200, "display.width", 250):
        print(res.drop(columns="estado").round(4).to_string(index=False))
    print("\nEstado de cada modelo:")
    for r in res.itertuples():
        print(f"   {r.run_id:<22} {r.estado}")

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
