#!/usr/bin/env python3
"""
Consolida la informacion de iNaturalist en una tabla por imagen con su estado.

Une tres fuentes:
  - registro_licencias.csv (descarga): licencia, atribucion, coordenadas
  - yolo_inat.csv (exploracion): confianza del YOLO de COCO
  - revision_manual_inat.csv (revision): categoria asignada a mano

Estados posibles:
  positivo                      oso visible confirmado a mano
  positivo_conf_alta_sin_revisar  YOLO >= 0.5, no revisado (se revisa al etiquetar)
  excluido_rastro / excluido_sin_oso / excluido_duda / excluido_cautivo
  pendiente_revision            no tiene revision ni confianza alta (p. ej. GIF originales)

Uso:
    python consolidar_inat.py --registro data/raw/inaturalist/registro_licencias.csv \
        --yolo docs/comprension_datos/yolo_inat.csv \
        --revision docs/comprension_datos/revision_manual_inat.csv \
        --out docs/comprension_datos/inat_estado.csv
"""

import argparse
import re

import pandas as pd

PATRON = re.compile(r"^inat_(\d+)_(\d+)")
ECU_LAT = (-5.1, 1.7)
ECU_LON = (-81.2, -75.1)
MAPA_ESTADO = {"oso": "positivo", "oso_cautivo": "excluido_cautivo", "rastro": "excluido_rastro",
               "sin_oso": "excluido_sin_oso", "duda": "excluido_duda"}


def ids_desde_nombre(nombre):
    m = PATRON.match(str(nombre))
    return (int(m.group(1)), int(m.group(2))) if m else (None, None)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--registro", required=True)
    ap.add_argument("--yolo", required=True)
    ap.add_argument("--revision", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    reg = pd.read_csv(args.registro).drop_duplicates("archivo")
    yolo = pd.read_csv(args.yolo)[["archivo", "n_cajas_oso", "conf_max_oso"]]
    rev = pd.read_csv(args.revision)[["archivo", "categoria_manual", "lote"]]
    rev = rev.drop_duplicates("archivo", keep="last")

    archivos = pd.DataFrame({"archivo": sorted(set(yolo["archivo"]) | set(rev["archivo"]))})
    archivos["es_fotograma_gif"] = ~archivos["archivo"].isin(yolo["archivo"])
    ids = archivos["archivo"].map(ids_desde_nombre)
    archivos["observacion_id"] = [i[0] for i in ids]
    archivos["foto_id"] = [i[1] for i in ids]

    # Licencia y coordenadas por (observacion, foto), para cubrir tambien los fotogramas
    cols_reg = ["observacion_id", "foto_id", "licencia", "atribucion", "usuario",
                "fecha_observada", "latitud", "longitud"]
    cols_reg = [c for c in cols_reg if c in reg.columns]
    reg_ids = reg[cols_reg].drop_duplicates(["observacion_id", "foto_id"])
    df = archivos.merge(reg_ids, on=["observacion_id", "foto_id"], how="left")
    df = df.merge(yolo, on="archivo", how="left").merge(rev, on="archivo", how="left")

    lat = pd.to_numeric(df.get("latitud"), errors="coerce")
    lon = pd.to_numeric(df.get("longitud"), errors="coerce")
    df["en_ecuador"] = lat.between(*ECU_LAT) & lon.between(*ECU_LON)

    def estado(r):
        if pd.notna(r["categoria_manual"]):
            return MAPA_ESTADO.get(r["categoria_manual"], "excluido_otro")
        if pd.notna(r["conf_max_oso"]) and r["conf_max_oso"] >= 0.5:
            return "positivo_conf_alta_sin_revisar"
        return "pendiente_revision"

    df["estado"] = df.apply(estado, axis=1)
    df.to_csv(args.out, index=False)

    print(f"Imagenes en la tabla: {len(df)} (fotogramas de GIF: {int(df['es_fotograma_gif'].sum())})")
    print("\nPor estado:")
    for e, n in df["estado"].value_counts().items():
        print(f"  {e}: {n}")
    pos = df[df["estado"].str.startswith("positivo")]
    print(f"\nPositivos totales (confirmados + conf. alta): {len(pos)}")
    print(f"  Observaciones distintas: {pos['observacion_id'].nunique()}")
    print(f"  Con coordenadas en Ecuador: {int(pos['en_ecuador'].sum())}")
    if "licencia" in pos:
        print(f"  Licencias: { {str(k): int(v) for k, v in pos['licencia'].value_counts().items()} }")
    sin_lic = pos["licencia"].isna().sum() if "licencia" in pos else len(pos)
    if sin_lic:
        print(f"  ATENCION: {int(sin_lic)} positivos sin licencia asociada (revisar nombres de archivo)")
    pend = df[df["estado"] == "pendiente_revision"]
    if len(pend):
        print(f"\nPendientes de revision ({len(pend)}), deberian ser los GIF originales:")
        for a in pend["archivo"].head(30):
            print(f"  {a}")
    print(f"\nTabla escrita en {args.out}")


if __name__ == "__main__":
    main()
