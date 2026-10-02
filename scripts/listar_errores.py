"""
listar_errores.py

Análisis de errores sobre validación (paso 4 del plan). Para cada modelo de umbral_opcion_b.csv, con el
umbral de la regla B de ese modelo, busca:
  - fotos con oso que quedan sin aviso (confianza máxima por debajo del umbral), y
  - fotos sin oso con aviso (falsas alarmas).
Usa la misma comparación que calcular_umbral.py: hay aviso si la confianza máxima de la foto (columna
maxconf_<RUN_ID> de umbral_imagenes.csv) es mayor o igual que el umbral de la regla B.

Uso, desde la raíz del proyecto:
    python scripts\\listar_errores.py

Entradas:
  docs/modelado/umbral_imagenes.csv, umbral_opcion_b.csv y umbral_detecciones.csv.gz
  data/processed/images/val y data/processed/labels/val
  data/raw/ena24/ena24.json (opcional: da la especie de los negativos de ENA24)

Salidas:
  docs/modelado/errores_val.csv   una fila por foto con error en al menos un modelo
  data/errores_val/osos_perdidos/  copias con cajas dibujadas: verde, caja real; naranja, detecciones del
  data/errores_val/falsas_alarmas/ modelo que falla con mayor confianza en esa foto (rótulo: modelo y confianza)
  data/errores_val/mosaico_*.jpg   hojas de 12 miniaturas para revisar de un vistazo
Las copias van dentro de data/, que no se versiona. No sobrescribe nada salvo con --sobrescribir.
"""

import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont

ORDEN_PART = {"ena24_dia": 0, "ena24_noche": 1, "inat": 2}
NOMBRE_PART = {"ena24_dia": "ENA24 dia", "ena24_noche": "ENA24 noche", "inat": "iNaturalist"}  # sin tildes: la letra por defecto de Pillow no siempre las tiene
VERDE, NARANJA = (0, 220, 0), (255, 140, 0)
TESELA_W, TESELA_H, PIE, COLUMNAS, FILAS = 480, 360, 30, 4, 3


def corto(run_id):
    t = run_id.split("_")[0]
    return t[0].upper() + t[1:]


def fuente_tipo(tam):
    try:
        return ImageFont.load_default(size=tam)  # Pillow 10.1 o posterior
    except TypeError:
        return ImageFont.load_default()


def leer_cajas(ruta_txt):
    if not ruta_txt.exists():
        return []
    cajas = []
    for linea in ruta_txt.read_text().splitlines():
        p = linea.split()
        if len(p) >= 5:
            cajas.append(tuple(float(v) for v in p[1:5]))  # cx, cy, w, h normalizadas
    return cajas


def tamano_cajas(cajas, W, H):
    """Área relativa (%) y lado menor en píxeles tal como lo ve la red a 640 (lado mayor de la foto = 640)."""
    escala = 640 / max(W, H)
    areas = [w * h * 100 for _, _, w, h in cajas]
    lados = [min(w * W, h * H) * escala for _, _, w, h in cajas]
    return min(areas), min(lados)


def especies_ena24(ruta_json):
    if not ruta_json.exists():
        print(f"AVISO: no encuentro {ruta_json}; la especie de los negativos de ENA24 queda en blanco.")
        return {}
    try:
        with open(ruta_json, encoding="utf-8") as f:
            data = json.load(f)
        cats = {c["id"]: c["name"] for c in data["categories"]}
        nombres = {im["id"]: Path(im["file_name"]).name for im in data["images"]}
        esp = {}
        for a in data["annotations"]:
            n = nombres.get(a["image_id"])
            if n is not None:
                esp.setdefault(n, set()).add(cats.get(a["category_id"], "?"))
        return {k: ", ".join(sorted(v)) for k, v in esp.items()}
    except Exception as e:  # el resto del análisis no depende de esto
        print(f"AVISO: no pude leer las especies de {ruta_json}: {e}")
        return {}


def dibujar(ruta, cajas, dets, titulo, destino):
    with Image.open(ruta) as im:
        im = im.convert("RGB")
    W, H = im.size
    d = ImageDraw.Draw(im)
    g = max(2, round(min(W, H) / 250))
    letra = fuente_tipo(max(14, round(min(W, H) / 28)))
    for cx, cy, w, h in cajas:
        d.rectangle([(cx - w / 2) * W, (cy - h / 2) * H, (cx + w / 2) * W, (cy + h / 2) * H], outline=VERDE, width=g)
    for x1, y1, x2, y2, rotulo in dets:
        d.rectangle([x1 * W, y1 * H, x2 * W, y2 * H], outline=NARANJA, width=g)
        pos = (x1 * W + 2 * g, y1 * H + g)
        d.rectangle(d.textbbox(pos, rotulo, font=letra), fill=(0, 0, 0))  # fondo para que el rótulo se lea
        d.text(pos, rotulo, fill=NARANJA, font=letra)
    alto_banda = round(min(W, H) / 28 * 1.6)
    lienzo = Image.new("RGB", (W, H + alto_banda), (0, 0, 0))
    lienzo.paste(im, (0, alto_banda))
    ImageDraw.Draw(lienzo).text((g * 2, alto_banda * 0.15), titulo, fill=(255, 255, 255), font=letra)
    lienzo.save(destino, quality=90)
    return lienzo


def mosaicos(teselas, prefijo, carpeta):
    letra = fuente_tipo(18)
    por_hoja = COLUMNAS * FILAS
    hojas = []
    for i in range(0, len(teselas), por_hoja):
        grupo = teselas[i:i + por_hoja]
        hoja = Image.new("RGB", (COLUMNAS * TESELA_W, FILAS * (TESELA_H + PIE)), (40, 40, 40))
        d = ImageDraw.Draw(hoja)
        for j, (im, pie) in enumerate(grupo):
            x, y = (j % COLUMNAS) * TESELA_W, (j // COLUMNAS) * (TESELA_H + PIE)
            mini = im.copy()
            mini.thumbnail((TESELA_W - 4, TESELA_H - 4))
            hoja.paste(mini, (x + (TESELA_W - mini.width) // 2, y + (TESELA_H - mini.height) // 2))
            d.text((x + 6, y + TESELA_H + 5), pie, fill=(255, 255, 255), font=letra)
        destino = carpeta / f"mosaico_{prefijo}_{i // por_hoja + 1:02d}.jpg"
        hoja.save(destino, quality=85)
        hojas.append(destino)
    return hojas


def main():
    ap = argparse.ArgumentParser(description="Errores de los modelos sobre validación con el umbral de la regla B.")
    ap.add_argument("--dir", type=Path, default=Path("docs") / "modelado")
    ap.add_argument("--datos", type=Path, default=Path("data") / "processed")
    ap.add_argument("--ena24-json", type=Path, default=Path("data") / "raw" / "ena24" / "ena24.json")
    ap.add_argument("--salida", type=Path, default=Path("data") / "errores_val")
    ap.add_argument("--sobrescribir", action="store_true")
    args = ap.parse_args()

    csv_salida = args.dir / "errores_val.csv"
    if not args.sobrescribir and (csv_salida.exists() or (args.salida.exists() and any(args.salida.iterdir()))):
        sys.exit(f"ERROR: ya existen {csv_salida} o {args.salida}. Usa --sobrescribir para rehacerlos.")
    for p in ("umbral_imagenes.csv", "umbral_opcion_b.csv", "umbral_detecciones.csv.gz"):
        if not (args.dir / p).exists():
            sys.exit(f"ERROR: no encuentro {args.dir / p}. Ejecuta el script desde la raíz del proyecto.")

    img = pd.read_csv(args.dir / "umbral_imagenes.csv")
    img = img[img.conjunto == "val"].copy()
    for c in ("positiva", "nocturna"):
        img[c] = img[c].astype(str).str.lower().isin(["true", "1"])
    ob = pd.read_csv(args.dir / "umbral_opcion_b.csv")
    umbral = ob.groupby("run_id").umbral_regla_B.first().to_dict()
    modelos = [c[len("maxconf_"):] for c in img.columns if c.startswith("maxconf_")]
    sin_umbral = [m for m in modelos if m not in umbral]
    if sin_umbral:
        sys.exit(f"ERROR: estos modelos no tienen umbral en umbral_opcion_b.csv: {sin_umbral}")
    det = pd.read_csv(args.dir / "umbral_detecciones.csv.gz")
    det = det[det.conjunto == "val"]

    app = next((m for m in modelos if corto(m) == "E1"), modelos[0])  # modelo de la aplicación (D19)
    img["particion"] = np.where(img.fuente == "inat", "inat", np.where(img.nocturna, "ena24_noche", "ena24_dia"))
    for m in modelos:
        mc = img["maxconf_" + m]
        img["falla_" + m] = np.where(img.positiva, mc < umbral[m], mc >= umbral[m])
    img["n_modelos"] = img[["falla_" + m for m in modelos]].sum(axis=1)

    # Control: los conteos deben ser los mismos de umbral_opcion_b.csv
    distintos = []
    for m in modelos:
        for part, mask in (("ena24", img.fuente == "ena24"), ("inat", img.fuente == "inat")):
            sub = img[mask]
            r = ob[(ob.run_id == m) & (ob.particion == part)].iloc[0]
            perdidos = int((sub.positiva & sub["falla_" + m]).sum())
            alarmas = int((~sub.positiva & sub["falla_" + m]).sum())
            if perdidos != r.n_pos - r.pos_detectadas or alarmas != r.neg_con_alarma:
                distintos.append((m, part))
    print("Control contra umbral_opcion_b.csv:", "conteos idénticos" if not distintos else f"DIFERENCIAS en {distintos}")

    # Tamaño de las cajas reales de todas las fotos con oso (para comparar perdidas frente a todas)
    imgs_dir, lbl_dir = args.datos / "images" / "val", args.datos / "labels" / "val"
    area, lado, ancho, alto = {}, {}, {}, {}
    for r in img.itertuples():
        ruta = imgs_dir / r.archivo
        if not ruta.exists():
            sys.exit(f"ERROR: no encuentro {ruta}")
        with Image.open(ruta) as im:
            ancho[r.archivo], alto[r.archivo] = im.size
        if r.positiva:
            cajas = leer_cajas(lbl_dir / (Path(r.archivo).stem + ".txt"))
            if cajas:
                area[r.archivo], lado[r.archivo] = tamano_cajas(cajas, ancho[r.archivo], alto[r.archivo])
    img["ancho"], img["alto"] = img.archivo.map(ancho), img.archivo.map(alto)
    img["area_min_pct"], img["lado_min_px640"] = img.archivo.map(area), img.archivo.map(lado)

    especies = especies_ena24(args.ena24_json)

    def tipo_negativo(nombre):
        if nombre.startswith("ena24_"):
            return especies.get(nombre[len("ena24_"):], "")
        return "otra especie (iNat)" if nombre.startswith("inatneg_") else "rastro o paisaje (iNat)"

    err = img[img.n_modelos > 0].copy()
    err["tipo"] = np.where(err.positiva, "oso_perdido", "falsa_alarma")
    err["modelos_que_fallan"] = [" ".join(corto(m) for m in modelos if fila["falla_" + m]) for _, fila in err.iterrows()]
    err["negativo"] = [tipo_negativo(a) if not p else "" for a, p in zip(err.archivo, err.positiva)]
    err["_orden"] = err.particion.map(ORDEN_PART)
    err = err.sort_values(["tipo", "_orden", "n_modelos", "archivo"], ascending=[False, True, False, True])
    if especies:
        en_json = sum(1 for a in img.archivo if a.startswith("ena24_") and a[len("ena24_"):] in especies)
        n_ena = int((img.fuente == "ena24").sum())
        print(f"Fotos de ENA24 de validación encontradas en el JSON: {en_json} de {n_ena}")

    # Copias dibujadas y mosaicos
    if args.salida.exists() and args.sobrescribir:
        shutil.rmtree(args.salida)
    carpetas = {"oso_perdido": args.salida / "osos_perdidos", "falsa_alarma": args.salida / "falsas_alarmas"}
    for c in carpetas.values():
        c.mkdir(parents=True, exist_ok=True)
    teselas = {"oso_perdido": [], "falsa_alarma": []}
    ref_modelo, ref_conf = [], []
    for i, (_, r) in enumerate(err.iterrows(), start=1):
        fallan = [m for m in modelos if r["falla_" + m]]
        ref = max(fallan, key=lambda m: r["maxconf_" + m])  # el que falla con mayor confianza
        d = det[(det.run_id == ref) & (det.archivo == r.archivo)].sort_values("conf", ascending=False)
        d = d[d.conf >= (0.05 if r.positiva else umbral[ref])].head(3)
        dets = [(q.x1, q.y1, q.x2, q.y2, f"{corto(ref)} {q.conf:.2f}") for q in d.itertuples()]
        cajas = leer_cajas(lbl_dir / (Path(r.archivo).stem + ".txt")) if r.positiva else []
        ref_modelo.append(corto(ref))
        ref_conf.append(round(r["maxconf_" + ref], 4))
        titulo = (f"{r.archivo} | {NOMBRE_PART[r.particion]} | falla en {r.n_modelos}/{len(modelos)}: "
                  f"{r.modelos_que_fallan}" + ("" if dets or not r.positiva else " | sin detecciones >= 0.05"))
        nombre = f"{i:03d}_{r.n_modelos}de{len(modelos)}_{r.particion}_{r.archivo}"  # el número conserva el orden
        lienzo = dibujar(imgs_dir / r.archivo, cajas, dets, titulo, carpetas[r.tipo] / nombre)
        extra = r.negativo[:18] if not r.positiva else f"{corto(app)} {r['maxconf_' + app]:.2f}"
        teselas[r.tipo].append((lienzo, f"{Path(r.archivo).stem[:26]} | {r.n_modelos}/{len(modelos)} | {extra}"))
    err["modelo_dibujado"], err["conf_dibujada"] = ref_modelo, ref_conf
    hojas = mosaicos(teselas["oso_perdido"], "osos_perdidos", args.salida)
    hojas += mosaicos(teselas["falsa_alarma"], "falsas_alarmas", args.salida)

    columnas = (["archivo", "tipo", "particion", "n_modelos", "modelos_que_fallan", "n_cajas", "area_min_pct",
                 "lado_min_px640", "ancho", "alto", "negativo", "modelo_dibujado", "conf_dibujada"]
                + ["maxconf_" + m for m in modelos])
    err[columnas].round(4).to_csv(csv_salida, index=False)

    # Resumen
    pd.set_option("display.width", 200)
    print(f"\nUmbrales de la regla B: " + ", ".join(f"{corto(m)} {umbral[m]}" for m in modelos))
    pos, neg = img[img.positiva], img[~img.positiva]
    print("\nFotos con oso sin aviso, por modelo:")
    print(pd.DataFrame({corto(m): pos[pos["falla_" + m]].groupby("particion").size() for m in modelos})
          .reindex(list(ORDEN_PART)).fillna(0).astype(int).to_string())
    print("\nFalsas alarmas, por modelo:")
    print(pd.DataFrame({corto(m): neg[neg["falla_" + m]].groupby("particion").size() for m in modelos})
          .reindex(list(ORDEN_PART)).fillna(0).astype(int).to_string())
    for tipo, sub in (("Fotos con oso", pos), ("Fotos sin oso", neg)):
        s = sub[sub.n_modelos > 0]
        print(f"\n{tipo} con error en al menos un modelo, según en cuántos modelos falla:")
        print(pd.crosstab(s.particion, s.n_modelos, margins=True, margins_name="total").to_string())

    print("\nTamaño del oso tal como lo ve la red (lado menor de la caja más pequeña, en px a 640):")
    tramos = [0, 16, 32, 64, 10000]
    etiquetas = ["< 16", "16-32", "32-64", ">= 64"]
    for fuente in ("ena24", "inat"):
        p = pos[pos.fuente == fuente]
        tabla = pd.DataFrame({
            "todas": pd.cut(p.lado_min_px640, tramos, labels=etiquetas, right=False).value_counts(),
            "perdidas por >= 1 modelo": pd.cut(p[p.n_modelos > 0].lado_min_px640, tramos, labels=etiquetas,
                                               right=False).value_counts(),
            f"perdidas por {corto(app)}": pd.cut(p[p["falla_" + app]].lado_min_px640, tramos, labels=etiquetas,
                                              right=False).value_counts(),
        }).reindex(etiquetas).fillna(0).astype(int)
        print(f"  {'ENA24' if fuente == 'ena24' else 'iNaturalist'}:")
        print("    " + tabla.to_string().replace("\n", "\n    "))

    s = err[(err.tipo == "falsa_alarma")]
    print("\nFalsas alarmas por tipo de negativo (fotos distintas):")
    print("    " + s.negativo.replace("", "(sin dato)").value_counts().to_string().replace("\n", "\n    "))
    print(f"\nTabla: {csv_salida} ({len(err)} fotos)")
    print(f"Copias dibujadas: {carpetas['oso_perdido']} y {carpetas['falsa_alarma']}")
    print("Mosaicos:")
    for h in hojas:
        print(f"   {h}")


if __name__ == "__main__":
    main()
