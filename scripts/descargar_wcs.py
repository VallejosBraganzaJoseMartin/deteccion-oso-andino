"""
descargar_wcs.py

Descarga solo las imágenes de WCS Camera Traps (LILA BC) etiquetadas como oso andino, a partir de la tabla
que escribe explorar_wcs.py, y prepara copias para revisarlas a ojo. No toca el dataset del proyecto.

Uso, desde la raíz del proyecto:
    python scripts\\descargar_wcs.py

Entrada:
  docs/wcs/wcs_oso_andino.csv        de explorar_wcs.py, versión 2

Salidas:
  data/raw/wcs/imagenes/<file_name>  las imágenes tal como bajan, con la ruta de LILA (D8: no se modifican)
  docs/wcs/wcs_descarga.csv          una fila por imagen: de qué nube bajó, formato, tamaño, brillo, escala de
                                     grises y nombre de su copia de revisión
  data/revisar/wcs/                  copias reducidas a 1600 px con las cajas de LILA en verde y una banda con
                                     cámara, secuencia, foto, hora y estado en LILA. Se clasifican moviéndolas a
                                     subcarpetas oso/, sin_oso/ y duda/, y se registran con registrar_revision.py
  data/wcs_mosaicos/                 una hoja por secuencia (hasta 10 fotos por hoja), en orden de foto

Las imágenes que ya están en disco y se abren bien no se vuelven a bajar: si la descarga se corta, basta con
correrlo otra vez. Si alguna falla, no se escriben las copias ni la tabla. Las copias y los mosaicos no se
sobrescriben salvo con --sobrescribir, y nunca si ya hay fotos clasificadas en las subcarpetas.

Versión 1 ejecutada con los datos reales el 2 de octubre de 2026 (462 imágenes, todas JPEG). Versión 2: corrige el
número de foto que muestran las bandas y los mosaicos, porque WCS cuenta las fotos de cada secuencia desde 1 y la
versión 1 suponía que desde 0. Las descargas y la tabla no cambian. La versión 2 solo se probó con datos sintéticos.
"""

import argparse
import concurrent.futures
import json
import re
import sys
import time
from pathlib import Path

import pandas as pd
import requests
from PIL import Image, ImageChops, ImageDraw, ImageFont, ImageStat

# Las tres nubes que da lila.science para WCS (consultado el 2 oct 2026). Se prueban en este orden.
BASES = {
    "gcp": "https://storage.googleapis.com/public-datasets-lila/wcs-unzipped/",
    "azure": "https://lilawildlife.blob.core.windows.net/lila-wildlife/wcs-unzipped/",
    "aws": "http://us-west-2.opendata.source.coop.s3.amazonaws.com/agentmorris/lila-wildlife/wcs-unzipped/",
}
CATEGORIAS = ["oso", "oso_cautivo", "rastro", "sin_oso", "duda"]  # las que reconoce registrar_revision.py
ESTADO_CORTO = {"caja de oso": "caja LILA", "caja de otra clase": "caja de otra clase",
                "vacía según cajas": "vacia segun LILA", "sin revisar": "sin revisar en LILA"}  # sin tildes
VERDE = (0, 220, 0)
LADO_COPIA = 1600
TESELA_W, TESELA_H, PIE, CABECERA, COLUMNAS, FILAS = 384, 288, 26, 34, 5, 2


def fuente_tipo(tam):
    try:
        return ImageFont.load_default(size=tam)  # Pillow 10.1 o posterior
    except TypeError:
        return ImageFont.load_default()


def describir(ruta):
    """Formato, tamaño y color de la imagen; lanza una excepción si no se puede leer."""
    fila = {"bytes": ruta.stat().st_size}
    with Image.open(ruta) as im:
        fila["formato"], fila["modo"] = im.format, im.mode
        fila["ancho_real"], fila["alto_real"] = im.size
        im.draft("RGB", (400, 400))  # decodifica a tamaño reducido, como describir_dataset.py
        peq = im.convert("RGB").resize((64, 64))
    r, g, b = peq.split()
    dif = (ImageStat.Stat(ImageChops.difference(r, g)).mean[0]
           + ImageStat.Stat(ImageChops.difference(g, b)).mean[0]) / 2
    fila["brillo_medio"] = round(ImageStat.Stat(peq.convert("L")).mean[0], 1)
    fila["dif_canales"] = round(dif, 2)
    fila["escala_grises"] = dif < 2.0  # mismo criterio que describir_dataset.py (infrarrojo nocturno)
    return fila


def bajar(file_name, destino, intentos=3):
    """Devuelve (estado, nube o mensaje de error)."""
    if destino.exists():
        try:
            describir(destino)
            return "ya estaba", ""
        except Exception:
            destino.unlink()  # archivo incompleto de una corrida anterior
    destino.parent.mkdir(parents=True, exist_ok=True)
    temporal = destino.with_name(destino.name + ".part")
    error = ""
    for nube, base in BASES.items():
        for intento in range(intentos):
            try:
                r = requests.get(base + file_name, timeout=60)
                if r.status_code == 200 and r.content:
                    temporal.write_bytes(r.content)
                    describir(temporal)  # si no es una imagen legible, cuenta como fallo
                    temporal.replace(destino)
                    return "descargada", nube
                error = f"{nube}: HTTP {r.status_code}"
                if r.status_code in (403, 404):
                    break  # no está o no se permite en esta nube; se prueba la siguiente
            except Exception as e:
                error = f"{nube}: {str(e)[:100]}"
            time.sleep(1 + 2 * intento)
    if temporal.exists():
        temporal.unlink()
    return "error", error


def limpio(texto):
    return re.sub(r"[^A-Za-z0-9-]+", "-", str(texto)).strip("-")


def entero(v, defecto):
    try:
        return int(v)
    except (TypeError, ValueError):
        return defecto


def copia_revision(ruta, cajas_rel, titulo, destino):
    with Image.open(ruta) as im:
        im = im.convert("RGB")
    esc = min(1.0, LADO_COPIA / max(im.size))
    if esc < 1:
        im = im.resize((round(im.width * esc), round(im.height * esc)), Image.LANCZOS)
    W, H = im.size
    d = ImageDraw.Draw(im)
    grosor = max(2, round(min(W, H) / 250))
    for x, y, w, h in cajas_rel:  # [x, y, ancho, alto] relativos, origen arriba a la izquierda
        d.rectangle([x * W, y * H, (x + w) * W, (y + h) * H], outline=VERDE, width=grosor)
    letra = fuente_tipo(max(14, round(min(W, H) / 32)))
    alto_banda = round(min(W, H) / 32 * 1.7)
    lienzo = Image.new("RGB", (W, H + alto_banda), (0, 0, 0))
    lienzo.paste(im, (0, alto_banda))
    ImageDraw.Draw(lienzo).text((2 * grosor, alto_banda * 0.15), titulo, fill=(255, 255, 255), font=letra)
    lienzo.save(destino, quality=88)
    return lienzo


def mosaicos_secuencia(teselas, cabecera, prefijo, carpeta):
    por_hoja = COLUMNAS * FILAS
    letra, letra_cab = fuente_tipo(16), fuente_tipo(18)
    hojas = []
    total = (len(teselas) + por_hoja - 1) // por_hoja
    for i in range(0, len(teselas), por_hoja):
        hoja = Image.new("RGB", (COLUMNAS * TESELA_W, CABECERA + FILAS * (TESELA_H + PIE)), (40, 40, 40))
        d = ImageDraw.Draw(hoja)
        d.text((8, 8), f"{cabecera} | hoja {i // por_hoja + 1} de {total}", fill=(255, 255, 255), font=letra_cab)
        for j, (im, pie) in enumerate(teselas[i:i + por_hoja]):
            x, y = (j % COLUMNAS) * TESELA_W, CABECERA + (j // COLUMNAS) * (TESELA_H + PIE)
            mini = im.copy()
            mini.thumbnail((TESELA_W - 4, TESELA_H - 4))
            hoja.paste(mini, (x + (TESELA_W - mini.width) // 2, y + (TESELA_H - mini.height) // 2))
            d.text((x + 6, y + TESELA_H + 4), pie, fill=(255, 255, 255), font=letra)
        destino = carpeta / f"{prefijo}_{i // por_hoja + 1}.jpg"
        hoja.save(destino, quality=85)
        hojas.append(destino)
    return hojas


def imagenes_en(carpeta):
    return [p for p in carpeta.rglob("*") if p.suffix.lower() in (".jpg", ".jpeg", ".png")] if carpeta.exists() else []


def main():
    ap = argparse.ArgumentParser(description="Descarga las imágenes de oso andino de WCS y prepara su revisión.")
    ap.add_argument("--tabla", type=Path, default=Path("docs") / "wcs" / "wcs_oso_andino.csv")
    ap.add_argument("--imagenes", type=Path, default=Path("data") / "raw" / "wcs" / "imagenes")
    ap.add_argument("--revision", type=Path, default=Path("data") / "revisar" / "wcs")
    ap.add_argument("--mosaicos", type=Path, default=Path("data") / "wcs_mosaicos")
    ap.add_argument("--salida", type=Path, default=Path("docs") / "wcs" / "wcs_descarga.csv")
    ap.add_argument("--hilos", type=int, default=8)
    ap.add_argument("--sobrescribir", action="store_true")
    args = ap.parse_args()

    if not args.tabla.exists():
        sys.exit(f"ERROR: no encuentro {args.tabla}. Corre antes scripts\\explorar_wcs.py desde la raíz del proyecto.")
    df = pd.read_csv(args.tabla, dtype={"location": str, "seq_id": str, "image_id": str})
    df["seq_id"] = df.seq_id.fillna("sin-secuencia-" + df.image_id)  # en WCS todas tienen; por si acaso
    for c in ("country_code", "estado_lila", "cajas_oso_rel"):
        if c not in df.columns:
            sys.exit(f"ERROR: a {args.tabla} le falta la columna {c}. Vuelve a correr explorar_wcs.py (versión 2) "
                     f"con --sobrescribir.")
    clasificadas = [p for c in CATEGORIAS for p in imagenes_en(args.revision / c)]
    if clasificadas:
        sys.exit(f"ERROR: ya hay {len(clasificadas)} fotos clasificadas en las subcarpetas de {args.revision}. "
                 f"No las toco. Si de verdad quieres rehacer las copias, saca antes esas subcarpetas de ahí.")
    if not args.sobrescribir and (args.salida.exists() or imagenes_en(args.revision) or imagenes_en(args.mosaicos)):
        sys.exit(f"ERROR: ya existen {args.salida}, copias en {args.revision} o mosaicos en {args.mosaicos}. "
                 f"Usa --sobrescribir para rehacerlos.")

    # ------------------------------------------------------------------ 1. Descarga
    print(f"Imágenes en la tabla: {len(df)}. Destino: {args.imagenes}")
    rutas = {r.image_id: args.imagenes / Path(r.file_name) for r in df.itertuples()}
    resultados, hechas, t0 = {}, 0, time.time()
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.hilos) as ex:
        futuros = {ex.submit(bajar, r.file_name, rutas[r.image_id]): r.image_id for r in df.itertuples()}
        for fut in concurrent.futures.as_completed(futuros):
            resultados[futuros[fut]] = fut.result()
            hechas += 1
            if hechas % 50 == 0 or hechas == len(df):
                print(f"  {hechas}/{len(df)} ({time.time() - t0:.0f} s)")
    df["descarga"] = df.image_id.map(lambda k: resultados[k][0])
    df["nube_o_error"] = df.image_id.map(lambda k: resultados[k][1])
    print(f"Descarga: {df.descarga.value_counts().to_dict()}")
    nubes = df[df.descarga == "descargada"].nube_o_error.value_counts().to_dict()
    if nubes:
        print(f"  desde: {nubes}")
    errores = df[df.descarga == "error"]
    if len(errores):
        print(f"ERROR: {len(errores)} imágenes no se pudieron bajar. Primeras:")
        for r in errores.head(10).itertuples():
            print(f"  {r.file_name}  {r.nube_o_error}")
        sys.exit("Vuelve a correr el script (retoma donde quedó). Si el error se repite, pégame estas líneas.")

    # ------------------------------------------------------------------ 2. Comprobación de las imágenes
    desc = pd.DataFrame([describir(rutas[k]) | {"image_id": k} for k in df.image_id])
    df = df.merge(desc, on="image_id")
    df["ruta_local"] = [str(rutas[k]) for k in df.image_id]
    print(f"\nTamaño en disco: {df.bytes.sum() / 1e6:.0f} MB")
    print(f"Formato real (PIL): {df.formato.value_counts().to_dict()}; modo: {df.modo.value_counts().to_dict()}")
    if (df.formato != "JPEG").any():
        print(f"ATENCIÓN: {int((df.formato != 'JPEG').sum())} imágenes no son JPEG por dentro (ver columna formato).")
    distinto = df[(df.ancho_real != df.width) | (df.alto_real != df.height)]
    print(f"Ancho x alto distinto del que dice el JSON: {len(distinto)}")
    print(f"En escala de grises (posible infrarrojo nocturno): {int(df.escala_grises.sum())}")
    print(f"Brillo medio: mediana {df.brillo_medio.median():.0f}; oscuras (< 60): {int((df.brillo_medio < 60).sum())}")
    if df.escala_grises.any():
        g = df[df.escala_grises]
        print(f"  horas del reloj de esas fotos en escala de grises: {g.hora.value_counts().sort_index().to_dict()}")

    # ------------------------------------------------------------------ 3. Copias de revisión y mosaicos
    for carpeta in (args.revision, args.mosaicos):
        if carpeta.exists():
            for p in carpeta.glob("*"):
                if p.is_file():
                    p.unlink()
        carpeta.mkdir(parents=True, exist_ok=True)
    df = df.sort_values(["location", "fecha", "hora", "seq_id", "frame_num"]).reset_index(drop=True)
    # El formato COCO Camera Traps numera las fotos de una secuencia desde 0, pero WCS lo hace desde 1 (tiene
    # frame_num 10 en secuencias de 10 fotos). Se detecta para mostrar siempre "foto 1 de N".
    desde = 0 if (pd.to_numeric(df.frame_num, errors="coerce") == 0).any() else 1
    num_seq = {s: i + 1 for i, s in enumerate(dict.fromkeys(df.seq_id))}
    copias, hojas = [], []
    for s, g in df.groupby("seq_id", sort=False):
        teselas = []
        n_seq = entero(g.fotos_seq_metadatos.iloc[0], len(g))
        for r in g.itertuples():
            orden = len(copias) + 1
            foto = entero(r.frame_num, desde)
            nombre = f"wcs_{orden:03d}_{limpio(r.location)}_s{num_seq[s]:02d}_f{foto:02d}_{Path(r.file_name).stem}.jpg"
            cajas = json.loads(r.cajas_oso_rel) if isinstance(r.cajas_oso_rel, str) and r.cajas_oso_rel else []
            hora_txt = str(r.datetime)[11:19]
            estado = ESTADO_CORTO.get(r.estado_lila, "?")
            titulo = (f"camara {r.location} | secuencia {num_seq[s]} ({s}) | foto {foto + 1 - desde} de {n_seq} | "
                      f"{r.fecha} {hora_txt} | {estado}")
            lienzo = copia_revision(rutas[r.image_id], cajas, titulo, args.revision / nombre)
            teselas.append((lienzo, f"foto {foto + 1 - desde} | {estado}"))
            copias.append(nombre)
        cab = (f"camara {g.location.iloc[0]} | secuencia {num_seq[s]} ({s}) | {g.fecha.iloc[0]} | "
               f"{len(g)} fotos etiquetadas de {n_seq}")
        hojas += mosaicos_secuencia(teselas, cab, f"mosaico_s{num_seq[s]:02d}_{limpio(g.location.iloc[0])}",
                                    args.mosaicos)
    df["copia_revision"] = copias
    df["num_secuencia"] = df.seq_id.map(num_seq)

    columnas = ["image_id", "file_name", "ruta_local", "copia_revision", "num_secuencia", "location", "seq_id",
                "frame_num", "estado_lila", "descarga", "nube_o_error", "bytes", "formato", "modo", "ancho_real",
                "alto_real", "brillo_medio", "dif_canales", "escala_grises"]
    args.salida.parent.mkdir(parents=True, exist_ok=True)
    df[columnas].to_csv(args.salida, index=False)

    print(f"\nPor estado en LILA: {df.estado_lila.value_counts().to_dict()}")
    print(f"Secuencias: {len(num_seq)}. Copias de revisión: {len(copias)} en {args.revision}")
    print(f"Mosaicos: {len(hojas)} hojas en {args.mosaicos}")
    print(f"Tabla: {args.salida}")
    print("\nPara registrar la revisión, cuando termines de mover las copias a oso, sin_oso y duda:")
    print(f"  python scripts\\registrar_revision.py --dir {args.revision} --lote wcs_oso_andino "
          f"--out docs\\wcs\\revision_manual_wcs.csv")


if __name__ == "__main__":
    main()
