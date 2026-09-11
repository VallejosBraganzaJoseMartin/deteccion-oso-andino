#!/usr/bin/env python3
"""
Descarga fotos de iNaturalist de OTRAS especies para usarlas como negativos
(imagenes sin oso, con el mismo estilo fotografico que los positivos).

Derivado de descargar_inaturalist.py: misma API v1, mismos filtros de
licencia, mismo registro de atribucion. Diferencias:
  - Varias especies en una corrida, con cuota de fotos por especie y un
    maximo de fotos por observacion (para no llenar la cuota con 20 fotos
    del mismo perro).
  - Recuadro geografico (por defecto Ecuador continental) con los parametros
    nelat/nelng/swlat/swlng de la API.
  - Especies "cautivas" (animales domesticos): en iNaturalist las
    observaciones de animales domesticos suelen marcarse como cautivas y por
    eso son de grado "casual", asi que para ellas no se exige grado de
    investigacion ni captive=false.
  - Archivos nombrados inatneg_<observacion>_<foto>.jpg, en su propia carpeta,
    y una columna 'taxon' en el CSV.

Uso (desde la raiz del proyecto):
    python scripts/descargar_inat_negativos.py --out data/raw/inaturalist_negativos \
        --taxones "Lycalopex culpaeus,Tapirus pinchaque,Puma concolor,Odocoileus virginianus" \
        --taxones-cautivos "Canis familiaris,Bos taurus,Equus caballus,Ovis aries" \
        --max-fotos-por-taxon 70

Se puede relanzar: las fotos ya descargadas se omiten y no se vuelven a
registrar. Para ampliar el recuadro (p. ej. a los Andes del norte) usar
--bbox SWLAT SWLNG NELAT NELNG.

NO PROBADO contra la API real desde este entorno (solo la logica con
respuestas simuladas). Nombres de parametros verificados en la
documentacion publica de la API v1.
"""

import argparse
import csv
import os
import sys
import time

import requests

API_URL = "https://api.inaturalist.org/v1/observations"
USER_AGENT = "jmvallejosb@utn.edu.ec"

CAMPOS_CSV = [
    "archivo", "taxon", "observacion_id", "foto_id", "url", "licencia", "atribucion",
    "usuario", "fecha_observada", "hora_observada", "lugar", "cautivo",
    "calidad", "latitud", "longitud",
]


def obtener_pagina(sesion, params):
    r = sesion.get(API_URL, params=params, timeout=60)
    if r.status_code != 200:
        print(f"Error {r.status_code} en la API: {r.text[:300]}", file=sys.stderr)
        return None
    return r.json()


def descargar(sesion, url, destino):
    r = sesion.get(url, timeout=60, stream=True)
    if r.status_code != 200:
        return False
    with open(destino, "wb") as f:
        for chunk in r.iter_content(chunk_size=65536):
            f.write(chunk)
    return True


def descargar_taxon(sesion, escritor, taxon, cautivo, args, ya_registradas, ya_del_taxon):
    """Completa la cuota de args.max_fotos_por_taxon para un taxon (contando las ya
    registradas en corridas anteriores). Devuelve (fotos nuevas, obs vistas)."""
    licencias_ok = {l.strip() for l in args.licencias.split(",") if l.strip()}
    swlat, swlng, nelat, nelng = args.bbox
    params = {
        "taxon_name": taxon, "photos": "true", "photo_license": args.licencias,
        "swlat": swlat, "swlng": swlng, "nelat": nelat, "nelng": nelng,
        "per_page": 200, "page": 1,
    }
    if not cautivo:
        params["quality_grade"] = "research"
        params["captive"] = "false"

    fotos, nuevas, obs_vistas = ya_del_taxon, 0, 0
    if fotos >= args.max_fotos_por_taxon:
        print(f"[{taxon}] ya tiene {fotos} fotos registradas; cuota cubierta")
        return 0, 0
    while fotos < args.max_fotos_por_taxon:
        datos = obtener_pagina(sesion, params)
        time.sleep(args.pausa)
        if not datos or not datos.get("results"):
            break
        if params["page"] == 1:
            print(f"[{taxon}] observaciones que reporta la API en el recuadro: {datos.get('total_results')}")
        for obs in datos["results"]:
            obs_vistas += 1
            geo = obs.get("geojson") or {}
            coords = geo.get("coordinates") or [None, None]
            fotos_obs = 0
            for foto in obs.get("photos") or []:
                if fotos >= args.max_fotos_por_taxon or fotos_obs >= args.max_fotos_por_obs:
                    break
                lic = foto.get("license_code")
                if lic not in licencias_ok or not foto.get("url"):
                    continue
                nombre = f"inatneg_{obs['id']}_{foto['id']}.jpg"
                if nombre in ya_registradas:
                    fotos_obs += 1
                    continue
                url = foto["url"].replace("square", args.tamano)
                destino = os.path.join(args.out, nombre)
                if not os.path.exists(destino):
                    if not descargar(sesion, url, destino):
                        print(f"No se pudo descargar {url}", file=sys.stderr)
                        continue
                    time.sleep(args.pausa)
                escritor.writerow({
                    "archivo": nombre, "taxon": taxon,
                    "observacion_id": obs["id"], "foto_id": foto["id"], "url": url,
                    "licencia": lic, "atribucion": foto.get("attribution", ""),
                    "usuario": (obs.get("user") or {}).get("login", ""),
                    "fecha_observada": obs.get("observed_on", ""),
                    "hora_observada": obs.get("time_observed_at", ""),
                    "lugar": obs.get("place_guess", ""), "cautivo": obs.get("captive", ""),
                    "calidad": obs.get("quality_grade", ""),
                    "latitud": coords[1], "longitud": coords[0],
                })
                ya_registradas.add(nombre)
                fotos += 1
                nuevas += 1
                fotos_obs += 1
            if fotos >= args.max_fotos_por_taxon:
                break
        if len(datos["results"]) < params["per_page"]:
            break
        params["page"] += 1
    return nuevas, obs_vistas


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/raw/inaturalist_negativos")
    ap.add_argument("--taxones", default="", help="Especies silvestres, separadas por comas")
    ap.add_argument("--taxones-cautivos", default="",
                    help="Especies domesticas (sin exigir grado de investigacion ni captive=false)")
    ap.add_argument("--licencias", default="cc0,cc-by,cc-by-nc")
    ap.add_argument("--tamano", default="large", choices=["medium", "large", "original"])
    ap.add_argument("--max-fotos-por-taxon", type=int, default=70)
    ap.add_argument("--max-fotos-por-obs", type=int, default=2)
    ap.add_argument("--bbox", nargs=4, type=float, default=[-5.1, -81.2, 1.7, -75.1],
                    metavar=("SWLAT", "SWLNG", "NELAT", "NELNG"),
                    help="Recuadro geografico; por defecto Ecuador continental")
    ap.add_argument("--pausa", type=float, default=1.0)
    args = ap.parse_args()

    taxones = [(t.strip(), False) for t in args.taxones.split(",") if t.strip()]
    taxones += [(t.strip(), True) for t in args.taxones_cautivos.split(",") if t.strip()]
    if not taxones:
        raise SystemExit("Indica al menos una especie con --taxones o --taxones-cautivos")

    os.makedirs(args.out, exist_ok=True)
    csv_path = os.path.join(args.out, "registro_licencias.csv")
    ya_registradas, ya_por_taxon = set(), {}
    if os.path.exists(csv_path):
        with open(csv_path, newline="", encoding="utf-8") as f:
            for r in csv.DictReader(f):
                ya_registradas.add(r["archivo"])
                ya_por_taxon[r["taxon"]] = ya_por_taxon.get(r["taxon"], 0) + 1
    csv_nuevo = not os.path.exists(csv_path)

    sesion = requests.Session()
    sesion.headers.update({"User-Agent": USER_AGENT})

    resumen = []
    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        escritor = csv.DictWriter(f, fieldnames=CAMPOS_CSV)
        if csv_nuevo:
            escritor.writeheader()
        for taxon, cautivo in taxones:
            n, obs = descargar_taxon(sesion, escritor, taxon, cautivo, args, ya_registradas,
                                     ya_por_taxon.get(taxon, 0))
            f.flush()
            resumen.append((taxon, n, obs))
            print(f"[{taxon}] {n} fotos nuevas de {obs} observaciones revisadas")

    print("\nResumen:")
    for taxon, n, obs in resumen:
        print(f"  {taxon}: {n} fotos nuevas")
    print(f"Total registrado en {csv_path}: {len(ya_registradas)} fotos")


if __name__ == "__main__":
    main()
