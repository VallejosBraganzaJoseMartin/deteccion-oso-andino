#!/usr/bin/env python3
"""
Descarga fotos de iNaturalist para una especie, filtrando por licencia,
y registra fuente, licencia y atribución de cada foto en un CSV.

Uso:
    pip install requests
    python descargar_inaturalist.py --taxon "Tremarctos ornatus" --out data/raw/inaturalist

Notas:
- Parámetros de la API basados en la v1 de iNaturalist. Verificar nombres y
  valores contra https://api.inaturalist.org/v1/docs antes de usar.
- iNaturalist pide no superar ~1 petición por segundo y usar un User-Agent
  identificable. Cambia el correo en USER_AGENT.
- Por defecto excluye observaciones marcadas como cautivas/cultivadas
  (captive=false), lo que elimina buena parte de las fotos de zoológico.
- Se puede relanzar: las fotos ya descargadas se omiten.
"""

import argparse
import csv
import os
import sys
import time

import requests

API_URL = "https://api.inaturalist.org/v1/observations"
USER_AGENT = "jmvallejosb@utn.edu.ec"
PAUSA_SEGUNDOS = 1.0

CAMPOS_CSV = [
    "archivo", "observacion_id", "foto_id", "url", "licencia", "atribucion",
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--taxon", default="Tremarctos ornatus")
    ap.add_argument("--out", default="data/raw/inaturalist")
    ap.add_argument("--licencias", default="cc0,cc-by,cc-by-nc",
                    help="Lista separada por comas de codigos de licencia aceptados")
    ap.add_argument("--tamano", default="large", choices=["medium", "large", "original"])
    ap.add_argument("--max-obs", type=int, default=3000)
    ap.add_argument("--incluir-cautivos", action="store_true")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    licencias_ok = {l.strip() for l in args.licencias.split(",") if l.strip()}
    csv_path = os.path.join(args.out, "registro_licencias.csv")
    csv_nuevo = not os.path.exists(csv_path)

    sesion = requests.Session()
    sesion.headers.update({"User-Agent": USER_AGENT})

    params = {
        "taxon_name": args.taxon,
        "quality_grade": "research",
        "photos": "true",
        "photo_license": args.licencias,
        "per_page": 200,
        "page": 1,
    }
    if not args.incluir_cautivos:
        params["captive"] = "false"

    total_fotos = 0
    total_obs = 0
    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        escritor = csv.DictWriter(f, fieldnames=CAMPOS_CSV)
        if csv_nuevo:
            escritor.writeheader()

        while total_obs < args.max_obs:
            datos = obtener_pagina(sesion, params)
            time.sleep(PAUSA_SEGUNDOS)
            if not datos or not datos.get("results"):
                break
            if params["page"] == 1:
                print(f"Observaciones que reporta la API: {datos.get('total_results')}")

            for obs in datos["results"]:
                total_obs += 1
                geo = obs.get("geojson") or {}
                coords = geo.get("coordinates") or [None, None]
                for foto in obs.get("photos") or []:
                    lic = foto.get("license_code")
                    if lic not in licencias_ok or not foto.get("url"):
                        continue
                    url = foto["url"].replace("square", args.tamano)
                    nombre = f"inat_{obs['id']}_{foto['id']}.jpg"
                    destino = os.path.join(args.out, nombre)
                    if not os.path.exists(destino):
                        if not descargar(sesion, url, destino):
                            print(f"No se pudo descargar {url}", file=sys.stderr)
                            continue
                        time.sleep(PAUSA_SEGUNDOS)
                    escritor.writerow({
                        "archivo": nombre,
                        "observacion_id": obs["id"],
                        "foto_id": foto["id"],
                        "url": url,
                        "licencia": lic,
                        "atribucion": foto.get("attribution", ""),
                        "usuario": (obs.get("user") or {}).get("login", ""),
                        "fecha_observada": obs.get("observed_on", ""),
                        "hora_observada": obs.get("time_observed_at", ""),
                        "lugar": obs.get("place_guess", ""),
                        "cautivo": obs.get("captive", ""),
                        "calidad": obs.get("quality_grade", ""),
                        "latitud": coords[1],
                        "longitud": coords[0],
                    })
                    total_fotos += 1
                if total_obs >= args.max_obs:
                    break

            print(f"Pagina {params['page']}: {total_obs} observaciones, {total_fotos} fotos")
            params["page"] += 1

    print(f"Listo. {total_fotos} fotos registradas en {csv_path}")


if __name__ == "__main__":
    main()
