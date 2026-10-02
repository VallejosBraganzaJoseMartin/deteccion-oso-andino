"""
medir_velocidad_cpu.py

Mide la velocidad y la memoria RAM de un modelo YOLO26 en CPU, procesando las fotos
una por una, como lo hará la aplicación de escritorio.

Uso, desde la raíz del proyecto, un modelo por ejecución:
    python scripts\\medir_velocidad_cpu.py --pesos modelos\\e1_n_deg0_s0_best.pt --conf 0.363

Cada modelo se mide en un proceso aparte para que la memoria de uno no se sume a la del otro.

Qué mide (comprobado en el código de Ultralytics 8.4.138):
- La carga del modelo y la primera foto, por separado. La primera llamada a predict prepara
  el predictor: fusiona capas, elige el formato de memoria, fija los hilos de PyTorch y
  calienta la red. No entra en las estadísticas.
- Calentamiento: 10 fotos (5 de ENA24 y 5 de iNaturalist) que no se miden.
- Muestra medida: 100 fotos de validación (50 de ENA24 y 50 de iNaturalist), elegidas con
  semilla fija y distintas de las de calentamiento. Todas las ejecuciones usan las mismas.
- Por foto:
  * total: reloj de pared alrededor de model.predict(ruta). Incluye leer y decodificar el
    JPEG del disco, preparar la imagen, la red y el posproceso. Es lo que espera el usuario.
  * preproceso, inferencia y posproceso: los tiempos que Ultralytics guarda en Results.speed.
    No incluyen la lectura del disco: en 8.4.138 la foto se lee en el iterador del dataset,
    fuera de sus cronómetros (engine/predictor.py, stream_inference).
- RAM: memoria residente del proceso (RSS) antes y después de importar las librerías, con el
  modelo cargado y después de cada foto, más el pico del proceso que informa Windows
  (peak_wset de psutil).

Salida: resumen en pantalla y una fila por foto en docs/modelado/velocidad_cpu_<RUN_ID>.csv.
No sobrescribe ese archivo salvo con --sobrescribir.
"""

import argparse
import csv
import platform
import random
import statistics
import sys
import time
from pathlib import Path

import psutil

PROC = psutil.Process()


def rss_mb():
    return PROC.memory_info().rss / 2**20


RSS_ARRANQUE = rss_mb()  # antes de importar torch y ultralytics

import torch  # noqa: E402
import ultralytics  # noqa: E402
from ultralytics import YOLO  # noqa: E402

RSS_IMPORTS = rss_mb()

RAIZ_DATOS = Path("data") / "processed"
LISTAS = {"ENA24": RAIZ_DATOS / "val_ena24.txt", "iNaturalist": RAIZ_DATOS / "val_inat.txt"}
N_MEDIDAS_POR_FUENTE = 50
N_CALENT_POR_FUENTE = 5
SEMILLA = 0
IMGSZ = 640
META_MS = 1000.0
META_RAM_MB = 4 * 1024


def leer_lista(txt):
    """Rutas de una lista por fuente. './' se resuelve respecto de la carpeta del .txt, igual que Ultralytics."""
    if not txt.is_file():
        sys.exit(f"ERROR: no encuentro {txt}. Ejecuta el script desde la raíz del proyecto.")
    rutas = []
    for linea in txt.read_text(encoding="utf-8").splitlines():
        linea = linea.strip()
        if not linea:
            continue
        p = Path(linea)
        rutas.append(p if p.is_absolute() else txt.parent / p)
    faltan = [p for p in rutas if not p.is_file()]
    if faltan:
        sys.exit(f"ERROR: {len(faltan)} rutas de {txt.name} no existen, por ejemplo {faltan[0]}")
    return sorted(rutas)


def elegir_fotos():
    rng = random.Random(SEMILLA)
    calentamiento, medidas = [], []
    for fuente, txt in LISTAS.items():
        rutas = leer_lista(txt)
        necesarias = N_CALENT_POR_FUENTE + N_MEDIDAS_POR_FUENTE
        if len(rutas) < necesarias:
            sys.exit(f"ERROR: {txt.name} tiene {len(rutas)} fotos y hacen falta {necesarias}.")
        elegidas = rng.sample(rutas, necesarias)
        calentamiento += [(fuente, p) for p in elegidas[:N_CALENT_POR_FUENTE]]
        medidas += [(fuente, p) for p in elegidas[N_CALENT_POR_FUENTE:]]
    rng.shuffle(medidas)  # alterna las fuentes para que un cambio de temperatura de la CPU no afecte solo a una
    return calentamiento, medidas


def formato_memoria(modelo):
    """Indica si Ultralytics dejó la red en formato channels_last (lo activa solo en CPU x86 con oneDNN)."""
    try:
        red = modelo.predictor.model.backend.model
        for m in red.modules():
            if isinstance(m, torch.nn.Conv2d) and m.weight.shape[1] > 1 and m.weight.shape[2] > 1:
                w = m.weight
                if w.is_contiguous(memory_format=torch.channels_last) and not w.is_contiguous():
                    return "sí"
                return "no"
    except Exception:
        pass
    return "no determinado"


def estadisticas(valores):
    v = sorted(valores)
    p95 = statistics.quantiles(v, n=20, method="inclusive")[18] if len(v) > 1 else v[0]
    return statistics.median(v), statistics.fmean(v), p95, v[-1]


def main():
    ap = argparse.ArgumentParser(description="Velocidad y RAM de un modelo YOLO en CPU, foto por foto.")
    ap.add_argument("--pesos", required=True, type=Path, help="archivo <RUN_ID>_best.pt")
    ap.add_argument("--conf", required=True, type=float, help="umbral de la regla B de ese modelo")
    ap.add_argument("--salida", type=Path, default=Path("docs") / "modelado", help="carpeta del CSV")
    ap.add_argument("--sobrescribir", action="store_true", help="reemplaza el CSV si ya existe")
    args = ap.parse_args()

    if not args.pesos.is_file():
        sys.exit(f"ERROR: no encuentro los pesos {args.pesos}")
    if not args.salida.is_dir():
        sys.exit(f"ERROR: no existe la carpeta de salida {args.salida}")
    run_id = args.pesos.stem.removesuffix("_best")
    csv_salida = args.salida / f"velocidad_cpu_{run_id}.csv"
    if csv_salida.exists() and not args.sobrescribir:
        sys.exit(f"ERROR: {csv_salida} ya existe. Usa --sobrescribir si quieres repetir la medición.")

    calentamiento, medidas = elegir_fotos()

    t0 = time.perf_counter()
    modelo = YOLO(str(args.pesos))
    t_carga = time.perf_counter() - t0
    rss_modelo = rss_mb()

    kw = dict(imgsz=IMGSZ, conf=args.conf, device="cpu", verbose=False)

    print(f"Calentando {run_id} con {len(calentamiento)} fotos...", flush=True)
    t0 = time.perf_counter()
    modelo.predict(source=str(calentamiento[0][1]), **kw)
    t_primera = time.perf_counter() - t0
    for _, ruta in calentamiento[1:]:
        modelo.predict(source=str(ruta), **kw)
    predictor = modelo.predictor
    hilos = torch.get_num_threads()  # Ultralytics los fija al preparar el predictor en CPU
    rss_max = rss_mb()

    print(f"Midiendo {len(medidas)} fotos...", flush=True)
    filas = []
    for i, (fuente, ruta) in enumerate(medidas, start=1):
        t0 = time.perf_counter()
        r = modelo.predict(source=str(ruta), **kw)[0]
        total_ms = (time.perf_counter() - t0) * 1000
        rss = rss_mb()
        rss_max = max(rss_max, rss)
        sp = r.speed
        suma = sp["preprocess"] + sp["inference"] + sp["postprocess"]
        filas.append(
            {
                "orden": i,
                "fuente": fuente,
                "archivo": ruta.name,
                "ancho": r.orig_shape[1],
                "alto": r.orig_shape[0],
                "total_ms": round(total_ms, 2),
                "preproceso_ms": round(sp["preprocess"], 2),
                "inferencia_ms": round(sp["inference"], 2),
                "posproceso_ms": round(sp["postprocess"], 2),
                "resto_ms": round(total_ms - suma, 2),
                "detecciones": len(r.boxes),
                "rss_mb": round(rss, 1),
            }
        )
        if i % 25 == 0:
            print(f"  {i}/{len(medidas)}", flush=True)

    predictor_estable = modelo.predictor is predictor
    mi = PROC.memory_info()
    pico_mb = getattr(mi, "peak_wset", None)
    pico_mb = pico_mb / 2**20 if pico_mb is not None else None

    with open(csv_salida, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(filas[0]))
        w.writeheader()
        w.writerows(filas)

    try:
        from ultralytics.utils.torch_utils import get_cpu_info

        cpu = get_cpu_info()
    except Exception:
        cpu = platform.processor() or "desconocida"

    n_fuente = {f: sum(1 for x in filas if x["fuente"] == f) for f in LISTAS}
    print()
    print("=" * 74)
    print(f"Velocidad en CPU: {args.pesos.name}")
    print(f"CPU: {cpu} | {psutil.cpu_count(logical=False)} núcleos físicos, {psutil.cpu_count()} lógicos")
    print(f"Python {platform.python_version()} | torch {torch.__version__} | ultralytics {ultralytics.__version__}")
    print(f"Hilos de PyTorch en la inferencia: {hilos} | channels_last: {formato_memoria(modelo)}")
    nombres = modelo.names if len(modelo.names) <= 5 else f"{len(modelo.names)} clases"
    print(f"Clases del modelo: {nombres} | imgsz {IMGSZ} | conf {args.conf}")
    print(f"Carga del modelo: {t_carga:.2f} s | primera foto (prepara el predictor): {t_primera:.2f} s")
    print(
        f"Muestra: {len(filas)} fotos ({n_fuente['ENA24']} ENA24, {n_fuente['iNaturalist']} iNaturalist), "
        f"tras {len(calentamiento)} de calentamiento"
    )
    print()
    print(f"{'Tiempo por foto (ms)':<34}{'mediana':>9}{'media':>9}{'p95':>9}{'máx':>9}")
    filas_tabla = [
        ("Total, ENA24", "total_ms", "ENA24"),
        ("Total, iNaturalist", "total_ms", "iNaturalist"),
        ("Total, todas", "total_ms", None),
        ("  preproceso (Ultralytics)", "preproceso_ms", None),
        ("  inferencia (Ultralytics)", "inferencia_ms", None),
        ("  posproceso (Ultralytics)", "posproceso_ms", None),
        ("  resto (leer JPEG y llamada)", "resto_ms", None),
    ]
    resumen = {}
    for etiqueta, clave, fuente in filas_tabla:
        valores = [x[clave] for x in filas if fuente is None or x["fuente"] == fuente]
        med, media, p95, mx = estadisticas(valores)
        resumen[(clave, fuente)] = (med, media, p95, mx)
        print(f"{etiqueta:<34}{med:>9.1f}{media:>9.1f}{p95:>9.1f}{mx:>9.1f}")
    print()
    con_det = sum(1 for x in filas if x["detecciones"] > 0)
    print(f"Fotos con al menos una detección: {con_det} de {len(filas)}")
    print(
        f"RAM (RSS, MB): {RSS_ARRANQUE:.0f} al arrancar, {RSS_IMPORTS:.0f} tras importar, "
        f"{rss_modelo:.0f} con el modelo cargado, {rss_max:.0f} máximo durante la medición"
    )
    pico_txt = f"{pico_mb:.0f} MB" if pico_mb is not None else "no disponible en este sistema"
    print(f"Pico de memoria del proceso (Windows): {pico_txt}")
    p95_total = resumen[("total_ms", None)][2]
    ram_ref = pico_mb if pico_mb is not None else rss_max
    print(f"Meta < 1 s por foto (p95 del total): {'CUMPLE' if p95_total < META_MS else 'NO CUMPLE'}")
    print(f"Meta < 4 GB de RAM (pico): {'CUMPLE' if ram_ref < META_RAM_MB else 'NO CUMPLE'}")
    if not predictor_estable:
        print("AVISO: Ultralytics rehizo el predictor durante la medición; los tiempos incluyen esa preparación.")
    print(f"Detalle por foto: {csv_salida}")


if __name__ == "__main__":
    main()
