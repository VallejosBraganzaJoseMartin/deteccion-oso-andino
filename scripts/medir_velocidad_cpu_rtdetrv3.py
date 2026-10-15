"""
medir_velocidad_cpu_rtdetrv3.py

Mide la velocidad y la memoria RAM de RT-DETRv3 (implementación oficial, PaddlePaddle) en CPU,
procesando las fotos una por una, con las mismas fotos y el mismo protocolo que
medir_velocidad_cpu.py usó con YOLO26 el 30 de septiembre.

Se corre con el Python 3.12 del entorno de Paddle, desde la raíz del proyecto, un modelo por
ejecución (cada uno en su propio proceso, para que la memoria de uno no se sume a la del otro):

    & C:\\Users\\Martin\\venv_paddle\\Scripts\\python.exe scripts\\medir_velocidad_cpu_rtdetrv3.py `
        --repo C:\\Users\\Martin\\RT-DETRv3 `
        --pesos modelos\\e6_rtdetrv3r18_s0\\e6_rtdetrv3r18_s0_best.pdparams --conf 0.842

Qué hace (verificado en el código de RT-DETRv3, commit 349e7d9, y de PaddlePaddle 3.3.0):
- Arma el modelo con la configuración oficial rtdetrv3_r18vd_6x_coco.yml y num_classes = 1, que es lo
  único de la configuración del proyecto (rtdetrv3_r18vd_oso.yml) que cambia la red. Comprueba que el
  archivo de pesos tenga todos los pesos del modelo, con su forma, y que no sobre ninguno. Los carga con
  la misma función que tools/eval.py (ppdet.utils.checkpoint.load_pretrain_weight).
- Prepara cada foto con las operaciones del TestReader de la configuración oficial: Decode, Resize a
  640 x 640 sin conservar la proporción (interpolación cúbica), NormalizeImage y Permute. Son las mismas
  del EvalReader con el que se exportaron las detecciones de validación.
- Inferencia en modo evaluación y sin gradientes (paddle.no_grad), como la evaluación de PaddleDetection.
- Hilos: Paddle 3.3.0 deja por defecto en 1 los hilos de su OpenMP (comprobado con threadpoolctl en
  Linux). El script los fija con paddle.base.core.set_num_threads (--hilos, 8 por defecto: los mismos que
  Ultralytics usó en este equipo con YOLO26) e informa los hilos de cada librería.
- oneDNN: no se toca. En modo dinámico, encender FLAGS_use_mkldnn no cambió nada en Paddle 3.3.0 (probado
  en Linux: mismo tiempo de una convolución y ninguna traza de oneDNN), así que se mide la configuración
  por defecto de Paddle, como con YOLO26 se midió la de Ultralytics.
- Fotos: las de medir_velocidad_cpu.py. Se sortean con la misma semilla y el mismo código (10 de
  calentamiento y 100 medidas, mitad de cada fuente). Si existe la tabla de E1
  (docs/modelado/velocidad_cpu_e1_n_deg0_s0.csv), las fotos medidas y su orden se toman de ella, y se
  avisa si el sorteo no coincide.
- Por foto:
  * total: reloj de pared desde leer el archivo hasta tener las cajas filtradas por --conf;
  * lectura: leer y decodificar el JPEG (operación Decode);
  * preproceso: Resize, NormalizeImage, Permute y paso a tensores;
  * red: la llamada al modelo sin su posproceso. Es lo comparable con la "inferencia" de Ultralytics;
  * posproceso: el DETRPostProcess del modelo (las 300 mejores consultas pasadas a cajas de la foto
    original), pasar a numpy y filtrar por --conf.
- RAM: memoria residente del proceso (RSS) antes y después de importar las librerías, con el modelo
  cargado y después de cada foto, más el pico del proceso que informa Windows (peak_wset de psutil).
- Control: si existe umbral_imagenes.csv de la evaluación sobre validación (--comparar), compara la
  confianza máxima de cada foto con la que dio la evaluación en GPU. Si los pesos o el preproceso
  estuvieran mal, no coincidirían.

Salida: resumen en pantalla y una fila por foto en docs/modelado/velocidad_cpu_<RUN_ID>.csv.
No sobrescribe ese archivo salvo con --sobrescribir.
"""

import argparse
import csv
import os
import platform
import random
import statistics
import subprocess
import sys
import time
from pathlib import Path

import psutil

os.environ.setdefault("MPLBACKEND", "Agg")  # ppdet importa matplotlib; sin ventana

PROC = psutil.Process()


def rss_mb():
    return PROC.memory_info().rss / 2**20


RAIZ_DATOS = Path("data") / "processed"
LISTAS = {"ENA24": RAIZ_DATOS / "val_ena24.txt", "iNaturalist": RAIZ_DATOS / "val_inat.txt"}
N_MEDIDAS_POR_FUENTE = 50
N_CALENT_POR_FUENTE = 5
SEMILLA = 0
META_MS = 1000.0
META_RAM_MB = 4 * 1024
COMMIT = "349e7d99a5065e7b684118912e6a74178d4f4625"
CONFIG_OFICIAL = Path("configs") / "rtdetrv3" / "rtdetrv3_r18vd_6x_coco.yml"
TRANSFORMS_ESPERADAS = ["Decode", "Resize", "NormalizeImage", "Permute"]
TOL_CONF = 0.01  # diferencia de confianza máxima tolerada frente a la evaluación en GPU (elegida, no medida)


# ---------- fotos: mismo código que medir_velocidad_cpu.py ----------

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


def fotos_de_referencia(csv_ref, medidas_sorteo):
    """Fotos medidas y su orden según la tabla de YOLO26, si existe. Devuelve (medidas, mensaje)."""
    if not csv_ref.is_file():
        return medidas_sorteo, f"no encuentro {csv_ref}; uso el sorteo (mismo código y semilla)"
    por_nombre = {}
    for fuente, txt in LISTAS.items():
        for p in leer_lista(txt):
            por_nombre[p.name] = (fuente, p)
    with open(csv_ref, newline="", encoding="utf-8") as f:
        filas = sorted(csv.DictReader(f), key=lambda r: int(r["orden"]))
    faltan = [r["archivo"] for r in filas if r["archivo"] not in por_nombre]
    if faltan:
        sys.exit(f"ERROR: {len(faltan)} fotos de {csv_ref.name} no están en las listas de validación, "
                 f"por ejemplo {faltan[0]}")
    medidas = [por_nombre[r["archivo"]] for r in filas]
    for r, (fuente, _) in zip(filas, medidas):
        if r["fuente"] != fuente:
            sys.exit(f"ERROR: {r['archivo']} figura como {r['fuente']} en {csv_ref.name} y como {fuente} en las listas")
    igual = [(f, p.name) for f, p in medidas] == [(f, p.name) for f, p in medidas_sorteo]
    msg = (f"las {len(medidas)} de {csv_ref.name}, en su orden; "
           + ("el sorteo da exactamente las mismas" if igual else
              "AVISO: el sorteo de este Python NO da las mismas; se usan las de la tabla"))
    return medidas, msg


# ---------- utilidades ----------

def estadisticas(valores):
    v = sorted(valores)
    p95 = statistics.quantiles(v, n=20, method="inclusive")[18] if len(v) > 1 else v[0]
    return statistics.median(v), statistics.fmean(v), p95, v[-1]


def nombre_cpu():
    if platform.system() == "Windows":
        try:
            import winreg
            with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r"HARDWARE\DESCRIPTION\System\CentralProcessor\0") as k:
                return winreg.QueryValueEx(k, "ProcessorNameString")[0].strip()
        except OSError:
            pass
    try:
        for linea in Path("/proc/cpuinfo").read_text().splitlines():
            if linea.startswith("model name"):
                return linea.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or "desconocida"


def commit_del_repo(repo):
    try:
        r = subprocess.run(["git", "-C", str(repo), "rev-parse", "HEAD"], capture_output=True, text=True, timeout=30)
        if r.returncode == 0:
            return r.stdout.strip()
    except (OSError, subprocess.SubprocessError):
        pass
    return "no determinado (¿git en el PATH?)"


def hilos_por_libreria():
    try:
        from threadpoolctl import threadpool_info
        return [f"{Path(i.get('filepath', '?')).name} ({i.get('internal_api')}): {i.get('num_threads')}"
                for i in threadpool_info()]
    except Exception as e:  # noqa: BLE001
        return [f"no determinado ({e})"]


class RelojPosproceso:
    """Envuelve el DETRPostProcess del modelo (un objeto Python, no una capa) para medir su tiempo aparte."""

    def __init__(self, original):
        self.original = original
        self.segundos = 0.0

    def __call__(self, *args, **kwargs):
        t0 = time.perf_counter()
        r = self.original(*args, **kwargs)
        self.segundos += time.perf_counter() - t0
        return r


def comparar_con_evaluacion(csv_imgs, run_id, filas):
    """Confianza máxima por foto: esta medición en CPU frente a la evaluación en GPU (umbral_imagenes.csv)."""
    if csv_imgs is None:
        return ("sin control: no encontré docs/modelado/eval_e7/umbral_imagenes.csv ni "
                "docs/modelado/eval_rtdetrv3/umbral_imagenes.csv (se puede indicar con --comparar)")
    if not csv_imgs.is_file():
        return f"sin control: no encuentro {csv_imgs}"
    col = f"maxconf_{run_id}"
    with open(csv_imgs, newline="", encoding="utf-8") as f:
        lector = csv.DictReader(f)
        if col not in (lector.fieldnames or []):
            return f"sin control: {csv_imgs.name} no tiene la columna {col}"
        ref = {r["archivo"]: float(r[col]) for r in lector if r.get("conjunto", "val") == "val"}
    difs = [abs(x["conf_max"] - ref[x["archivo"]]) for x in filas if x["archivo"] in ref]
    if len(difs) != len(filas):
        return f"sin control completo: solo {len(difs)} de {len(filas)} fotos están en {csv_imgs.name}"
    grandes = sum(d > TOL_CONF for d in difs)
    veredicto = "coincide" if grandes == 0 else "REVISAR"
    return (f"confianza máxima por foto frente a {csv_imgs} (GPU): diferencia mediana {statistics.median(difs):.4f}, "
            f"máxima {max(difs):.4f}; {grandes} de {len(difs)} fotos por encima de {TOL_CONF} -> {veredicto}")


def referencias_yolo(carpeta, propio):
    """Medianas de las mediciones de YOLO26 del 30 de septiembre (mismas fotos), para comparar."""
    lineas = []
    for p in sorted(carpeta.glob("velocidad_cpu_*.csv")):
        if p.name == propio.name:
            continue
        try:
            with open(p, newline="", encoding="utf-8") as f:
                filas = list(csv.DictReader(f))
            tot = [float(r["total_ms"]) for r in filas]
            red = [float(r.get("inferencia_ms") or r.get("red_ms")) for r in filas]
            rss = max(float(r["rss_mb"]) for r in filas)
            _, _, p95, _ = estadisticas(tot)
            lineas.append(f"   {p.stem.removeprefix('velocidad_cpu_'):<26} total mediana {statistics.median(tot):7.1f} ms "
                          f"(p95 {p95:7.1f}) | red mediana {statistics.median(red):7.1f} ms | RSS máx. {rss:6.0f} MB")
        except (KeyError, ValueError, TypeError):
            continue
    return lineas


# ---------- programa ----------

def main():
    ap = argparse.ArgumentParser(description="Velocidad y RAM de RT-DETRv3 (PaddlePaddle) en CPU, foto por foto.")
    ap.add_argument("--repo", required=True, type=Path, help="carpeta del repositorio clxia12/RT-DETRv3")
    ap.add_argument("--pesos", required=True, type=Path, help="archivo <RUN_ID>_best.pdparams")
    ap.add_argument("--conf", required=True, type=float, help="umbral de la regla B de ese modelo")
    ap.add_argument("--hilos", type=int, default=8, help="hilos de CPU de Paddle (8, como YOLO26 en este equipo)")
    ap.add_argument("--clases", type=int, default=1, help="num_classes con que se entrenó (1: oso)")
    ap.add_argument("--salida", type=Path, default=Path("docs") / "modelado", help="carpeta del CSV")
    ap.add_argument("--fotos-de", type=Path, default=Path("docs") / "modelado" / "velocidad_cpu_e1_n_deg0_s0.csv",
                    help="tabla de YOLO26 de la que se toman las fotos medidas y su orden")
    ap.add_argument("--comparar", type=Path, default=None,
                    help="umbral_imagenes.csv de una evaluación que incluya este modelo (por defecto, el de eval_e7 "
                         "o el de eval_rtdetrv3 si existen)")
    ap.add_argument("--sobrescribir", action="store_true", help="reemplaza el CSV si ya existe")
    args = ap.parse_args()

    rss_arranque = rss_mb()  # antes de importar paddle y ppdet

    config = args.repo / CONFIG_OFICIAL
    if not config.is_file():
        sys.exit(f"ERROR: no encuentro {config}. ¿--repo apunta a la carpeta del repositorio?")
    if not args.pesos.is_file() or args.pesos.suffix != ".pdparams":
        sys.exit(f"ERROR: no encuentro el archivo de pesos .pdparams {args.pesos}")
    if not args.salida.is_dir():
        sys.exit(f"ERROR: no existe la carpeta de salida {args.salida}")
    run_id = args.pesos.stem.removesuffix("_best")
    csv_salida = args.salida / f"velocidad_cpu_{run_id}.csv"
    if csv_salida.exists() and not args.sobrescribir:
        sys.exit(f"ERROR: {csv_salida} ya existe. Usa --sobrescribir si quieres repetir la medición.")
    if args.comparar is None:
        for c in (Path("docs/modelado/eval_e7/umbral_imagenes.csv"), Path("docs/modelado/eval_rtdetrv3/umbral_imagenes.csv")):
            if c.is_file():
                args.comparar = c
                break

    calentamiento, medidas_sorteo = elegir_fotos()
    medidas, msg_fotos = fotos_de_referencia(args.fotos_de, medidas_sorteo)

    # --- librerías (después de medir la RAM de arranque) ---
    sys.path.insert(0, str(args.repo.resolve()))
    import numpy as np
    import cv2
    import paddle

    from ppdet.core.workspace import create, load_config, merge_config
    from ppdet.data import transform as T
    from ppdet.utils.checkpoint import load_pretrain_weight, match_state_dict

    paddle.set_device("cpu")
    hilos_antes = hilos_por_libreria()
    paddle.base.core.set_num_threads(args.hilos)
    rss_imports = rss_mb()

    # --- modelo ---
    t0 = time.perf_counter()
    cfg = load_config(str(config))
    merge_config({"num_classes": args.clases})
    if cfg.architecture != "RTDETRV3":
        sys.exit(f"ERROR: la configuración arma {cfg.architecture}, no RTDETRV3")
    modelo = create(cfg.architecture)
    modelo.eval()
    sd_modelo = modelo.state_dict()
    sd_archivo = paddle.load(str(args.pesos))
    emparejados = match_state_dict(sd_modelo, sd_archivo)
    sin_cargar = [k for k in sd_modelo if k not in emparejados]
    sobran = len(sd_archivo) - len(emparejados)
    if sin_cargar or sobran:
        sys.exit(f"ERROR: los pesos no corresponden a esta red: {len(sin_cargar)} pesos del modelo sin cargar "
                 f"(p. ej. {sin_cargar[:3]}) y {sobran} del archivo sin usar. ¿Otra configuración u otro --clases?")
    load_pretrain_weight(modelo, str(args.pesos.with_suffix("")))
    del sd_archivo, emparejados
    n_param = sum(int(np.prod(p.shape)) for p in modelo.parameters() if not p.stop_gradient)
    reloj_post = RelojPosproceso(modelo.post_process)
    modelo.post_process = reloj_post
    t_carga = time.perf_counter() - t0
    rss_modelo = rss_mb()

    nombres_ops = [list(t)[0] for t in cfg["TestReader"]["sample_transforms"]]
    if nombres_ops != TRANSFORMS_ESPERADAS:
        sys.exit(f"ERROR: el TestReader de la configuración tiene {nombres_ops}, se esperaba {TRANSFORMS_ESPERADAS}")
    ops = [getattr(T, k)(**v) for t in cfg["TestReader"]["sample_transforms"] for k, v in t.items()]
    decodificar, resto_ops = ops[0], ops[1:]

    def procesar(ruta):
        reloj_post.segundos = 0.0
        t_ini = time.perf_counter()
        muestra = decodificar({"im_file": str(ruta)})
        t_lec = time.perf_counter()
        for op in resto_ops:
            muestra = op(muestra)
        entrada = {
            "image": paddle.to_tensor(muestra["image"][None]),
            "im_shape": paddle.to_tensor(muestra["im_shape"][None]),
            "scale_factor": paddle.to_tensor(muestra["scale_factor"][None]),
        }
        t_pre = time.perf_counter()
        with paddle.no_grad():
            salida = modelo(entrada)
        t_mod = time.perf_counter()
        cajas = salida["bbox"].numpy()
        detecciones = int((cajas[:, 1] >= args.conf).sum())
        conf_max = float(cajas[:, 1].max()) if len(cajas) else 0.0
        t_fin = time.perf_counter()
        post = reloj_post.segundos + (t_fin - t_mod)
        red = (t_mod - t_pre) - reloj_post.segundos
        partes = (t_lec - t_ini, t_pre - t_lec, red, post)
        return {
            "ancho": int(muestra["w"]), "alto": int(muestra["h"]),
            "total_ms": round((t_fin - t_ini) * 1000, 2),
            "lectura_ms": round(partes[0] * 1000, 2),
            "preproceso_ms": round(partes[1] * 1000, 2),
            "red_ms": round(partes[2] * 1000, 2),
            "posproceso_ms": round(partes[3] * 1000, 2),
            "resto_ms": round(((t_fin - t_ini) - sum(partes)) * 1000, 2),
            "detecciones": detecciones,
            "conf_max": round(conf_max, 4),
            "n_cajas_red": int(len(cajas)),
        }

    print(f"Calentando {run_id} con {len(calentamiento)} fotos...", flush=True)
    t0 = time.perf_counter()
    procesar(calentamiento[0][1])
    t_primera = time.perf_counter() - t0
    for _, ruta in calentamiento[1:]:
        procesar(ruta)
    rss_max = rss_mb()

    print(f"Midiendo {len(medidas)} fotos...", flush=True)
    filas = []
    for i, (fuente, ruta) in enumerate(medidas, start=1):
        r = procesar(ruta)
        rss = rss_mb()
        rss_max = max(rss_max, rss)
        filas.append({"orden": i, "fuente": fuente, "archivo": ruta.name, **r, "rss_mb": round(rss, 1)})
        if i % 25 == 0:
            print(f"  {i}/{len(medidas)}", flush=True)

    mi = PROC.memory_info()
    pico_mb = getattr(mi, "peak_wset", None)
    pico_mb = pico_mb / 2**20 if pico_mb is not None else None

    with open(csv_salida, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(filas[0]))
        w.writeheader()
        w.writerows(filas)

    commit = commit_del_repo(args.repo)
    parche = "ppdet/modeling/transformers/rtdetr_transformerv3.py"
    parche_txt = (args.repo / parche).read_text(encoding="utf-8", errors="replace")
    con_parche = "Parche del proyecto oso andino v2" in parche_txt
    n_fuente = {f: sum(1 for x in filas if x["fuente"] == f) for f in LISTAS}

    print()
    print("=" * 78)
    print(f"Velocidad en CPU: {args.pesos.name}")
    print(f"CPU: {nombre_cpu()} | {psutil.cpu_count(logical=False)} núcleos físicos, {psutil.cpu_count()} lógicos")
    print(f"Python {platform.python_version()} | paddle {paddle.__version__} | numpy {np.__version__} | "
          f"opencv {cv2.__version__}")
    print(f"Repositorio: {args.repo} | commit {commit}"
          f"{'' if commit == COMMIT else '  <-- ATENCIÓN: no es el commit fijado ' + COMMIT[:7]}"
          f" | parche D24: {'aplicado' if con_parche else 'no aplicado (no afecta a la inferencia)'}")
    print(f"Red: RTDETRV3, {args.clases} clase(s), {n_param / 1e6:.3f} M parámetros entrenables; pesos completos, "
          f"sin sobrantes")
    print(f"Flags: FLAGS_use_mkldnn = {paddle.get_flags(['FLAGS_use_mkldnn'])['FLAGS_use_mkldnn']} | "
          f"hilos pedidos: {args.hilos}")
    print("Hilos por librería, antes de fijarlos: " + "; ".join(hilos_antes))
    print("Hilos por librería, durante la medición: " + "; ".join(hilos_por_libreria()))
    print(f"Preproceso: {' > '.join(nombres_ops)} (640 x 640 sin conservar la proporción) | conf {args.conf}")
    print(f"Carga del modelo: {t_carga:.2f} s | primera foto: {t_primera:.2f} s")
    print(f"Fotos: {msg_fotos}")
    print(f"Muestra: {len(filas)} fotos ({n_fuente['ENA24']} ENA24, {n_fuente['iNaturalist']} iNaturalist), "
          f"tras {len(calentamiento)} de calentamiento")
    print()
    print(f"{'Tiempo por foto (ms)':<34}{'mediana':>9}{'media':>9}{'p95':>9}{'máx':>9}")
    filas_tabla = [
        ("Total, ENA24", "total_ms", "ENA24"),
        ("Total, iNaturalist", "total_ms", "iNaturalist"),
        ("Total, todas", "total_ms", None),
        ("  lectura del JPEG", "lectura_ms", None),
        ("  preproceso", "preproceso_ms", None),
        ("  red (sin posproceso)", "red_ms", None),
        ("  posproceso", "posproceso_ms", None),
        ("  resto", "resto_ms", None),
    ]
    resumen = {}
    for etiqueta, clave, fuente in filas_tabla:
        valores = [x[clave] for x in filas if fuente is None or x["fuente"] == fuente]
        med, media, p95, mx = estadisticas(valores)
        resumen[(clave, fuente)] = (med, media, p95, mx)
        print(f"{etiqueta:<34}{med:>9.1f}{media:>9.1f}{p95:>9.1f}{mx:>9.1f}")
    print()
    con_det = sum(1 for x in filas if x["detecciones"] > 0)
    print(f"Fotos con al menos una detección (conf >= {args.conf}): {con_det} de {len(filas)}")
    print(f"RAM (RSS, MB): {rss_arranque:.0f} al arrancar, {rss_imports:.0f} tras importar, "
          f"{rss_modelo:.0f} con el modelo cargado, {rss_max:.0f} máximo durante la medición")
    pico_txt = f"{pico_mb:.0f} MB" if pico_mb is not None else "no disponible en este sistema"
    print(f"Pico de memoria del proceso (Windows): {pico_txt}")
    p95_total = resumen[("total_ms", None)][2]
    ram_ref = pico_mb if pico_mb is not None else rss_max
    print(f"Meta < 1 s por foto (p95 del total): {'CUMPLE' if p95_total < META_MS else 'NO CUMPLE'}")
    print(f"Meta < 4 GB de RAM (pico): {'CUMPLE' if ram_ref < META_RAM_MB else 'NO CUMPLE'}")
    print(f"Control: {comparar_con_evaluacion(args.comparar, run_id, filas)}")
    refs = referencias_yolo(args.salida, csv_salida)
    if refs:
        print("Otras mediciones en la misma carpeta (mismas fotos):")
        print("\n".join(refs))
    print(f"Detalle por foto: {csv_salida}")


if __name__ == "__main__":
    main()
