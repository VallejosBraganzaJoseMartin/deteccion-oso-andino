"""Lanza tools/train.py del repositorio oficial de RT-DETRv3 fijando antes la semilla.
El repositorio solo permite la semilla 0 (opción --enable_ce); esto llama a la misma función
(ppdet.engine.set_random_seed) con la semilla que se elija, sin modificar el código oficial.
Uso: python entrenar_con_semilla.py RUTA_REPO SEMILLA [argumentos de tools/train.py ...]"""
import runpy
import sys

repo, semilla, resto = sys.argv[1], int(sys.argv[2]), sys.argv[3:]
if "--enable_ce" in resto:
    raise SystemExit("No uses --enable_ce aquí: volvería a fijar la semilla en 0.")
sys.path.insert(0, repo)
from ppdet.engine import set_random_seed
set_random_seed(semilla)
print(f"Semilla fijada en {semilla} (paddle, random y numpy)", flush=True)
sys.argv = [f"{repo}/tools/train.py"] + resto
runpy.run_path(f"{repo}/tools/train.py", run_name="__main__")
