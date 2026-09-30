# Registro de experimentos. Fase de modelado, iteración 1 (dataset proxy)

Este documento tiene dos partes. La primera es el diseño de pruebas, escrito el 16 de septiembre de 2026 antes de entrenar nada, y no se modifica después salvo para añadir bloques de revisión con fecha. La segunda es el registro de corridas, una entrada por entrenamiento, que se va llenando.

---

## Parte 1. Diseño de pruebas

### 1.1 Datos

- Dataset proxy v1: 3832 imágenes, 2207 cajas, una clase (`oso`). División registrada imagen por imagen en `docs/preparacion_datos/division.csv` (semilla 42, agrupamiento por Hamming con umbral 12, decisión D13).
- Reparto: entrenamiento 2656, validación 598 (338 ENA24 + 260 iNaturalist), prueba 578 (318 ENA24 + 260 iNaturalist).
- Particiones por fuente para la evaluación separada (D15): listas de rutas en la raíz de `data/processed` (`val_inat.txt`, `val_ena24.txt`, `test_inat.txt`, `test_ena24.txt`) y un yaml por lista en `data/processed/particiones/`. Generadas con `scripts/particiones_por_fuente.py`. Verificado el 15 de septiembre que Ultralytics 8.4.138 lee las listas y encuentra las etiquetas (validación de prueba sobre `test_inat`: 260 imágenes, 190 cajas, cero descartes).
- El 15 de septiembre se reescribieron como JPEG 35 imágenes de iNaturalist que tenían extensión `.jpg` pero estaban codificadas como GIF animado (14) o PNG (21). Ultralytics descarta los GIF al escanear el dataset, sin detener el entrenamiento, así que sin la corrección habría entrenado con 14 imágenes menos de las declaradas. Inventario en `docs/preparacion_datos/formatos_no_jpeg.txt`. Las cifras del dataset no cambian.
- Archivo comprimido para Kaggle: `data/dataset_proxy_v1.zip`, 1.59 GB, 7682 entradas, con `images/`, `labels/`, `dataset.yaml`, `particiones/` y las cuatro listas en la raíz.

### 1.2 Entorno

- Entrenamiento: Kaggle, GPU del plan gratuito. GPU concreta, versión de Python y de PyTorch: anotar en la primera corrida.
- Ultralytics fijado en **8.4.138**, la versión instalada en el equipo local (`pip install ultralytics==8.4.138` en el cuaderno). Motivo: los valores por defecto de los argumentos cambian entre versiones y todo lo que sigue se apoya en ellos.
- Latencia en CPU: se mide en el equipo local (AMD Ryzen 5 3600, sin GPU, Python 3.14.2, torch 2.12.1+cpu), no en Kaggle, porque el escenario de uso es un computador sin GPU.

> **Revisión 2026-09-16, tras E0.** Entorno real de Kaggle: Tesla T4 (el cuaderno ofrece dos, se usa una con `device=0`), Python 3.12.13, torch 2.10.0+cu128, Ultralytics 8.4.138. La ruta del dataset en el cuaderno es `/kaggle/input/datasets/josemartinvallejos/deteccionoso`. `/kaggle/input` es de solo lectura, así que Ultralytics no puede guardar la caché de etiquetas y lo avisa en cada corrida; no afecta. Kaggle trae instalado el paquete `albumentations`, ver revisión de 1.4.

### 1.3 Protocolo común a todas las corridas de YOLO26

Valores tomados de `ultralytics/cfg/default.yaml` de la versión 8.4.138, comprobados el 16 de septiembre. Solo se listan los que condicionan la comparación. Todo lo que no aparece aquí queda en su valor por defecto.

| Argumento | Valor | Comentario |
|---|---|---|
| `epochs` | 100 | Mismo número para todos los modelos, como pide el protocolo de comparación justa |
| `patience` | 100 | Igual a `epochs`, así no hay parada temprana y todas las corridas entrenan lo mismo |
| `imgsz` | 640 | Resolución base del protocolo. El experimento a 800 o 960 px es opcional y se registra aparte |
| `batch` | 16 | Por defecto. Si la GPU de Kaggle no lo soporta con el modelo m, se anota el valor usado |
| `seed` / `deterministic` | 0 / True | Reproducibilidad |
| `pretrained` | pesos COCO (`yolo26n.pt`, `yolo26s.pt`, `yolo26m.pt`) | Transfer learning desde COCO y no desde ImageNet como dice el anteproyecto |
| `optimizer` / `lr0` / `lrf` | auto / 0.01 / 0.01 | Por defecto. Con `auto`, Ultralytics elige el optimizador según el número de iteraciones. Anotar cuál eligió en cada corrida, lo imprime al inicio |
| `warmup_epochs` | 3.0 | Por defecto |
| `close_mosaic` | 10 | Mosaic se apaga en las últimas 10 épocas |
| `amp` | True | Precisión mixta |

> **Revisión 2026-09-16, tras E0.** Con `optimizer=auto` Ultralytics ignora `lr0` y `momentum` y elige el optimizador según las iteraciones nominales, `ceil(n_train / nbs) × epochs` con `nbs=64` (regla en `ultralytics/engine/trainer.py`, versión 8.4.138): AdamW con `lr=0.002 × 5 / (4 + nc)` si son 10000 o menos, MuSGD con lr 0.01 si son más. Para este dataset son 42 × 100 = 4200, así que **todas las corridas de 100 épocas usan AdamW con lr 0.002 y momentum 0.9**, lo mismo que E0. Se añade `cache="ram"` al protocolo: guarda las imágenes reescaladas en memoria (2.2 GB para train) y no altera el resultado, aunque Ultralytics avisa de que puede hacer el entrenamiento no determinista. En lugar de perseguir un determinismo que en GPU no está garantizado ni con `deterministic=True`, se mide el ruido entre corridas con una repetición de la línea base con otra semilla (E1b). Ultralytics usa 2 procesos de carga de datos en la configuración T4 x2 de Kaggle (4 CPU repartidas entre 2 GPU), lo que probablemente limita la velocidad del modelo nano; queda anotado por si conviene probar la configuración P100 (1 GPU, 4 procesos) para los tamaños mayores.


### 1.4 Aumento de datos

El anteproyecto compromete rotaciones, ajustes de brillo y saturación y Mosaic. Correspondencia con los argumentos de Ultralytics 8.4.138 y valores usados:

| Técnica | Argumento | Por defecto | En este trabajo | Comentario |
|---|---|---|---|---|
| Brillo | `hsv_v` | 0.4 | 0.4 | Variación de hasta ±40 % en el canal de valor |
| Saturación | `hsv_s` | 0.7 | 0.7 | Hasta ±70 % |
| Tono | `hsv_h` | 0.015 | 0.015 | No lo nombra el anteproyecto, viene activo |
| Mosaic | `mosaic` | 1.0 | 1.0 | Cuatro imágenes en una, apagado en las últimas 10 épocas |
| Rotación | `degrees` | 0.0 | **0 frente a 10, ablación** | Ver abajo |
| Volteo horizontal | `fliplr` | 0.5 | 0.5 | No lo nombra el anteproyecto, viene activo |
| Escala | `scale` | 0.5 | 0.5 | De 0.5× a 1.5× |
| Traslación | `translate` | 0.1 | 0.1 | |
| Volteo vertical, cizalla, perspectiva, MixUp, CutMix | `flipud`, `shear`, `perspective`, `mixup`, `cutmix` | 0 | 0 | Un oso no aparece boca abajo. MixUp y CutMix no están prometidos y con una sola clase aportan poco |

Sobre la rotación. Ultralytics implementa la rotación en `RandomPerspective.apply_bboxes` (`ultralytics/data/augment.py`): rota las cuatro esquinas de cada caja y toma como nueva etiqueta la envolvente alineada a los ejes, que es siempre más grande que el animal. Para una caja apaisada 3:2 el área crece ×1.19 a 5°, ×1.37 a 10° y ×1.54 a 15°. Es ruido de etiqueta sistemático y penaliza justamente mAP50-95, el criterio principal del trabajo. Por eso la librería la trae en cero. A favor de una rotación pequeña está que las cámaras trampa se sujetan a troncos y no siempre quedan niveladas, aunque no tengo una fuente que cuantifique esa inclinación. La decisión (registro de decisiones, 16 de septiembre) es no fijarla a ciegas: se entrena YOLO26n dos veces, con `degrees=0` y con `degrees=10`, todo lo demás igual, y se compara sobre validación global y por fuente. Si 10° mejora, se adopta para s y m. Si no, se descarta y queda como resultado reportado.

El sobremuestreo de la clase rara que menciona el marco teórico (1.3.3) no se aplica: hay una sola clase y el 53 % de las imágenes son positivas.

> **Revisión 2026-09-16, tras E0.** En Kaggle está instalado `albumentations`, y cuando Ultralytics lo detecta añade cuatro transformaciones al pipeline de entrenamiento: `Blur`, `MedianBlur`, `ToGray` y `CLAHE`, cada una con probabilidad 0.01 (lo imprime al inicio como `albumentations: ...`). En el equipo local no ocurre porque el paquete no está instalado. Se conservan, porque su efecto es marginal y quitarlas exigiría desinstalar el paquete en cada sesión, pero forman parte del protocolo y se declaran. Versión de `albumentations` en Kaggle: 2.0.8.

### 1.5 Corridas planificadas

| ID | Modelo | `degrees` | Épocas | Propósito | Estado |
|---|---|---|---|---|---|
| E0 | YOLO26n | 0 | 10 a 20 | Verificar el pipeline de punta a punta en Kaggle: lectura de etiquetas, entrenamiento, validación, guardado de pesos. Las métricas no cuentan. Da el tiempo real por época para planificar la cuota | Hecho |
| E1 | YOLO26n | 0 | 100 | Línea base, candidato de despliegue en CPU | Hecho |
| E1b | YOLO26n | 0 | 100 | Repetición de E1 con `seed=1` para medir el ruido entre corridas; da la escala contra la que se lee la ablación | Hecho |
| E2 | YOLO26n | 10 | 100 | Ablación de rotación contra E1 | Hecho |
| E3 | YOLO26s | según E1/E2 | 100 | Punto intermedio velocidad/precisión | Hecho |
| E4 | YOLO26m | según E1/E2 | 100 | Emparejado en parámetros con RT-DETRv3-R18 (~20 M) para la comparación justa | Hecho |
| E3b | YOLO26s | 0 | 100 | Repetición de E3 con `seed=1` | Hecho |
| E4b | YOLO26m | 0 | 100 | Repetición de E4 con `seed=1` | Hecho |
| E5 | mejor de E1..E4 | según E1/E2 | 100 | Opcional: `imgsz` 800 o 960 por los osos pequeños de ENA24 | Solo si sobran horas |

Orden de ejecución: E0, E1, E2, E3, E4, E5. Estimación previa de GPU (sin fuente, se corrige con E0): n alrededor de 1 a 1.5 h por corrida, s de 2 a 3 h, m de 4 a 6 h.

RT-DETRv3 tiene su propio apartado en el plan (2.3, límite de dos días, D9) y se registra aquí como E6 en adelante cuando llegue.

> **Revisión 2026-09-16, tras E0.** Tiempo medido: YOLO26n tarda 31.7 s por época en T4 con validación incluida, unos 55 minutos por corrida de 100 épocas. Para s y m sigue sin haber medida; estimación corregida, 1.5 y 3 horas. Orden de ejecución: E1, E2 y E1b pueden correr en paralelo; E3 y E4 después, con el `degrees` que resulte.

### 1.6 Evaluación

- Selección de modelo y de umbrales: solo sobre validación. El conjunto de prueba se usa una vez por modelo final, al terminar. Ninguna decisión de configuración se toma mirando prueba.
- Cada modelo se valida tres veces sobre prueba (global con `dataset.yaml`, ENA24 con `dataset_test_ena24.yaml`, iNaturalist con `dataset_test_inat.yaml`) y las mismas tres sobre validación. Métricas de Ultralytics con los umbrales por defecto de `val`, confirmados en la salida de E0: `conf=0.001` implícito para el cálculo de mAP e `iou=0.7`.
- Recall a nivel de imagen (D15): proporción de imágenes con oso en las que el modelo pone al menos una detección por encima del umbral de operación, y proporción de negativos con alguna detección (falsas alarmas). El umbral de operación se elige sobre validación. Requiere un script propio a partir de las predicciones, pendiente de escribir.
- Además: número de parámetros, tiempo de inferencia por imagen en CPU local, y ruta de los pesos.
- Las métricas de la iteración 1 son preliminares (D1). El resultado principal es el de ENA24 (D15). Limitación declarada: la misma cámara de ENA24 aparece en entrenamiento y prueba con el mismo fondo, así que no se mide generalización a cámaras nuevas (D13).

### 1.7 Qué se anota por corrida

Fecha, ID, modelo y pesos de partida, argumentos que difieren del protocolo, versión de Ultralytics y GPU, optimizador elegido por `auto`, tiempo total y por época, época del mejor `best.pt` según Ultralytics, métricas sobre validación (global y por fuente), ruta de los pesos y del `results.csv`, y cualquier aviso que imprimiera el entrenamiento. Las métricas sobre prueba se añaden solo al cerrar el modelo.

---

## Parte 2. Registro de corridas

### E0. Verificación del pipeline
 
- Fecha: 2026-09-16.
- Entorno: Kaggle, Tesla T4 (una de dos), Python 3.12.13, torch 2.10.0+cu128, Ultralytics 8.4.138. Cuaderno `notebooks/e0_verificacion_yolo26n.ipynb`.
- Argumentos: `model=yolo26n.pt epochs=10 imgsz=640 batch=16 seed=0 deterministic=True cache=ram device=0`. Resto por defecto.
- Optimizador elegido por `auto`: AdamW, lr 0.002, momentum 0.9.
- Escaneo de etiquetas: train 2656 imágenes, 1274 fondos, 0 corruptas. val 598 imágenes, 272 fondos, 0 corruptas. Particiones: val_ena24 338 imágenes y 162 cajas, val_inat 260 imágenes y 189 cajas, ambas sin descartes.
- Tiempo: 31.7 s por época con validación; 6.9 minutos en total contando descarga de pesos, comprobación de AMP y caché.
- Avisos: caché de etiquetas no escribible en `/kaggle/input` (esperado); `cache='ram'` puede ser no determinista (ver revisión de 1.3); Albumentations activo (ver revisión de 1.4); un `UserWarning` de `ray` sobre `get_trial_id`, ajeno al entrenamiento.
- Resultado: funciona de punta a punta. Métricas sobre validación con `best.pt` (época 10), solo como comprobación del mecanismo, no como resultado:

  | Partición | P | R | mAP50 | mAP50-95 |
  |---|---|---|---|---|
  | Global (598) | 0.872 | 0.835 | 0.910 | 0.694 |
  | ENA24 (338) | 0.884 | 0.891 | 0.938 | 0.716 |
  | iNaturalist (260) | 0.863 | 0.798 | 0.886 | 0.681 |
 
- Observación: ENA24 sale por encima de iNaturalist, al contrario de lo que se anticipó en D15. Hipótesis a revisar con los modelos completos: el fondo fijo compartido entre train y val por la misma cámara de ENA24 (limitación declarada en D13).
- Pesos: `e0_yolo26n_best.pt` descargado del cuaderno, no se conserva en el repositorio. `results.csv` guardado como `docs/modelado/e0_results.csv` (por decidir si se versiona).

### E1. YOLO26n, línea base (`degrees=0`, `seed=0`)
 
- Fecha: 2026-09-17. Cuaderno `notebooks/entrenar_yolo26.ipynb`, celda 0 con `RUN_ID=e1_n_deg0_s0`.
- Entorno: Kaggle, Tesla T4, Ultralytics 8.4.138, Albumentations activo (Blur, MedianBlur, ToGray, CLAHE, p=0.01). Optimizador elegido por `auto`: AdamW, lr 0.002, momentum 0.9.
- Argumentos: protocolo 1.3 con `degrees=0`, `seed=0`, `cache=ram`. 100 épocas completas.
- Tiempo: 30.0 s por época, 50.1 min de entrenamiento, 52.3 min la celda.
- Mejor época: 99 (mAP50-95 val 0.7586). Última época: 0.7565.
- Validación con `best.pt`:
| Partición | P | R | mAP50 | mAP50-95 |
|---|---|---|---|---|
| Global (598) | 0.923 | 0.903 | 0.932 | 0.759 |
| ENA24 (338) | 0.939 | 0.951 | 0.955 | 0.786 |
| iNaturalist (260) | 0.916 | 0.866 | 0.908 | 0.732 |
 
- Curvas (`docs/modelado/curvas_nano_ablacion.png`): la pérdida de caja en validación baja hasta el final (mínimo en la época 99) y mAP50-95 sigue subiendo en las últimas 20 épocas (+0.006). La pérdida de clase en validación toca mínimo en la época 88 y sube levemente después (+0.015 en las últimas 20), sin que mAP se resienta. No hay señal de sobreajuste en la localización; hay una señal leve en la clasificación al final, absorbida por la selección de `best.pt`.
- Archivos: `docs/modelado/e1_n_deg0_s0_results.csv`, `e1_n_deg0_s0_args.yaml`. Pesos `e1_n_deg0_s0_best.pt` (5.4 MB) fuera del repositorio.
### E1b. YOLO26n, repetición de la línea base (`degrees=0`, `seed=1`)
 
- Fecha: 2026-09-17. `RUN_ID=e1b_n_deg0_s1`. Mismo entorno, optimizador y argumentos que E1 salvo `seed=1`.
- Tiempo: 29.2 s por época, 48.7 min de entrenamiento.
- Mejor época: 95 (mAP50-95 val 0.7565). Última época: 0.7498.
- Validación con `best.pt`:

| Partición | P | R | mAP50 | mAP50-95 |
|---|---|---|---|---|
| Global (598) | 0.912 | 0.890 | 0.929 | 0.758 |
| ENA24 (338) | 0.909 | 0.923 | 0.933 | 0.762 |
| iNaturalist (260) | 0.945 | 0.852 | 0.922 | 0.758 |
 
- Curvas: prácticamente superpuestas a las de E1 en pérdida de caja y mAP. La pérdida de clase en validación sube algo más al final (+0.06 en las últimas 20 épocas, mínimo en la 79).
- **Ruido entre corridas (E1 contra E1b)**: 0.001 en mAP50-95 global y 0.003 en mAP50 global. Por fuente, 0.024 en ENA24 y 0.026 en iNaturalist, en direcciones opuestas. Con dos corridas es una estimación gruesa, pero fija una regla de lectura: en las particiones por fuente (260 a 338 imágenes) las diferencias por debajo de 0.03 de mAP50-95 no se interpretan como efecto; en la validación global sí se puede resolver menos.
- Archivos: `docs/modelado/e1b_n_deg0_s1_results.csv`, `e1b_n_deg0_s1_args.yaml`.
### E2. YOLO26n, rotación (`degrees=10`, `seed=0`)
 
- Fecha: 2026-09-17. `RUN_ID=e2_n_deg10_s0`. Mismo entorno, optimizador y argumentos que E1 salvo `degrees=10`.
- Tiempo: 29.5 s por época, 49.1 min de entrenamiento.
- Mejor época: 70 (mAP50-95 val 0.7394). Última época: 0.7317.
- Validación con `best.pt`:

| Partición | P | R | mAP50 | mAP50-95 |
|---|---|---|---|---|
| Global (598) | 0.945 | 0.874 | 0.932 | 0.738 |
| ENA24 (338) | 0.926 | 0.924 | 0.955 | 0.753 |
| iNaturalist (260) | 0.946 | 0.841 | 0.916 | 0.727 |
 
- Curvas: la pérdida de caja, tanto en entrenamiento como en validación, va por encima de las líneas base durante las 100 épocas y no las alcanza. mAP50-95 se estanca desde la época 70 y baja levemente en las últimas 20 (−0.004).
- Archivos: `docs/modelado/e2_n_deg10_s0_results.csv`, `e2_n_deg10_s0_args.yaml`.
### Resultado de la ablación de rotación y decisión
 
| | mAP50 global | mAP50-95 global | mAP50-95 ENA24 | mAP50-95 iNat | R global |
|---|---|---|---|---|---|
| E1 (0°, s0) | 0.932 | 0.759 | 0.786 | 0.732 | 0.903 |
| E1b (0°, s1) | 0.929 | 0.758 | 0.762 | 0.758 | 0.890 |
| E2 (10°, s0) | 0.932 | 0.738 | 0.753 | 0.727 | 0.874 |
| E2 menos E1 | 0.000 | −0.020 | −0.032 | −0.006 | −0.029 |
| E2 menos E1b | +0.003 | −0.020 | −0.008 | −0.032 | −0.016 |
 
Con rotación de ±10° el modelo detecta igual (mAP50 idéntico) pero ajusta peor la caja (mAP50-95 0.020 por debajo de las dos líneas base, veinte veces el ruido entre semillas en global) y pierde recall. La mejor época llega en la 70 en lugar de al final, y la pérdida de caja nunca alcanza la de las líneas base. Es el patrón que predice el mecanismo descrito en 1.4: las etiquetas infladas por la envolvente de la caja rotada enseñan cajas holgadas. Con una corrida por condición no es una prueba estadística; es un resultado consistente en las tres particiones y explicable.
 
**Decisión (2026-09-17)**: `degrees=0` para E3 y E4. Se transfiere la decisión a los tamaños s y m bajo el supuesto, declarado, de que el efecto de la rotación no depende del tamaño del modelo. Si al leer E3 y E4 hiciera falta, la ablación se repetiría solo en medium.
 
Nota sobre el protocolo de 100 épocas sin parada temprana, a raíz de las curvas: en las dos líneas base la mejor época cae en la 95 y la 99 y la validación no se degrada, así que 100 épocas no sobreajustan a nano; si acaso, se queda algo corto. No se cambia el protocolo en esta iteración, porque el número de épocas es lo que hace comparables las corridas. Queda anotado como posible experimento (150 épocas en nano) si sobran horas.

### E3. YOLO26s (`degrees=0`, `seed=0`)
 
- Fecha: 2026-09-18. `RUN_ID=e3_s_deg0_s0`. Kaggle, Tesla T4, Ultralytics 8.4.138, Albumentations activo. Optimizador elegido por `auto`: AdamW, lr 0.002, momentum 0.9.
- Parámetros: 9,465,567 (modelo fusionado, 20.8 GFLOPs).
- Tiempo: 52.2 s por época, 87.0 min de entrenamiento, 88.9 min la celda.
- Mejor época: 91 (mAP50-95 val 0.7464 durante el entrenamiento). Última época: 0.7370.
- Validación con `best.pt` (celda 5):


| Partición | P | R | mAP50 | mAP50-95 |
|---|---|---|---|---|
| Global (598) | 0.939 | 0.866 | 0.935 | 0.743 |
| ENA24 (338) | 0.935 | 0.887 | 0.946 | 0.754 |
| iNaturalist (260) | 0.947 | 0.851 | 0.920 | 0.738 |
 
- Curvas (`docs/modelado/curvas_tamanos.png`): pérdida de caja por encima de nano durante todo el entrenamiento, en entrenamiento y en validación. mAP50-95 se aplana desde la época 90 y la pérdida de caja en validación sube levemente al final (+0.004 en las últimas 20 épocas); la pérdida de clase en validación baja. Sin señal clara de sobreajuste.
- Archivos: `docs/modelado/e3_s_deg0_s0_results.csv`, `e3_s_deg0_s0_args.yaml`. Pesos `e3_s_deg0_s0_best.pt` (20.3 MB) fuera del repositorio.
### E4. YOLO26m (`degrees=0`, `seed=0`)
 
- Fecha: 2026-09-18. `RUN_ID=e4_m_deg0_s0`. Mismo entorno y optimizador que E3.
- Parámetros: 20,350,223 (modelo fusionado, 68.1 GFLOPs).
- Tiempo: 99.7 s por época, 166.2 min de entrenamiento, 167.7 min la celda.
- Mejor época: 100 (mAP50-95 val 0.7520). Última época: la misma.
- Validación con `best.pt` (celda 5):

| Partición | P | R | mAP50 | mAP50-95 |
|---|---|---|---|---|
| Global (598) | 0.965 | 0.862 | 0.933 | 0.752 |
| ENA24 (338) | 0.967 | 0.913 | 0.957 | 0.767 |
| iNaturalist (260) | 0.941 | 0.846 | 0.912 | 0.746 |
 
- Curvas: la pérdida de caja más alta de los tres tamaños durante todo el entrenamiento. En las últimas 20 épocas mAP50-95 sigue subiendo (+0.008) y las dos pérdidas de validación siguen bajando, y la mejor época es la última. El modelo no había terminado de mejorar a las 100 épocas.
- Archivos: `docs/modelado/e4_m_deg0_s0_results.csv`, `e4_m_deg0_s0_args.yaml`. Pesos `e4_m_deg0_s0_best.pt` (44.0 MB) fuera del repositorio.

### E3b. YOLO26s, repetición con `seed=1`
 
- Fecha: 2026-09-18. `RUN_ID=e3b_s_deg0_s1`. Mismo entorno, optimizador (AdamW 0.002) y argumentos que E3 salvo `seed=1`.
- Tiempo: 47.0 s por época, 78.3 min de entrenamiento.
- Mejor época: 97 (0.7566). Última época: 0.7556.
- Validación con `best.pt` (celda 5):

| Partición | P | R | mAP50 | mAP50-95 |
|---|---|---|---|---|
| Global (598) | 0.948 | 0.906 | 0.941 | 0.758 |
| ENA24 (338) | 0.962 | 0.944 | 0.957 | 0.769 |
| iNaturalist (260) | 0.935 | 0.873 | 0.930 | 0.754 |
 
- Archivos: `docs/modelado/e3b_s_deg0_s1_results.csv`, `e3b_s_deg0_s1_args.yaml`.
### E4b. YOLO26m, repetición con `seed=1`
 
- Fecha: 2026-09-18. `RUN_ID=e4b_m_deg0_s1`. Mismo entorno, optimizador y argumentos que E4 salvo `seed=1`.
- Tiempo: 89.0 s por época, 148.3 min de entrenamiento.
- Mejor época: 100 (0.7560), la última, igual que en E4. Las dos pérdidas de validación siguen bajando en las últimas 20 épocas.
- Validación con `best.pt` (celda 5):

| Partición | P | R | mAP50 | mAP50-95 |
|---|---|---|---|---|
| Global (598) | 0.939 | 0.882 | 0.936 | 0.755 |
| ENA24 (338) | 0.945 | 0.926 | 0.959 | 0.765 |
| iNaturalist (260) | 0.937 | 0.847 | 0.917 | 0.755 |
 
- Archivos: `docs/modelado/e4b_m_deg0_s1_results.csv`, `e4b_m_deg0_s1_args.yaml`.

### Comparación de tamaños (validación, protocolo común de 100 épocas)
 
| Modelo | Corrida | s/época (T4) | Pesos | mAP50 global | mAP50-95 global | mAP50-95 ENA24 | mAP50-95 iNat |
|---|---|---|---|---|---|---|---|
| YOLO26n | E1 (semilla 0) | 30.0 | 5.4 MB | 0.932 | 0.759 | 0.786 | 0.732 |
| YOLO26n | E1b (semilla 1) | 29.2 | 5.4 MB | 0.929 | 0.758 | 0.762 | 0.758 |
| YOLO26s | E3 | 52.2 | 20.3 MB | 0.935 | 0.743 | 0.754 | 0.738 |
| YOLO26m | E4 | 99.7 | 44.0 MB | 0.933 | 0.752 | 0.767 | 0.746 |
 
### Comparación de tamaños (validación, protocolo común de 100 épocas, dos semillas por tamaño)
 
| Modelo | Semilla 0 | Semilla 1 | Media | Diferencia entre semillas | Parámetros (fusionado) | GFLOPs | s/época (T4) | Pesos |
|---|---|---|---|---|---|---|---|---|
| YOLO26n (E1, E1b) | 0.759 | 0.758 | 0.758 | 0.001 | 2,375,031 | 5.3 | 29 a 30 | 5.4 MB |
| YOLO26s (E3, E3b) | 0.743 | 0.758 | 0.751 | 0.014 | 9,465,567 | 20.8 | 47 a 52 | 20.3 MB |
| YOLO26m (E4, E4b) | 0.752 | 0.755 | 0.753 | 0.003 | 20,350,223 | 68.1 | 89 a 100 | 44.0 MB |
 
Cifras: mAP50-95 global sobre validación. mAP50 queda entre 0.929 y 0.941 en las seis corridas.
 
Lectura:
 
- Los tres tamaños se solapan. La diferencia entre semillas de small (0.014) es casi igual a la distancia entre small y nano que se había señalado con una sola corrida (0.015), así que esa distancia era ruido. La mejor corrida de cada tamaño queda en 0.758 o 0.759 para nano y small, y en 0.755 para medium.
- Conclusión con el protocolo común: aumentar el tamaño del modelo no mejora la detección en este dataset. No se puede afirmar que nano sea mejor; se puede afirmar que no es peor, y cuesta un tercio del tiempo de small y un octavo del peso de medium.
- Medium tiene su mejor época en la 100 en las dos semillas y sus pérdidas de validación siguen bajando. No había convergido. Se declara como limitación del protocolo.
- La diferencia entre semillas varía mucho de un tamaño a otro (de 0.001 a 0.014). Dos corridas por tamaño no bastan para estimar bien esa variación; la regla de lectura práctica para esta iteración es que diferencias de mAP50-95 global por debajo de unos 0.015 no se interpretan.
- ENA24 sigue por encima de iNaturalist en las siete corridas.
**Corrección a la lectura de la ablación de rotación (E2).** Tras E1 y E1b se escribió que la caída de E2 (0.020 en mAP50-95 global) era "veinte veces el ruido". Esa cifra se apoyaba en la diferencia entre las dos semillas de nano (0.001), que resultó ser la más pequeña de los tres tamaños. Con la variación observada en small (0.014), la caída de E2 sigue siendo mayor que cualquier diferencia entre semillas vista en la iteración, pero con un margen pequeño. La decisión de usar `degrees=0` se mantiene porque se apoya en tres cosas juntas: esa caída, el hecho de que la pérdida de caja de E2 va por encima de las líneas base durante las 100 épocas, y el mecanismo conocido de las cajas infladas. Se reporta como indicio consistente, no como efecto demostrado.
 
GPU consumida en la iteración hasta aquí: unas 11.5 horas.

### Evaluación sobre validación: umbral de confianza, recall por imagen y falsas alarmas (2026-09-30)
 
- **Herramientas**: `notebooks/evaluar_umbral_yolo26.ipynb` (Kaggle, Tesla T4, Ultralytics 8.4.138) obtiene las detecciones de los seis modelos sobre validación, con confianza de 0.001 en adelante y una imagen por vez. `scripts/calcular_umbral.py` aplica en local la regla del umbral. Archivos en `docs/modelado/`: `umbral_metricas.csv`, `umbral_imagenes.csv`, `umbral_detecciones.csv.gz` y `umbral_opcion_b.csv`.
- **Definiciones**: una detección acierta si su caja se solapa con una caja real con IoU de 0.5 o más. Se emparejan de mayor a menor confianza y cada caja real se usa una sola vez. Recall por imagen: fotos con oso en las que hay al menos una detección por encima del umbral. Falsa alarma: foto sin oso con al menos una detección por encima del umbral. Intervalos de confianza del 95 % por el método de Wilson.
- **Comprobaciones**: los conteos de imágenes, cajas, fuentes y estratos coinciden con `informe_preparacion_datos.md`. Las nocturnas de ENA24 (224) también. El cuaderno encontró además 9 fotos de iNaturalist en blanco y negro, que no se usan en las particiones de día y noche.
- **Control de calidad**: el mAP50 calculado por el cuaderno difiere del de Ultralytics en hasta 0.007 en global y en ENA24, con signo variable. En iNaturalist difiere en hasta 0.015, siempre por debajo. Causa probable, no verificada: Ultralytics valida en lotes de 16 fotos agrupadas por forma y con un borde extra (`pad=0.5` en `ultralytics/data/build.py`), y el cuaderno procesa cada foto sola con el borde mínimo. En iNaturalist, con fotos horizontales y verticales mezcladas, ese borde cambia más. No afecta el recall por imagen ni las falsas alarmas, que no dependen del emparejamiento de cajas.
- **Incidencia**: el primer intento se detuvo por falta de memoria en la GPU, porque las 598 rutas se pasaron a Ultralytics como lista y las procesó en un solo lote. Se corrigió pasando un archivo de texto con las rutas.
**ENA24, regla anterior (A) frente a regla B**
 
| Modelo | Umbral A | Fotos con oso detectadas (A) | Falsas alarmas (A) | Umbral B | Fotos con oso detectadas (B) | Falsas alarmas (B) | P caja (B) | R caja (B) |
|---|---|---|---|---|---|---|---|---|
| E1 nano s0 | 0.496 | 144/156 (0.92) | 3/182 | 0.363 | 149/156 (0.955) | 3/182 | 0.950 | 0.932 |
| E1b nano s1 | 0.611 | 147/156 (0.94) | 5/182 | 0.461 | 149/156 (0.955) | 6/182 | 0.919 | 0.914 |
| E3 small s0 | 0.452 | 145/156 (0.93) | 8/182 | 0.404 | 149/156 (0.955) | 9/182 | 0.893 | 0.926 |
| E3b small s1 | 0.756 | 142/156 (0.91) | 4/182 | 0.431 | 149/156 (0.955) | 4/182 | 0.962 | 0.944 |
| E4 medium s0 | 0.498 | 144/156 (0.92) | 2/182 | 0.256 | 149/156 (0.955) | 4/182 | 0.899 | 0.932 |
| E4b medium s1 | 0.679 | 145/156 (0.93) | 2/182 | 0.501 | 149/156 (0.955) | 2/182 | 0.943 | 0.926 |
 
**Regla B en las demás particiones** (fotos con oso detectadas · falsas alarmas)
 
| Modelo | iNaturalist | Global | ENA24 de día | ENA24 de noche |
|---|---|---|---|---|
| E1 nano s0 | 155/170 (0.91) · 3/90 | 304/326 (0.93) · 6/272 | 47/52 (0.90) · 2/62 | 102/104 (0.98) · 1/120 |
| E1b nano s1 | 149/170 (0.88) · 4/90 | 298/326 (0.91) · 10/272 | 48/52 (0.92) · 4/62 | 101/104 (0.97) · 2/120 |
| E3 small s0 | 158/170 (0.93) · 6/90 | 307/326 (0.94) · 15/272 | 49/52 (0.94) · 6/62 | 100/104 (0.96) · 3/120 |
| E3b small s1 | 153/170 (0.90) · 4/90 | 302/326 (0.93) · 8/272 | 47/52 (0.90) · 3/62 | 102/104 (0.98) · 1/120 |
| E4 medium s0 | 153/170 (0.90) · 4/90 | 302/326 (0.93) · 8/272 | 46/52 (0.88) · 3/62 | 103/104 (0.99) · 1/120 |
| E4b medium s1 | 150/170 (0.88) · 5/90 | 299/326 (0.92) · 7/272 | 49/52 (0.94) · 0/62 | 100/104 (0.96) · 2/120 |
 
**Lectura**
 
- Con la regla B, los seis modelos cumplen en validación todas las metas del apartado 1.1 medidas sobre ENA24: recall por imagen de 0.955, recall de cajas de 0.91 a 0.94, precisión de cajas de 0.89 a 0.96 y falsas alarmas de 1 a 5 % en ENA24 y de 3 a 7 % en iNaturalist. El 0.955 lo fija la regla, así que no es una estimación del rendimiento. La medida honesta sale del conjunto de prueba.
- Pasar de la regla A a la B recuperó entre 2 y 7 fotos con oso por modelo. A cambio, las falsas alarmas en ENA24 subieron como mucho en 2 fotos, y en tres modelos no subieron.
- Queda margen. El umbral más bajo que todavía mantendría las falsas alarmas por debajo del 10 % va de 0.007 (E1) a 0.23 (E4b), muy por debajo de los umbrales elegidos.
- Con el recall por imagen igualado, las falsas alarmas en ENA24 van de 2 a 9 de 182 y sus intervalos se solapan. Ningún tamaño es mejor que otro, lo mismo que se vio con el mAP.
- El umbral cambia mucho entre modelos y entre semillas del mismo tamaño (de 0.26 a 0.50). Cada archivo de pesos lleva su propio umbral.
- iNaturalist es más difícil que ENA24 en los seis modelos, al contrario de lo previsto en D15 (ver la revisión de esa ficha).
- En ENA24 se detecta mejor de noche que de día: de 0.96 a 0.99 frente a 0.88 a 0.94. Hay solo 52 fotos con oso de día, así que la diferencia es indicativa. Una hipótesis sin verificar: en entrenamiento hay el doble de osos nocturnos que diurnos (396 frente a 196).
- Con la regla B, cada modelo deja sin aviso 7 fotos con oso de ENA24. Cuáles son y qué tienen en común queda para el análisis de errores de la fase de evaluación. `umbral_imagenes.csv` tiene la información para identificarlas.

### Modelo de la aplicación (2026-09-30)
 
Se fija sobre validación, antes de tocar el conjunto de prueba: **E1 (YOLO26n, semilla 0) con umbral 0.363** (regla B). Motivos y alternativas en la ficha correspondiente de `registro_decisiones.md` (D19). Queda por medir su velocidad en CPU, que es el primer paso de la siguiente conversación.
 
---