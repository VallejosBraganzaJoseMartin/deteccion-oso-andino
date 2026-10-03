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

> **Revisión 2026-10-02.** Se añade como línea base el modelo COCO sin entrenar (`coco_n_sinentrenar`, pesos `yolo26n.pt`, solo la clase `bear` con `classes=[21]`). No se entrena: se evalúa con el mismo cuaderno y la misma regla de umbral que los demás modelos.

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

> **Revisión 2026-10-01.** E5 (resolución de 800 o 960 px) se descarta sin correrse (D20): el análisis de errores sobre validación mostró que casi ningún oso perdido es pequeño (ver "Análisis de errores sobre validación" en la parte 2). Se añaden tres corridas que no estaban en el plan, para la ablación por fuente (pregunta 22), con el protocolo 1.3 sin más cambio que los datos de entrenamiento: A1 y A1b (nano solo con ENA24, semillas 0 y 1) y A2 (nano solo con iNaturalist, semilla 0). Nombres: `a<N>[b]_<tamaño>_<fuente>_s<semilla>`.

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

> **Revisión 2026-09-30.** Velocidad medida en el equipo local: 52 ms por foto y 412 MB de RAM como máximo (ver "Velocidad en CPU"). La decisión se mantiene.
 
### Velocidad en CPU (2026-09-30)
 
- **Herramienta**: `scripts/medir_velocidad_cpu.py`, en el equipo local: AMD Ryzen 5 3600 (6 núcleos, 12 hilos), Windows, Python 3.14.2, torch 2.12.1+cpu, Ultralytics 8.4.138. Ultralytics usó 8 hilos de PyTorch y el formato de memoria channels_last, lo que hace por su cuenta en CPU (verificado en el código; ver `ESTADO_PROYECTO.md`, sección 7).
- **Protocolo**: un modelo por ejecución, para que la memoria de uno no se sume a la del otro. 100 fotos fijas de validación (50 de ENA24 y 50 de iNaturalist, semilla 0), después de 10 de calentamiento; 640 px; una foto por vez; umbral de la regla B de cada modelo. "Total" es el reloj alrededor de `predict` e incluye leer el JPEG; "red" es el tiempo de inferencia que guarda Ultralytics en `Results.speed`, que no incluye la lectura.

| Modelo | Total, mediana | Total, p95 | Total, máximo | Red, mediana | Red, p95 | RAM con el modelo cargado | Pico de RAM |
|---|---|---|---|---|---|---|---|
| E1 nano | 51.9 ms | 73.1 ms | 85.5 ms | 34.4 ms | 60.3 ms | 240 MB | 412 MB |
| E3 small | 81.8 ms | 108.4 ms | 127.2 ms | 67.6 ms | 96.7 ms | 273 MB | 492 MB |
| E4 medium | 183.1 ms | 225.8 ms | 265.6 ms | 169.3 ms | 218.5 ms | 324 MB | 597 MB |
 
- Por fuente (total, mediana y p95): E1, ENA24 53.6 y 70.4 ms, iNaturalist 50.0 y 79.2 ms; E3, 83.6 y 89.5 ms, 77.7 y 116.0 ms; E4, 184.6 y 195.1 ms, 178.3 y 240.6 ms. Carga del modelo: de 0.08 a 0.11 s. Primera foto, que prepara el predictor: de 0.17 a 0.33 s.
- **Lectura**: los tres tamaños cumplen las metas de los criterios (menos de 1 s por foto y menos de 4 GB) con mucho margen. Con la misma precisión, nano es unas 3.5 veces más rápido que medium en tiempo total y unas 4.9 veces en tiempo de red. Confirma D19.
- **Salvedad**: E1 se midió primero y su "resto" (lectura del JPEG y llamada) salió unos 4 ms mayor que en E3 y E4. Lo más probable es la caché de disco de Windows con las mismas fotos; no está verificado. Para comparar tamaños, la columna limpia es la de red.
- **Coherencia**: la referencia del plan, 9 ms de red a 320 px, equivale a unos 36 ms a 640 px (cuatro veces los píxeles). Medido: 34.4 ms.
- **Pendiente**: fotos de mayor resolución (las de Angochagua pueden pasar de 3 MP) y MP4.
- **Archivos**: `docs/modelado/velocidad_cpu_e1_n_deg0_s0.csv`, `velocidad_cpu_e3_s_deg0_s0.csv` y `velocidad_cpu_e4_m_deg0_s0.csv`.

### Análisis de errores sobre validación (2026-09-30 y 2026-10-01)
 
- **Herramienta**: `scripts/listar_errores.py`, con el umbral de la regla B de cada modelo y la misma comparación que `calcular_umbral.py` (los conteos coinciden con `umbral_opcion_b.csv`). Salida: `docs/modelado/errores_val.csv`, con 79 fotos, y copias con las cajas dibujadas y mosaicos en `data/errores_val/`. Los mosaicos se revisaron a ojo con el estudiante. La especie de los negativos de ENA24 sale de `ena24.json` y la de los de iNaturalist, de la columna `taxon` de su `registro_licencias.csv`.

Fotos con oso sin aviso, por modelo:
 
| | E1 | E1b | E3 | E3b | E4 | E4b |
|---|---|---|---|---|---|---|
| ENA24 de día (52) | 5 | 4 | 3 | 5 | 6 | 3 |
| ENA24 de noche (104) | 2 | 3 | 4 | 2 | 1 | 4 |
| iNaturalist (170) | 15 | 21 | 12 | 17 | 17 | 20 |
 
Falsas alarmas, por modelo:
 
| | E1 | E1b | E3 | E3b | E4 | E4b |
|---|---|---|---|---|---|---|
| ENA24 de día (62) | 2 | 4 | 6 | 3 | 3 | 0 |
| ENA24 de noche (120) | 1 | 2 | 3 | 1 | 1 | 2 |
| iNaturalist (90) | 3 | 4 | 6 | 4 | 4 | 5 |
 
Fotos distintas con error, según en cuántos de los seis modelos fallan:
 
| | 1 | 2 | 3 | 4 | 5 | 6 | Total |
|---|---|---|---|---|---|---|---|
| Con oso, ENA24 de día | 7 | 2 | 1 | 0 | 0 | 2 | 12 |
| Con oso, ENA24 de noche | 7 | 1 | 1 | 1 | 0 | 0 | 10 |
| Con oso, iNaturalist | 12 | 5 | 2 | 6 | 4 | 5 | 34 |
| Sin oso, ENA24 de día | 4 | 2 | 2 | 1 | 0 | 0 | 9 |
| Sin oso, ENA24 de noche | 2 | 0 | 1 | 0 | 1 | 0 | 4 |
| Sin oso, iNaturalist | 6 | 1 | 0 | 0 | 0 | 3 | 10 |
 
Tamaño del oso: lado menor de la caja más pequeña de la foto, en píxeles tal como lo ve la red a 640 (el lado mayor de la foto pasa a 640):
 
| | < 16 | 16–32 | 32–64 | ≥ 64 |
|---|---|---|---|---|
| ENA24, todas las fotos con oso | 0 | 0 | 18 | 138 |
| ENA24, perdidas por algún modelo | 0 | 0 | 2 | 20 |
| ENA24, perdidas por E1 | 0 | 0 | 1 | 6 |
| iNaturalist, todas las fotos con oso | 2 | 4 | 27 | 137 |
| iNaturalist, perdidas por algún modelo | 2 | 1 | 1 | 30 |
| iNaturalist, perdidas por E1 | 2 | 1 | 0 | 12 |
 
En las fotos con varios osos se mide el más pequeño, lo que exagera lo pequeño; solo 3 de las 56 fotos perdidas tienen más de un oso.
 
- **Causas** (clasificación a ojo sobre los mosaicos):
  - ENA24, 22 fotos: oso muy cerca de la cámara o cortado por el borde 14 (en 8 el oso ocupa más de un cuarto de la foto y 6 están cortados por el borde); mancha de luz en el lente y oso movido 1 (9472); oso completo a pleno sol en una ráfaga de 3 fotos, cada una perdida por un modelo distinto, 3 (9483 a 9485); lejanos 2 (9000 y 9002, misma escena al anochecer); otros 2. En todo ENA24, 312 de las 959 cajas ocupan más de un cuarto de la foto, así que 8 de 22 no es desproporcionado: describe cómo son los fallos, no prueba que los primeros planos sean más difíciles. Fallan en los seis modelos la 9472 y la 9618 (solo asoma el hocico).
  - iNaturalist, 34 fotos: tapados por vegetación u objetos 13, poca visibilidad (contraluz, niebla, sombra o foto borrosa) 8, lejanos 5, primer plano 3, nadando 1, sin causa evidente 4. De las 15 que fallan en 4 modelos o más: tapados 7, poca visibilidad 4, lejanos 3, nadando 1. Hay dos observaciones con dos fotos cada una entre los fallos (268470121 y 351866524). El caso más llamativo es `inat_48660817_77200359`: un oso bien visible en la hierba al que los seis modelos dan casi 0.
  - E1: en ENA24, 7 (primer plano o cortado 5, mancha de luz 1, lejano 1); en iNaturalist, 15 (tapados 6, poca visibilidad 4, lejanos 3, nadando 1, sin causa evidente 1).

Falsas alarmas por tipo de negativo (fotos de validación):
 
| Negativo | Fotos | Con alarma en algún modelo | Con alarma en E1 |
|---|---|---|---|
| Tapir de montaña (iNat) | 5 | 3 | 2 |
| Cuervo americano (ENA24) | 2 | 2 | 1 |
| Vaca (iNat) | 9 | 1 | 1 |
| Gato doméstico (ENA24) | 22 | 3 | 1 |
| Lince rojo (ENA24) | 73 | 7 | 1 |
| Lobo de páramo (iNat) | 13 | 3 | 0 |
| Perro (ENA24 34, iNat 3) | 37 | 2 | 0 |
| Rastros y paisajes (iNat) | 23 | 1 | 0 |
| Coyote, venado, caballo, puma, oveja y otras | 88 | 1 | 0 |
 
- **Lectura de las falsas alarmas**: se concentran en animales oscuros y de cuerpo macizo. Las dos fotos de tapir de la misma observación (345028290) disparan en los seis modelos, con confianza de hasta 0.96; también dispara en los seis una vaca negra con blanco. Las 3 de gato son el mismo gato negro en una ráfaga (6621 a 6623). Tres alarmas no caen sobre un animal: una roca oscura, un zapato junto a un excremento y unas nubes. Los linces son 73 de los 182 negativos de ENA24 en validación (40 %), por el reparto por grupos, y en proporción no destacan. Perros (1 de 34) y coyotes (0 de 17), que el informe de comprensión anticipaba como negativos difíciles, casi no disparan.
- **Umbral de E1 y sensibilidad**: fotos con alguna caja por encima del umbral (aunque la caja no caiga sobre el oso):

| Umbral | Con oso, ENA24 | Con oso, iNat | Con oso, total | Falsas alarmas ENA24 | Falsas alarmas iNat |
|---|---|---|---|---|---|
| 0.363 (regla B) | 149/156 | 155/170 | 304/326 | 3/182 | 3/90 |
| 0.20 | 151/156 | 158/170 | 309/326 | 3/182 | 4/90 |
| 0.10 | 154/156 | 162/170 | 316/326 | 4/182 | 4/90 |
| 0.05 | 154/156 | 165/170 | 319/326 | 6/182 | 5/90 |
| 0.01 | 154/156 | 167/170 | 321/326 | 10/182 | 8/90 |
 
  En la mayoría de las fotos perdidas el modelo pone una caja débil, y en los mosaicos esa caja suele caer sobre el oso. No se cambia el umbral: el proxy no tiene fotos vacías y con ellas las falsas alarmas reales pueden subir. Se revisa en la iteración 2.
- **Etiquetas revisadas**: `inat_190412124_333788182_frame_000` (oso trepado y tapado por el tronco; la detección de la izquierda es una rama) e `inat_268470424_482568294` (oso tras unas barandas, no cautivo según su observación en iNaturalist). Las dos etiquetas se mantienen.
- **Decisión**: D20, no se corre el experimento de resolución.
- **Para la iteración 2**: incluir como negativos los animales oscuros que aparezcan en las cámaras de Angochagua.

### Ablación por fuente: A1, A1b y A2 (2026-09-30 y 2026-10-01)
 
- **Objetivo**: pregunta 22. Si las fotos de iNaturalist ayudan a detectar osos en cámara trampa, y qué datos conservar en el fine-tuning de la iteración 2.
- **Cuaderno**: `notebooks/entrenar_yolo26.ipynb`, versión 2, con el parámetro `FUENTE`. Entorno igual que en septiembre: Kaggle, Tesla T4, Python 3.12.13, torch 2.10.0+cu128, Ultralytics 8.4.138, albumentations 2.0.8.
- **Argumentos**: protocolo 1.3 con `degrees=0` y 100 épocas. Solo cambian los datos de entrenamiento. Validación durante el entrenamiento y elección de `best.pt` con la validación completa (598 imágenes), como en E1.
- **Escaneo**: A1 y A1b, entrenamiento 1437 imágenes, 845 de fondo; A2, 1219 imágenes, 429 de fondo; validación 598 (272 de fondo), ENA24 338 (182), iNaturalist 260 (90). Cero corruptas. Optimizador elegido por `auto` en las tres: AdamW, lr 0.002, momentum 0.9 (iteraciones nominales 2300 y 2000, por debajo de 10000).

| Corrida | Entrenado con | s/época | Minutos de entrenamiento | Mejor época (mAP50-95 global) | Última época |
|---|---|---|---|---|---|
| A1 `a1_n_ena24_s0` | ENA24, semilla 0 | 17.5 | 29.1 | 66 (0.5308) | 0.4917 |
| A1b `a1b_n_ena24_s1` | ENA24, semilla 1 | 17.7 | 29.5 | 57 (0.5188) | 0.4930 |
| A2 `a2_n_inat_s0` | iNaturalist, semilla 0 | 15.9 | 26.4 | 94 (0.6216) | 0.6112 |
 
Validación con `best.pt` (celda 5):
 
| Corrida | Partición | P | R | mAP50 | mAP50-95 |
|---|---|---|---|---|---|
| A1 | Global (598) | 0.836 | 0.668 | 0.743 | 0.532 |
| A1 | ENA24 (338) | 0.887 | 0.918 | 0.942 | 0.730 |
| A1 | iNaturalist (260) | 0.694 | 0.508 | 0.532 | 0.331 |
| A1b | Global (598) | 0.890 | 0.670 | 0.745 | 0.519 |
| A1b | ENA24 (338) | 0.927 | 0.920 | 0.932 | 0.716 |
| A1b | iNaturalist (260) | 0.712 | 0.522 | 0.564 | 0.330 |
| A2 | Global (598) | 0.862 | 0.709 | 0.791 | 0.622 |
| A2 | ENA24 (338) | 0.739 | 0.612 | 0.637 | 0.470 |
| A2 | iNaturalist (260) | 0.923 | 0.841 | 0.914 | 0.746 |
 
Comparación con E1 y E1b (mismo protocolo, ambas fuentes):
 
| Entrenado con | Corridas | ENA24 mAP50 | ENA24 mAP50-95 | iNat mAP50 | iNat mAP50-95 |
|---|---|---|---|---|---|
| Ambas fuentes | E1, E1b | 0.933–0.955 | 0.762–0.786 | 0.908–0.922 | 0.732–0.758 |
| Solo ENA24 | A1, A1b | 0.932–0.942 | 0.716–0.730 | 0.532–0.564 | 0.330–0.331 |
| Solo iNaturalist | A2 | 0.637 | 0.470 | 0.914 | 0.746 |
 
- **Lectura provisional**:
  - Sin fotos de cámara trampa, el modelo no sirve en cámara trampa: A2 cae a 0.64 de mAP50 en ENA24. Hipótesis sin verificar: 104 de las 156 fotos con oso de ENA24 en validación son nocturnas, e iNaturalist casi no tiene fotos de noche (4 en escala de grises en total).
  - Sin iNaturalist, en ENA24 la detección (mAP50) queda dentro del rango de E1 y E1b, pero el ajuste de las cajas (mAP50-95) baja unos 0.05: por encima de la regla de ruido de 0.03 de las particiones por fuente, y sin que los rangos se toquen.
  - Ninguna fuente estorba a la otra: en iNaturalist, entrenar con ambas da lo mismo que entrenar solo con iNaturalist.
  - A la inversa, sin fotos de iNaturalist el modelo no sirve en iNaturalist: A1 y A1b caen a 0.33 de mAP50-95 en esa partición.
- **Salvedades**:
  - A1 y A1b eligieron su mejor época con la validación completa, que incluye las fotos de iNaturalist donde fallan. Sus mejores épocas (66 y 57) llegan mucho antes que las de E1 y E1b (99 y 95), y después la métrica global baja. Puede que una época elegida mirando solo ENA24 acortara la diferencia en mAP50-95. Se puede comprobar evaluando su `last.pt`.
  - Cada modelo de la ablación entrena con cerca de la mitad de las imágenes de E1, con otra proporción de negativos (1.43 negativos por positivo con ENA24, 0.54 con iNaturalist, 0.92 en E1), y A2 tiene una sola semilla.
  - Las P y R de Ultralytics están en el umbral de máximo F1 de cada modelo y no se comparan entre modelos.
- **Respuesta provisional a la pregunta 22**: conservar ambas fuentes en el fine-tuning de la iteración 2. ENA24 no se puede quitar, porque sin ella se pierde la cámara trampa, e iNaturalist no cuesta nada, aporta la especie y probablemente mejora el ajuste de las cajas.
- **Evaluación por imagen:** hecha el 2 de octubre (sección siguiente). La respuesta provisional se confirma.
- **GPU**: unos 85 minutos de entrenamiento entre las tres corridas.
- **Archivos**: `docs/modelado/a1_n_ena24_s0_*`, `a1b_n_ena24_s1_*` y `a2_n_inat_s0_*` (`results.csv`, `args.yaml`, `val_por_particion.csv`). Pesos `*_best.pt` (5.4 MB cada uno) y `*_run.zip` fuera del repositorio.

### Evaluación con la regla B del modelo COCO sin entrenar y de la ablación (2026-10-02)
 
- **Herramientas:**
  - `notebooks/evaluar_umbral_yolo26.ipynb`, versión 2, corrido en Kaggle (Tesla T4, Ultralytics 8.4.138).
  - `scripts/calcular_umbral.py`, versión 2, en local.
  - Archivos en `docs/modelado/eval_coco_ablacion/`.
- **Modelos:** el COCO sin entrenar, E1 y E1b (control), y A1, A1b y A2. Los `last.pt` de A1 y A1b no se subieron, así que no se evaluaron.
- **Control:** E1 y E1b repiten exactamente las cifras del 30 de septiembre:
  - Umbrales A de 0.496 y 0.611.
  - Umbrales B de 0.363 y 0.461.
  - Los mismos conteos en las cinco particiones.
  La versión 2 del cuaderno no cambió los resultados.
- **Control de mAP50** (este cuaderno frente a Ultralytics):
  - E1 y E1b: igual que en septiembre (hasta 0.0065 en global y en ENA24, y hasta −0.015 en iNaturalist).
  - Ablación: difiere más: −0.013 y −0.014 en global para A1 y A1b; −0.017, −0.035 y −0.019 en iNaturalist para A1, A1b y A2; +0.0145 en ENA24 para A2.
  - Las referencias de la ablación tienen 3 decimales. Hipótesis sin verificar: en particiones donde el modelo es malo, cambios pequeños de caja mueven más el AP (pregunta 30).
  - No afecta al recall por imagen ni a las falsas alarmas.
**Umbrales:**
 
| Modelo | Umbral A | Umbral por recall de imagen | Umbral mínimo por falsas alarmas | Umbral B | Estado |
|---|---|---|---|---|---|
| COCO sin entrenar | 0.0014 | inalcanzable | 0.1688 | 0.1688 | Recall por imagen de 0.95 inalcanzable |
| E1 | 0.4960 | 0.3629 | 0.0071 | 0.3629 | Cumple |
| E1b | 0.6108 | 0.4608 | 0.0111 | 0.4608 | Cumple |
| A1 | 0.4436 | 0.1931 | 0.7380 | 0.7380 | No cumple: falsas alarmas en iNaturalist |
| A1b | 0.3294 | 0.1824 | 0.6419 | 0.6419 | No cumple: falsas alarmas en iNaturalist |
| A2 | inalcanzable | inalcanzable | 0.0212 | 0.0212 | Ninguna meta de recall alcanzable |
 
En el modelo COCO, el recall de cajas máximo en ENA24 con confianza de 0.001 o más es 0.914.
 
**Regla B** (fotos con oso detectadas · falsas alarmas):
 
| Modelo | ENA24 | iNaturalist | ENA24 de día | ENA24 de noche | P y R de cajas en ENA24 |
|---|---|---|---|---|---|
| COCO sin entrenar | 103/156 (0.66) · 18/182 | 135/170 (0.79) · 8/90 | 41/52 · 16/62 | 62/104 · 2/120 | 0.780 · 0.636 |
| E1 | 149/156 (0.96) · 3/182 | 155/170 (0.91) · 3/90 | 47/52 · 2/62 | 102/104 · 1/120 | 0.950 · 0.932 |
| E1b | 149/156 (0.96) · 6/182 | 149/170 (0.88) · 4/90 | 48/52 · 4/62 | 101/104 · 2/120 | 0.919 · 0.914 |
| A1 | 126/156 (0.81) · 1/182 | 51/170 (0.30) · 8/90 | 37/52 · 1/62 | 89/104 · 0/120 | 0.977 · 0.796 |
| A1b | 137/156 (0.88) · 4/182 | 74/170 (0.44) · 8/90 | 40/52 · 3/62 | 97/104 · 1/120 | 0.933 · 0.858 |
| A2 | 110/156 (0.71) · 18/182 | 163/170 (0.96) · 6/90 | 32/52 · 11/62 | 78/104 · 7/120 | 0.629 · 0.660 |
 
**mAP50 calculado por el cuaderno** (global · ENA24 · iNaturalist):
- COCO: 0.789 · 0.737 · 0.846
- E1: 0.926 · 0.960 · 0.892
- E1b: 0.922 · 0.934 · 0.912
- A1: 0.730 · 0.944 · 0.515
- A1b: 0.731 · 0.933 · 0.529
- A2: 0.785 · 0.652 · 0.895

**Recall por imagen igualado** (`--max-falsas 1.0`: el umbral sale solo de las metas de recall en ENA24; archivo `umbral_solo_recall.csv`):
 
| Modelo | Umbral | ENA24 · falsas alarmas | P · R de cajas en ENA24 | iNaturalist · falsas alarmas | ENA24 de día | ENA24 de noche |
|---|---|---|---|---|---|---|
| E1 | 0.3629 | 149/156 · 3/182 | 0.950 · 0.932 | 155/170 (0.91) · 3/90 | 47/52 · 2/62 | 102/104 · 1/120 |
| E1b | 0.4608 | 149/156 · 6/182 | 0.919 · 0.914 | 149/170 (0.88) · 4/90 | 48/52 · 4/62 | 101/104 · 2/120 |
| A1 | 0.1931 | 149/156 · 9/182 | 0.831 · 0.938 | 90/170 (0.53) · 18/90 | 45/52 · 5/62 | 104/104 · 4/120 |
| A1b | 0.1824 | 149/156 · 8/182 | 0.857 · 0.926 | 100/170 (0.59) · 21/90 | 45/52 · 5/62 | 104/104 · 3/120 |
 
Con el umbral mínimo (0.0001), ni el COCO ni A2 llegan a 0.95 en ENA24:
- COCO: 146/156 con 112/182 falsas alarmas. No pone ninguna caja de oso en 10 fotos con oso.
- A2: 132/156 (0.85) con 66/182 falsas alarmas, igual de día (44/52) que de noche (88/104).

**Lectura:**
 
- **Lo que aporta el entrenamiento.** Frente al modelo COCO, E1 pasa de 0.66 a 0.96 de recall por imagen en ENA24 y de 18 a 3 falsas alarmas de 182. La mayor diferencia está de noche (0.60 frente a 0.98). En iNaturalist la ganancia es menor (0.79 frente a 0.91, y 8 frente a 3 falsas alarmas de 90): el modelo COCO ya reconoce osos en fotografías comunes, pero no en el infrarrojo de cámara trampa.
- **Sin cámaras trampa (A2):** no se alcanza la meta en cámara trampa, ni de día ni de noche. La hipótesis de que A2 fallaba sobre todo de noche no se sostiene.
- **Sin iNaturalist (A1, A1b), en cámara trampa:** con el mismo recall por imagen, da algo más de falsas alarmas (8 y 9 frente a 3 y 6 de 182, con intervalos solapados) y cajas menos precisas (0.83 y 0.86 frente a 0.95 y 0.92). Coincide con la caída de mAP50-95 vista el 1 de octubre, pero la diferencia en falsas alarmas no está demostrada.
- **Sin iNaturalist, en fotos comunes:** detecta 0.53 y 0.59 de los osos andinos y da 20 y 23 % de falsas alarmas con otras especies. Con la regla B completa, eso obliga a subir el umbral a 0.64–0.74, y el recall en ENA24 cae a 0.81–0.88.
- **Lo que no separa la ablación:** cuánto se debe a las fotos de oso de iNaturalist y cuánto a las de otras especies.
- **Respuesta a la pregunta 22:** conservar las dos fuentes en el fine-tuning de la iteración 2 (D21). ENA24 aporta la cámara trampa y la noche. iNaturalist aporta la especie y las otras especies andinas, que son las que verá el modelo en Angochagua.
- **Salvedades:**
  - La elección de época de A1 y A1b sigue sin comprobarse, aunque con el recall igualado su diferencia con E1 en ENA24 ya es pequeña.
  - A2 tiene una sola semilla.
  - Todo es sobre validación, y el umbral de cada modelo se eligió con esos mismos datos.
 

---