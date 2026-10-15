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

> **Revisión 2026-10-05.** Desde el 3 de octubre la imagen de Kaggle es otra: Python 3.13.15, torch 2.11.0+cu128, numpy 2.1.3 y controlador con CUDA 13.0. Todo lo que se corra con Ultralytics desde ahora usará esa imagen, y todavía no se ha comprobado que Ultralytics 8.4.138 dé lo mismo en ella: E1 y E1b como control en cada evaluación lo dirán. RT-DETRv3 corre con PaddlePaddle 3.3.0 en un entorno virtual aparte, con Python 3.12.13 (sección de RT-DETRv3 en la parte 2). Para medir RT-DETRv3 en CPU hará falta otro Python en el equipo local, porque PaddlePaddle no tiene versión para Python 3.14.

> **Revisión 2026-10-07.** La evaluación de E7 corrió en la imagen nueva de Kaggle (Python 3.13.15, torch 2.11.0+cu128, numpy 2.1.3, pandas 2.3.3 y pycocotools 2.0.11) y repitió al cuarto decimal las cifras de septiembre (E1, E1b, E4 y E4b) y del 6 de octubre (también E6 y E6b): la imagen nueva no cambia la evaluación. Para el entrenamiento no hay comparación, porque E7 corrió en la imagen de septiembre. RT-DETRv3 se midió en la CPU local con Python 3.12.9, PaddlePaddle 3.3.0 para CPU y las mismas 67 versiones que en Kaggle (parte 2, "Velocidad de RT-DETRv3 en CPU").

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

> **Revisión 2026-10-05.** Tres datos de Ultralytics 8.4.138, verificados en el código, que importan para comparar con RT-DETRv3:
> - Con `batch=16`, Ultralytics acumula los gradientes de 4 lotes y actualiza los pesos cada 64 fotos (`nbs=64`; `engine/trainer.py`, línea 302).
> - La mejor época (`best.pt`) se elige solo por mAP50-95: `fitness` le da peso 1 y 0 a las demás métricas (`utils/metrics.py`). Es el mismo criterio con que PaddleDetection elige `best_model`.
> - Con 200 épocas, `optimizer=auto` seguiría eligiendo AdamW con lr 0.002 (42 × 200 = 8400 iteraciones nominales, por debajo de 10 000). Con 239 épocas o más pasaría a MuSGD.


> **Revisión 2026-10-06.** Para E7 (200 épocas, regla de D25) se añade `patience=EPOCHS` (versión 3 de `entrenar_yolo26.ipynb`). Hasta ahora `patience=100` era igual a `epochs`. Con 200 épocas, Ultralytics 8.4.138 cortaría el entrenamiento tras 100 épocas sin mejora (`utils/torch_utils.py`, `EarlyStopping`). Con 100 épocas el cambio no altera nada.


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

> **Revisión 2026-10-05.** RT-DETRv3 usa el aumento de datos de su receta oficial (`configs/rtdetrv3/_base_/rtdetr_reader.yml`), no el de Ultralytics: cambios de color con probabilidad 0.8, expansión del lienzo, recorte aleatorio con probabilidad 0.8, volteo horizontal y una escala por lote elegida al azar entre 480 y 800 px, sin conservar la proporción. No usa Mosaic ni rotación. Se deja como viene, porque cambiarlo sería modificar el método (D23), y la diferencia se declara.

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

> **Revisión 2026-10-05.** RT-DETRv3-R18 se registra como E6 (semilla 0) y E6b (semilla 1), con una corrida corta de verificación, E6v. No sigue el protocolo 1.3: usa la receta oficial de RT-DETRv3 con los cambios de D23. Lo común con YOLO26 son los datos y la división, los pesos de partida COCO, las 100 épocas, la mejor época por mAP50-95 de validación y la evaluación. Su apartado en el plan es el 2.2, no el 2.3 que dice el párrafo de esta sección que empieza "RT-DETRv3 tiene su propio apartado". Detalle en la parte 2.

> **Revisión 2026-10-06.** Se añade E7 (`e7_m_deg0_s0`): YOLO26m con 200 épocas y semilla 0, como corrida adicional, porque se cumplió la regla de D25. El protocolo principal sigue siendo de 100 épocas. Una segunda semilla (E7b) solo se entrena si E7 queda a menos de 0.03 de la media de RT-DETRv3 (ficha D25).

> **Revisión 2026-10-07.** E7 se evaluó y queda 0.081 por debajo de RT-DETRv3 en mAP50-95 global de pycocotools, más del límite de 0.03, así que E7b no se entrena (ficha D25). E7 queda con una sola semilla.

### 1.6 Evaluación

- Selección de modelo y de umbrales: solo sobre validación. El conjunto de prueba se usa una vez por modelo final, al terminar. Ninguna decisión de configuración se toma mirando prueba.
- Cada modelo se valida tres veces sobre prueba (global con `dataset.yaml`, ENA24 con `dataset_test_ena24.yaml`, iNaturalist con `dataset_test_inat.yaml`) y las mismas tres sobre validación. Métricas de Ultralytics con los umbrales por defecto de `val`, confirmados en la salida de E0: `conf=0.001` implícito para el cálculo de mAP e `iou=0.7`.
- Recall a nivel de imagen (D15): proporción de imágenes con oso en las que el modelo pone al menos una detección por encima del umbral de operación, y proporción de negativos con alguna detección (falsas alarmas). El umbral de operación se elige sobre validación. Requiere un script propio a partir de las predicciones, pendiente de escribir.
- Además: número de parámetros, tiempo de inferencia por imagen en CPU local, y ruta de los pesos.
- Las métricas de la iteración 1 son preliminares (D1). El resultado principal es el de ENA24 (D15). Limitación declarada: la misma cámara de ENA24 aparece en entrenamiento y prueba con el mismo fondo, así que no se mide generalización a cámaras nuevas (D13).

> **Revisión 2026-10-05.** RT-DETRv3 no se valida con Ultralytics. Sus detecciones de validación se exportan desde PaddleDetection y el cuaderno de evaluación las procesa igual que las de YOLO26: mismo emparejamiento, misma regla B y mismas particiones. Para comparar el mAP entre arquitecturas se usa el que calcule el cuaderno con el mismo código para todos los modelos (pregunta 32). El de cada librería queda como control.

> **Revisión 2026-10-06.** El mAP que se compara entre arquitecturas lo calcula pycocotools 2.0.11 con sus parámetros estándar (D26), en el cuaderno de evaluación versión 3. El cálculo propio del cuaderno (el emparejamiento en 10 umbrales de IoU y `compute_ap`) y las cifras de Ultralytics y de PaddleDetection quedan como control.

> **Revisión 2026-10-07.** RT-DETRv3 produce 300 cajas por foto y YOLO26 unas 2. Antes de la evaluación final se comprueba si la ventaja de RT-DETRv3 en mAP50-95 depende de eso, recalculando el mAP con las k cajas de mayor confianza por foto (pregunta 33; lectura fijada en la ficha D26).

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

> **Revisión 2026-10-07.** RT-DETRv3 se midió con el mismo protocolo y las mismas 100 fotos: 1.41 s por foto (ver "Velocidad de RT-DETRv3 en CPU", al final de la sección de RT-DETRv3).

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
 
 ### RT-DETRv3-R18 con la implementación oficial (2026-10-03 a 2026-10-05)

- **Objetivo:** la comparación de arquitecturas que dispuso el director (D9). El par emparejado en parámetros es RT-DETRv3-R18 frente a YOLO26m (E4 y E4b).
- **Fuentes verificadas el 3 de octubre:**
  - Repositorio oficial `clxia12/RT-DETRv3`, implementación del artículo de WACV 2025, licencia Apache-2.0, último commit `349e7d99a5065e7b684118912e6a74178d4f4625` del 18 de marzo de 2025. Es PaddlePaddle, con una copia de PaddleDetection dentro. No hay versión en PyTorch (issue #6, abierto).
  - Configuración `configs/rtdetrv3/rtdetrv3_r18vd_6x_coco.yml` y sus bases (`optimizer_6x.yml`, `rtdetrv3_r50vd.yml` y `rtdetr_reader.yml`). Según el README, R18 tiene 20 M de parámetros, 60 GFLOPs y 48.1 AP en COCO.
  - `requirements.txt` no fija la versión de Paddle. En PyPI, `paddlepaddle-gpu` llega hasta la 2.6.2. La 3.3.0 está en el índice propio de Paddle, para Python 3.9 a 3.13.
  - Ultralytics 8.4.138 solo trae el RT-DETR original (`rtdetr-l`, `rtdetr-x`, `rtdetr-resnet50` y `rtdetr-resnet101`), con pesos para L y X. No hizo falta usarlo (D22).
- **Receta oficial, E6 y YOLO26** (verificado en las configuraciones de RT-DETRv3 y en el código de Ultralytics 8.4.138):

| Aspecto | Receta oficial de RT-DETRv3-R18 | E6 | YOLO26 (protocolo 1.3) |
|---|---|---|---|
| Pesos de partida | ImageNet (ResNet18-vd) | COCO (RT-DETRv3-R18 del README) | COCO |
| Épocas | 72 | 100 | 100 |
| Fotos por actualización de pesos | 64 (4 GPU × 16) | 8 (2 T4 × 4) | 64 (lotes de 16, gradientes acumulados) |
| Optimizador | AdamW, lr 0.0004, weight decay 0.0001, recorte del gradiente a norma 0.1 | Igual, con lr 0.00005 (0.0004 × 8 / 64) | AdamW, lr 0.002, elegido por `optimizer=auto` |
| Tasa de aprendizaje | Calentamiento lineal de 2000 pasos y después constante | Igual: 2000 pasos son unas 6 épocas | Calentamiento de 3 épocas y bajada lineal hasta el 1 % |
| EMA de los pesos | Sí, 0.9999 | Igual | Sí, el de Ultralytics |
| Normalización | BatchNorm sincronizada entre GPU | Sincronizada entre las 2 T4 (8 fotos) | BatchNorm por lote (16 fotos) |
| Precisión | Completa | Completa | Mixta |
| Aumento de datos | Color (p 0.8), expansión, recorte (p 0.8), volteo, escala de 480 a 800 px | Igual | Sección 1.4 |
| Evaluación | Foto reescalada a 640 × 640 sin conservar la proporción | Igual | 640 conservando la proporción, con bordes |
| Mejor época | mAP50-95 de validación (pycocotools), pesos EMA | Igual, elegida entre sesiones | mAP50-95 de validación (`fitness`) |
| Fotos sin oso | Descartadas salvo con `allow_empty: true` | Incluidas | Incluidas |

- **Entorno en Kaggle:** T4 x2, entorno virtual con Python 3.12.13 (instalado con `uv`), PaddlePaddle 3.3.0 para CUDA 12.6 desde el índice de Paddle y 67 paquetes con versión fija (`requisitos_rtdetrv3.txt`). Paddle funciona en la T4: su lista de arquitecturas compiladas incluye la 7.5. Detalle en `ESTRUCTURA_Y_SCRIPTS.md`, sección 3.
- **Problemas resueltos para que corriera:**

| # | Problema | Dónde apareció | Arreglo |
|---|---|---|---|
| 1 | PaddlePaddle instalado en el Python de Kaggle reemplaza las librerías de CUDA y cuDNN de torch por versiones que pip marca como incompatibles | Kaggle, 3 oct | Paddle va siempre en un entorno virtual aparte |
| 2 | El código necesita numpy anterior a 2 (`imgaug` usa `np.sctypes`), y la imagen nueva de Kaggle trae Python 3.13 con numpy 2.1.3. numpy 1.26 no tiene versión para Python 3.13 | CPU y Kaggle, 3 oct | Entorno virtual con Python 3.12.13 y numpy 1.26.4 |
| 3 | setuptools 81 o posterior ya no trae `pkg_resources`, que PaddleDetection importa | CPU, 3 oct | setuptools 80.10.2 |
| 4 | El `sklearn==0.0` de `requirements.txt` instala un paquete vacío, y el código hace `import sklearn` | CPU, 3 oct | scikit-learn 1.9.1 |
| 5 | El enlace de Baidu a los pesos COCO de R18 da 404 | Kaggle, 3 oct | Enlace de Google Drive del README, con el md5 anotado |
| 6 | El entorno virtual hereda el `MPLBACKEND` que Kaggle fija para el cuaderno, y matplotlib no lo reconoce | Kaggle, 3 oct | `MPLBACKEND=Agg` en cada llamada |
| 7 | `--enable_ce` necesita un valor y solo fija la semilla 0 | CPU, 3 oct | Lanzador propio `entrenar_con_semilla.py` |
| 8 | `print_params=True` falla con Paddle 3 (`IndexError`) | CPU, 3 oct | Parámetros contados aparte |
| 9 | `infer.py` baja una fuente tipográfica de Baidu para dibujar las cajas, y la configuración de prueba pide el `instances_val2017.json` de COCO | CPU, 3 oct | Script propio que imprime las cajas sin dibujarlas |
| 10 | El lector COCO descarta las imágenes sin cajas | Lectura del código, 3 oct | `allow_empty: true`. El registro confirma 2656 imágenes válidas y 0 descartadas |
| 11 | Un lote sin ninguna caja hace fallar al modelo (`paddle.concat` con `None` en `rtdetr_transformerv3.py`) | CPU, 3 oct | Parche versión 1 (saltar el lote) y, desde el 5 de octubre, versión 2 (entrenarlo sin denoising, D24) |
| 12 | Con memoria compartida, las fotos sin oso tumban la carga de datos (`Tensor holds no memory`) | Kaggle, 5 oct | `use_shared_memory: false` |
| 13 | Sin memoria con 16 fotos por GPU: pidió 2.86 GiB con 12.55 GiB ocupados | Kaggle, 5 oct | 8 fotos |
| 14 | Sin memoria con 8 fotos por GPU: pidió 1.43 GiB con 13.82 GiB ocupados | Kaggle, 5 oct | 4 fotos por GPU |
| 15 | En las opciones `-o`, `5e-05` se lee como texto (YAML 1.1) | CPU, 5 oct | Tasa escrita con decimales fijos |

  Seis aparecieron solo en Kaggle, siete solo en las pruebas previas en CPU, uno en los dos sitios y uno leyendo el código. Paddle imprime la memoria en "GB", pero son GiB: 16 fotos × 8 cabezas × 2450 × 2450 consultas × 4 bytes dan exactamente 2.862215 GiB, la cifra del error.

- **Prueba con los pesos COCO sin entrenar** (3 de octubre, 1 T4, tres fotos de entrenamiento): en la foto de iNaturalist detecta `bear` (0.86 la caja más alta). En la de ENA24 con oso, ninguna de las 5 cajas de mayor puntaje es `bear` (la primera es `elephant`, con 0.26). En la de ENA24 sin oso, `bear` con 0.54, una falsa alarma. Es el mismo patrón de YOLO26n sin entrenar: los modelos COCO no conocen la cámara trampa.
- **Parámetros** (configuración de una clase, contados en CPU con PaddleDetection): 22.214 M al entrenar. ResNet18-vd 11.200 M, codificador híbrido 4.965 M, transformer 4.781 M y rama auxiliar (`aux_o2m_head`) 1.268 M. Al detectar se usan unos 20.1 M, porque la rama auxiliar y las cabezas del codificador de los grupos extra de consultas (unos 0.8 M) solo existen en el entrenamiento. Emparejado con YOLO26m (20.35 M).

#### E6v. Corrida corta de verificación (2026-10-05)

- Cuaderno interactivo `rtdetrv3_entorno` en Kaggle. 1 T4, 4 fotos por paso, lr 0.000025 (0.0004 × 4 / 64), precisión completa, semilla 0 con el lanzador y parche versión 1. 3 épocas, con validación en cada una.
- Comando: `entrenar_con_semilla.py RT-DETRv3 0 -c rtdetrv3_r18vd_oso.yml --eval -o epoch=3 save_dir=… log_iter=20`.
- Carga: 2656 imágenes de entrenamiento válidas y 0 descartadas; 598 de validación.
- 664 pasos por época. Las 3 épocas (1992 pasos) caen dentro del calentamiento de 2000 pasos.
- El parche saltó 106 lotes sin cajas en 3 épocas, unos 35 por época: cerca del 11 % de las fotos sin oso de cada época. Por eso se escribió la versión 2.
- Pérdida total: de 52.3 en el primer registro a 19.0 en el último, sin NaN.
- Tiempo: 719 y 710 s por época con la validación; 36.1 minutos en total.
- Memoria: 10 641 MiB como máximo según nvidia-smi (la T4 tiene 15 360). Según PaddleDetection, 10 134 MiB reservados y 8 372 MiB ocupados como máximo (su registro dice "MB", pero divide entre 1024²).
- Validación según PaddleDetection, época por época:

| Época | mAP50-95 | mAP50 | AR (100 detecciones) |
|---|---|---|---|
| 1 | 0.088 | 0.111 | 0.665 |
| 2 | 0.365 | 0.475 | 0.863 |
| 3 | 0.544 | 0.681 | 0.883 |

- Detecciones exportadas: 179 400 con confianza ≥ 0.001 (300 por foto, las 598 fotos). Confianza máxima por foto: mediana 0.32 y máximo 0.69, todavía en calentamiento.
- La evaluación de PaddleDetection dio 23.5 fotos por segundo en la T4, con lotes de 16. No se compara con la velocidad en CPU.
- **Lectura:** solo comprueba que el entorno entrena, evalúa y exporta. No es un resultado: la tasa seguía en calentamiento y el parche era el de la versión 1. Cumple el criterio de "corre" fijado de antemano (D22).
- **Archivos:** no se conservaron. La sesión interactiva se cerró y estas cifras salen de la salida que se copió en la conversación.

#### Sondeo de velocidad (2026-10-05)

- Cuaderno `notebooks/rtdetrv3_sondeo.ipynb`, T4 x2, parche versión 2. Cada configuración entrena 6 minutos, sin validación, con 4 fotos por GPU y la tasa oficial escalada al lote total. Del registro se toman los segundos por paso y la espera de datos, sin los dos primeros registros. La memoria se mide con nvidia-smi cada 15 segundos. Los minutos por época suman 40 s de validación, medidos en E6v.
- Regla fijada antes de ver los números: la configuración más fiel a la receta oficial con la que 100 épocas quepan en 11 horas. Orden de fidelidad: 2 GPU en precisión completa, 1 GPU en precisión completa, 2 GPU en precisión mixta y 1 GPU en precisión mixta.

| Configuración | Lote total | Pasos por época | s por paso | Espera de datos | Fotos por s | min por época | h para 100 épocas | NaN | Memoria máx. (MiB) |
|---|---|---|---|---|---|---|---|---|---|
| 1 GPU, precisión completa | 4 | 664 | 1.034 | 0 % | 3.9 | 12.1 | 20.2 | No | 10 511 |
| 2 GPU, precisión completa | 8 | 332 | 1.465 | 0 % | 5.5 | 8.8 | 14.6 | No | 11 081 y 10 899 |
| 1 GPU, precisión mixta | 4 | 664 | 0.872 | 0 % | 4.6 | 10.3 | 17.2 | No | 11 069 |
| 2 GPU, precisión mixta | 8 | 332 | 1.330 | 0 % | 6.0 | 8.0 | 13.4 | No | 11 801 y 11 235 |

- **Lectura:**
  - Lo que limita es la GPU: la espera de datos es 0 % en las cuatro.
  - Con dos GPU cada época tarda 1.38 veces menos, no la mitad. Cada paso con dos GPU dura unos 0.43 s más que con una, porque las dos se coordinan después de cada paso (pérdidas, gradientes y BatchNorm sincronizada).
  - La precisión mixta acelera poco: 1.19 veces con una GPU y 1.10 con dos. Tampoco ahorra memoria: la softmax de la atención sigue en precisión completa (`paddle/amp/amp_lists.py`).
  - Ninguna configuración cabe en 11 horas, así que la regla no se puede cumplir tal cual. El cuaderno propuso la más rápida (2 GPU, precisión mixta, 13.4 h). Como las dos más rápidas necesitan de todos modos dos sesiones y se diferencian en 1.2 horas, se eligió la más fiel: 2 GPU en precisión completa (D23).
- **GPU:** unos 35 minutos de sondeo más unos 10 de preparación.
- **Archivos:** `docs/modelado/rtdetrv3/sondeo_rtdetrv3.csv` y `sondeo_registros.zip`.

#### E6 y E6b. RT-DETRv3-R18, semillas 0 y 1 (2026-10-05 y 2026-10-06)

- `RUN_ID`: `e6_rtdetrv3r18_s0` y `e6b_rtdetrv3r18_s1`. Cuaderno `notebooks/rtdetrv3_entrenar.ipynb`, T4 x2, en dos cuentas de Kaggle. Entre las dos solo cambian `RUN_ID` y `SEMILLA`.
- Configuración (D23): 2 T4 × 4 fotos, lr 0.00005, precisión completa, BatchNorm sincronizada, 100 épocas con validación en cada una, semilla fijada con el lanzador, parche versión 2 y pesos COCO.
- Comando: `python -m paddle.distributed.launch --devices 0,1 --log_dir <RUN_ID>/entreno/log entrenar_con_semilla.py /tmp/RT-DETRv3 <SEMILLA> -c rtdetrv3_r18vd_oso.yml --eval -o epoch=100 save_dir=<RUN_ID>/entreno log_iter=50`, con `-r …/ultimo/ultimo` en la sesión 2.
- Lanzadas en paralelo el 5 de octubre hacia las 16:15 (hora de Ecuador). La sesión 2 de cada una corrió el 6 de octubre.

| | Sesión 1 | Sesión 2 | Horas de entrenamiento (registro) | min por época, mediana por sesión (rango) | Mejor época (mAP50-95 de PaddleDetection) |
|---|---|---|---|---|---|
| E6 | épocas 1–73 | 74–100 | 10.81 + 4.16 = 14.97 | 8.8 y 9.2 (8.5–9.7) | 85 (0.8307), de la sesión 2 |
| E6b | épocas 1–68 | 69–100 | 10.71 + 4.92 = 15.63 | 9.5 y 9.1 (8.9–9.9) | 44 (0.8270), de la sesión 1 |

- **Fin de cada sesión.** Las sesiones 1 se detuvieron por tiempo antes de las 11 horas. Las sesiones 2 terminaron las 100 épocas (`model_final`). Las tres líneas de error del registro de cada sesión 1 (`C++ Traceback`, `Error Message Summary` y `FatalError: Termination signal is detected by the operating system`) son el aviso de Paddle al recibir la señal de parada que manda el cuaderno. El texto está en `libpaddle.so` de Paddle 3.3.0. En las sesiones 2 no hay ninguna.
- **Reanudación**, verificada en `train_s2.log`. La sesión 2 carga pesos, EMA y optimizador ("Exchange model and ema_model to load" … "Finish resuming model weights", `ppdet/utils/checkpoint.py`, `load_weight`) y empieza desde el paso 0 la época en que se cortó la sesión 1 (la 73 en E6 y la 68 en E6b, contadas desde 0). La tasa sigue en 0.000050, así que no repite el calentamiento. En el CSV no hay salto entre sesiones: 0.828 y 0.828 en E6, y 0.825 y 0.825 en E6b.
- **NaN:** ninguno. El único valor vacío del CSV es `min_epoca` de la época 100, que no tiene una época siguiente con la que medirse.
- **Memoria máxima** (nvidia-smi, sesión 2): 12 055 y 11 135 MiB (E6) y 11 321 y 11 135 MiB (E6b), de 15 360.
- **Curva** (`<RUN_ID>_epocas.csv`, métricas de PaddleDetection):
  - Sube rápido: 0.79 de mAP50-95 en la época 8. Después se aplana. E6 queda entre 0.828 y 0.831 desde la época 57, y E6b entre 0.821 y 0.827 desde la 43.
  - El mAP50 llega antes a su máximo (0.970 en la época 30 de E6 y 0.971 en la 23 de E6b) y termina en 0.962 y 0.963. Con más épocas las cajas ajustan algo mejor y la detección baja un poco. La mejor época se elige por mAP50-95, como en YOLO26.
  - En E6, la época 85 ganó a la 96 y a la 98 por menos de 0.0003. Con una curva plana, la mejor época se decide por el cuarto decimal.
  - La mejor época queda unos 0.002 por encima del promedio de la meseta (0.8307 frente a 0.8291 en E6, y 0.8270 frente a 0.8246 en E6b).
- **GPU:** E6, unas 15.4 horas. E6b, unas 16.1. Son las horas de entrenamiento más unos 13 minutos por sesión de preparación y cierre, estimados.
- **Archivos:** `docs/modelado/rtdetrv3/<RUN_ID>_epocas.csv`, `_best.json` y `_detecciones_val.csv.gz`, y los registros de cada sesión en `docs/modelado/rtdetrv3/<RUN_ID>/`. Los pesos `<RUN_ID>_best.pdparams` (unos 89 MB), fuera del repositorio.

#### Evaluación de E6 y E6b sobre validación (2026-10-06)

- **Herramientas:** `notebooks/evaluar_umbral_yolo26.ipynb` versión 3 (Kaggle, Tesla T4, Ultralytics 8.4.138 y pycocotools 2.0.11) y `scripts/calcular_umbral.py` versión 2 (local, sin cambios). Archivos en `docs/modelado/eval_rtdetrv3/`.
- **Modelos:** E1 y E1b (control), E4 y E4b (YOLO26m, el par emparejado en parámetros) y E6 y E6b (detecciones exportadas, tipo `externo`).
- **Fijado en la celda 0 antes de correrlo:** la regla de D25 con su precisión del 6 de octubre (media de E6 y E6b frente a la de E4 y E4b, en mAP50-95 global de pycocotools, con un margen de 0.015).
- **Detecciones:** YOLO26, de 1018 a 1340 por modelo con confianza ≥ 0.001. RT-DETRv3, 156 160 y 178 231 (hasta 300 por foto). Recall de cajas máximo en ENA24: de 0.969 a 0.988 en YOLO26 y 1.000 en RT-DETRv3.
- **Controles:**
  - Conteos de imágenes, cajas, fuentes y particiones iguales a los del informe de preparación. Para pycocotools, 351 cajas reales, ninguna recortada ni descartada, y ninguna foto con orientación EXIF.
  - E1, E1b, E4 y E4b repiten exactamente septiembre: el umbral A (0.496, 0.611, 0.498 y 0.679) y, con él, las fotos con oso detectadas y las falsas alarmas en ENA24. Con `calcular_umbral.py`, también los umbrales B (0.3629, 0.4608, 0.2559 y 0.5009) y los conteos de las cinco particiones.
  - mAP50 del cuaderno frente a Ultralytics: igual que en septiembre (de −0.0065 a +0.0054 en global y en ENA24, y hasta −0.0152 en iNaturalist).
  - pycocotools sobre las detecciones exportadas frente a PaddleDetection: 0.8308 frente a 0.8307 (E6) y 0.8270 frente a 0.8270 (E6b). +0.0001 en las dos.

**mAP común (pycocotools, y entre paréntesis el cálculo propio del cuaderno):**

| Modelo | mAP50-95 global | mAP50 global | mAP50-95 ENA24 | mAP50-95 iNat | mAP50-95 de la librería |
|---|---|---|---|---|---|
| E1 (YOLO26n) | 0.7494 (0.7519) | 0.9218 | 0.7763 | 0.7290 | 0.759 (Ultralytics) |
| E1b (YOLO26n) | 0.7388 (0.7412) | 0.9178 | 0.7483 | 0.7375 | 0.758 (Ultralytics) |
| E4 (YOLO26m) | 0.7434 (0.7461) | 0.9274 | 0.7526 | 0.7439 | 0.752 (Ultralytics) |
| E4b (YOLO26m) | 0.7468 (0.7495) | 0.9291 | 0.7605 | 0.7446 | 0.755 (Ultralytics) |
| E6 (RT-DETRv3) | 0.8308 (0.8339) | 0.9616 | 0.8318 | 0.8293 | 0.8307 (PaddleDetection) |
| E6b (RT-DETRv3) | 0.8270 (0.8302) | 0.9631 | 0.8237 | 0.8291 | 0.8270 (PaddleDetection) |

- El cálculo propio sale entre 0.0024 y 0.0032 por encima de pycocotools en los seis modelos, porque integra la curva de otra forma y no corta en 100 detecciones por foto. La diferencia es parecida entre arquitecturas, algo mayor en RT-DETRv3, que da más detecciones por foto.
- El de Ultralytics sale entre 0.008 y 0.019 por encima del de pycocotools en YOLO26. Por eso, en la tesis, las cifras de YOLO26 de septiembre y las de la comparación de arquitecturas no se mezclan.

**Regla B** con `calcular_umbral.py` (fotos con oso detectadas · falsas alarmas):

| Modelo | Umbral B | ENA24 | P · R de cajas en ENA24 | iNaturalist | Global | ENA24 de día | ENA24 de noche |
|---|---|---|---|---|---|---|---|
| E1 | 0.3629 | 149/156 · 3/182 | 0.950 · 0.932 | 155/170 · 3/90 | 304/326 · 6/272 | 47/52 · 2/62 | 102/104 · 1/120 |
| E1b | 0.4608 | 149/156 · 6/182 | 0.919 · 0.914 | 149/170 · 4/90 | 298/326 · 10/272 | 48/52 · 4/62 | 101/104 · 2/120 |
| E4 | 0.2559 | 149/156 · 4/182 | 0.899 · 0.932 | 153/170 · 4/90 | 302/326 · 8/272 | 46/52 · 3/62 | 103/104 · 1/120 |
| E4b | 0.5009 | 149/156 · 2/182 | 0.943 · 0.926 | 150/170 · 5/90 | 299/326 · 7/272 | 49/52 · 0/62 | 100/104 · 2/120 |
| E6 | 0.8420 | 149/156 · 3/182 | 0.968 · 0.938 | 161/170 · 1/90 | 310/326 · 4/272 | 48/52 · 3/62 | 101/104 · 0/120 |
| E6b | 0.8213 | 149/156 · 4/182 | 0.962 · 0.938 | 155/170 · 2/90 | 304/326 · 6/272 | 48/52 · 3/62 | 101/104 · 1/120 |

Las dos semillas de RT-DETRv3 cumplen las dos metas de recall con falsas alarmas por debajo del 10 %. Su umbral mínimo por falsas alarmas es 0.075 y 0.119, muy por debajo del elegido.

**Lectura:**
- **mAP50-95:** RT-DETRv3 supera a YOLO26m por 0.084 (medias de 0.8289 y 0.7451), más de cinco veces el margen de lectura, y en las dos fuentes (+0.07 en ENA24 y +0.085 en iNaturalist). Sus dos semillas difieren en 0.004.
- **mAP50:** la distancia es menor (0.962 frente a 0.928). RT-DETRv3 detecta algo más, pero sobre todo ajusta mejor las cajas.
- **Regla B en ENA24:** empate. Con el recall por imagen que fija la regla (149/156), las falsas alarmas de RT-DETRv3 (3 y 4 de 182) quedan dentro del rango de los seis YOLO26 de septiembre (de 2 a 9). Lo mismo de día (48/52) y de noche (101/104). Las cajas son más precisas: 0.96 y 0.97, frente a 0.90 a 0.95.
- **Regla B en iNaturalist:** indicio a favor de RT-DETRv3. E6 detecta 161 de 170 con 1 falsa alarma, más que cualquiera de los seis YOLO26 (de 149 a 158, con 3 a 6). E6b detecta 155, con 2. Los intervalos se solapan. Es la partición donde YOLO26 fallaba con osos tapados o con poca visibilidad.
- **Umbral:** el de RT-DETRv3 (0.82 y 0.84) es mucho más alto que el de YOLO26 (de 0.26 a 0.50). Las confianzas de las dos arquitecturas no se comparan, y cada modelo lleva su umbral.
- **Modelo de la aplicación:** D19 no cambia por ahora. En la métrica principal RT-DETRv3 empata, falta su velocidad en CPU y necesitaría otro Python en el equipo del usuario. Se decide con el director (pregunta 24).
- **Regla de D25:** se cumple. Condición 1: mejores épocas 85 y 44. Condición 2: +0.0838 frente a un mínimo de 0.015. Se entrena E7.
- **Salvedades:**
  - Todo es validación: el umbral y la mejor época de cada modelo se eligieron con estas mismas fotos.
  - Las recetas son distintas. En 100 épocas, RT-DETRv3 actualizó sus pesos unas 33 200 veces, con lotes de 8 fotos. YOLO26m, unas 4 200, con 64 fotos por actualización, porque Ultralytics acumula gradientes (`engine/trainer.py`).
  - La mejor época de RT-DETRv3 salió de una meseta, lo que le da una ventaja de unos 0.002 sobre su promedio. YOLO26m casi no la tiene, porque su mejor época fue la última.
  - YOLO26m no había convergido a las 100 épocas. Para eso es E7.

#### E7. YOLO26m con 200 épocas (`e7_m_deg0_s0`, regla de D25)

- Cuaderno `notebooks/entrenar_yolo26.ipynb` versión 3: `MODEL = yolo26m.pt`, `DEGREES = 0`, `SEED = 0`, `EPOCHS = 200`, `FUENTE = "ambas"` y `patience=EPOCHS`. El resto, como E4.
- Entorno: Kaggle, Tesla T4 (una de dos), Python 3.12.13, torch 2.10.0+cu128, Ultralytics 8.4.138 y albumentations 2.0.8. Es el entorno de septiembre, no la imagen que vio el cuaderno de RT-DETRv3 el 3 de octubre.
- Optimizador elegido por `auto`: AdamW, lr 0.002, momentum 0.9 (8400 iteraciones nominales).
- 200 épocas hechas. 98.6 s por época, 328.5 minutos de entrenamiento y 335.6 minutos la celda. GPU: unas 5.7 horas.
- Terminó el 7 de octubre. Revisada y evaluada esa noche (las tres secciones siguientes).

#### E7: curva, mejor época y tiempo (2026-10-07)

- Archivos: `docs/modelado/e7_m_deg0_s0_results.csv`, `_args.yaml` y `_val_por_particion.csv`. Pesos `e7_m_deg0_s0_best.pt` en `yolodeteccionoso` (carpeta `e7_m_deg0_s0/`).
- 200 épocas sin incidentes. Tiempo: la primera época, 128.9 s; las demás, de 96.6 a 100.5 s (98.4 de mediana; 96.8 con Mosaic apagado). 328.5 minutos de entrenamiento.
- Tasa de aprendizaje: baja en línea recta desde 0.002 hasta 0.00003 en la época 200, y desde la época 4 coincide exactamente con la fórmula de Ultralytics. En la época 100, E7 tenía 0.00102, mientras que E4 terminaba con 0.00004.
- **Mejor época: la 165** (mAP50-95 0.7631, mAP50 0.9468). Última: 0.7531. El mAP50 máximo, 0.9479, está en la 164.
- mAP50-95 medio por tramos de 20 épocas:

| Épocas | 81–100 | 101–120 | 121–140 | 141–160 | 161–180 | 181–200 |
|---|---|---|---|---|---|---|
| mAP50-95 | 0.720 | 0.733 | 0.738 | 0.747 | 0.754 | 0.754 |

- **Meseta:** de la época 166 a la 200, media de 0.754 (entre 0.750 y 0.758). Desde la 157, todas las épocas quedan a 0.015 o menos de la mejor.
- La época 165 queda 0.009 por encima de la media de sus diez vecinas (0.754). Es una ventaja de elección mayor que la de RT-DETRv3 (0.002).
- **Pérdidas de validación:** la de caja toca su mínimo en la época 169 (0.758) y la L1 en la 172 (0.0128), y después quedan planas. La de clasificación toca su mínimo en la 134 (0.490) y sube hasta 0.589 en la 200, mientras la de entrenamiento sigue bajando. Es la señal leve de sobreajuste en la clasificación que ya se vio en E1 y E1b, algo más marcada, y el mAP no se resiente.
- **Mosaic apagado** (épocas 191 a 200): las pérdidas de entrenamiento caen de golpe (caja de 0.587 a 0.464 y clasificación de 0.346 a 0.193, en medias de diez épocas), pero la validación no mejora: el mAP50-95 queda igual (0.754), el mAP50 baja 0.004 y las pérdidas de validación suben un poco.
- Tres caídas puntuales al principio (épocas 13, 23 y 45), cada una recuperada en la época siguiente.
- En la época 100, E7 da 0.743 y E4 terminó en 0.752. No se comparan: por la tasa de aprendizaje, son momentos distintos del entrenamiento.
- Validación con `best.pt` (celda 5, Ultralytics):

| Partición | P | R | mAP50 | mAP50-95 | mAP50-95 de E4 / E4b |
|---|---|---|---|---|---|
| Global (598) | 0.954 | 0.883 | 0.944 | 0.763 | 0.752 / 0.755 |
| ENA24 (338) | 0.956 | 0.936 | 0.958 | 0.784 | 0.767 / 0.765 |
| iNaturalist (260) | 0.913 | 0.887 | 0.924 | 0.743 | 0.746 / 0.755 |

- Frente a la media de E4 y E4b: +0.0095 en global, +0.018 en ENA24 y −0.007 en iNaturalist. Ninguna diferencia se interpreta (márgenes de 0.015 en global y de 0.03 por fuente).
- **Lectura:** medium se estabiliza con 200 épocas. Su meseta (0.754) está al nivel en que terminaron E4 y E4b, así que entrenar el doble no subió el techo de una forma que se distinga del azar. Lo que le da algo más a E7 es sobre todo la época 165.
- Verificado en el código de Ultralytics 8.4.138 (`ESTADO_PROYECTO.md`, sección 7): `best.pt` se elige solo por mAP50-95 y, si dos épocas empatan, se queda con la más tardía; la tasa baja en línea recta a lo largo de las épocas configuradas; Mosaic se apaga en la época 191; `time` es acumulado e incluye la validación; `l1_loss` aparece porque YOLO26 no usa DFL.

#### Evaluación de E7 sobre validación y reglas de después de E7 (2026-10-07)

- **Herramientas:** `notebooks/evaluar_umbral_yolo26.ipynb` versión 3 (Kaggle, Tesla T4, imagen nueva: Python 3.13.15, torch 2.11.0+cu128, Ultralytics 8.4.138, pycocotools 2.0.11) y `scripts/calcular_umbral.py` versión 2 (local). Archivos en `docs/modelado/eval_e7/`.
- **Modelos:** E1 y E1b (control), E4 y E4b, E7 (de la versión nueva de `yolodeteccionoso`) y E6 y E6b (detecciones exportadas).
- **Controles:**
  - E1 y E1b repiten septiembre: umbral A de 0.496 y 0.611, con 144 y 147 fotos con oso y 3 y 5 falsas alarmas en ENA24. E4 y E4b, también.
  - RT-DETRv3 frente a PaddleDetection: +0.0001 en las dos semillas.
  - Los seis modelos del 6 de octubre repiten al cuarto decimal su mAP común (global, ENA24 e iNaturalist) y los umbrales y conteos de la regla B en las cinco particiones.
  - E7 frente a Ultralytics, comparado a mano con `e7_m_deg0_s0_val_por_particion.csv`: mAP50 −0.0063 en global, +0.0080 en ENA24 y −0.0109 en iNaturalist. El +0.0080 pasa un poco el máximo visto hasta ahora en global y ENA24 (0.0065), con el mismo patrón. En mAP50-95, pycocotools da 0.0152 menos que Ultralytics (el rango de YOLO26 es de 0.008 a 0.019) y el cálculo propio del cuaderno, 0.0028 más (rango de 0.0024 a 0.0032).
- **Umbrales de E7:** regla A 0.7191 (recall de cajas máximo en ENA24: 0.981), regla B 0.4552 (la fija el recall por imagen) y umbral mínimo por falsas alarmas 0.0510. Cumple las dos metas de recall con falsas alarmas por debajo del 10 %.

**mAP común (pycocotools; entre paréntesis, el cálculo propio del cuaderno):**

| Modelo | mAP50-95 global | mAP50 global | mAP50-95 ENA24 | mAP50-95 iNat | mAP50-95 de la librería |
|---|---|---|---|---|---|
| E4 (YOLO26m, 100 épocas) | 0.7434 (0.7461) | 0.9274 | 0.7526 | 0.7439 | 0.752 (Ultralytics) |
| E4b (YOLO26m, 100 épocas) | 0.7468 (0.7495) | 0.9291 | 0.7605 | 0.7446 | 0.755 (Ultralytics) |
| E7 (YOLO26m, 200 épocas) | 0.7478 (0.7506) | 0.9335 | 0.7656 | 0.7400 | 0.763 (Ultralytics) |
| E6 (RT-DETRv3) | 0.8308 (0.8339) | 0.9616 | 0.8318 | 0.8293 | 0.8307 (PaddleDetection) |
| E6b (RT-DETRv3) | 0.8270 (0.8302) | 0.9631 | 0.8237 | 0.8291 | 0.8270 (PaddleDetection) |

ENA24 de día y de noche (pycocotools):

| Modelo | mAP50-95 de día | mAP50-95 de noche | mAP50 de día | mAP50 de noche |
|---|---|---|---|---|
| E4 | 0.747 | 0.758 | 0.943 | 0.957 |
| E4b | 0.781 | 0.755 | 0.990 | 0.952 |
| E7 | 0.748 | 0.778 | 0.943 | 0.975 |
| E6 | 0.813 | 0.843 | 0.978 | 0.974 |
| E6b | 0.806 | 0.836 | 0.966 | 0.975 |

**Regla B** (fotos con oso detectadas · falsas alarmas):

| Modelo | Umbral B | ENA24 | P · R de cajas en ENA24 | iNaturalist | Global | ENA24 de día | ENA24 de noche |
|---|---|---|---|---|---|---|---|
| E4 | 0.2559 | 149/156 · 4/182 | 0.899 · 0.932 | 153/170 · 4/90 | 302/326 · 8/272 | 46/52 · 3/62 | 103/104 · 1/120 |
| E4b | 0.5009 | 149/156 · 2/182 | 0.943 · 0.926 | 150/170 · 5/90 | 299/326 · 7/272 | 49/52 · 0/62 | 100/104 · 2/120 |
| E7 | 0.4552 | 149/156 · 4/182 | 0.956 · 0.944 | 153/170 · 4/90 | 302/326 · 8/272 | 47/52 · 3/62 | 102/104 · 1/120 |
| E6 | 0.8420 | 149/156 · 3/182 | 0.968 · 0.938 | 161/170 · 1/90 | 310/326 · 4/272 | 48/52 · 3/62 | 101/104 · 0/120 |
| E6b | 0.8213 | 149/156 · 4/182 | 0.962 · 0.938 | 155/170 · 2/90 | 304/326 · 6/272 | 48/52 · 3/62 | 101/104 · 1/120 |

**Reglas de después de E7 (ficha D25), aplicadas tal como se fijaron:**
1. E7 se evaluó igual que los demás modelos, con E1 y E1b de control, y los controles pasaron.
2. E7 frente a la media de RT-DETRv3 (0.8289): 0.0811 por debajo, más que el margen de 0.015. Se lee que la ventaja de RT-DETRv3 no venía solo de que medium no hubiera convergido. Frente a la media de E4 y E4b (0.7451): +0.0027, por debajo del margen; 100 épocas más no dan una ganancia que se distinga del azar. La distancia con RT-DETRv3 pasa de 0.0838 a 0.0811.
3. **E7b no se entrena:** E7 queda a 0.081 de RT-DETRv3, más del límite de 0.03. Se declara que E7 tiene una sola semilla.
4. La mejor época fue la 165, no la última, así que no hay nada que declarar sobre la convergencia.
5. El protocolo principal sigue en 100 épocas, y E7 es un punto de comparación adicional.

**Lectura:**
- Con la regla B, E7 se comporta como E4 en las cinco particiones. Sus cajas son algo más precisas que las de E4 y E4b (P de cajas en ENA24 de 0.956, frente a 0.899 y 0.943) y se acercan a las de RT-DETRv3 (0.962 y 0.968).
- En iNaturalist sigue el indicio a favor de RT-DETRv3 (161 y 155 fotos, frente a 153 de E7), con intervalos solapados.
- D19 no cambia. E7 no se midió en CPU: tiene la misma red que E4 (183 ms por foto).
- **¿Es el resultado esperado?** Está bien medido (mismos datos, misma división, mismo evaluador y controles que pasan), pero no es el que predecían las fuentes:
  - En COCO, YOLO26m (53.1 de mAP50-95, Tabla 2 del marco) supera a RT-DETRv3-R18 (48.1 según su README).
  - El estudio [39] del apartado 1.5.2 encontró a RT-DETR (el original) por debajo de YOLOv8 en mAP50 (0.84 frente a 0.89) y sin ninguna detección en una escena nocturna. Aquí RT-DETRv3 tiene más mAP50 (0.962 frente a valores de 0.927 a 0.934 en YOLO26m). De noche, en ENA24, empata en detección por imagen (101 de 104 fotos, frente a 100 a 103) y tiene más mAP50-95 (0.84 frente a valores de 0.755 a 0.778).
  - No indica un error: son otras versiones y otros datos, y en [39] el caso nocturno es una sola escena. El capítulo 3 tiene que discutirlo.
- **Comprobación pendiente** (pregunta 33): RT-DETRv3 produce 300 cajas por foto (con confianza ≥ 0.001 quedan entre 261 y 298 de media) y pycocotools usa hasta 100, las de mayor confianza. YOLO26 da unas 2 por foto. En el mAP de COCO, las cajas de confianza baja pueden sumar aciertos al final de la curva y casi no restan: solo bajan la precisión de los aciertos con confianza todavía menor. No se sabe cuánta de la ventaja de 0.08 viene de ahí. No afecta a la regla B, que solo usa las cajas por encima del umbral.
- **Salvedades:**
  - Todo es validación.
  - E7 tiene una sola semilla, y su mejor época sobresale 0.009 sobre su meseta. Si esa época tuvo algo de suerte, E7 estaría todavía más lejos de RT-DETRv3.
  - Las recetas siguen siendo distintas: E7 actualizó sus pesos unas 8 400 veces, con 64 fotos por actualización; RT-DETRv3, unas 33 200 en 100 épocas, con 8 fotos. E7 responde a la duda de la convergencia, no a la de la receta.

#### Velocidad de RT-DETRv3 en CPU (2026-10-07)

- **Herramienta:** `scripts/medir_velocidad_cpu_rtdetrv3.py` (nuevo), en el equipo local (AMD Ryzen 5 3600, 6 núcleos y 12 hilos, Windows), con el Python 3.12.9 de `C:\Users\Martin\venv_paddle`: PaddlePaddle 3.3.0 para CPU y las 67 versiones de `requisitos_rtdetrv3.txt`. Código oficial en el commit `349e7d9`, con el parche de D24, que no actúa al detectar.
- **Entorno local:** creado con `py -3.12 -m venv` y `pip install --only-binary=:all: -r docs\modelado\rtdetrv3\requisitos_rtdetrv3.txt paddlepaddle==3.3.0`. Todo tenía rueda para Windows en PyPI y `pip check` salió limpio.
- **Protocolo**, el de YOLO26 del 30 de septiembre:
  - Las mismas 100 fotos, tomadas de `velocidad_cpu_e1_n_deg0_s0.csv` en su orden (el sorteo con la semilla 0 dio las mismas), tras 10 de calentamiento, una por una.
  - Con la librería de RT-DETRv3, en modo dinámico y sin exportar el modelo: el mismo código que produjo las detecciones de validación. Preproceso del `TestReader` oficial (640 × 640 sin conservar la proporción).
  - 8 hilos, como Ultralytics con YOLO26. Paddle trae 1 por defecto, y `libiomp5md.dll` pasó de 1 a 8. oneDNN sin activar, que es como viene Paddle (activarlo no cambió el tiempo de una convolución en una prueba en Linux).
  - Umbral de la regla B de E6: 0.842.
- **Control:** la confianza máxima de cada foto coincide con la de la evaluación en GPU (`eval_e7/umbral_imagenes.csv`): diferencia mediana de 0.0000 y máxima de 0.0003, sin ninguna foto por encima de 0.01. Los pesos y el preproceso son los de la evaluación.
- **Resultados** (E6, `e6_rtdetrv3r18_s0_best.pdparams`):

| Tiempo por foto (ms) | Mediana | p95 | Máximo |
|---|---|---|---|
| Total, ENA24 | 1417.7 | 1458.9 | 1490.4 |
| Total, iNaturalist | 1398.0 | 1444.1 | 1449.8 |
| Total, todas | 1410.0 | 1450.1 | 1490.4 |
| Lectura del JPEG | 22.6 | 36.5 | 81.0 |
| Preproceso | 5.0 | 6.1 | 7.1 |
| Red, sin posproceso | 1378.0 | 1420.7 | 1454.3 |
| Posproceso | 0.7 | 0.8 | 1.0 |

  - Carga del modelo: 1.49 s; primera foto: 1.52 s.
  - RAM (RSS): 21 MB al arrancar, 327 tras importar, 427 con el modelo cargado y 460 como máximo durante la medición. Pico del proceso (Windows): 614 MB.
  - 56 de las 100 fotos tienen alguna detección con confianza ≥ 0.842. La red entrega siempre 300 cajas.

Comparación con YOLO26 (mismas fotos, mismo protocolo):

| Modelo | Total, mediana (p95) | Red, mediana | Pico de RAM |
|---|---|---|---|
| E1 YOLO26n | 51.9 ms (73.1) | 34.4 ms | 412 MB |
| E3 YOLO26s | 81.8 ms (108.4) | 67.6 ms | 492 MB |
| E4 YOLO26m | 183.1 ms (225.8) | 169.3 ms | 597 MB |
| E6 RT-DETRv3-R18 | 1410.0 ms (1450.1) | 1378.0 ms | 614 MB |

- **Lectura:**
  - No cumple la meta de menos de 1 s por foto: las 100 fotos pasan de 1.33 s. Cumple la de RAM, con un pico casi igual al de YOLO26m.
  - Frente a YOLO26m, con los mismos parámetros, es unas 8 veces más lento (7.7 en el total y 8.1 en la red). Frente a E1, unas 27 veces en el total.
  - Una carpeta de 10 000 fotos tardaría unas 3.9 horas con RT-DETRv3 y unos 9 minutos con E1.
  - Casi todo el tiempo es la red: la lectura y el preproceso suman unos 28 ms.
- **Salvedad:** la diferencia mezcla arquitectura y librería (Paddle en modo dinámico frente a PyTorch con Ultralytics, que además prepara la red: fusiona capas y elige el formato de memoria). Los GFLOPs publicados son del mismo orden (60 de RT-DETRv3-R18 según su README y 68.1 de YOLO26m según Ultralytics), aunque puede que no se cuenten igual. Ninguno de los dos se midió en un formato optimizado (exportado, ONNX u OpenVINO). Se declara así.
- **Indicio, sin confirmar en el equipo del estudiante:** en el entorno de Claude (2 núcleos, 2 hilos, pesos aleatorios), el tiempo de RT-DETRv3 se reparte así: backbone ResNet18, 57 %; codificador híbrido, 22 %; decodificador, 21 %; cabeza y posproceso, 0.1 %. Lo más lento es la parte convolucional, no la atención. Si se confirma con 8 hilos en el Ryzen, buena parte de la diferencia con YOLO26m vendría de cómo Paddle ejecuta las convoluciones en CPU en modo dinámico.
- **Decisión:** D19 se mantiene (revisión del 7 de octubre de la ficha). RT-DETRv3 queda como arquitectura de comparación.
- **Archivos:** `docs/modelado/velocidad_cpu_e6_rtdetrv3r18_s0.csv`.

GPU consumida en RT-DETRv3: unas 4 horas hasta el lanzamiento de E6 (entorno, E6v y sondeo), unas 15.4 en E6 y unas 16.1 en E6b. En E7, unas 5.7. E7b no se entrenó.

---