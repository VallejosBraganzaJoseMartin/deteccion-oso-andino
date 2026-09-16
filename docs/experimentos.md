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

### 1.5 Corridas planificadas

| ID | Modelo | `degrees` | Épocas | Propósito | Estado |
|---|---|---|---|---|---|
| E0 | YOLO26n | 0 | 10 a 20 | Verificar el pipeline de punta a punta en Kaggle: lectura de etiquetas, entrenamiento, validación, guardado de pesos. Las métricas no cuentan. Da el tiempo real por época para planificar la cuota | Pendiente |
| E1 | YOLO26n | 0 | 100 | Línea base, candidato de despliegue en CPU | Pendiente |
| E2 | YOLO26n | 10 | 100 | Ablación de rotación contra E1 | Pendiente |
| E3 | YOLO26s | según E1/E2 | 100 | Punto intermedio velocidad/precisión | Pendiente |
| E4 | YOLO26m | según E1/E2 | 100 | Emparejado en parámetros con RT-DETRv3-R18 (~20 M) para la comparación justa | Pendiente, condicionado a la cuota |
| E5 | mejor de E1..E4 | según E1/E2 | 100 | Opcional: `imgsz` 800 o 960 por los osos pequeños de ENA24 | Solo si sobran horas |

Orden de ejecución: E0, E1, E2, E3, E4, E5. Estimación previa de GPU (sin fuente, se corrige con E0): n alrededor de 1 a 1.5 h por corrida, s de 2 a 3 h, m de 4 a 6 h.

RT-DETRv3 tiene su propio apartado en el plan (2.3, límite de dos días, D9) y se registra aquí como E6 en adelante cuando llegue.

### 1.6 Evaluación

- Selección de modelo y de umbrales: solo sobre validación. El conjunto de prueba se usa una vez por modelo final, al terminar. Ninguna decisión de configuración se toma mirando prueba.
- Cada modelo se valida tres veces sobre prueba (global con `dataset.yaml`, ENA24 con `dataset_test_ena24.yaml`, iNaturalist con `dataset_test_inat.yaml`) y las mismas tres sobre validación. Métricas de Ultralytics: precisión, recall, mAP50 y mAP50-95, con los umbrales por defecto de `val` (`conf` 0.001 e `iou` 0.7 para el cálculo de mAP, verificar que sigan siendo esos en 8.4.138 al escribir el cuaderno).
- Recall a nivel de imagen (D15): proporción de imágenes con oso en las que el modelo pone al menos una detección por encima del umbral de operación, y proporción de negativos con alguna detección (falsas alarmas). El umbral de operación se elige sobre validación. Requiere un script propio a partir de las predicciones, pendiente de escribir.
- Además: número de parámetros, tiempo de inferencia por imagen en CPU local, y ruta de los pesos.
- Las métricas de la iteración 1 son preliminares (D1). El resultado principal es el de ENA24 (D15). Limitación declarada: la misma cámara de ENA24 aparece en entrenamiento y prueba con el mismo fondo, así que no se mide generalización a cámaras nuevas (D13).

### 1.7 Qué se anota por corrida

Fecha, ID, modelo y pesos de partida, argumentos que difieren del protocolo, versión de Ultralytics y GPU, optimizador elegido por `auto`, tiempo total y por época, época del mejor `best.pt` según Ultralytics, métricas sobre validación (global y por fuente), ruta de los pesos y del `results.csv`, y cualquier aviso que imprimiera el entrenamiento. Las métricas sobre prueba se añaden solo al cerrar el modelo.

---

## Parte 2. Registro de corridas

### E0. Verificación del pipeline

- Fecha:
- Entorno (GPU, Python, torch, Ultralytics):
- Argumentos: `model=yolo26n.pt epochs=?? imgsz=640 batch=16 seed=0`
- Optimizador elegido por `auto`:
- Tiempo por época:
- Escaneo de etiquetas (imágenes, fondos, corruptas) en train y val:
- Resultado: funciona / no funciona. Avisos:
- Pesos guardados en:

### E1. YOLO26n, línea base

(pendiente)

### E2. YOLO26n, rotación 10°

(pendiente)
