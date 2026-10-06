# Resumen de avance para la reunión con el director

Trabajo de Integración Curricular: detección automatizada del oso andino con Deep Learning en cámaras trampa de Angochagua.
Para: PhD. Iván Danilo García Santillán. Corte: 2 de octubre de 2026. Todas las cifras son sobre el conjunto de validación: el conjunto de prueba no se ha usado.

## 1. Situación

- Sin el material de Angochagua (renuncia del ingeniero), se siguió la indicación de entrenar con imágenes de internet. Dataset proxy de 3832 imágenes y 2207 cajas, con licencias registradas por foto: iNaturalist (oso andino y otras especies de Ecuador) y ENA24 de LILA BC (cámaras trampa con oso negro americano, de día y en infrarrojo).
- CRISP-DM en dos iteraciones: la 1, con el proxy, está hecha hasta la evaluación sobre validación; la 2 será el ajuste fino con el material real.

## 2. Resultados de la iteración 1

| | Modelo COCO sin entrenar* | YOLO26n entrenado (E1) |
|---|---|---|
| Cámara trampa (ENA24): fotos con oso detectadas | 103/156 (0.66) | 149/156 (0.96) |
| … de ellas, de noche (infrarrojo) | 62/104 (0.60) | 102/104 (0.98) |
| Cámara trampa: falsas alarmas | 18/182 (0.10) | 3/182 (0.02) |
| Fotos de iNaturalist: osos detectados | 135/170 (0.79) | 155/170 (0.91) |

\* Con su mejor umbral sin pasar del 10 % de falsas alarmas. No alcanza la meta de 0.95 con ningún umbral.

- **Tamaño del modelo:** nano, small y medium empatan (mAP50-95 medio de 0.758, 0.751 y 0.753 con dos semillas cada uno). Se eligió nano (E1): 52 ms por foto y 412 MB de RAM en un equipo sin GPU.
- **Umbral:** 0.363, elegido para avisar en el 95 % de las fotos con oso de cámara trampa (regla B, D18). El 0.96 de la tabla lo fija esa regla; la cifra honesta saldrá del conjunto de prueba.
- **Ablación por fuente:** sin fotos de cámara trampa, el modelo no alcanza la meta en cámara trampa (como máximo 0.85, con 36 % de falsas alarmas). Sin iNaturalist, en cámara trampa rinde parecido, pero en fotos de oso andino detecta solo la mitad y da entre 20 y 23 % de falsas alarmas con otras especies. Conclusión: conservar ambas fuentes en la iteración 2.
- **Errores:** los osos perdidos son sobre todo cercanos, cortados por el borde o tapados, no pequeños (D20). Las falsas alarmas se concentran en animales oscuros y macizos: tapir de montaña, una vaca negra con blanco, cuervos y un gato negro.

## 3. Hallazgo: oso andino en cámaras trampa reales (WCS Camera Traps, LILA BC)

- 462 fotos etiquetadas como *Tremarctos ornatus*, con la misma licencia que ENA24 (CDLA-Permissive-1.0). Revisadas a ojo: 362 muestran el oso, 41 no y 59 son dudosas.
- Vienen de Bolivia: 8 cámaras, 40 secuencias, abril a junio de 2012, que equivalen a 13 combinaciones de cámara y día. 107 fotos traen caja.
- Usos posibles:
  - Evaluación externa del modelo: por imagen, ya; con cajas, habría que dibujar 255.
  - Plan B de la iteración 2.
  - Fuente de fotos vacías reales de cámaras andinas, que el proxy no tiene.

## 4. Decisiones tomadas sin consulta, para validar

Kaggle en lugar de Colab; pesos COCO y no de ImageNet; X-AnyLabeling y no LabelImg; una sola clase `oso` (D11); negativos de iNaturalist (D14); rotación descartada tras probarla (D16); regla B del umbral (D18); YOLO26n como modelo de la aplicación (D19); sin experimento de resolución mayor (D20).

## 5. Preguntas

1. ¿Existe material ya capturado por las 15 cámaras? ¿Quién lo custodia y de quién son las cámaras? (preguntas 1, 2 y 6)
2. ¿Se validan el dataset proxy y la estrategia de dos iteraciones? (5)
3. ¿Se aceptan la regla B del umbral y YOLO26n como modelo de la aplicación? (23 y 24)
4. ¿Cómo se usa WCS: evaluación externa, plan B o ambas? ¿Se descargan también fotos vacías de esas cámaras? (29 y 13)
5. ¿Se mantiene el 92.6 % de mAP de la referencia [4] como mínimo esperado, aunque venga de otro dataset y otras especies? (9)
6. Si la implementación oficial de RT-DETRv3 no corre, ¿se acepta el RT-DETR de Ultralytics? (11)
7. ¿Hay tapir, ganado oscuro o perros en la zona de las cámaras? (27)

La lista completa está en `preguntas_pendientes.md`.

## 6. Próximos pasos

RT-DETR (máximo dos días para la versión oficial, D9), evaluación final sobre el conjunto de prueba una sola vez, aplicación de escritorio, e iteración 2 cuando haya material.
