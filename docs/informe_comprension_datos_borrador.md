# Informe de comprensión de los datos (CRISP-DM, iteración 1: dataset proxy)


## 1. Contexto

El material de las cámaras trampa de Angochagua no está disponible durante el periodo de vacaciones (agosto y septiembre de 2026) por la renuncia del ingeniero responsable en la comunidad. Por indicación verbal del director, la primera iteración de CRISP-DM se realiza con un dataset proxy construido a partir de fuentes públicas, y el material real se incorporará en una segunda iteración mediante ajuste fino del modelo ya entrenado. Esta decisión se apoya en la referencia [16] del marco teórico (Shepley et al., 2021), que obtuvo detectores competitivos entrenando con imágenes de Flickr e iNaturalist y evaluando sobre cámaras trampa.

## 2. Recolección inicial

Se usaron dos fuentes, cada una para cubrir una parte del problema.

**iNaturalist.** Fotografías de *Tremarctos ornatus* descargadas a través de la API pública (v1), con los filtros de grado de investigación, observaciones no marcadas como cautivas y licencias CC0, CC BY o CC BY-NC. Aporta los rasgos visuales de la especie, sobre todo la máscara facial y el pelaje, en entornos naturales. Por cada foto se guardó la licencia, la atribución que exige iNaturalist, el usuario, la fecha y las coordenadas en `registro_licencias.csv`.

**ENA24 (LILA BC).** Dataset de cámaras trampa del este de Norteamérica con cajas delimitadoras ya anotadas en formato COCO Camera Traps. No contiene oso andino. Se incluye porque aporta lo que iNaturalist no tiene: el dominio de la cámara trampa, con más de la mitad de las imágenes en infrarrojo nocturno, y una especie de pelaje oscuro (oso negro americano) que sirve como aproximación visual. Aporta además negativos difíciles del mismo dominio, en particular perros y caballos.

## 3. Descripción de los datos

### iNaturalist

Se obtuvieron 1424 archivos de imagen correspondientes a 693 observaciones. Una observación agrupa varias fotos del mismo encuentro (2.05 fotos en promedio, con un máximo de 20), y 253 observaciones tienen más de una foto. Este dato importa para la división posterior en entrenamiento, validación y prueba, que deberá hacerse por observación y no por foto.

Licencias: 1287 fotos CC BY-NC, 101 CC BY y 36 CC0. Las tres permiten el uso académico con atribución. Procedencia: 1064 fotos tienen coordenadas dentro de Ecuador continental y 17 no tienen coordenadas. El campo de texto libre "lugar" solo identifica 631 como ecuatorianas, así que las coordenadas son el dato confiable para describir la procedencia.

Resolución mediana de 0.70 megapíxeles, con 1024x683 como tamaño más frecuente, lo que corresponde al tamaño "large" de iNaturalist. Orientación: 982 horizontales, 423 verticales y 19 cuadradas. Solo 4 imágenes están en escala de grises y 26 tienen brillo medio bajo (menor a 60 sobre 255), es decir, casi no hay fotos nocturnas de oso andino en esta fuente.

### ENA24

En disco hay 8789 imágenes. El JSON de anotaciones referencia 9676 imágenes, 11 596 cajas y 23 categorías. Las 887 imágenes que faltan coinciden exactamente con la categoría "Human", que LILA BC excluye de la descarga por privacidad. No se perdió nada relevante.

Resolución mediana de 3.15 megapíxeles, con 2048x1536 como tamaño más frecuente. Todas las imágenes son horizontales. 4686 (53.3 %) están en escala de grises, lo que corresponde a disparos infrarrojos nocturnos, y 1733 tienen brillo medio bajo.

La categoría "American Black Bear" tiene 893 imágenes y 959 cajas. El área relativa de la caja de oso tiene una mediana de 12.6 % de la imagen. Hay 56 cajas menores al 1 % (oso lejano) y 312 mayores al 25 % (oso muy cerca de la cámara). Como referencia para negativos: perro 751 imágenes, gato doméstico 491, venado 350, caballo 338, coyote 334, lince rojo 328. No hay imágenes sin anotación, así que ENA24 no aporta imágenes vacías de cámara trampa.

El JSON no incluye campos de ubicación, secuencia ni fecha, solo nombre de archivo y dimensiones. Esto se compensa con los grupos de duplicados descritos en la exploración.

## 4. Exploración

### Duplicados y ráfagas

El hash perceptual encontró 1009 grupos de imágenes casi idénticas, con 2902 archivos implicados. En iNaturalist son solo 30 archivos en 15 grupos, seguramente fotos repetidas entre observaciones. En ENA24 son 2872 archivos en 994 grupos, con un grupo de 55 imágenes idénticas y varios de 10 a 15. Corresponden a ráfagas: varios disparos seguidos de la misma escena tras una activación del sensor. Como el JSON no trae secuencias, estos grupos por hash se usarán como pseudosecuencias para que las imágenes de una misma ráfaga queden siempre del mismo lado en la división de datos. Queda pendiente refinar el agrupamiento con una distancia de Hamming entre hashes, porque el criterio actual solo captura coincidencias exactas.

### Detección zero-shot con YOLO26n preentrenado en COCO

COCO incluye la clase "bear", así que el modelo YOLO26n de Ultralytics detecta osos sin haber sido entrenado con oso andino. Se corrió sobre las 1424 fotos de iNaturalist con umbral de confianza 0.05 para ver también las detecciones dudosas. Resultado: 761 imágenes con oso detectado con confianza mayor o igual a 0.5, 285 con confianza menor a 0.5 y 378 sin ninguna detección de oso. Este resultado sirve como referencia de partida sin entrenamiento y sirvió para dirigir la revisión manual hacia las imágenes donde el modelo falló o dudó.

[[Pendiente, opcional: correr la misma exploración sobre ENA24 y comparar con las cajas reales del JSON, lo que daría el mismo valor de referencia pero sobre imágenes de cámara trampa.]]

## 5. Verificación de calidad

No se encontraron archivos corruptos en ninguna de las dos fuentes.

Se revisaron a mano dos lotes de iNaturalist: las imágenes donde el modelo de COCO no vio oso y aquellas donde lo vio con confianza baja. Cada imagen se asignó a una de cinco categorías: oso visible, oso en cautiverio evidente, rastro (huellas, excrementos, restos), sin oso, y duda.

| Lote | Revisadas | Oso | Cautivo | Rastro | Sin oso | Duda |
|---|---|---|---|---|---|---|
| Sin detección (conf. 0) | 381 | 109 | 0 | 110 | 65 | 97 |
| Confianza baja (< 0.5) | 289 | 266 | 1 | 8 | 3 | 11 |
| Total revisado | 670 | 375 | 1 | 118 | 68 | 108 |

Las 761 imágenes con confianza alta no se revisaron una a una en esta fase, porque en la fase de etiquetado cada caja se corregirá manualmente y todas pasarán por revisión.

Durante la revisión se detectó que 8 archivos con extensión .jpg eran en realidad GIF animados. De cada uno se extrajeron entre uno y tres fotogramas con el oso en posiciones distintas, nombrados con el prefijo del archivo original más `_frame_NNN` para conservar el vínculo con la observación y con el registro de licencias. Por eso el número de imágenes revisadas (670) supera al de imágenes seleccionadas (663).

Balance provisional de iNaturalist: unas 1136 imágenes con oso visible (375 revisadas más 761 de confianza alta), y quedan fuera del dataset 118 rastros, 68 sin oso, 108 dudas y 1 de cautiverio.

Las 109 fotos de oso que el modelo de COCO no detectó son las más valiosas del conjunto: osos lejanos, en penumbra o parcialmente tapados por vegetación, que es el tipo de caso que el modelo entrenado deberá resolver.

## 6. Decisiones tomadas en esta fase

Cada una es revertible y queda pendiente de validación con el director al retomar clases.

1. Se excluyen del dataset de la iteración 1 los rastros, las imágenes sin oso, las dudas y la de cautiverio. Las dudas no se usan ni como positivas ni como negativas: una imagen donde un revisor humano no puede decidir no sirve para entrenar ni para evaluar. Si sobra tiempo se les dará una segunda pasada.
2. La división en entrenamiento, validación y prueba se hará por `observacion_id` en iNaturalist y por grupo de hash perceptual en ENA24.
3. Las 761 imágenes con detección de confianza alta se aceptan provisionalmente como positivas y se verifican durante el etiquetado.
4. Los fotogramas extraídos de GIF se tratan como fotos de la misma observación.
5. Los negativos de la iteración 1 saldrán de ENA24, priorizando perro, caballo y otras especies de tamaño medio y pelaje oscuro. La cantidad se define en la fase de preparación.
6. La carpeta `data/raw` se mantiene intacta como bajó de cada fuente. Todo lo derivado (fotogramas, copias de revisión, dataset final) vive en otras carpetas.

## 7. Vacíos identificados

- El proxy no tiene imágenes vacías de cámara trampa (fondo sin animal), que son la mayoría del material real según el marco teórico. Opciones: buscar otro dataset de LILA BC que las incluya, o cubrirlas en la iteración 2 con el material de Angochagua. [[decidir en la fase de preparación]]
- Casi no hay fotos nocturnas de oso andino (4 en escala de grises). Las nocturnas de ENA24 son de otra especie.
- ENA24 no trae metadatos de secuencia ni ubicación, lo que obliga al agrupamiento por hash.
- Falta confirmar la licencia exacta de ENA24.


## 8. Siguiente fase

Preparación de los datos: construir el dataset en formato YOLO a partir de las imágenes seleccionadas, etiquetar las de iNaturalist corrigiendo las cajas propuestas por el modelo de COCO, convertir las cajas de ENA24 desde el JSON, definir la división por observación y por pseudosecuencia, y aplicar el aumento de datos previsto en el anteproyecto.

## Preguntas para el director y el asesor (para el primer día de clases)

- ¿Existe material ya capturado por las 15 cámaras y quién lo custodia?
- ¿Qué modelo de cámara se usa? Define si la fecha y hora vienen en EXIF o grabadas sobre la imagen, y qué metadatos traen los MP4.
- ¿Existe la tabla de código de cámara con sus coordenadas?
- ¿Se valida el uso del dataset proxy y la estrategia de dos iteraciones?
- ¿Qué implementación de RT-DETRv3 se acepta para la comparación? [[completar cuando se haya probado]]
