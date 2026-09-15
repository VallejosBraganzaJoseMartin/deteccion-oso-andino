# Informe de comprensión de los datos (CRISP-DM, iteración 1: dataset proxy)

Nota sobre las cifras. Este informe se escribió al cerrar la fase, el 3 de septiembre, y se corrigió el 10 de septiembre con los datos definitivos que salieron al consolidar la revisión manual. Los números que cambiaron aparecen ya corregidos, y donde el cambio tiene alguna implicación se explica en el texto. La diferencia principal es que el conteo original trataba los ocho archivos GIF como una sola imagen cada uno, cuando en realidad aportaron veinte fotogramas.

## 1. Contexto

El material de las cámaras trampa de Angochagua no está disponible durante el periodo de vacaciones (agosto y septiembre de 2026) por la renuncia del ingeniero responsable en la comunidad. Por indicación verbal del director, la primera iteración de CRISP-DM se realiza con un dataset proxy construido a partir de fuentes públicas, y el material real se incorporará en una segunda iteración mediante ajuste fino del modelo ya entrenado. Esta decisión se apoya en la referencia [16] del marco teórico (Shepley et al., 2021), que obtuvo detectores competitivos entrenando con imágenes de Flickr e iNaturalist y evaluando sobre cámaras trampa.

## 2. Recolección inicial

Se usaron dos fuentes, cada una para cubrir una parte del problema.

**iNaturalist.** Fotografías de *Tremarctos ornatus* descargadas a través de la API pública (v1), con los filtros de grado de investigación, observaciones no marcadas como cautivas y licencias CC0, CC BY o CC BY-NC. Aporta los rasgos visuales de la especie, sobre todo la máscara facial y el pelaje, en entornos naturales. Por cada foto se guardó la licencia, la atribución que exige iNaturalist, el usuario, la fecha y las coordenadas en `registro_licencias.csv`.

**ENA24 (LILA BC).** Dataset de cámaras trampa del este de Norteamérica con cajas delimitadoras ya anotadas en formato COCO Camera Traps. No contiene oso andino. Se incluye porque aporta lo que iNaturalist no tiene: el dominio de la cámara trampa, con más de la mitad de las imágenes en infrarrojo nocturno, y una especie de pelaje oscuro (oso negro americano) que sirve como aproximación visual. Aporta además negativos difíciles del mismo dominio, en particular perros y coyotes.

En la fase de preparación se añadió una tercera fuente, que no existía cuando se escribió este informe: fotografías de otras especies descargadas del mismo iNaturalist para usarlas como negativos. El motivo y el detalle están en `informe_preparacion_datos.md` y en la ficha del 9 de septiembre del registro de decisiones.

## 3. Descripción de los datos

### iNaturalist

Se obtuvieron 1424 archivos de imagen correspondientes a 693 observaciones. Una observación agrupa varias fotos del mismo encuentro (2.05 fotos en promedio, con un máximo de 20), y 253 observaciones tienen más de una foto. Este dato importa para la división posterior en entrenamiento, validación y prueba, que deberá hacerse por observación y no por foto.

Licencias: 1287 fotos CC BY-NC, 101 CC BY y 36 CC0. Las tres permiten el uso académico con atribución. Procedencia: 1064 fotos tienen coordenadas dentro de Ecuador continental y 17 no tienen coordenadas. El campo de texto libre "lugar" solo identifica 631 como ecuatorianas, así que las coordenadas son el dato confiable para describir la procedencia.

Resolución mediana de 0.70 megapíxeles, con 1024x683 como tamaño más frecuente, lo que corresponde al tamaño "large" de iNaturalist. Orientación: 982 horizontales, 423 verticales y 19 cuadradas. Solo 4 imágenes están en escala de grises y 26 tienen brillo medio bajo (menor a 60 sobre 255), es decir, casi no hay fotos nocturnas de oso andino en esta fuente.

Sobre las verticales conviene aclarar algo, porque durante la preparación surgió la duda: no hace falta descartarlas ni recortarlas. Ultralytics redimensiona cada imagen conservando la proporción hasta que el lado mayor alcanza la resolución de entrada y rellena el resto del lienzo, de modo que una fotografía vertical entra sin deformarse, solo ocupando menos superficie útil.

### ENA24

En disco hay 8789 imágenes. El JSON de anotaciones referencia 9676 imágenes, 11 596 cajas y 23 categorías. Las 887 imágenes que faltan coinciden exactamente con la categoría "Human", que LILA BC excluye de la descarga por privacidad. No se perdió nada relevante para el oso, aunque sí para algunos negativos, como se explica más abajo.

Resolución mediana de 3.15 megapíxeles, con 2048x1536 como tamaño más frecuente. Todas las imágenes son horizontales. 4686 (53.3 %) están en escala de grises, lo que corresponde a disparos infrarrojos nocturnos, y 1733 tienen brillo medio bajo.

La categoría "American Black Bear" tiene 893 imágenes y 959 cajas, y todas están en disco. El área relativa de la caja de oso tiene una mediana de 12.6 % de la imagen. Hay 56 cajas menores al 1 % (oso lejano) y 312 mayores al 25 % (oso muy cerca de la cámara).

Los conteos por categoría del JSON no coinciden con lo que hay en disco, y la diferencia importa a la hora de elegir negativos. Una imagen puede tener anotaciones de dos categorías, así que las que contienen una persona se excluyeron de la descarga aunque también aparezca otro animal. El caso extremo es el caballo: el JSON declara 338 imágenes y en disco quedan 60, porque casi siempre hay alguien montándolo. Las cifras reales disponibles para negativos son perro 703, gato doméstico 482, venado 350, coyote 334, lince rojo 328 y caballo 60. No hay imágenes sin anotación, así que ENA24 no aporta imágenes vacías de cámara trampa.

El JSON no incluye campos de ubicación, secuencia ni fecha, solo nombre de archivo y dimensiones. Esto se compensa con los grupos de duplicados descritos en la exploración.

## 4. Exploración

### Duplicados y ráfagas

El hash perceptual encontró 1009 grupos de imágenes casi idénticas, con 2902 archivos implicados. En iNaturalist son solo 30 archivos en 15 grupos, seguramente fotos repetidas entre observaciones. En ENA24 son 2872 archivos en 994 grupos, con un grupo de 55 imágenes idénticas y varios de 10 a 15. Corresponden a ráfagas: varios disparos seguidos de la misma escena tras una activación del sensor. Como el JSON no trae secuencias, estos grupos por hash se usarán como pseudosecuencias para que las imágenes de una misma ráfaga queden siempre del mismo lado en la división de datos.

Aquí quedó anotado como pendiente refinar el agrupamiento con una distancia de Hamming entre hashes, porque el criterio del hash exacto solo captura coincidencias perfectas. El pendiente se resolvió en la fase de preparación, y resultó no ser un refinamiento opcional: midiendo las fugas sobre la primera división construida aparecieron 1999 pares de imágenes casi idénticas repartidas entre conjuntos distintos, que afectaban a 408 de las 963 imágenes de validación y prueba. El detalle está en el informe de preparación.

### Detección zero-shot con YOLO26n preentrenado en COCO

COCO incluye la clase "bear", así que el modelo YOLO26n de Ultralytics detecta osos sin haber sido entrenado con oso andino. Se corrió sobre las 1424 fotos de iNaturalist con umbral de confianza 0.05 para ver también las detecciones dudosas. Resultado: 761 imágenes con oso detectado con confianza mayor o igual a 0.5, 285 con confianza menor a 0.5 y 378 sin ninguna detección de oso. Este resultado sirve como referencia de partida sin entrenamiento y sirvió para dirigir la revisión manual hacia las imágenes donde el modelo falló o dudó.

Al consolidar la tabla de estados, el recuento de imágenes de confianza alta dio 763 en lugar de 761. Los dos casos de diferencia deben estar en el borde del umbral y lo más probable es que se trate del redondeo con el que el script escribe la confianza en el CSV, pero no se ha comprobado. No afecta a nada, porque esas imágenes se revisaron una a una durante el etiquetado como todas las demás.

[[Pendiente, opcional: correr la misma exploración sobre ENA24 y comparar con las cajas reales del JSON, lo que daría el mismo valor de referencia pero sobre imágenes de cámara trampa.]]

## 5. Verificación de calidad

No se encontraron archivos corruptos en ninguna de las dos fuentes.

Se revisaron a mano dos lotes de iNaturalist: las imágenes donde el modelo de COCO no vio oso y aquellas donde lo vio con confianza baja. Cada imagen se asignó a una de cinco categorías: oso visible, oso en cautiverio evidente, rastro (huellas, excrementos, restos), sin oso, y duda.

| Lote | Revisadas | Oso | Cautivo | Rastro | Sin oso | Duda |
|---|---|---|---|---|---|---|
| Sin detección (conf. 0) | 381 | 109 | 0 | 110 | 65 | 97 |
| Confianza baja (< 0.5) | 292 | 266 | 1 | 8 | 3 | 14 |
| Total revisado | 673 | 375 | 1 | 118 | 68 | 111 |

Las imágenes con confianza alta no se revisaron una a una en esta fase, porque en la fase de etiquetado cada caja se corregiría manualmente y todas pasarían por revisión. Así ocurrió.

Durante la revisión se detectó que 8 archivos con extensión .jpg eran en realidad GIF animados. De cada uno se extrajeron entre uno y tres fotogramas con el oso en posiciones distintas, 20 en total, nombrados con el prefijo del archivo original más `_frame_NNN` para conservar el vínculo con la observación y con el registro de licencias. Los ocho archivos originales se conservan en `data/raw` intactos y quedan fuera del dataset, sustituidos por sus fotogramas. Por eso el número de imágenes revisadas (673) supera al de archivos descargados que se revisaron.

Tres imágenes del lote de confianza baja se borraron de la carpeta de revisión durante el trabajo, antes de quedar clasificadas, porque no encajaban con claridad en ninguna categoría. Como `data/raw` permanece intacta, se recuperaron desde ahí y se registraron como duda, que es lo que corresponde cuando el revisor no puede decidir. Ese es el motivo de que las dudas pasaran de 108 a 111. El episodio confirma el valor de la decisión de no tocar nunca la carpeta de origen.

Balance de iNaturalist tras consolidar: 1138 imágenes con oso visible (375 revisadas más 763 de confianza alta), repartidas en 561 observaciones distintas, de las cuales 869 tienen coordenadas dentro de Ecuador continental. Licencias del subconjunto positivo: 1037 CC BY-NC, 76 CC BY y 25 CC0. Quedan fuera 118 rastros, 68 imágenes sin oso, 111 dudas y 1 de cautiverio.

Las 109 fotos de oso que el modelo de COCO no detectó son las más valiosas del conjunto: osos lejanos, en penumbra o parcialmente tapados por vegetación, que es el tipo de caso que el modelo entrenado deberá resolver.

## 6. Decisiones tomadas en esta fase

Cada una es revertible y queda pendiente de validación con el director al retomar clases. Las que cambiaron después llevan la nota correspondiente, y todas tienen ficha en `registro_decisiones.md`.

1. Se excluyen del dataset de la iteración 1 los rastros, las imágenes sin oso, las dudas y la de cautiverio. *(Revisada el 9 de septiembre: los rastros y las imágenes sin oso vuelven al dataset como negativos, con archivo de etiquetas vacío. Las dudas y el cautiverio siguen fuera.)*
2. La división en entrenamiento, validación y prueba se hará por `observacion_id` en iNaturalist y por grupo de hash perceptual en ENA24. *(Revisada el 10 de septiembre: al grupo por observación se suma la unión de hashes a distancia de Hamming menor o igual a 12.)*
3. Las imágenes con detección de confianza alta se aceptan provisionalmente como positivas y se verifican durante el etiquetado.
4. Los fotogramas extraídos de GIF se tratan como fotos de la misma observación.
5. Los negativos de la iteración 1 saldrán de ENA24, priorizando perro, caballo y otras especies de tamaño medio y pelaje oscuro. *(Ampliada el 9 de septiembre con negativos de iNaturalist, por el problema de correlación entre fuente y etiqueta.)*
6. La carpeta `data/raw` se mantiene intacta como bajó de cada fuente. Todo lo derivado (fotogramas, copias de revisión, dataset final) vive en otras carpetas.

## 7. Observaciones de la revisión manual

Lo que sigue no sale de ningún script, sino de haber mirado las 673 imágenes de los dos lotes.

Dentro de la categoría rastro hay bastante más variedad de lo que sugiere el nombre. La mayoría son arañazos en troncos de árboles, que es la marca que el oso deja al trepar, y luego huellas en suelo blando y excrementos. Muchas incluyen objetos puestos ahí para dar escala: monedas, cintas métricas, la mano del observador. Son fotografías de trabajo de campo, tomadas de cerca y apuntando al suelo o al tronco, un encuadre que no se parece en nada al de una foto de oso.

En la categoría sin oso también aparecieron cosas que no esperaba. Además de paisajes y árboles, que era lo previsible, hay fotografías de pantallas de monitor y de la pantalla trasera de una cámara donde sí se ve un oso, y algún montaje tipo collage. Al principio las clasifiqué como sin oso porque en la escena real no hay ningún animal, pero conviene tratarlas aparte: el oso está en la imagen aunque sea a través de una pantalla, y usarlas como negativos le enseñaría al modelo justo lo contrario de lo que se busca. En la fase de preparación se separaron y quedaron fuera del dataset por completo.

Las dudas son casi siempre el mismo tipo de caso: una mancha oscura entre vegetación densa o en penumbra que podría ser un oso, o podría ser una sombra, una roca o un tronco. En ninguna de ellas pude decidir con confianza, y esa es exactamente la razón por la que no sirven para entrenar ni para evaluar.

Un último detalle que apareció al revisar: hay observaciones cuyas fotos son casi idénticas entre sí, tomadas en ráfaga o recortes distintos de la misma imagen. Refuerza la decisión de dividir por observación, porque tratarlas como imágenes independientes sería engañarse.

## 8. Vacíos identificados

- El proxy no tiene imágenes vacías de cámara trampa (fondo sin animal), que son la mayoría del material real según el marco teórico. La decisión que se tomó en la fase de preparación fue no resolverlo ahora: las vacías son muy específicas de cada cámara y su fondo fijo, así que quinientas vacías de Norteamérica enseñan bastante menos que quinientas de Angochagua. La consecuencia es que en la iteración 1 no se puede medir la tasa de falsos positivos sobre escena vacía, que es el número que más le importará al investigador. Queda cubierto por el umbral de confianza ajustable y por la iteración 2.
- Casi no hay fotos nocturnas de oso andino (4 en escala de grises). Las nocturnas del dataset son todas de oso negro americano.
- ENA24 no trae metadatos de secuencia ni ubicación, lo que obliga al agrupamiento por hash. Ninguna agrupación por hash evita que la misma cámara aparezca en entrenamiento y en prueba con el mismo fondo, de modo que las métricas de la iteración 1 no miden generalización a cámaras nuevas. La referencia [15] documenta esa caída y hay que declararla como limitación.
- Falta confirmar la licencia exacta de ENA24 en lila.science.

## 9. Siguiente fase

Preparación de los datos: construir el dataset en formato YOLO a partir de las imágenes seleccionadas, etiquetar las de iNaturalist corrigiendo las cajas propuestas por el modelo de COCO, convertir las cajas de ENA24 desde el JSON, definir la división por observación y por pseudosecuencia, y aplicar el aumento de datos previsto en el anteproyecto. Esa fase está cerrada y su entregable es `informe_preparacion_datos.md`.

## Preguntas para el director y el asesor (para el primer día de clases)

El listado completo está en `preguntas_pendientes.md`. Las que nacen de esta fase:

- ¿Existe material ya capturado por las 15 cámaras y quién lo custodia?
- ¿Qué modelo de cámara se usa? Define si la fecha y hora vienen en EXIF o grabadas sobre la imagen, y qué metadatos traen los MP4.
- ¿Existe la tabla de código de cámara con sus coordenadas?
- ¿Se valida el uso del dataset proxy y la estrategia de dos iteraciones?
- ¿Qué implementación de RT-DETRv3 se acepta para la comparación? [[completar cuando se haya probado]]
