# Preguntas Pendientes

Listado y seguimiento de dudas, preguntas de investigación, requerimientos por clarificar y bloqueantes.

Última actualización: 15 de noviembre de 2026, al cerrar la conversación de la pregunta 33.

---

Estados posibles: Pendiente, En curso, Resuelta, Descartada. El 16 de octubre hubo reunión con el director, y desde entonces le ha dicho al estudiante que tome las decisiones que crea mejores para el proyecto. La única instrucción explícita sigue siendo la de su correo del 7 de octubre: el conjunto de prueba solo al final, para las métricas finales. Por eso, el 15 de noviembre, las preguntas metodológicas que tenían una decisión registrada se cerraron con su ficha, y las demás pasaron a tener al estudiante como responsable. Las de material y logística se llevan ahora con la Prefectura de Imbabura, que maneja el material desde la reunión del 4 de noviembre.

## Abiertas

| ID | Pregunta / Asunto | Área | Estado | Responsable / Fuente | Resolución |
|---|---|---|---|---|---|
| 1 | ¿Existe material ya capturado por las cámaras trampa y quién lo custodia? | Logística | En curso | Estudiante / Asesor / Coordinación de la carrera | 4 nov: el material lo maneja la Prefectura de Imbabura, con más de 40 cámaras en la provincia (no solo en Angochagua), y está dispuesta a compartirlo con una solicitud formal. Por indicación del director, el oficio sale de la Coordinación de la carrera; el estudiante lo pidió: material de Angochagua o, si no se puede separar, el de toda la provincia. Al 15 nov, a la espera |
| 2 | ¿De quién son las cámaras? El anteproyecto dice propiedad institucional y el estado del proyecto decía propiedad de la comunidad o parroquia de Angochagua | Logística | En curso | Prefectura de Imbabura | Desde el 4 nov ya no condiciona a quién se pide el material (a la Prefectura). Queda por saber si las 15 cámaras del anteproyecto son parte de las de la Prefectura, para escribirlo bien en la tesis |
| 3 | ¿Qué modelo de cámara trampa se usa? Define si la fecha y hora vienen en EXIF o solo grabadas sobre la imagen, y qué metadatos traen los MP4 | Técnica / Logística | Pendiente | Prefectura (Andrés Laguna) | Preguntarlo al recibir el material, o verlo en sus metadatos |
| 4 | ¿Existe una tabla que asocie el código de cada cámara con sus coordenadas? La aplicación debe mostrar ese dato y no viene en los archivos | Logística | Pendiente | Prefectura | |
| 6 | ¿Quién asume la logística tras la renuncia del ingeniero y hay una fecha estimada? | Logística | En curso | Estudiante / Asesor | La gestión la lleva el estudiante con el asesor, ante la Prefectura. Sin fecha: depende del oficio y de la respuesta |
| 7 | ¿Cuántas imágenes se esperan y en qué proporción de día y noche? Sin ese dato, las metas de la iteración 2 son una conjetura | Metodología | Pendiente | Prefectura | |
| 9 | ¿Se mantiene el 92.6 % de mAP de la referencia [4] como rendimiento mínimo esperado, sabiendo que proviene de un dataset mayor y de otras especies? | Metodología | Pendiente | Estudiante | Decidir en el protocolo de la evaluación final (D27) cómo se presenta. No es comparable con la iteración 1 |
| 10 | ¿Qué precisión mínima le resulta tolerable a quien revisará el material con la aplicación? | Metodología | Pendiente | Prefectura / Asesor | Debería responderlo quien revisa el material a mano |
| 13 | El proxy no tiene imágenes vacías de cámara trampa, que en el material real son la mayoría | Técnica | Pendiente | Estudiante | El material de la Prefectura traerá vacías reales. WCS (29) queda como alternativa si no llega |
| 16 | ¿Vale la pena una segunda pasada sobre las 111 imágenes marcadas como duda? | Técnica | Pendiente | Estudiante | Opcional, solo si sobra tiempo |
| 17 | El anteproyecto y el marco mencionan Google Colaboratory, pesos de ImageNet y LabelImg. Corresponde corregirlos por Kaggle, pesos de COCO y X-AnyLabeling, e informarlo al director | Redacción | Pendiente | Estudiante | Es la actividad 1.7 de la matriz de Titulación 2. Falta averiguar si el anteproyecto aprobado se puede modificar o si basta con corregir el capítulo I y declarar los cambios en la tesis |
| 19 | En la iteración 2, ¿se reserva alguna cámara completa para prueba, aun a costa de tener menos eventos de oso para entrenar? | Metodología | Pendiente | Estudiante | Depende del material (preguntas 7 y 34) |
| 20 | ¿Un modelo entrenado con imágenes CC BY-NC hereda la restricción de uso no comercial? | Legal / Redacción | Pendiente | Director / asesoría legal | No es una decisión técnica: la delegación no la resuelve. No afecta al uso académico. La licencia de ENA24 (CDLA-Permissive-1.0) no impone restricciones sobre los resultados, así que la duda se limita a las fotos CC BY-NC de iNaturalist |
| 27 | ¿Hay tapir de montaña, ganado oscuro o perros en la zona de las cámaras? En validación son los negativos que más falsas alarmas dan | Datos | Pendiente | Prefectura (Andrés Laguna) | Si los hay, incluirlos como negativos en la iteración 2 |
| 28 | ¿El Kaggle Dataset `deteccionoso` y el repositorio son privados? Si alguno es público, se publican imágenes y hacen falta los avisos de licencia | Legal | Pendiente | Estudiante | Las 80 imágenes de `docs/preparacion_datos/muestra_cajas` entran en git |
| 29 | ¿Se usan las fotos de oso andino de WCS Camera Traps (362 con oso, 41 sin oso y 59 dudosas; Bolivia, 8 cámaras, 13 combinaciones de cámara y día)? Como evaluación externa de la iteración 1, como plan B de la iteración 2, o ambas | Metodología / Datos | Pendiente | Estudiante | Decidir en el protocolo de la evaluación final si entra como evaluación externa: es la única medida disponible con oso andino en cámara trampa y con cámaras nuevas. El plan B pierde peso si llega el material de la Prefectura. Detalle en `docs/wcs/informe_wcs_oso_andino.md` |
| 30 | En el control de mAP50 del 2 de octubre, la ablación difiere del mAP50 de Ultralytics más que YOLO26 (hasta 0.035). ¿Por qué? | Técnica | Pendiente | Estudiante | Opcional. Decidir en el protocolo si se explica antes de tocar prueba o se descarta. No afecta al recall por imagen ni a las falsas alarmas |
| 34 | Si llega material de otras zonas de Imbabura además de Angochagua, ¿cómo se usa? | Metodología | Pendiente | Estudiante | Nueva (15 nov). Idea para decidir al recibirlo: entrenar con todo y reservar cámaras de Angochagua para prueba, para que las métricas finales sean del lugar del título (relacionada con la 19) |

## Resueltas

| ID | Pregunta / Asunto | Área | Fecha | Resolución |
|---|---|---|---|---|
| 14 | ¿Se conservaron los archivos GIF originales en `data/raw` al extraer los fotogramas, o se borraron? | Técnica | 7 sep | Se conservaron. Los 8 archivos siguen en `data/raw` y aparecen como pendientes de revisión en la tabla consolidada, que es lo correcto: quedan fuera del dataset sustituidos por sus 20 fotogramas. Lo que sí se había borrado eran tres copias de la carpeta de revisión, que se recuperaron desde `data/raw` y se registraron como duda |
| 15 | ¿Conviene refinar el agrupamiento de duplicados con distancia de Hamming entre hashes? | Técnica | 10 sep | Sí, y no era opcional. Con hash exacto quedaban 1999 pares de imágenes casi idénticas repartidas entre conjuntos distintos, que afectaban a 408 de las 963 imágenes de validación y prueba. Se adoptó la unión por distancia menor o igual a 12, verificada con `fugas_hamming.py`. Ficha del 10 de septiembre en `registro_decisiones.md` |
| 12 | ¿Bajo qué licencia se distribuye ENA24? Falta anotarla en el informe de comprensión de los datos y en el anexo de licencias | Técnica | 30 sep | CDLA-Permissive-1.0, verificada en lila.science. Ver la revisión del 30 de septiembre de la ficha de fuentes con licencia rastreable en `registro_decisiones.md`. Falta reescribir con voz propia el párrafo del informe de comprensión |
| 22 | ¿Conviene correr el experimento de ablación (solo iNaturalist, solo ENA24 y ambas fuentes)? | Técnica | 2 oct | Sí, y se hizo (`experimentos.md`). Sin ENA24, el modelo no alcanza la meta en cámara trampa (como máximo 0.85, con 36 % de falsas alarmas). Sin iNaturalist rinde parecido en cámara trampa, pero con fotos de oso andino y de otras especies falla: detecta 0.53–0.59 y da 20–23 % de falsas alarmas. Se conservan ambas fuentes en la iteración 2 (D21). La pregunta hablaba de evaluar sobre prueba: se hizo sobre validación y se repetirá en la evaluación final |
| 11 | ¿Qué implementación de RT-DETRv3 se acepta para la comparación, si la oficial en PaddlePaddle no llega a funcionar en el entorno disponible? | Técnica | 5 oct | La oficial funciona en Kaggle (D22), así que el caso que planteaba la pregunta no se dio y no hizo falta el RT-DETR de Ultralytics. Lo que queda por validar con el director es su configuración adaptada a dos T4 (pregunta 31) |
| 32 | El mAP50-95 de PaddleDetection (pycocotools) y el de Ultralytics no se calculan igual. ¿Cómo se comparan RT-DETRv3 y YOLO26? | Técnica | 6 oct | Con pycocotools 2.0.11 sobre las detecciones de todos los modelos, en el cuaderno de evaluación versión 3 (D26). El cálculo propio del cuaderno y la cifra de cada librería quedan como control. pycocotools repite el mAP de PaddleDetection (+0.0001). Para YOLO26, el de Ultralytics sale entre 0.008 y 0.019 más alto |
| 26 | YOLO26m no convergió en 100 épocas. ¿Hay que entrenarlo más tiempo para compararlo con RT-DETR? | Técnica | 7 oct | Se entrenó E7 (200 épocas) por la regla de D25. Medium se estabiliza desde la época 160, más o menos, al nivel en que terminaron E4 y E4b. Su mAP50-95 global con pycocotools es 0.748, frente a 0.745 de E4 y E4b y 0.829 de RT-DETRv3: la ventaja de RT-DETRv3 no venía solo de la falta de convergencia. E7b no se entrena (ficha D25) |
| 33 | ¿La ventaja de RT-DETRv3 en mAP50-95 se mantiene si se compara con pocas cajas por foto? | Técnica | 15 nov | Sí. Con las 3 cajas de mayor confianza por foto, la media de RT-DETRv3 queda 0.0804 por encima de la de YOLO26m (margen de 0.015); con una sola, 0.071. La variante con el mismo filtro para todos da la misma lectura. Se reporta el mAP estándar (ficha D26; `experimentos.md`) |
| 5 | ¿Se valida el uso del dataset proxy y la estrategia de dos iteraciones de CRISP-DM? | Metodología | 15 nov | El director se mostró conforme por correo (7 oct) y delegó las decisiones en el estudiante. Se mantiene D1 |
| 8 | ¿Se acepta el recall como métrica principal, con la justificación del costo asimétrico de los errores? | Metodología | 15 nov | Decisión del estudiante con la delegación del director: se mantiene (criterios de éxito, apartado 0) |
| 18 | ¿Se acepta la mitigación de la correlación entre fuente y etiqueta (negativos de iNaturalist y métricas por fuente)? | Metodología | 15 nov | Se mantiene (D14 y D15) y se declara como limitación |
| 21 | ¿Se acepta que los criterios numéricos se lean sobre la partición de ENA24 y no sobre la global? | Metodología | 15 nov | Se mantiene (D15) |
| 23 | ¿Se acepta la regla B para el umbral de confianza? | Metodología | 15 nov | Se mantiene (D18) |
| 24 | ¿Se acepta YOLO26n (E1) como modelo de la aplicación? | Metodología | 15 nov | Se mantiene (D19): empata en la métrica principal, cumple la meta de velocidad (52 ms por foto) y no necesita otro Python. RT-DETRv3 tarda 1.41 s por foto en la misma CPU |
| 25 | ¿Se acepta la desviación de haber descartado la rotación prometida en el anteproyecto? | Metodología / Redacción | 15 nov | Se mantiene y se declara (D16) |
| 31 | ¿Se acepta RT-DETRv3 con la configuración adaptada a dos T4 (lote total 8, 100 épocas, pesos COCO y el parche para lotes sin cajas)? | Metodología | 15 nov | Se mantiene (D22 a D24) y se declaran las desviaciones |

## Notas sobre preguntas que cambiaron de forma

La pregunta 13, sobre las imágenes vacías de cámara trampa, se mantiene abierta pero con una decisión provisional tomada durante la fase de preparación: no resolverla en la iteración 1. El motivo es que una imagen vacía es muy específica de su cámara y de su fondo fijo, de modo que unos cientos de vacías de Norteamérica enseñarían bastante menos que las de Angochagua, y además la descarga selectiva desde LILA BC exige averiguar cómo obtener un subconjunto sin bajar archivos de decenas de gigabytes, algo que no se ha investigado. La consecuencia queda declarada como limitación: en la iteración 1 no se puede medir la tasa de falsos positivos sobre escena vacía.

La pregunta 13 cambió el 2 de octubre. La nota anterior decía que la descarga selectiva desde LILA BC no se había investigado. Con WCS se resolvió: cada foto se baja por HTTP con su ruta. Lo que queda es decidir si vale la pena, junto con la pregunta 29.

La pregunta 16 decía 108 dudas y ahora son 111, por las tres imágenes recuperadas que se describen en la resolución de la pregunta 14.

La pregunta 17 incorpora LabelImg, que en la versión anterior de este listado aparecía solo como una observación del estado del proyecto. La herramienta quedó descartada por un fallo al crear cajas en versiones recientes de Python, así que la corrección en el anteproyecto ya no es opcional.

La pregunta 11 se cerró el 5 de octubre porque la implementación oficial de RT-DETRv3 funcionó. La parte que sí necesita al director, aceptar la configuración con que se entrena en Kaggle, pasó a la pregunta 31.

La pregunta 31 perdió su segunda parte el 6 de octubre: con E6 y E6b terminadas, ya no se plantea reportar una sola semilla de RT-DETRv3.

La pregunta 26 se cerró el 7 de octubre con E7 y las reglas de la ficha D25. La pregunta 33 nació el mismo día, al revisar si el resultado de RT-DETRv3 era el esperado.

El 15 de noviembre se cerraron ocho preguntas metodológicas que ya tenían una decisión registrada (5, 8, 18, 21, 23, 24, 25 y 31), además de la 33. Se cierran con la decisión del estudiante, tomada con la delegación del director; si el director o el tribunal plantean otra cosa, se reabren. Las de material (1 a 4, 6, 7, 10 y 27) pasan a tener a la Prefectura de Imbabura como fuente. Se añadió la 34.
