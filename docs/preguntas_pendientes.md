# Preguntas Pendientes

Listado y seguimiento de dudas, preguntas de investigación, requerimientos por clarificar y bloqueantes.

Última actualización: 30 de septiembre de 2026, al cerrar la conversación de modelado de YOLO26.

---

Estados posibles: Pendiente, En curso, Resuelta, Descartada. Las preguntas dirigidas al director o al asesor no pueden plantearse hasta el retorno a clases, previsto para finales de septiembre o inicios de octubre de 2026. Las de área técnica puede resolverlas el estudiante por su cuenta.

## Abiertas

| ID | Pregunta / Asunto | Área | Estado | Responsable / Fuente | Resolución |
|---|---|---|---|---|---|
| 1 | ¿Existe material ya capturado por las 15 cámaras trampa y quién lo custodia físicamente? La renuncia del ingeniero afecta la recolección futura, pero el material histórico debería existir en tarjetas SD o discos | Logística | Pendiente | Director / Asesor FICAYA | |
| 2 | ¿De quién son las cámaras? El anteproyecto dice propiedad institucional y el registro de estado del proyecto dice propiedad de la comunidad o parroquia de Angochagua. Determina a quién se le pide el material | Logística | Pendiente | Director / anteproyecto | |
| 3 | ¿Qué modelo de cámara trampa se usa? Define si la fecha y hora vienen en metadatos EXIF o solo grabadas sobre la imagen, y qué metadatos traen los archivos MP4 | Técnica / Logística | Pendiente | Asesor FICAYA | |
| 4 | ¿Existe una tabla que asocie el código de cada cámara con sus coordenadas geográficas? La aplicación debe mostrar ese dato y no viene dentro de los archivos | Logística | Pendiente | Asesor FICAYA | |
| 5 | ¿Se valida el uso del dataset proxy y la estrategia de dos iteraciones de CRISP-DM? | Metodología | Pendiente | Director | |
| 6 | ¿Quién asume la logística con la comunidad tras la renuncia y hay una fecha estimada de solución? | Logística | Pendiente | Director | |
| 7 | ¿Cuántas imágenes se esperan de Angochagua y en qué proporción de día y noche? Sin ese dato, las metas de la iteración 2 en los criterios de éxito son una conjetura | Metodología | Pendiente | Asesor FICAYA | |
| 8 | ¿Se acepta el recall como métrica principal por encima de la precisión, con la justificación del costo asimétrico de los errores? | Metodología | Pendiente | Director | Ver `criterios_exito_provisional.md`, apartado 0 |
| 9 | ¿Se mantiene el 92.6 % de mAP de la referencia [4] como rendimiento mínimo esperado, sabiendo que proviene de un dataset mayor y de otras especies? | Metodología | Pendiente | Director / anteproyecto | |
| 10 | ¿Qué umbral mínimo de precisión resulta tolerable para el investigador que usará la aplicación? Debería responderlo quien revisa el material a mano | Metodología | Pendiente | Asesor FICAYA | |
| 11 | ¿Qué implementación de RT-DETRv3 se acepta para la comparación, si la oficial en PaddlePaddle no llega a funcionar en el entorno disponible? | Técnica | Pendiente | Director | Depende del resultado del intento con límite de dos días |
| 13 | El dataset proxy no contiene imágenes vacías de cámara trampa, que en el material real son la mayoría. ¿Se busca otro dataset de LILA BC que las incluya o se cubre en la iteración 2? | Técnica | Pendiente | Estudiante | Se decidió posponerlo (ver nota abajo), pero sigue abierto como opción si sobra tiempo |
| 16 | ¿Vale la pena una segunda pasada sobre las 111 imágenes marcadas como duda en la revisión manual? | Técnica | Pendiente | Estudiante | Opcional, solo si sobra tiempo |
| 17 | El anteproyecto y el marco teórico mencionan Google Colaboratory, pesos de ImageNet y LabelImg. Corresponde corregirlos por Kaggle, pesos de COCO y X-AnyLabeling, e informarlo al director | Redacción | Pendiente | Estudiante / Director | Pendiente hasta la revisión final de los documentos |
| 18 | En el dataset proxy, todos los positivos de una fuente y casi todos los negativos de la otra crean una correlación entre fuente y etiqueta que el modelo podría aprovechar. ¿Se acepta la mitigación aplicada (negativos de iNaturalist más métricas separadas por fuente) o el director prefiere otro enfoque? | Metodología | Pendiente | Director | Ver ficha del 9 de septiembre en `registro_decisiones.md` |
| 19 | La misma cámara de ENA24 aparece en entrenamiento y en prueba con el mismo fondo fijo, así que las métricas de la iteración 1 no miden generalización a cámaras nuevas. En la iteración 2, ¿se reserva alguna cámara completa de Angochagua para el conjunto de prueba, aun a costa de tener menos eventos de oso para entrenar? | Metodología | Pendiente | Director | Depende de la respuesta a la pregunta 7 |
| 20 | ¿Un modelo entrenado con imágenes bajo licencia CC BY-NC hereda la restricción de uso no comercial? No afecta al uso académico del trabajo, pero sí a una eventual publicación del modelo o del dataset | Legal / Redacción | Pendiente | Director | El estudiante no puede responderlo; es una consulta legal. La licencia de ENA24 (CDLA-Permissive-1.0) no impone restricciones sobre los resultados, así que la duda queda limitada a las fotos CC BY-NC de iNaturalist |
| 21 | ¿Se acepta que los criterios numéricos de éxito se lean sobre la partición de ENA24 y no sobre la métrica global del conjunto de prueba? | Metodología | Pendiente | Director | Ver `criterios_exito_provisional.md`, apartado 1.0 |
| 22 | ¿Conviene correr el experimento de ablación (entrenar solo con iNaturalist, solo con ENA24 y con ambas fuentes, evaluando las tres sobre el conjunto de prueba de ENA24)? Respondería con números si la combinación de fuentes ayudó | Técnica | En curso | Estudiante | Entrenada el 30 de septiembre y el 1 de octubre (A1, A1b y A2). Primera lectura con el mAP de Ultralytics: sin cámaras trampa el modelo no sirve en cámara trampa; sin iNaturalist, en ENA24 detecta igual pero ajusta peor las cajas (con una salvedad sobre la elección de época); ninguna fuente estorba a la otra. Respuesta provisional: conservar ambas fuentes. Falta la evaluación con la regla B (`experimentos.md`)|
| 23 | ¿Se acepta la revisión de la regla del umbral de confianza: fijarlo por el recall por imagen de 0.95, con las falsas alarmas como restricción, en lugar de por el recall de cajas de 0.90? | Metodología | Pendiente | Director | Ficha del 30 de septiembre en `registro_decisiones.md` y `criterios_exito_provisional.md`, apartado 1.1 Dato nuevo (1 oct): con E1, bajar el umbral de 0.363 a 0.10 avisaría 12 fotos con oso más en validación con 2 falsas alarmas más; se mantiene 0.363 porque el proxy no tiene fotos vacías|
| 24 | ¿Se acepta YOLO26n (E1) como modelo de la aplicación, dado que en validación los tres tamaños empatan y nano es el pensado para equipos sin GPU? | Metodología | Pendiente | Director | Ficha del 30 de septiembre en `registro_decisiones.md` Velocidad medida (30 sep): 52 ms por foto y 412 MB como máximo en el equipo local, frente a 183 ms de medium|
| 25 | El anteproyecto compromete rotaciones como aumento de datos. La ablación mostró que empeoran el ajuste de las cajas y se descartaron. ¿Se acepta esa desviación documentada? | Metodología / Redacción | Pendiente | Director | `experimentos.md`, sección 1.4, y ficha del 16 de septiembre |
| 26 | YOLO26m no convergió en 100 épocas. ¿Hay que entrenarlo más tiempo para compararlo con RT-DETR? | Técnica | Pendiente | Estudiante | Decidir al ver cómo se comporta RT-DETR |
| 27 | ¿Hay tapir de montaña, ganado oscuro o perros en la zona de las cámaras de Angochagua? En validación son los negativos que más falsas alarmas dan (el tapir dispara en 3 de 5 fotos) | Datos | Pendiente | Asesor FICAYA / Director | Si los hay, conviene incluirlos como negativos en la iteración 2 |
| 28 | ¿El Kaggle Dataset `deteccionoso` y el repositorio son privados? Si alguno es público se están publicando imágenes, y hay que incluir los avisos de licencia: nombre o enlace de la CDLA para ENA24 y atribución por foto para iNaturalist | Legal | Pendiente | Estudiante | Las 80 imágenes de `docs/preparacion_datos/muestra_cajas` entran en git |
| 29 | ¿Se usan las imágenes de oso andino del dataset WCS Camera Traps (LILA BC, 462 etiquetadas como *Tremarctos ornatus*)? Como evaluación externa de la iteración 1, como plan B de la iteración 2 si no llega el material de Angochagua, o ambas | Metodología / Datos | Pendiente | Director | Antes, comprobar cuántas tienen caja, cuántas muestran de verdad el oso (las etiquetas son en parte por secuencia) y de dónde son |

## Resueltas

| ID | Pregunta / Asunto | Área | Fecha | Resolución |
|---|---|---|---|---|
| 14 | ¿Se conservaron los archivos GIF originales en `data/raw` al extraer los fotogramas, o se borraron? | Técnica | 7 sep | Se conservaron. Los 8 archivos siguen en `data/raw` y aparecen como pendientes de revisión en la tabla consolidada, que es lo correcto: quedan fuera del dataset sustituidos por sus 20 fotogramas. Lo que sí se había borrado eran tres copias de la carpeta de revisión, que se recuperaron desde `data/raw` y se registraron como duda |
| 15 | ¿Conviene refinar el agrupamiento de duplicados con distancia de Hamming entre hashes? | Técnica | 10 sep | Sí, y no era opcional. Con hash exacto quedaban 1999 pares de imágenes casi idénticas repartidas entre conjuntos distintos, que afectaban a 408 de las 963 imágenes de validación y prueba. Se adoptó la unión por distancia menor o igual a 12, verificada con `fugas_hamming.py`. Ficha del 10 de septiembre en `registro_decisiones.md` |
| 12 | ¿Bajo qué licencia se distribuye ENA24? Falta anotarla en el informe de comprensión de los datos y en el anexo de licencias | Técnica | Pendiente | Estudiante / lila.science | |

## Notas sobre preguntas que cambiaron de forma

La pregunta 13, sobre las imágenes vacías de cámara trampa, se mantiene abierta pero con una decisión provisional tomada durante la fase de preparación: no resolverla en la iteración 1. El motivo es que una imagen vacía es muy específica de su cámara y de su fondo fijo, de modo que unos cientos de vacías de Norteamérica enseñarían bastante menos que las de Angochagua, y además la descarga selectiva desde LILA BC exige averiguar cómo obtener un subconjunto sin bajar archivos de decenas de gigabytes, algo que no se ha investigado. La consecuencia queda declarada como limitación: en la iteración 1 no se puede medir la tasa de falsos positivos sobre escena vacía.

La pregunta 16 decía 108 dudas y ahora son 111, por las tres imágenes recuperadas que se describen en la resolución de la pregunta 14.

La pregunta 17 incorpora LabelImg, que en la versión anterior de este listado aparecía solo como una observación del estado del proyecto. La herramienta quedó descartada por un fallo al crear cajas en versiones recientes de Python, así que la corrección en el anteproyecto ya no es opcional.
