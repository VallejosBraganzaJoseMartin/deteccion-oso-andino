# Registro de Decisiones de Arquitectura y Proyecto (ADR)

Este documento registra las decisiones técnicas y metodológicas clave tomadas durante el desarrollo del proyecto de Detección de Oso Andino.

---

## Estructura de Registro

### [YYYY-MM-DD] Título de la Decisión
- **Contexto**: Descripción del problema o necesidad.
- **Alternativas consideradas**: Opciones evaluadas.
- **Decisión tomada**: Justificación de la alternativa seleccionada.
- **Consecuencias**: Impacto positivo, limitaciones o compromisos asumidos.

Cuando una decisión se modifica después, la ficha original se conserva tal como se escribió y se le añade un bloque de revisión con la fecha. Así queda a la vista qué se pensó primero y por qué cambió, que es justamente lo que interesa mostrar en el Capítulo 2.

---

## Nota sobre este registro

Las decisiones con ficha completa son las que habrá que sostener ante el tribunal o que resultarían costosas de revertir. Las de menor alcance se agrupan al final en la tabla de decisiones operativas. Todas las decisiones de septiembre de 2026 se tomaron durante el periodo de vacaciones, sin posibilidad de consultar al director, y quedan pendientes de validación al iniciar Titulación 2.

**Nota del 15 de noviembre de 2026.** En la reunión del 16 de octubre y en las conversaciones siguientes, el director le dijo al estudiante que tome las decisiones que crea mejores para el proyecto. La única instrucción explícita sigue siendo la de su correo del 7 de octubre: el conjunto de prueba solo al final, para reportar las métricas finales, sin usarlo antes para nada. Las fichas de este registro, escritas como pendientes de validar con el director, quedan como decisiones del estudiante tomadas con esa delegación. No se reescriben: se conservan como se escribieron.

---

## [2026-09-01] Iteración 1 de CRISP-DM con dataset proxy

- **Contexto**: El ingeniero responsable en la comunidad de Angochagua renunció y no quedó nadie que entregue el material de las 15 cámaras trampa. El director indicó que la solución tomará tiempo y que, mientras tanto, se entrenen los modelos con imágenes de internet para luego hacer ajuste fino con el material real.
- **Alternativas consideradas**:
  1. Esperar el material real y dedicar las vacaciones solo a redacción.
  2. Entrenar con un dataset proxy de fuentes públicas y ajustar después (indicación del director).
  3. Buscar imágenes de oso andino en otra organización que fototrampee en Ecuador.
- **Decisión tomada**: Se adopta la opción 2, estructurada como dos iteraciones de CRISP-DM. La iteración 1 usa el dataset proxy y la iteración 2 incorpora el material de Angochagua mediante ajuste fino. La opción 1 desperdiciaría tres semanas de trabajo. La opción 3 queda como plan de contingencia y requiere gestiones que no pueden hacerse en vacaciones. CRISP-DM contempla explícitamente el carácter cíclico del proceso, así que la estructura en dos iteraciones es coherente con la metodología declarada en el anteproyecto. La referencia [16] del marco teórico (Shepley et al.) respalda el enfoque: entrenó detectores con imágenes de Flickr e iNaturalist y obtuvo buen desempeño al evaluarlos sobre cámaras trampa, con mejoras adicionales al inyectar datos del dominio real.
- **Consecuencias**: Permite avanzar sin depender de la logística. Las métricas de la iteración 1 no son los resultados del trabajo y deben presentarse siempre como preliminares. Obliga a explicar la estrategia en el Capítulo 2 y a documentar la situación que la motivó. Abre además una posibilidad: medir la caída de desempeño entre dominio proxy y dominio real, y cuánto la recupera el ajuste fino, lo que constituye un resultado propio.

## [2026-09-01] Fuentes con licencia rastreable

- **Contexto**: El dataset proxy necesita imágenes de oso andino y de cámaras trampa. La forma rápida sería descargar resultados de un buscador de imágenes.
- **Alternativas consideradas**: descarga masiva desde buscadores, frente a fuentes con licencia explícita y API documentada.
- **Decisión tomada**: Solo se usan fuentes con licencia verificable. Para el oso andino, iNaturalist vía su API pública, filtrando por licencias CC0, CC BY y CC BY-NC, con grado de investigación y excluyendo observaciones marcadas como cautivas. Para el dominio de cámara trampa, ENA24 de LILA BC. Se guarda licencia, atribución, usuario, fecha y coordenadas de cada fotografía en `registro_licencias.csv`.
- **Consecuencias**: El origen de cada imagen es defendible ante una pregunta del tribunal y el registro sirve como anexo. El costo es que el volumen disponible es menor que con descarga indiscriminada. Las licencias CC BY-NC restringen el uso comercial, lo que no afecta a un trabajo académico pero conviene mencionarlo.

**Revisión del 2026-09-09.** La descarga de negativos de otras especies (ver ficha del 2026-09-09) mantiene el mismo criterio y genera su propio `registro_licencias.csv`, con una columna adicional para el taxón. Una diferencia: para las especies domésticas se levantó el filtro de grado de investigación y el de cautividad, porque iNaturalist marca como cautivas casi todas las observaciones de perros, vacas, caballos y ovejas, y con el filtro puesto no habría material. Esa excepción solo aplica a los negativos, donde el interés es el fondo y el animal que no es un oso. Queda pendiente anotar la licencia exacta de ENA24, que sigue sin verificarse en lila.science.

**Revisión del 2026-09-30.** Licencia de ENA24 verificada en su ficha de lila.science: Community Data License Agreement, variante permisiva, versión 1.0 (CDLA-Permissive-1.0, https://cdla.dev/permissive-1-0/). Permite usar, copiar, adaptar y analizar los datos (sección 2) y no impone obligaciones ni restricciones sobre los resultados, que define como lo que se obtiene al procesar los datos con un computador (sección 3.4). Un modelo entrenado y sus métricas encajan en esa definición, aunque esa lectura no es la de un abogado. Si se publican las imágenes, hay que incluir el nombre o el enlace de la licencia, avisar de las modificaciones (aquí, el cambio de nombre y la conversión de las cajas) y conservar la atribución (sección 3). LILA pide citar: Yousif, H., Kays, R. y He, Z. (2019). Dynamic Programming Selection of Object Proposals for Sequence-Level Animal Species Classification in the Wild. IEEE Transactions on Circuits and Systems for Video Technology. Su BibTeX no trae volumen ni páginas. Con esto, la duda sobre el uso no comercial (pregunta 20) queda limitada a las fotos CC BY-NC de iNaturalist.

## [2026-09-03] Exclusión de rastros, imágenes sin oso y casos dudosos

- **Contexto**: La revisión manual de 670 fotografías de iNaturalist encontró 118 registros de rastros (huellas, excrementos, restos), 68 imágenes sin oso, 108 casos donde no se pudo decidir y 1 de cautiverio evidente. El anteproyecto define la detección del oso, no de sus rastros.
- **Alternativas consideradas**:
  1. Excluir todo lo que no sea un oso visible.
  2. Incluir los rastros como clase adicional.
  3. Usar las imágenes dudosas como positivas de baja confianza.
- **Decisión tomada**: Opción 1. Los rastros no entran porque el objetivo declarado es detectar al animal, y una clase adicional ampliaría el alcance sin respaldo en el anteproyecto. Las dudas no se usan ni como positivas ni como negativas: una imagen donde un revisor humano no puede decidir no aporta señal para entrenar y contamina la evaluación.
- **Consecuencias**: El dataset queda más limpio a costa de unas 295 imágenes. Los archivos siguen disponibles en `data/revisar`, así que revertir es barato. Los rastros podrían recuperarse más adelante como negativos si el modelo confunde troncos y sombras con osos. Queda pendiente una segunda pasada sobre las 108 dudas si sobra tiempo.

**Revisión del 2026-09-09.** La decisión se mantiene en lo que respecta a las clases del modelo, que sigue teniendo una sola, pero se revierte la parte de la exclusión. Los rastros y las imágenes sin oso vuelven al dataset como negativos, es decir, con archivo de etiquetas vacío. El motivo aparece en la ficha de negativos de iNaturalist del mismo día: con todos los positivos viniendo de una fuente y todos los negativos de otra, el modelo puede aprender a distinguir el estilo fotográfico en lugar del animal. Entraron 118 rastros y 63 paisajes. Quedan fuera las 111 dudas, la imagen de cautiverio y las que resultaron ser fotos de pantallas de monitor o collages, porque contienen un oso aunque sea de forma indirecta y como negativos enseñarían justo lo contrario de lo que se busca. Lo que ya se había anticipado aquí —que los rastros podrían recuperarse como negativos— es lo que terminó ocurriendo, solo que por un motivo distinto del previsto.

## [2026-09-03] División de datos por grupo y no por imagen

- **Contexto**: En iNaturalist, 253 observaciones tienen más de una fotografía del mismo encuentro, con un máximo de 20. En ENA24 el análisis por hash perceptual encontró 994 grupos de imágenes casi idénticas, uno de ellos con 55, que corresponden a ráfagas de una misma activación del sensor. Una división aleatoria por imagen pondría fotografías casi iguales en entrenamiento y en prueba.
- **Alternativas consideradas**: división aleatoria simple, división por grupo, o eliminación de duplicados antes de dividir.
- **Decisión tomada**: La unidad de división es el grupo. En iNaturalist el grupo es `observacion_id`, y los fotogramas extraídos de GIF heredan el de su archivo original. En ENA24, como el JSON no incluye campos de secuencia ni ubicación, el grupo es el hash perceptual calculado por `describir_dataset.py`. No se eliminan los duplicados porque las ráfagas aportan variación real de pose dentro del mismo evento.
- **Consecuencias**: Evita la fuga entre conjuntos, que inflaría las métricas y produciría resultados no defendibles. Obliga a mantener la trazabilidad del grupo en todo el pipeline. El agrupamiento por hash solo captura coincidencias exactas del hash y puede refinarse con distancia de Hamming si hay tiempo. Si esta decisión se revirtiera después de entrenar, habría que repetir el entrenamiento.

**Revisión del 2026-09-10.** El refinamiento que aquí quedaba como opcional resultó ser necesario, y tiene ficha propia más abajo. Con la primera división construida se midieron las fugas que el agrupamiento por hash exacto no llegó a evitar: 1999 pares de imágenes casi idénticas repartidos entre conjuntos distintos, que afectaban a 408 de las 963 imágenes de validación y prueba. Casi todas eran de ENA24. La unidad de división pasa a ser el grupo formado por unión transitiva de hashes a distancia de Hamming menor o igual a 12, y en iNaturalist esa unión se suma a la agrupación por observación en lugar de sustituirla.

**Nota del 2026-10-07.** El director, en su respuesta por correo al pedido de reunión, pidió dividir el dataset en 70/15/15 u 80/10/10 y dejar el conjunto de prueba exclusivamente para reportar las métricas finales, sin usarlo antes para nada. La división vigente ya es 70/15/15 por grupo: 2656, 598 y 578 imágenes (69.3, 15.6 y 15.1 %). No sale exacta porque las fotos de una misma observación y las casi idénticas van juntas al mismo conjunto. El conjunto de prueba no se ha usado para nada, como fija la decisión operativa del 16 de septiembre.

## [2026-09-01] Límite de tiempo para la implementación oficial de RT-DETRv3

- **Contexto**: El director dispuso añadir RT-DETRv3 al trabajo para comparar arquitecturas. La implementación oficial está en PaddlePaddle, un framework distinto de Ultralytics, que es el previsto en el anteproyecto para YOLO. No hay confirmación de que Ultralytics soporte la versión 3.
- **Alternativas consideradas**:
  1. Montar el entorno de PaddlePaddle en la nube y usar la implementación oficial.
  2. Usar una reimplementación no oficial en PyTorch, cuya fidelidad al artículo no está garantizada.
  3. Comparar contra la variante de RT-DETR que Ultralytics sí soporta.
- **Decisión tomada**: Se intenta la opción 1 con un límite de dos días de trabajo. Si a ese plazo el entorno no funciona, se pasa a la opción 3 documentando el error exacto que impidió la primera vía, y se consulta al director al retomar clases. La opción 2 queda descartada salvo que se identifique una reimplementación con evidencia de reproducir los resultados publicados.
- **Consecuencias**: Acota el riesgo de perder la vacación en problemas de instalación. Si se acaba usando la opción 3, la comparación deja de ser estrictamente contra RT-DETRv3 y eso debe declararse en la redacción. Por otro lado, RT-DETRv3-R18 tiene alrededor de 20 millones de parámetros frente a los 2.4 de YOLO26n, así que en cualquier escenario la comparación debe emparejar presupuestos computacionales o declarar explícitamente que el eje es precisión frente a costo.

**Nota del 2026-09-10.** Si la implementación oficial exige anotaciones en formato COCO, la conversión desde los archivos de texto que produce `dividir_dataset.py` es corta de escribir, porque `division.csv` ya guarda el conjunto y las cajas de cada imagen. Ese trabajo no debe contarse dentro del límite de dos días, que se reserva para el entorno.

**Resultado del 2026-10-05.** La opción 1 funcionó dentro del plazo. Llevó unas 3.5 horas de trabajo de entorno entre la noche del 3 y el 5 de octubre, y cumplió el criterio de "corre" fijado de antemano: una corrida corta de 3 épocas sin NaN que deja las detecciones de validación exportadas. No hizo falta la opción 3. RT-DETRv3-R18 tiene, con una clase, 20.1 M de parámetros en las partes que usa al detectar (22.2 M al entrenar), así que queda emparejado con YOLO26m (20.35 M). Las decisiones que salieron de aquí son D22 a D24, y el detalle está en `experimentos.md`.

## [2026-09-04] Entrenamiento en Kaggle en lugar de Google Colaboratory

- **Contexto**: El anteproyecto y el marco teórico establecen Google Colaboratory como entorno de entrenamiento. Durante la preparación del trabajo se optó por Kaggle.
- **Alternativas consideradas**: Colaboratory en su versión gratuita, Colaboratory de pago, y Kaggle.
- **Decisión tomada**: El entrenamiento se realiza en Kaggle, por la cuota de GPU. En un proyecto anterior trabajé con la versión gratuita de Colaboratory y las interrupciones por límite de uso impidieron terminar sesiones de entrenamiento, que es la razón por la que probé Kaggle y me quedé ahí. La cuota semanal de GPU que Kaggle ofrece en su plan gratuito (alrededor de 30 horas, dato que hay que verificar en la documentación vigente antes de citarlo en la memoria) da margen suficiente para entrenar dos arquitecturas y repetir corridas. No es una decisión de fondo del trabajo y no cambia nada de la metodología. Si en algún momento la cuota gratuita se queda corta, la alternativa prevista es pasar a Colaboratory de pago.
- **Consecuencias**: Obliga a corregir el anteproyecto, el marco teórico y cualquier documento que mencione Colab, y a informarlo al director. Cambia la forma de cargar los datos, que pasan a subirse como Kaggle Dataset en lugar de montarse desde Google Drive. Hay que verificar en la documentación vigente de Kaggle los límites de sesión, la cuota semanal de GPU, el tamaño máximo de un dataset y la necesidad de habilitar el acceso a internet en el cuaderno para instalar dependencias, porque esas restricciones condicionan la planificación del entrenamiento y el intento con PaddlePaddle. El `dataset.yaml` que genera `dividir_dataset.py` lleva una ruta absoluta del equipo local, así que en el cuaderno hay que reemplazar el campo `path`.

## [2026-09-07] Una sola clase para oso andino y oso negro americano

- **Contexto**: El dataset proxy combina fotografías de oso andino de iNaturalist con imágenes de oso negro americano de ENA24. Son especies distintas, y el objetivo del trabajo es detectar únicamente la primera. Había que decidir si el modelo distingue entre ambas o si las trata como una sola categoría.
- **Alternativas consideradas**:
  1. Una clase única, `oso`, que agrupe las dos especies.
  2. Dos clases, `oso_andino` y `oso_negro`, y quedarse con la primera al evaluar.
  3. Usar solo iNaturalist como positivo y ENA24 únicamente como fuente de negativos.
- **Decisión tomada**: Opción 1. El oso negro no está en el dataset porque interese detectarlo, sino porque aporta lo que iNaturalist no tiene: un animal grande de pelaje oscuro visto por una cámara trampa, de noche y en infrarrojo, que es la condición donde el modelo final tendrá que trabajar. Separar las especies obligaría al modelo a aprender una distinción que no sirve para nada en Angochagua, donde no hay osos negros, y repartiría los ejemplos entre dos clases con menos datos cada una. La opción 3 desperdicia 893 imágenes de cámara trampa con cajas ya anotadas, que es el recurso más escaso del proyecto.
- **Consecuencias**: El modelo de la iteración 1 detecta osos en general y no oso andino en particular, y eso debe decirse en el Capítulo 3 sin rodeos. En la iteración 2, el ajuste fino con material de Angochagua especializa el modelo hacia la especie real, y como en la zona no hay otros úrsidos, la clase única deja de ser una aproximación y pasa a ser exacta. Si en la evaluación el modelo confunde con oso a otros animales oscuros, habrá que revisar si la clase única contribuyó a esa confusión. El archivo `dataset.yaml` declara `nc: 1` y `names: ['oso']`, así que cambiar de criterio implica volver a generar todas las etiquetas y repetir el entrenamiento.

## [2026-09-07] Herramienta de anotación: X-AnyLabeling

- **Contexto**: El anteproyecto menciona LabelImg como herramienta de etiquetado, anotada en su momento como tentativa. Había que corregir a mano las 1091 cajas propuestas por el modelo preentrenado y dibujar las que faltaban, sobre 1138 imágenes.
- **Alternativas consideradas**:
  1. LabelImg, la herramienta prevista.
  2. Una plataforma de anotación en la nube con etiquetado asistido.
  3. X-AnyLabeling, aplicación de escritorio con soporte de formato YOLO y trazado asistido por SAM.
- **Decisión tomada**: Opción 3. LabelImg quedó descartada por un motivo técnico: el proyecto no recibe mantenimiento desde hace años y la versión que se instala desde pip falla al crear una caja en versiones recientes de Python, por un error de tipos en la capa de PyQt. Las soluciones que circulan pasan por volver a Python 3.9 o editar el código de la librería a mano, y ninguna de las dos parece razonable para una herramienta que iba a usarse durante horas. La opción 2 se descartó por licencias: la mayoría de las imágenes son CC BY-NC y los planes gratuitos de esas plataformas suelen publicar los proyectos, lo que abriría una discusión que con una herramienta local no existe. X-AnyLabeling funciona en local, se mantiene activo, lee y escribe etiquetas en formato YOLO e incorpora trazado asistido. Se instaló la versión 4.0.6 desde el paquete `x-anylabeling-cvhub` sobre Python 3.14.2, en un entorno virtual separado para que su dependencia de OpenCV no interfiera con la de Ultralytics.
- **Consecuencias**: Hay que corregir la mención a LabelImg en el anteproyecto y en el marco teórico. La herramienta guarda su trabajo en archivos JSON propios junto a las imágenes y exporta a formato YOLO en un paso aparte, lo que obliga a mantener separadas las cajas propuestas y las revisadas, algo que además resultó útil para medir cuánto corrigió el revisor. El entorno virtual `venv_etiquetado` no se versiona.

## [2026-09-09] Negativos de iNaturalist para romper la correlación entre fuente y etiqueta

- **Contexto**: Con la primera versión del dataset, todas las imágenes de iNaturalist eran positivas y todos los negativos venían de ENA24. Un detector puede aprovechar esa correlación y aprender a reconocer el estilo fotográfico de cada fuente en lugar del animal, y como la correlación se repite en validación y prueba, las métricas no lo delatarían.
- **Alternativas consideradas**:
  1. Dejarlo así y declararlo como limitación.
  2. Recuperar como negativos los rastros y paisajes de iNaturalist que ya estaban en disco.
  3. Descargar además fotografías de otras especies del mismo hábitat, con los mismos criterios de licencia.
- **Decisión tomada**: Se aplican las opciones 2 y 3 juntas. La 2 sola no basta, porque los rastros son primeros planos de suelo y de troncos, un subestilo tan reconocible que el modelo podría seguir resolviendo la tarea sin mirar animales. La 3 es la que de verdad obliga a distinguir un oso de otro animal dentro del mismo estilo de fotografía. Se descargaron 70 fotos de cada una de ocho especies, dentro del recuadro geográfico de Ecuador continental: lobo de páramo, tapir de montaña, puma y venado por ser fauna que comparte hábitat con el oso, y perro, vaca, caballo y oveja por ser los animales que una cámara trampa en zona de conflicto ganadero va a captar con más frecuencia. De las 560 descargadas se descartaron 171 tras revisarlas a ojo: 128 en entornos urbanos o interiores, que no se parecen en nada a lo que verá el modelo, 2 fotografías de pantallas de cámara y 41 archivos que resultaron ser GIF, sustituidos cada uno por un fotograma. El dataset queda con 611 negativos de iNaturalist frente a 1128 positivos de la misma fuente.
- **Consecuencias**: El atajo deja de ser perfecto, aunque no desaparece: la proporción dentro de iNaturalist sigue inclinada hacia los positivos. Por eso la mitigación se completa del lado de la evaluación, con métricas separadas por fuente. Se añade una fuente más de imágenes al proyecto, con su propio registro de licencias que debe ir al anexo. Queda como trabajo opcional un experimento de ablación, entrenando solo con iNaturalist, solo con ENA24 y con ambas fuentes, y evaluando las tres sobre el conjunto de prueba de ENA24, que respondería con números si la combinación ayudó.

**Resultado 2026-10-02.** El experimento de ablación que esta ficha dejaba como opcional se hizo (A1, A1b y A2). Con el mismo recall por imagen en cámara trampa, los modelos entrenados sin iNaturalist dan 18 y 21 falsas alarmas de 90 en las fotos sin oso de iNaturalist, frente a 3 y 4 de los entrenados con ambas fuentes. La ablación no separa cuánto se debe a los negativos de iNaturalist y cuánto a sus positivos. Detalle en `experimentos.md`.

## [2026-09-10] Agrupamiento por distancia de Hamming con umbral 12

- **Contexto**: La división por hash exacto dejaba pasar fugas. El diagnóstico sobre la primera versión del dataset encontró 1999 pares de imágenes casi idénticas repartidas entre conjuntos distintos a distancia de Hamming menor o igual a 8, que implicaban a 408 de las 963 imágenes de validación y prueba. Al subir el umbral a 8 quedaban todavía casos claros: dos fotografías de la cámara PR6 tomadas a las 06:35:42 y 06:35:43 del mismo día, con el mismo venado, una en entrenamiento y otra en prueba, y dos de la cámara Cam1 separadas por nueve segundos con la misma cabalgata pasando.
- **Alternativas consideradas**: umbrales de 0 (equivalente al hash exacto), 4, 8, 12 y 16 para la unión de grupos.
- **Decisión tomada**: Umbral 12. Con 8 seguían apareciendo pares de la misma activación del sensor separados por segundos, comprobados uno a uno abriendo las imágenes y leyendo la marca de tiempo que la cámara graba sobre la fotografía. Con 12 desaparecen por completo. No se sube a 16 porque a esa distancia empiezan a emparejarse 216 pares de fotografías de iNaturalist que solo se parecen de lejos, señal de que el hash deja de discriminar y de que la unión arrastraría imágenes sin relación real.
- **Consecuencias**: El conjunto de prueba ya no contiene ninguna imagen casi idéntica a una de entrenamiento, verificado con el mismo script que midió el problema. Las dos tablas, antes y después, sirven como evidencia en el Capítulo 2. El costo es que la unión es transitiva y forma algunos grupos grandes: el mayor reúne 391 imágenes nocturnas de ENA24, casi todas negativas, y cae entero en entrenamiento. Eso desequilibra un poco el reparto, aunque validación y prueba conservan 104 y 86 positivos nocturnos, suficientes para medir. Queda una fuga que ninguna agrupación por hash resuelve y que hay que declarar como limitación: la misma cámara de ENA24 aparece en entrenamiento y en prueba con el mismo fondo fijo, de modo que las métricas de la iteración 1 no miden generalización a cámaras nuevas. La referencia [15] del marco teórico documenta esa caída.

## [2026-09-10] Evaluación con métricas separadas por fuente

- **Contexto**: Ultralytics evalúa un conjunto completo y devuelve un número por métrica. El conjunto de prueba mezcla 260 imágenes de iNaturalist con 318 de ENA24, que son dos problemas distintos: fotografía con cámara en mano frente a cámara trampa fija, día frente a infrarrojo nocturno. Un promedio de los dos no describe bien ninguno.
- **Alternativas consideradas**: reportar solo la métrica global, reportar solo la de ENA24 por ser el dominio de interés, o reportar las tres.
- **Decisión tomada**: Se reportan las tres. La métrica sobre ENA24 es el resultado principal de la iteración 1, porque el dominio de cámara trampa es el del trabajo. La de iNaturalist se presenta como complementaria y optimista, ya que la correlación entre fuente y etiqueta la favorece. La global se incluye por comparabilidad con la literatura. Se añade además una métrica a nivel de imagen, que mide si el modelo pone al menos una detección por encima del umbral en una imagen que contiene oso, porque es la que corresponde al uso real de la aplicación, donde el investigador quiere saber qué carpetas revisar y no cuán ajustada quedó la caja.
- **Consecuencias**: Obliga a preparar listas de rutas separadas por fuente y a correr la validación tres veces por modelo, lo que alarga poco el trabajo. El conjunto de prueba de ENA24 con oso tiene 145 imágenes, así que las cifras por fuente llevan un margen de error de varios puntos y deben presentarse como indicativas. Los criterios de éxito quedan formulados sobre la métrica de ENA24 y no sobre la global.

> **Revisión 2026-09-30.** La ficha presenta la métrica de iNaturalist como complementaria y optimista. En validación ocurrió lo contrario en los seis modelos: con el umbral de la regla B, el recall por imagen fue de 0.88 a 0.93 en iNaturalist y de 0.955 en ENA24, y el mAP50-95 de ENA24 quedó por encima del de iNaturalist en todas las corridas. Hay dos hipótesis compatibles, sin verificar: ENA24 comparte cámaras y fondos entre entrenamiento y validación (limitación de D13), y en iNaturalist hay más osos lejanos, tapados o en encuadres variados. La decisión de reportar las tres métricas no cambia. Cambia la lectura: iNaturalist no se presentará como optimista sin antes confirmarlo con el conjunto de prueba y el análisis de errores.

## [2026-09-16] Aumento de datos: valores por defecto de Ultralytics y ablación de la rotación
 
- **Contexto**: El anteproyecto compromete cuatro técnicas de aumento: rotaciones, ajustes de brillo, saturación y Mosaic. Ultralytics aplica el aumento durante el entrenamiento mediante argumentos, y en la versión instalada (8.4.138) brillo (`hsv_v` 0.4), saturación (`hsv_s` 0.7) y Mosaic (`mosaic` 1.0) vienen activos, mientras que la rotación (`degrees`) viene en 0. Al mirar cómo la librería implementa la rotación (`RandomPerspective.apply_bboxes`) se vio que rota las esquinas de cada caja y toma como etiqueta la envolvente alineada a los ejes, que es siempre más grande que el animal: ×1.37 de área a 10° para una caja apaisada 3:2. Eso introduce ruido de etiqueta y castiga mAP50-95, el criterio principal del trabajo.
- **Alternativas consideradas**:
  1. Dejar `degrees=0` y justificar por escrito la desviación respecto del anteproyecto.
  2. Fijar un valor pequeño (10°) sin probarlo, para cumplir lo declarado.
  3. Entrenar el modelo nano dos veces, con 0° y con 10°, y decidir con el resultado.
- **Decisión tomada**: La opción 3. La línea base usa los valores por defecto de Ultralytics 8.4.138, que ya cubren tres de las cuatro técnicas prometidas y además activan volteo horizontal, escala y traslación. La rotación se somete a una ablación controlada sobre YOLO26n (misma semilla, mismas 100 épocas, solo cambia `degrees`), comparada sobre validación global y por fuente. Si 10° mejora se adopta para los tamaños s y m. Si no, se descarta con datos. Se elige 10° y no más porque a partir de ahí la inflación de las cajas supera el tercio del área, y porque una cámara trampa mal nivelada rara vez se inclina más que eso, aunque para esto último no tengo una fuente. No se prueban volteo vertical, MixUp ni CutMix: no están prometidos y no tienen sentido para el problema. Tampoco se sobremuestrea la clase rara como sugiere el marco teórico, porque hay una sola clase y las positivas son el 53 % de las imágenes.
- **Consecuencias**: Cuesta un entrenamiento extra del modelo más barato. Convierte el compromiso del anteproyecto en un resultado reportado en el Capítulo 2 en lugar de en una desviación silenciosa. Obliga a declarar en la memoria todos los argumentos de aumento con sus valores, porque "se aplicó aumento de datos" sin cifras no es reproducible. La correspondencia completa está en `docs/experimentos.md`, sección 1.4.
> **Resultado 2026-09-17, revisado el 18.** E2 (10°) quedó 0.020 por debajo de E1 y E1b en mAP50-95 global, con el mismo mAP50 y con la pérdida de caja más alta durante las 100 épocas. Se fija `degrees=0` para los demás tamaños. La variación entre semillas medida después en small (0.014) deja la caída por encima de cualquier diferencia por azar observada, pero con poco margen. Se reporta como un indicio consistente con el mecanismo de las cajas infladas, no como un efecto demostrado. Detalle en `experimentos.md`.
## [2026-09-16] Tamaños de YOLO26 en la iteración 1: nano, small y medium
 
- **Contexto**: El marco teórico (Tabla 2) presenta las cinco variantes de YOLO26 y declara candidatas a nano y small por la falta de cómputo especializado y por el escenario de uso en CPU. El protocolo de comparación justa del plan pide emparejar el presupuesto de parámetros con RT-DETRv3-R18, que tiene unos 20 millones.
- **Alternativas consideradas**: solo nano; nano y small; nano, small y medium; añadir large.
- **Decisión tomada**: Nano, small y medium, en ese orden. Nano es el candidato de despliegue en CPU y la base de la ablación de rotación. Medium (20.4 M de parámetros) es el único tamaño que empareja a RT-DETRv3-R18; sin él la comparación enfrenta 2.4 M contra 20 M y solo puede leerse como precisión frente a costo, un argumento más débil. Small entra porque el marco teórico lo declara candidato y porque, si nano y medium difieren mucho, dice dónde está el punto de equilibrio. Large y extra-large quedan fuera: no sirven para CPU y la referencia [28] del marco teórico (Safonova et al.) encontró que las variantes medianas y compactas superaron a las pesadas en su conjunto de prueba. Medium queda condicionado a que la cuota semanal de GPU alcance, que se sabrá tras medir el tiempo por época en la corrida de verificación.
- **Consecuencias**: Entre cuatro y cinco entrenamientos completos de YOLO26 (nano dos veces por la ablación, small, medium y quizá uno a mayor resolución), estimados en 9 a 14 horas de GPU antes de medir. Si la cuota no da para medium, la comparación con RT-DETR se declara como precisión frente a costo y se anota como limitación.
> **Resultado 2026-09-18.** Con dos semillas por tamaño, el mAP50-95 global medio fue 0.758 en nano, 0.751 en small y 0.753 en medium, con diferencias entre semillas de hasta 0.014. Los tres empatan. Medium tuvo su mejor época en la 100 con las dos semillas, así que no había convergido, y se declara como limitación para la comparación con RT-DETR. La cuota de GPU alcanzó para las seis corridas.

## [2026-09-30] Umbral de confianza elegido por el recall por imagen
 
- **Contexto**: Los criterios de éxito fijaban el umbral de confianza como el más alto que conservara un recall de cajas de 0.90 en validación. Aplicada a los seis modelos de YOLO26 de la iteración 1, esa regla dejó el recall por imagen de ENA24 entre 0.91 y 0.94. Entre 9 y 14 de las 156 fotos con oso quedaban sin aviso, y la meta de los mismos criterios es 0.95. Las falsas alarmas, en cambio, estaban entre 1 y 4 %, muy lejos del límite de 10 %. La regla usaba una métrica secundaria para fijar el umbral y dejaba sin cumplir la métrica principal, que es la que corresponde al uso real de la aplicación.
- **Alternativas consideradas**:
  1. Mantener la regla y reportar que la meta de recall por imagen no se alcanza en ese punto.
  2. Fijar el umbral por el recall por imagen (0.95), conservar el recall de cajas de 0.90 como condición y usar las falsas alarmas como restricción.
- **Decisión tomada**: La opción 2, porque es la que sigue la jerarquía del principio rector de los criterios: el recall manda y la precisión es una restricción. La regla queda así: sobre ENA24 de validación, el umbral más alto que cumpla a la vez recall por imagen de 0.95 o más y recall de cajas de 0.90 o más, con falsas alarmas por debajo del 10 % en los negativos de ENA24 y en los de iNaturalist por separado. Con los seis modelos, el umbral bajó de 0.45–0.76 a 0.26–0.50. Cada modelo recuperó entre 2 y 7 fotos con oso de ENA24, y las falsas alarmas en ENA24 subieron como mucho en 2 fotos por modelo (en tres modelos no subieron). La partición sobre la que se elige el umbral (ENA24 y no la global) se fijó antes de ver resultados.
- **Consecuencias**: El recall por imagen de 0.955 en validación no es una estimación del rendimiento, porque el umbral se ajustó justamente para alcanzarlo. La estimación honesta sale del conjunto de prueba y probablemente sea algo menor. El umbral depende de cada archivo de pesos (de 0.26 a 0.50 entre modelos y semillas), así que la aplicación tiene que guardar el umbral junto con los pesos que carga. Es un cambio de criterio hecho sin poder consultar al director. Se decidió sobre validación, sin mirar el conjunto de prueba, y queda como pregunta 7 del apartado 5 de los criterios.

## [2026-09-30] Modelo de la aplicación: YOLO26n (E1) con umbral 0.363
 
- **Contexto**: En validación, los tres tamaños de YOLO26 empatan en mAP y, con la regla B, también en recall por imagen. La aplicación debe funcionar en el equipo del investigador, que puede no tener GPU (criterios de éxito, apartado 2). Hay que fijar el modelo de despliegue antes de evaluar sobre el conjunto de prueba, porque si se elige después mirando esas cifras, dejan de ser una evaluación limpia.
- **Alternativas consideradas**:
  1. E1 (nano, semilla 0).
  2. E1b (nano, semilla 1).
  3. Un modelo mayor (small o medium).
  4. Decidir después de ver el conjunto de prueba.
- **Decisión tomada**: E1. Nano, porque es el tamaño pensado para equipos sin GPU y no rinde peor que los mayores: mAP50-95 global medio de 0.758, frente a 0.751 de small y 0.753 de medium. Entre las dos semillas de nano, E1 tiene menos falsas alarmas en ENA24 con la regla B (3 frente a 6 de 182) y mayor recall por imagen en iNaturalist (0.91 frente a 0.88), con un mAP prácticamente igual. La opción 4 se descarta porque elegir el mejor de seis modelos parecidos mirando el conjunto de prueba haría que su cifra reflejara en parte la suerte. El umbral, 0.363, es el que da la regla B sobre ENA24 de validación.
- **Consecuencias**: E1 es el modelo por defecto de la aplicación y el punto de partida del fine-tuning de la iteración 2. Se puede cambiar si un experimento posterior (ablación por fuente, resolución, RT-DETR) da un candidato mejor, siempre que se decida sobre validación y antes de la evaluación final. Queda por medir su velocidad en CPU. Si no cumpliera la meta de menos de 1 segundo por foto, habría que revisar la decisión. Es improbable: en la primera validación local, nano tardó 9 ms de inferencia por foto a 320 px. Pendiente de validar con el director.

**Revisión del 2026-09-30.** Velocidad medida en el equipo local: E1 tarda 52 ms por foto (mediana con la lectura del JPEG; 73 ms en el percentil 95) y usa 412 MB de RAM como máximo, frente a 82 ms y 492 MB de small y 183 ms y 597 MB de medium. Cumple las metas de los criterios con margen. La decisión se mantiene.

**Revisión del 2026-10-07.** RT-DETRv3 era la otra candidata (pregunta 24). En validación da más mAP50-95 (0.83 frente a 0.75, con pycocotools) y, con la regla B, empata en ENA24, con un indicio a favor en iNaturalist. En el equipo local tarda 1.41 s por foto (mediana; 614 MB de pico), así que no cumple la meta de menos de 1 s, y necesita un Python aparte (3.12) con PaddlePaddle. E1 tarda 52 ms. La decisión se mantiene y RT-DETRv3 queda como arquitectura de comparación. La velocidad se midió con la librería de cada modelo en su modo por defecto; un formato optimizado podría acelerar a los dos, y no se probó. Pendiente de validar con el director.

## [2026-10-01] Sin experimento de resolución mayor (D20)
 
- **Contexto**: El plan dejaba como condicional entrenar a 800 o 960 px (E5 en `experimentos.md`), solo si el análisis de errores mostraba que se escapaban osos pequeños o lejanos. El análisis se hizo sobre validación con los seis modelos y la regla B.
- **Alternativas consideradas**:
  1. Entrenar nano a 800 o 960 px y compararlo con E1.
  2. No correrlo y dejarlo para la iteración 2, si el material real lo justifica.
- **Decisión tomada**: No se corre. En ENA24 de validación ningún oso mide menos de 32 px de lado tal como lo ve la red a 640, y de las 22 fotos de ENA24 que pierde algún modelo solo 2 son osos lejanos, de la misma escena. E1 pierde 22 fotos en validación y a lo sumo 3 son osos pequeños, todas de iNaturalist. Los fallos se deben sobre todo a osos tapados, con poca visibilidad, muy cerca de la cámara o cortados por el borde, que una resolución mayor no resuelve.
- **Consecuencias**: Se ahorra GPU y el protocolo sigue en 640 px. Si el material de Angochagua trae osos lejanos o pequeños, se reconsidera en la iteración 2. Pendiente de validar con el director.

## [2026-10-02] Ambas fuentes del proxy en el fine-tuning de la iteración 2 (D21)
 
- **Contexto:** la iteración 2 hará fine-tuning desde E1 con el material real. Si los datos reales son pocos, conviene mezclar parte del proxy, y había que saber qué fuente conservar (pregunta 22). La ablación por fuente (A1 y A1b solo con ENA24, A2 solo con iNaturalist) se evaluó sobre validación con la regla B y con el recall por imagen igualado.
- **Alternativas consideradas:**
  1. Conservar solo ENA24, que es cámara trampa como el material real.
  2. Conservar solo iNaturalist, que es la especie.
  3. Conservar ambas.
- **Decisión tomada:** la opción 3.
  - **Sin ENA24 (A2):** el modelo no alcanza la meta en cámara trampa. Aun aceptando todos sus avisos detecta como máximo 0.85 de las fotos con oso, con 36 % de falsas alarmas, de día y de noche por igual.
  - **Sin iNaturalist (A1, A1b):** con el mismo recall en cámara trampa rinde parecido, con algo más de falsas alarmas (sin diferencia demostrada) y cajas menos precisas. En fotografías comunes detecta solo 0.53–0.59 de los osos andinos y da 20–23 % de falsas alarmas con otras especies andinas.
  - El material de Angochagua tendrá oso andino y fauna andina, así que lo que aporta iNaturalist es pertinente.
- **Consecuencias:**
  - El fine-tuning de la iteración 2 usará las dos fuentes del proxy junto al material real, si se mezcla.
  - Queda sin comprobar la elección de época de A1 y A1b, porque sus `last.pt` no se evaluaron.
  - Todo es sobre validación; la ablación se repetirá sobre prueba en la evaluación final.
  - Pendiente de validar con el director.

## [2026-10-05] RT-DETRv3 con la implementación oficial en PaddlePaddle (D22)

- **Contexto:** D9 daba dos días de trabajo de entorno para la implementación oficial de RT-DETRv3, con el RT-DETR de Ultralytics como respaldo.
  - El repositorio oficial (`clxia12/RT-DETRv3`, último commit del 18 de marzo de 2025) no fija la versión de PaddlePaddle.
  - El 3 de octubre la imagen de Kaggle pasó a Python 3.13 con numpy 2, y el código oficial necesita numpy anterior a 2.
  - Ultralytics 8.4.138 solo trae el RT-DETR original. Su variante más pequeña con pesos, RT-DETR-L, tiene unos 32 M de parámetros según el repositorio oficial de RT-DETR, así que no se empareja con YOLO26m.
- **Alternativas consideradas:**
  1. La implementación oficial con PaddlePaddle 3.3.0, del índice de Paddle, en un entorno virtual con Python 3.12.
  2. La misma con `paddlepaddle-gpu` 2.6.2 de PyPI.
  3. El RT-DETR de Ultralytics (respaldo de D9).
  4. Una reimplementación no oficial en PyTorch (descartada en D9).
- **Decisión tomada:** la opción 1. Paddle 3.3.0 funciona en la T4 y el código oficial corre con ella tras resolver quince problemas de entorno, de memoria y de datos (tabla en `experimentos.md`). La única modificación del código oficial es el parche de D24. La opción 2 no hizo falta probarla. La 3 habría cambiado la arquitectura comparada (RT-DETR original en lugar de v3) y el emparejamiento de parámetros.
- **Consecuencias:**
  - La comparación es contra RT-DETRv3, como pidió el director, y no hay que declarar un cambio de arquitectura.
  - YOLO26 y RT-DETRv3 se entrenan con librerías distintas. Por eso RT-DETRv3 se evalúa a partir de sus detecciones exportadas, con el mismo cuaderno y la misma regla B, y el mAP que se compare entre arquitecturas lo calcula el cuaderno igual para todos (pregunta 32). El aumento de datos y el reescalado al evaluar son los de cada librería y se declaran.
  - El entorno de Paddle se rehace en cada sesión de Kaggle, unos 9.5 minutos.
  - Medir RT-DETRv3 en CPU exige otro Python en el equipo local, porque PaddlePaddle no tiene versión para Python 3.14.
  - Los issues recientes del repositorio no tienen respuesta del autor (#13 y #22, vistos el 3 de octubre): lo que falle hay que resolverlo por cuenta propia.

## [2026-10-05] Configuración de la corrida completa de RT-DETRv3 (D23)

- **Contexto:**
  - Receta oficial de RT-DETRv3-R18: 72 épocas, 4 GPU con 16 fotos cada una (64 por paso), AdamW con lr 0.0004, calentamiento de 2000 pasos, EMA, BatchNorm sincronizada, precisión completa y pesos de ImageNet.
  - En una T4 de 15 GiB no caben ni 16 ni 8 fotos por GPU (dos intentos fallidos por falta de memoria); 4 fotos usan entre 10.3 y 11.5 GiB.
  - Con una GPU y 4 fotos, cada época tarda unos 12 minutos: 100 épocas serían unas 20 horas, más que una sesión de Kaggle (12 horas como máximo).
  - YOLO26 se entrenó con pesos COCO, 100 épocas y la mejor época por mAP50-95 de validación.
- **Alternativas consideradas:**
  - Épocas: las 72 de la receta o las 100 del protocolo de YOLO26.
  - Pesos de partida: ImageNet, como la receta, o COCO, como YOLO26.
  - Tasa de aprendizaje: la de YOLO26 (0.002), la oficial sin cambiar (0.0004) o la oficial escalada al lote total.
  - Hardware y precisión: 1 o 2 T4, en precisión completa o mixta. Se eligió con un sondeo de 6 minutos por configuración y una regla fijada antes de ver los números: la más fiel a la receta con la que 100 épocas quepan en 11 horas.
- **Decisión tomada:**
  - Igual que YOLO26, para que la diferencia medida venga de la arquitectura: datos y división con las fotos sin oso, pesos COCO, 100 épocas, validación en cada época y mejor época por mAP50-95 de validación, y la misma evaluación.
  - Propio de RT-DETRv3, porque los modelos tipo DETR son sensibles a la receta y una ajena mediría una mala receta y no la arquitectura: optimizador, calentamiento, EMA, aumento de datos y reescalado al evaluar. La tasa oficial se escala en proporción al lote total: 0.0004 × 8 / 64 = 0.00005.
  - 2 T4 con 4 fotos cada una, precisión completa y BatchNorm sincronizada (unas 14.6 horas). Ninguna configuración cabía en 11 horas, así que la regla no se podía cumplir tal cual. Las dos más rápidas necesitaban de todos modos dos sesiones y se diferenciaban en 1.2 horas, y se eligió la más fiel en lugar de la más rápida (precisión mixta, 13.4 horas).
  - La corrida se reparte en dos sesiones de Kaggle que retoman desde el último punto de control, y el mejor modelo se elige entre las épocas de todas las sesiones.
  - Semilla 0 (E6) primero. La semilla 1 (E6b), cuando haya cuota.
- **Consecuencias:**
  - Unas 15 horas de GPU por semilla, frente a 2.5 a 2.8 de YOLO26m.
  - Desviaciones de la receta oficial que hay que declarar: lote total 8 en lugar de 64 (la BatchNorm sincronizada calcula sus estadísticas con 8 fotos y el calentamiento de 2000 pasos dura unas 6 épocas), 100 épocas en lugar de 72, pesos COCO en lugar de ImageNet, 2 T4 en lugar de 4 GPU, y el parche de D24. Puede que el lote pequeño le reste algo de precisión. Es una limitación, no un error.
  - Si la cuota no alcanza, se reportará una sola semilla de RT-DETRv3 frente a dos de YOLO26m (pregunta 31).
  - La sesión 2 depende de una reanudación probada en CPU y no en GPU.
  - Revertir cuesta otras 15 horas de GPU.

  
**Resultado del 2026-10-06.** Las dos semillas se entrenaron en paralelo desde el 5 de octubre, en dos sesiones cada una: E6 con 73 y 27 épocas, y E6b con 68 y 32. La reanudación funcionó en GPU: la sesión 2 retomó optimizador, tasa y EMA. Sin NaN, con 12 055 MiB de memoria como máximo. Unas 15.4 horas de GPU en E6 y 16.1 en E6b. Mejores épocas: la 85 y la 44, con las curvas planas desde la 57 y la 43. No hizo falta reportar una sola semilla. Detalle en `experimentos.md`.

## [2026-10-05] Lotes sin cajas: parche versión 2 al código oficial (D24)

- **Contexto:**
  - Con `allow_empty: true` entran las 1274 fotos sin oso del entrenamiento, el 48 %.
  - Si en un lote ninguna foto tiene cajas, la parte de denoising no devuelve nada y el código de RT-DETRv3 falla al unir piezas (`paddle.concat` con `None` en `rtdetr_transformerv3.py`). El RT-DETR original sí contempla ese caso; la versión 3, con varios grupos de consultas, no.
  - Con 16 fotos por lote es raro (1 de cada 134 000 lotes, más o menos). Con 4 fotos por GPU pasa en uno de cada 19 lotes, más o menos: en E6v, 106 lotes en 3 épocas.
- **Alternativas consideradas:**
  1. Saltar esos lotes (versión 1, usada en E6v). Pierde cerca del 11 % de las fotos sin oso en cada época, que son las que enseñan a no dar falsas alarmas. Con 2 GPU congelaría el entrenamiento: una GPU saltaría su lote mientras la otra la espera.
  2. Entrenar esos lotes sin denoising, como hace el RT-DETR original (versión 2).
  3. Lotes más grandes, que harían el caso casi imposible. No caben en la T4.
- **Decisión tomada:** la versión 2. Toca tres sitios: `_get_decoder_input` y `forward` de `rtdetr_transformerv3.py`, y `DINOv3Head.forward` de `detr_head.py`. En los grupos sin denoising calcula la pérdida sin esa parte y repite el conteo de cajas entre GPU que habría hecho la pérdida de denoising, para que las dos GPU hagan las mismas operaciones en común.
  - Probada en CPU con lotes solo de fotos sin oso, con lotes mixtos y con dos procesos, en los que a menudo uno tenía cajas y el otro no: terminó sin fallar ni congelarse.
  - En Kaggle con 2 GPU, el sondeo y los primeros 532 pasos de E6 corrieron sin fallar ni congelarse. Por la proporción de negativos debieron pasar por decenas de esos lotes. Es una estimación: el parche no los cuenta.
- **Consecuencias:**
  - Es la única modificación al código oficial y se declara como desviación. Afecta a una parte pequeña de los pasos, cerca del 5 % de los lotes de 4 fotos.
  - Las fotos sin oso se usan todas en cada época.
  - E6v se entrenó con la versión 1. Solo era una verificación.
  - El parche queda en `scripts/parche_lotes_sin_cajas_v2.py` y el `git diff` de cada sesión, en `entorno_s<N>.txt`.
  - Revertirlo obliga a reentrenar.

**Resultado del 2026-10-06.** E6 y E6b completaron sus 100 épocas con 2 GPU sin fallar ni congelarse. Por la proporción de negativos, cada corrida debió pasar por unos 3 500 lotes sin cajas (estimación: cerca del 5 % de los lotes de 4 fotos, en 332 pasos por época, 2 GPU y 100 épocas). El parche no los cuenta.

## [2026-10-03] Regla para entrenar YOLO26m con más épocas (D25)

- **Contexto:** YOLO26m (E4 y E4b) tuvo su mejor época en la 100 con las dos semillas, con las pérdidas de validación todavía bajando: no había convergido. Si RT-DETRv3 converge antes de 100 épocas y supera a YOLO26m, la comparación puede leerse como sesgada a su favor. El plan dejaba la decisión para cuando se viera RT-DETR, y decidir después de ver los resultados permite elegir la opción que convenga.
- **Alternativas consideradas:**
  1. Decidir al ver los resultados de RT-DETRv3.
  2. Entrenar siempre YOLO26m con más épocas.
  3. Fijar una regla antes de ver RT-DETRv3.
- **Decisión tomada:** la opción 3, fijada el 3 de octubre, antes de entrenar RT-DETRv3. Se entrena YOLO26m 200 épocas, como corrida adicional, si se cumplen dos condiciones:
  - RT-DETRv3 tiene su mejor época en la 90 o antes con sus dos semillas;
  - y supera a YOLO26m en mAP50-95 de validación calculado por el cuaderno de evaluación.

  Si no se cumplen, se comparan a 100 épocas y se declara. Con 200 épocas, Ultralytics sigue eligiendo AdamW con lr 0.002, como en E4 (8400 iteraciones nominales; desde 239 épocas cambiaría a MuSGD). Costaría unas 5 a 6 horas de T4.
- **Consecuencias:** las 100 épocas siguen siendo el protocolo principal. Si la regla se cumple, YOLO26m con 200 épocas es un punto de comparación adicional y correría en la imagen nueva de Kaggle (Python 3.13, torch 2.11). Hace falta que el cuaderno de evaluación calcule mAP50-95, porque el de PaddleDetection y el de Ultralytics no se calculan igual.

**Precisiones pendientes (2026-10-05), a confirmar antes de evaluar E6 con el cuaderno versión 3.** La regla no dice qué hacer en dos casos. Propuesta:
1. Si al evaluar E6 todavía no hay E6b, la regla se aplica con E6 sola, de forma provisional, y se revisa cuando llegue E6b. Si E6b no llega a entrenarse, la decisión tomada con E6 queda como definitiva y se declara.
2. "Supera" significa que el mAP50-95 global de validación de RT-DETRv3 queda por encima de la media de E4 y E4b por más de 0.015, la regla de lectura de esta iteración. Dentro de ese margen se leen como empate y se comparan a 100 épocas.


**Precisiones fijadas (2026-10-06), antes de correr el cuaderno de evaluación versión 3.**
1. La precisión 1 ya no hace falta: E6 y E6b terminaron a la vez.
2. "Supera" significa que la media del mAP50-95 global de validación de E6 y E6b queda por encima de la media de E4 y E4b por más de 0.015. Las cuatro cifras las calcula pycocotools 2.0.11 en el cuaderno (D26), con las detecciones de confianza ≥ 0.001 (hasta 300 por foto) y las cajas recortadas al borde en los dos modelos. Se mantiene lo que se escribió el 5 de octubre, cuando todavía no había resultados de RT-DETRv3: la métrica global, la media de E4 y E4b y el margen de 0.015. Solo se añade cómo se usan las dos semillas y con qué código se calcula.
3. La regla solo se aplica si pasan dos controles: E1 y E1b repiten septiembre, y pycocotools repite el mAP50-95 de PaddleDetection de E6 y E6b (de −0.001 a +0.010).

Estos parámetros se escribieron en la celda 0 del cuaderno antes de correrlo.

**Aplicación (2026-10-06).** Los dos controles pasaron. Condición 1: mejores épocas 85 (E6) y 44 (E6b). Condición 2: la media de RT-DETRv3 es 0.8289 y la de YOLO26m 0.7451, una diferencia de +0.0838. Se cumplen las dos, así que se entrena YOLO26m con 200 épocas (E7, semilla 0).

Nota sobre la condición 1: con una curva plana, la mejor época se decide por el cuarto decimal. En E6, la época 85 ganó a la 96 y a la 98 por menos de 0.0003. Si hubiera ganado una de esas, la condición habría fallado por la letra con la misma curva. Aquí la letra y la intención coinciden, porque las dos curvas se aplanan desde la época 57 y la 43. En reglas futuras conviene definir la convergencia por la meseta y no por la mejor época.

**Después de E7 (2026-10-07), fijado antes de revisar E7.**
1. E7 se evalúa igual que las demás: cuaderno versión 3 y `calcular_umbral.py`, con E1 y E1b de control.
2. Se compara el mAP50-95 global de validación (pycocotools) de E7 con la media de E6 y E6b (0.8289), con el margen de 0.015. Si E7 queda a 0.015 o menos, se lee que con 200 épocas YOLO26m alcanza a RT-DETRv3. Si queda más abajo, que la ventaja de RT-DETRv3 no venía solo de que medium no había convergido. También se compara con la media de E4 y E4b (0.7451), para medir lo que ganó medium con 100 épocas más.
3. E7b (YOLO26m, 200 épocas, semilla 1) solo se entrena si E7 queda a menos de 0.03 de la media de RT-DETRv3. La lectura cambia en 0.015, y la mayor diferencia entre dos semillas vista en la iteración fue 0.014: cerca de ese punto, otra semilla podría cambiar la conclusión. Si E7 queda a 0.03 o más, una semilla basta y se declara.
4. Medium no se entrena más de 200 épocas en esta iteración, aunque la mejor época de E7 vuelva a ser la última. Si pasa, se declara. Con 239 épocas o más, `optimizer=auto` pasaría a MuSGD y cambiaría la receta.
5. El protocolo principal sigue siendo de 100 épocas. E7 es un punto de comparación adicional.

**Resultado después de E7 (2026-10-07).** Los controles pasaron: E1 y E1b repiten septiembre, RT-DETRv3 repite PaddleDetection (+0.0001) y los seis modelos del 6 de octubre repiten sus cifras.
1. E7 se evaluó igual que los demás modelos.
2. mAP50-95 global de validación (pycocotools): E7, 0.7478. Frente a la media de RT-DETRv3 (0.8289), −0.0811: la ventaja de RT-DETRv3 no venía solo de que medium no hubiera convergido. Frente a la media de E4 y E4b (0.7451), +0.0027: 100 épocas más no dan una ganancia que se distinga del azar.
3. E7b no se entrena: E7 queda a 0.081 de RT-DETRv3, más del límite de 0.03.
4. La mejor época de E7 fue la 165, no la última. Medium se estabiliza desde la época 160, más o menos, al nivel en que terminaron E4 y E4b.
5. El protocolo principal sigue en 100 épocas.

La mejor época de E7 sobresale 0.009 sobre su meseta. Si eso fue suerte, E7 estaría todavía más lejos de RT-DETRv3, así que no cambia la lectura. Las recetas siguen siendo distintas: E7 descarta la falta de convergencia, pero no separa arquitectura y receta. Detalle en `experimentos.md`.


## [2026-10-06] mAP común entre arquitecturas calculado con pycocotools (D26)

- **Contexto:** YOLO26 y RT-DETRv3 vienen de librerías que calculan el mAP de forma distinta. Ultralytics valida en lotes con un borde extra y usa su propia integración de la curva. PaddleDetection usa pycocotools, con hasta 100 detecciones por foto y las cajas sin recortar. Comparar la cifra de cada librería mezclaría la diferencia entre arquitecturas con la diferencia entre cálculos (pregunta 32). La regla de D25 pedía un mAP50-95 calculado por el cuaderno, igual para todos.
- **Alternativas consideradas:**
  1. Reportar la cifra de cada librería.
  2. Calcularlo con el código del cuaderno: el emparejamiento que ya usa para el mAP50, repetido en los 10 umbrales de IoU, con `compute_ap` de Ultralytics.
  3. Calcularlo con pycocotools sobre las detecciones de todos los modelos.
- **Decisión tomada:** la opción 3, con pycocotools 2.0.11 y sus parámetros estándar (IoU de 0.50 a 0.95, 101 puntos de recall y hasta 100 detecciones por foto).
  - Es el evaluador oficial de COCO, el que usan los artículos de RT-DETRv3 y de YOLO, y es más fácil de defender que un cálculo propio.
  - Es el mismo con que PaddleDetection eligió la mejor época de RT-DETRv3, así que sirve de control: sobre las detecciones exportadas repite su cifra (+0.0001).
  - Aplica el mismo código a todos. Entran las detecciones con confianza ≥ 0.001 (hasta 300 por foto) y las cajas recortadas al borde, como hace Ultralytics con las de YOLO26.
  - La opción 2 se calcula igual, como cruce. Sale entre 0.0024 y 0.0032 por encima, con el mismo signo en los seis modelos.
- **Consecuencias:**
  - El mAP50-95 de YOLO26 con pycocotools sale entre 0.008 y 0.019 por debajo del de Ultralytics. En la tesis, las tablas de septiembre (comparación de tamaños) y la comparación de arquitecturas usan cifras distintas, y hay que decirlo.
  - Cierra la pregunta 32.
  - El cuaderno instala pycocotools 2.0.11 en cada sesión de Kaggle.
  - Si hiciera falta, se recalcula en minutos con el cuaderno.
  - Pendiente de validar con el director.

**Comprobación pendiente (2026-10-07), con la lectura fijada antes de calcularla.** RT-DETRv3 produce 300 cajas por foto (con confianza ≥ 0.001 quedan entre 261 y 298 de media) y pycocotools usa hasta 100 por foto, las de mayor confianza. YOLO26 da unas 2 por foto. En el mAP de COCO, las cajas de confianza baja pueden sumar aciertos al final de la curva y casi no restan, porque solo bajan la precisión de los aciertos de confianza todavía menor. Un modelo que entrega muchas cajas puede salir favorecido.
- Se recalcula el mAP de cada modelo con las k detecciones de mayor confianza por foto, k = 1, 3, 10 y 100 (pregunta 33).
- Control: con k = 100 tiene que acercarse a las cifras de `map_comun.csv`.
- Lectura: con k = 3, YOLO26 conserva casi todas sus cajas y los dos modelos compiten con un número parecido. Si la media de E6 y E6b en mAP50-95 global sigue por encima de la de E4 y E4b por más de 0.015, la ventaja de RT-DETRv3 no viene del número de cajas. Si queda a 0.015 o menos, la comparación de mAP entre arquitecturas se reporta con el mismo k para todos y se declara. E7 se informa aparte.
- No afecta a la regla B ni a D19.

**Resultado (2026-11-15).** La comprobación se hizo con la regla tal como se fijó el 7 de octubre.
- **Datos.** La lectura necesita las mismas detecciones con que el cuaderno calcula el mAP (confianza ≥ 0.001). `umbral_detecciones.csv.gz` guarda solo las de confianza ≥ 0.01, y con él el control no pasaba: YOLO26 salía hasta 0.010 por debajo en mAP50-95. Antes de calcular nada por k, el cuaderno pasó a la versión 4, que guarda todas las detecciones, y se volvió a correr: repitió la evaluación del 7 de octubre sin diferencias. El cambio de datos no toca la regla, que pide leer el mismo mAP del cuaderno.
- **Controles.** Con k = 100, el script repite `map_comun.csv` en los siete modelos y las tres particiones (diferencia 0). La variante con solo las cajas ≥ 0.01 para todos y las tolerancias de los controles (0.0005) se fijaron antes de ver ninguna cifra por k.
- **Resultado.** Con k = 3, mAP50-95 global de 0.8232 (E6), 0.8195 (E6b), 0.7391 (E4) y 0.7429 (E4b). Media de RT-DETRv3 0.8214 y de YOLO26m 0.7410: **+0.0804**, más que el margen de 0.015. La variante da +0.0854: la misma lectura.
- **Lectura.** La ventaja de RT-DETRv3 no viene del número de cajas por foto: con una sola caja por foto sigue en +0.071. La comparación de arquitecturas se reporta con el mAP estándar de pycocotools, y esta comprobación como prueba de robustez.
- E7, aparte: 0.7441 con k = 3; RT-DETRv3 queda +0.0773.
- No cambia la regla B ni D19. Detalle en `experimentos.md`, sección "Comprobación de las cajas por foto (pregunta 33, 2026-11-15)".

---
## Decisiones operativas

Decisiones de menor alcance, registradas para trazabilidad. Ninguna requiere justificación extensa.

| Fecha | Decisión | Motivo | Costo de revertir |
|---|---|---|---|
| 2026-09-03 | Las 761 imágenes con detección de confianza mayor o igual a 0.5 del modelo COCO se aceptan provisionalmente como positivas | Todas pasarán por revisión humana durante el etiquetado, así que la verificación no se omite, solo se difiere | Ninguno |
| 2026-09-03 | Los fotogramas extraídos de archivos GIF se nombran `inat_<observacion>_<foto>_frame_NNN.jpg` y se tratan como fotografías de la misma observación | Conserva el vínculo con el registro de licencias y con la agrupación para la división de datos | Ninguno |
| 2026-09-03 | Los negativos de la iteración 1 provienen de ENA24, priorizando perro, caballo y otras especies de tamaño medio y pelaje oscuro | Son negativos difíciles del mismo dominio de cámara trampa, comparables a los confusores esperados en Angochagua | Bajo |
| 2026-09-03 | La carpeta `data/raw` se mantiene intacta tal como se descargó de cada fuente y todo lo derivado vive en otras carpetas | Trazabilidad del origen de cada imagen | Ninguno |
| 2026-09-03 | Se usa el hash perceptual (`imagehash.phash`) para detectar imágenes casi idénticas | Identifica ráfagas de cámara trampa que el JSON de ENA24 no marca | Ninguno |
| 2026-09-07 | Se toman 1200 negativos de ENA24, repartidos en cuota igual entre perro, coyote, lince rojo, venado, gato doméstico y caballo, con mitad nocturnas y un máximo de dos por ráfaga | Diversidad de especies y de condiciones de luz sin llenar la muestra con fotos casi iguales | Bajo, hay unos 5600 candidatos sin usar |
| 2026-09-07 | El pre-etiquetado con el modelo COCO se corre con umbral de confianza 0.2 | Borrar una caja sobrante durante la revisión es más rápido que dibujar una que falta | Ninguno |
| 2026-09-07 | Las cajas propuestas por el modelo se conservan en `labels_pre` y las revisadas se exportan a `labels_rev` | Permite medir cuántas cajas se corrigieron, se borraron y se dibujaron de nuevo, y da una copia de seguridad del trabajo manual | Ninguno |
| 2026-09-09 | De cada GIF encontrado entre los negativos se conserva un solo fotograma, el central | Un negativo aporta la misma información en todos sus fotogramas, a diferencia de un positivo donde cada pose es un ejemplo nuevo | Ninguno |
| 2026-09-09 | Entre los negativos de otras especies se excluyen los entornos urbanos e interiores y se conservan los rurales y naturales | Un perro dentro de un auto no se parece a nada de lo que verá una cámara trampa en Angochagua | Bajo |
| 2026-09-09 | Los archivos de ENA24 se renombran con el prefijo `ena24_` al construir el dataset | La procedencia de cada imagen se lee en el nombre, sin consultar ninguna tabla | Ninguno |
| 2026-09-10 | La división usa semilla fija 42 y queda registrada imagen por imagen en `division.csv` | Reproducibilidad, y las dos arquitecturas se comparan sobre exactamente la misma partición | Ninguno |
| 2026-09-10 | Las etiquetas revisadas se copian a `docs/preparacion_datos/` para que entren en el control de versiones | Son horas de trabajo manual y `data/` está excluido del repositorio | Ninguno |
| 2026-09-10 | Tras cada construcción del dataset se dibujan las cajas sobre una muestra de imágenes de cada fuente y se revisan a ojo | Una conversión de coordenadas mal hecha produce cajas válidas en formato pero en el lugar equivocado, y solo se detecta mirando | Ninguno |
| 2026-09-15 | Las listas de rutas por fuente (`val_inat.txt`, `val_ena24.txt`, `test_inat.txt`, `test_ena24.txt`) se guardan en la raíz de `data/processed`, como hace COCO, y no en una subcarpeta | Ultralytics resuelve el prefijo `./` de cada línea relativo a la carpeta del `.txt`, no al campo `path` del yaml. Verificado en `ultralytics/data/base.py` de la versión 8.4.138 y con una validación de prueba | Ninguno |
| 2026-09-15 | 35 imágenes de iNaturalist con extensión `.jpg` pero codificadas como GIF animado (14, de 2 a 117 fotogramas) o PNG (21) se reescriben como JPEG en `data/processed`, primer fotograma, mismo nombre y dimensiones | Ultralytics descarta los GIF al escanear el dataset sin detener el entrenamiento. Las etiquetas están normalizadas y se dibujaron sobre el primer fotograma, así que siguen valiendo. `data/raw` no se toca. Inventario en `docs/preparacion_datos/formatos_no_jpeg.txt` | Ninguno |
| 2026-09-16 | La versión de Ultralytics queda fijada en 8.4.138, la instalada en el equipo local, y se instala esa misma en Kaggle | Los valores por defecto de los argumentos cambian entre versiones y el diseño de pruebas se apoya en ellos | Ninguno |
| 2026-09-16 | La selección de modelo y de umbrales se hace solo sobre validación; el conjunto de prueba se evalúa una vez por modelo final | Evitar que las métricas de prueba queden contaminadas por decisiones tomadas mirándolas | Ninguno |
| 2026-09-29 | En la evaluación, las imágenes se procesan de una en una (lote de 1), pasando a Ultralytics un archivo de texto con las rutas | Con una lista de Python, Ultralytics carga todas las imágenes en un solo lote y agota la memoria de la GPU (ocurrió en el primer intento). Procesarlas de una en una reproduce además lo que hará la aplicación | Ninguno |
| 2026-09-30 | Las particiones de día y noche se definen solo sobre ENA24 | `dividir_dataset.py` marcó todas las fotos de iNaturalist como diurnas. En validación hay 9 fotos de iNaturalist en blanco y negro que no entran en esas particiones | Ninguno |
| 2026-09-18 | Cada tamaño de YOLO26 se entrena dos veces, con semillas 0 y 1 | Dos entrenamientos iguales no dan el mismo resultado. Sin repetición no se puede saber si una diferencia entre modelos es efecto o azar. La repetición de small mostró que su aparente desventaja era azar | Ninguno |
| 2026-09-30 | La velocidad en CPU se mide con un modelo por ejecución, sobre 100 fotos fijas de validación (50 por fuente, semilla 0) tras 10 de calentamiento, a 640 px, foto por foto y con el umbral de la regla B de cada modelo. Se reporta el tiempo total con la lectura del JPEG y los tiempos de Ultralytics | Medir lo que espera quien use la app, sin sumar la memoria de dos modelos | Ninguno |
| 2026-09-30 | Ablación por fuente: A1 y A1b (solo ENA24, semillas 0 y 1) y A2 (solo iNaturalist), con el protocolo 1.3, 100 épocas y `best.pt` elegido con la validación completa, como E1. Nombres `a<N>[b]_<tamaño>_<fuente>_s<semilla>` | Mismo protocolo que E1. Dos semillas en la comparación principal (solo ENA24 frente a ambas fuentes, medida sobre ENA24) | Bajo |
| 2026-10-01 | Las dos dudas de etiqueta del análisis de errores (`inat_190412124_333788182_frame_000` e `inat_268470424_482568294`) se revisaron y las etiquetas se mantienen. Si apareciera un error de etiqueta en validación, se documenta y no se regenera el dataset en esta iteración | El umbral se fija sobre ENA24 y no depende de esas fotos; regenerar el dataset obliga a volver a subirlo a Kaggle | Ninguno |
| 2026-10-02 | WCS Camera Traps: se baja solo el subconjunto de oso andino (462 fotos) a `data/raw/wcs/`, se revisa a ojo con las categorías del protocolo (oso, sin oso, duda) y no se usa para entrenar ni evaluar hasta decidirlo con el director | Saber qué contiene antes de proponer su uso, sin tocar el proxy | Ninguno |
| 2026-10-02 | Si un modelo no alcanza la regla B, se reporta con el umbral más bajo que deja las falsas alarmas por debajo del 10 % y se dice qué meta falla. La ablación se lee además con el recall por imagen igualado | La regla no da un umbral para esos modelos y hay que compararlos de alguna forma explícita | Ninguno |
| 2026-10-02 | El modelo COCO sin entrenar se evalúa solo con su clase `bear` (`classes=[21]`), como línea base | Es la pregunta de cuánto aporta el entrenamiento. Las otras 79 clases no corresponden al problema | Ninguno |
| 2026-10-02 | Cada corrida nueva del cuaderno de evaluación incluye E1 y E1b como control | Si repiten las cifras de septiembre, los cambios en el cuaderno no alteraron los resultados | Ninguno |
| 2026-10-02 | `calcular_umbral.py` redondea hacia abajo el umbral A a 4 decimales | Las confianzas se guardan con 4 decimales; sin redondear, la detección que define el umbral podía quedar fuera. No cambia las cifras de septiembre | Ninguno |
| 2026-10-03 | PaddlePaddle se instala solo en un entorno virtual con Python 3.12.13 y 67 versiones fijas, nunca en el Python de Kaggle | En el Python de Kaggle reemplaza las librerías de CUDA y cuDNN de torch por versiones que pip marca como incompatibles. El código oficial necesita numpy anterior a 2, que no existe para Python 3.13 | Ninguno |
| 2026-10-03 | Las anotaciones COCO para RT-DETRv3 se generan en el propio cuaderno a partir de `labels/`, con todas las imágenes (las negativas, sin anotaciones), teniendo en cuenta la orientación EXIF y comparando los conteos con el informe de preparación | Misma división y mismas cajas que YOLO26, sin archivos intermedios que mantener | Ninguno |
| 2026-10-03 | La semilla de RT-DETRv3 se fija con un lanzador propio (`entrenar_con_semilla.py`) que llama a `ppdet.engine.set_random_seed` | La opción `--enable_ce` del repositorio solo admite la semilla 0. El lanzador no modifica el código oficial | Ninguno |
| 2026-10-03 | Nombres de las corridas de RT-DETRv3: `e6v_rtdetrv3r18` (verificación), `e6_rtdetrv3r18_s0` y `e6b_rtdetrv3r18_s1` | El mismo esquema de número, `b` y semilla que YOLO26; la arquitectura reemplaza al tamaño y la rotación | Ninguno |
| 2026-10-05 | La carga de datos de PaddleDetection va sin memoria compartida (`use_shared_memory: false`) | Con memoria compartida, las fotos sin oso tumban el lector (Paddle 3.3.0) | Ninguno: en el sondeo la espera de datos fue 0 % |
| 2026-10-05 | La tasa de aprendizaje se escribe con decimales fijos (`0.00005000`) en la configuración y en las opciones `-o` | YAML 1.1 lee `5e-05` como texto | Ninguno |
| 2026-10-05 | Solo se conservan los puntos de control de la última época completa y del mejor modelo | Cada época ocupa unos 356 MB; 100 serían unos 36 GB | Ninguno |
| 2026-10-05 | Una sesión de Kaggle no empieza una época que no termine antes de 11 horas. La corrida sigue en otra sesión desde el último punto de control, y el mejor modelo se elige entre las épocas de todas las sesiones (en empate, la más tardía) | 100 épocas de RT-DETRv3 no caben en una sesión de 12 horas | Ninguno |
| 2026-10-05 | Las detecciones de validación de RT-DETRv3 se exportan desde la evaluación de PaddleDetection (300 por foto) al formato del cuaderno de evaluación, con confianza ≥ 0.001 | Evaluar las dos arquitecturas con el mismo emparejamiento y la misma regla B | Ninguno |
| 2026-10-06 | Las detecciones exportadas de RT-DETRv3 y sus `_best.json` se suben como Kaggle Dataset (`rtdetrv3deteccionoso`), y el cuaderno de evaluación las lee como modelos de tipo `externo` | Evaluar las dos arquitecturas en el mismo cuaderno sin depender de las salidas de otras cuentas | Ninguno |
| 2026-10-06 | El control frente a PaddleDetection acepta de −0.001 a +0.010 | La exportación recorta al borde las cajas que PaddleDetection evaluó sin recortar, y eso solo puede subir el mAP. El control busca errores gruesos, como un archivo de otra corrida | Ninguno |
| 2026-10-06 | `entrenar_yolo26.ipynb` versión 3 pasa `patience=EPOCHS` | Con más de 100 épocas, el `patience=100` por defecto permitiría una parada temprana, y el protocolo es sin parada temprana | Ninguno: con 100 épocas no cambia nada |
| 2026-10-06 | E7 se llama `e7_m_deg0_s0` y su segunda semilla, si se entrena, `e7b_m_deg0_s1` | El patrón de YOLO26 | Ninguno |
| 2026-10-07 | La velocidad de RT-DETRv3 en CPU se mide con su propia librería (PaddlePaddle 3.3.0 en modo dinámico, sin exportar), con 8 hilos como YOLO26 y lo demás en la configuración por defecto de Paddle (oneDNN sin activar), con las mismas 100 fotos y el mismo protocolo del 30 de septiembre y un control frente a las confianzas de la evaluación en GPU | Medir cada arquitectura como la entrega su librería, sin perjudicar a ninguna por los hilos. Exportar las dos a un formato común sería otro experimento | Ninguno |
| 2026-10-07 | Paddle va en el equipo local en un entorno virtual aparte (`C:\Users\Martin\venv_paddle`, Python 3.12.9) con las 67 versiones de Kaggle, y el código oficial en `C:\Users\Martin\RT-DETRv3`, fuera del proyecto | PaddlePaddle no tiene versión para Python 3.14, y usar las mismas versiones que produjeron los pesos evita diferencias | Ninguno |
| 2026-10-07 | E7b no se entrena | Regla de D25: E7 queda a 0.081 de RT-DETRv3, más del límite de 0.03 | Ninguno |
| 2026-10-07 | El resumen para el director se actualiza el día anterior a la reunión, o antes si la reunión se adelanta | Que recoja todo lo hecho hasta la reunión | Ninguno |
| 2026-11-15 | El cuaderno de evaluación versión 4 guarda todas las detecciones de los modelos de Ultralytics, desde 0.001 y sin redondear (`detecciones_completas.csv.gz`). La segunda corrida va en `docs/modelado/eval_p33/` y `eval_e7/` queda como estaba | Con `umbral_detecciones.csv.gz` (≥ 0.01, 4 decimales) no se repite el mAP del cuaderno: en YOLO26 faltan hasta 0.010 de mAP50-95 | Ninguno |
| 2026-11-15 | La pregunta 33 se lee con las detecciones completas de todos los modelos; como variante, todos con solo las cajas ≥ 0.01. Si no coinciden, no se concluye. Tolerancias de los dos controles: 0.0005 | Fijado antes de ver ninguna cifra por k, para que el resultado no decida el método | Ninguno |
| 2026-11-15 | Las cifras de mAP con k < 100 cajas por foto solo sirven para la comprobación. La comparación de arquitecturas se reporta con el mAP estándar de pycocotools | Es el que usan los artículos y el que fija D26 | Ninguno |
| 2026-11-15 | Las preguntas metodológicas que ya tenían ficha se cierran con ella, como decisiones del estudiante con la delegación del director | El director delegó las decisiones; si él o el tribunal plantean otra cosa, se reabren | Ninguno |