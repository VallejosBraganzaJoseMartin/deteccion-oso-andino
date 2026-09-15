# Informe de preparación de los datos (CRISP-DM, iteración 1: dataset proxy)

Entregable de la tercera fase de CRISP-DM. Cubre del 7 al 10 de septiembre de 2026 y describe cómo se pasó de las imágenes descargadas y revisadas en la fase anterior a un dataset listo para entrenar. Todas las cifras salen de las salidas de los scripts y de los CSV que se citan en cada apartado.

Las decisiones que aquí se mencionan tienen ficha completa en `registro_decisiones.md`. Este informe explica qué se hizo; ese documento explica por qué se eligió esa opción y no otra.

## 1. Punto de partida y objetivo

La fase de comprensión dejó dos conjuntos de imágenes descritos y revisados, pero sin tocar: 1424 fotografías de oso andino de iNaturalist y 8789 imágenes de cámara trampa de ENA24 con sus anotaciones en formato COCO. El objetivo de esta fase era construir con ellas un dataset en el formato que espera Ultralytics, con las cajas delimitadoras verificadas por una persona y con una división en entrenamiento, validación y prueba que no permita que el modelo vea en evaluación nada parecido a lo que memorizó entrenando.

## 2. Selección de los datos

### 2.1 Positivos de iNaturalist

El script `consolidar_inat.py` cruza el registro de licencias, la detección zero-shot del modelo preentrenado en COCO y la revisión manual, y produce `inat_estado.csv` con una fila por imagen y un estado por fila. Sobre 1444 filas (los 1424 archivos descargados más los 20 fotogramas extraídos de ocho GIF) el reparto quedó así:

| Estado | Imágenes |
|---|---|
| Positivo confirmado en revisión manual | 375 |
| Positivo con detección de confianza alta, sin revisar | 763 |
| Excluido: rastro | 118 |
| Excluido: duda | 111 |
| Excluido: sin oso | 68 |
| Excluido: cautiverio | 1 |
| Pendiente: archivo GIF sustituido por sus fotogramas | 8 |

Los 1138 positivos provienen de 561 observaciones distintas, 869 tienen coordenadas dentro de Ecuador continental y sus licencias son 1037 CC BY-NC, 76 CC BY y 25 CC0. Los ocho archivos GIF originales quedan fuera del dataset pero no se borran: siguen en `data/raw` y aparecen en la tabla con estado propio, sustituidos por sus fotogramas.

### 2.2 Imágenes de ENA24

El script `seleccionar_ena24.py` toma como positivas las 893 imágenes con al menos una caja de oso negro americano, que suman 959 cajas, 586 de ellas nocturnas y repartidas en 850 grupos de duplicados.

Para los negativos se partió de las imágenes sin ninguna caja de oso que estuvieran en disco y no corruptas, con tres criterios: cuota igual por especie entre perro, coyote, lince rojo, venado, gato doméstico y caballo, mitad nocturnas y mitad diurnas dentro de cada especie, y un máximo de dos imágenes por ráfaga para que la muestra no se llenara de fotos casi iguales. Se pidieron 1200 y se obtuvieron 1200: perro 225, coyote 225, lince rojo 224, venado 224, gato doméstico 224, caballo 54 y otras especies 24, con 782 nocturnas repartidas en 1077 grupos.

El caballo no llegó a su cuota porque solo hay 56 candidatos, y ahí apareció algo que la fase anterior no había detectado: los conteos por categoría del JSON no coinciden con lo que hay en disco. LILA BC excluye de la descarga las imágenes que contienen personas, y como una imagen puede tener anotaciones de dos categorías, las fotos de caballo con jinete se fueron con ellas. De 338 imágenes de caballo que declara el JSON quedan 60 en disco. El faltante se repartió entre las demás especies.

Quedan unos 4400 candidatos a negativo sin usar, que es el margen disponible si más adelante hiciera falta subir la proporción de negativos.

### 2.3 Negativos de iNaturalist

Esta parte no estaba prevista en el plan y se añadió tras construir la primera versión del dataset. El problema que la motivó se explica en el apartado 6.2.

Con `descargar_inat_negativos.py`, derivado del script de descarga original, se bajaron 70 fotografías de cada una de ocho especies, dentro del recuadro geográfico de Ecuador continental y con los mismos filtros de licencia. Cuatro son fauna que comparte hábitat con el oso (lobo de páramo, tapir de montaña, puma y venado de cola blanca) y cuatro son animales domésticos que una cámara trampa en zona de conflicto ganadero capta con frecuencia (perro, vaca, caballo y oveja). Para los domésticos hubo que levantar los filtros de grado de investigación y de cautividad, porque iNaturalist marca como cautivas casi todas esas observaciones y con el filtro puesto no habría material.

De las 560 descargadas se descartaron 171 tras revisarlas una a una: 128 tomadas en entornos urbanos o interiores, 2 fotografías de la pantalla de una cámara y 41 archivos que resultaron ser GIF animados, cada uno sustituido por un fotograma. Quedan 430.

A estas se suman las imágenes de iNaturalist que la fase anterior había excluido y que ahora vuelven como negativos: 118 de rastro y 63 de paisaje. Las de cautiverio, las dudas y las que resultaron ser fotos de pantalla o montajes siguen fuera, porque todas contienen un oso de una forma u otra y como negativos enseñarían lo contrario de lo que se busca.

Total de negativos de iNaturalist: 611.

## 3. Limpieza de los datos

La limpieza de esta fase fue casi toda manual y consistió en sacar del dataset lo que no debía estar. Se hizo con listas de texto (`nombre_archivo;motivo`) que los scripts leen, en lugar de borrar archivos, de modo que cada exclusión queda registrada con su razón y `data/raw` permanece intacta.

Durante el etiquetado se descartaron 10 imágenes que habían entrado como positivas: 7 resultaron ser rastros que la revisión anterior no había detectado, 2 osos en cautiverio y 1 caso dudoso. Se movieron a `data/etiquetado/inat/descartadas` con sus etiquetas.

Entre los negativos recién descargados se descartaron las 171 ya mencionadas.

Un caso que conviene dejar escrito porque justifica una decisión metodológica: tres imágenes del lote de confianza baja se habían borrado de la carpeta de revisión durante la fase anterior, antes de quedar clasificadas, porque no encajaban con claridad en ninguna categoría. Como `data/raw` nunca se modifica, se recuperaron desde ahí y se registraron como duda, que es lo que corresponde cuando un revisor no puede decidir. Sin esa regla, el conteo habría quedado con un agujero sin explicación.

## 4. Construcción de los datos: etiquetado

### 4.1 Pre-etiquetado automático

`preetiquetar_inat.py` copió los 1138 positivos a una carpeta de trabajo y corrió YOLO26n preentrenado en COCO sobre ellos, guardando las detecciones de la clase "bear" en formato YOLO. El umbral se fijó en 0.2, más bajo que el habitual, con el criterio de que borrar una caja sobrante durante la revisión cuesta menos que dibujar una que falta.

Resultado: 935 imágenes recibieron al menos una caja, 131 de ellas más de una, y 203 quedaron sin ninguna. Se propusieron 1091 cajas en total. Las 203 sin caja pertenecen todas al grupo que ya se había revisado a mano en la fase anterior, es decir, los osos que el modelo de COCO no había detectado: lejanos, en penumbra o tapados por vegetación.

### 4.2 Corrección manual

La corrección se hizo con X-AnyLabeling 4.0.6 en local, imagen por imagen, siguiendo el protocolo que se detalla en `protocolo_anotacion.md`. Las cajas propuestas se conservaron en una carpeta aparte (`labels_pre`) y las revisadas se exportaron a otra (`labels_rev`), lo que permitió medir después cuánto había cambiado el revisor.

Sobre las 1128 imágenes que quedaron tras los descartes, el resultado final son 1248 cajas, 102 imágenes con más de una caja, y un área relativa mediana del 6.8 % de la imagen, con 128 cajas por debajo del 1 % y 234 por encima del 25 %. Comparado con ENA24, cuya mediana es del 12.6 %, el oso aparece más pequeño en las fotos de iNaturalist.

La comparación entre lo propuesto y lo revisado da la medida de cuánto aportó el pre-etiquetado:

| | Cajas |
|---|---|
| Propuestas por el modelo | 1082 |
| Aceptadas sin cambio apreciable | 851 |
| Ajustadas | 95 |
| Eliminadas | 136 |
| Dibujadas de cero por el revisor | 302 |

Dicho de otro modo, el pre-etiquetado ahorró alrededor del 70 % del trabajo de trazado, pero una de cada cuatro cajas del dataset final la puso una persona desde cero, y de las que propuso el modelo, una de cada ocho hubo que borrarla. La cifra sirve para dos cosas: dimensionar el trabajo equivalente en la iteración 2, y dejar claro en la redacción que el *ground truth* no lo generó un modelo.

### 4.3 Verificación de las etiquetas

`verificar_etiquetado_inat.py` comprobó las 1128 imágenes que quedaron: todas tienen su archivo de etiquetas, no hay archivos de etiquetas huérfanos, ninguna línea está mal formada ni tiene coordenadas fuera del rango válido, y ninguna imagen quedó sin caja por descuido. Esta verificación importa porque un archivo de etiquetas vacío es perfectamente válido para Ultralytics, que lo interpreta como imagen de fondo: un olvido del revisor no habría producido ningún error, solo un positivo convertido en negativo de forma silenciosa.

### 4.4 Conversión de las cajas de ENA24

Las cajas de ENA24 vienen en píxeles como esquina superior izquierda más ancho y alto, y hubo que pasarlas a coordenadas normalizadas de centro. La conversión la hace `dividir_dataset.py` al construir el dataset, comprobando de paso que las dimensiones declaradas en el JSON coincidan con las del archivo en disco y recortando cualquier caja que se salga del borde. Ninguna de las 959 cajas presentó discrepancia de dimensiones ni hubo que recortar ninguna.

Como estas cajas no pasan por revisión humana, un error de conversión habría producido cajas válidas en formato pero colocadas en el lugar equivocado, y nada lo habría delatado. Por eso el script dibuja las cajas finales sobre una muestra de imágenes de cada fuente y las guarda en `docs/preparacion_datos/muestra_cajas` para revisarlas a ojo. Se revisaron 80 imágenes en dos pasadas y todas tenían la caja bien situada.

## 5. Integración y formato

Las tres fuentes se integran en `data/processed` con la estructura que espera Ultralytics: carpetas `images` y `labels`, cada una con sus subcarpetas de entrenamiento, validación y prueba, más un archivo `dataset.yaml` que declara una sola clase llamada `oso`.

Los archivos de ENA24 se renombran con el prefijo `ena24_` para que la procedencia de cada imagen se lea en su nombre sin consultar ninguna tabla. Los de iNaturalist ya llevan un prefijo que identifica la observación y la fotografía, y los negativos de otras especies llevan el suyo propio.

Los negativos, tanto los de ENA24 como los de iNaturalist, reciben un archivo de etiquetas vacío. Así Ultralytics los usa como imágenes de fondo durante el entrenamiento.

## 6. División en entrenamiento, validación y prueba

### 6.1 Unidad de división

La división no se hace por imagen sino por grupo, porque fotografías casi idénticas repartidas entre conjuntos inflarían las métricas sin que nada lo delate. Un grupo se forma por dos criterios que se suman:

- En iNaturalist, todas las fotografías de una misma observación van juntas. Se detectaron 1011 observaciones entre positivos y negativos.
- En ambas fuentes, las imágenes cuyos hashes perceptuales están a una distancia de Hamming menor o igual a 12 se unen en el mismo grupo, de forma transitiva. Esa unión añadió 29 uniones nuevas en iNaturalist y 1545 en ENA24, lo que da idea de cuántas ráfagas de cámara trampa el criterio anterior no había reconocido.

El resultado son 1530 grupos para 3832 imágenes.

### 6.2 Estratificación

La división se hace dentro de cada estrato por separado, para que la composición de los tres conjuntos sea comparable. Los estratos combinan la fuente, el rol de la imagen y la condición de luz: positivos y negativos de iNaturalist, y positivos y negativos de ENA24 separados en diurnos y nocturnos.

La razón de separar por rol merece explicación, porque nace de un problema detectado al construir la primera versión del dataset. En aquella versión todas las imágenes de iNaturalist eran positivas y todos los negativos venían de ENA24, de modo que fuente y etiqueta quedaban correlacionadas. Un detector puede aprovechar eso y aprender a reconocer el estilo fotográfico de cada fuente (color y encuadre frente a infrarrojo y cámara fija) en lugar del animal, y como la correlación se repite en validación y prueba, las métricas no lo delatarían. La mitigación tiene dos partes: incorporar negativos de iNaturalist, descrita en el apartado 2.3, y evaluar con métricas separadas por fuente, descrita en el apartado 8.

### 6.3 Resultado

Proporciones de 70, 15 y 15 por ciento, con semilla fija 42. Cada imagen queda registrada en `docs/preparacion_datos/division.csv` con su fuente, rol, grupo, estrato y conjunto asignado, de manera que el experimento es reproducible y las dos arquitecturas se comparan sobre exactamente la misma partición.

| Conjunto | Imágenes | Positivas | Negativas | Cajas | iNaturalist | ENA24 | Nocturnas |
|---|---|---|---|---|---|---|---|
| Entrenamiento | 2656 | 1382 | 1274 | 1507 | 1219 | 1437 | 947 |
| Validación | 598 | 326 | 272 | 351 | 260 | 338 | 224 |
| Prueba | 578 | 313 | 265 | 349 | 260 | 318 | 197 |
| Total | 3832 | 2021 | 1811 | 2207 | 1739 | 2093 | 1368 |

Reparto por estrato:

| Estrato | Entrenamiento | Validación | Prueba |
|---|---|---|---|
| iNaturalist, positivas | 790 | 170 | 168 |
| iNaturalist, negativas | 429 | 90 | 92 |
| ENA24, positivas de día | 196 | 52 | 59 |
| ENA24, positivas de noche | 396 | 104 | 86 |
| ENA24, negativas de día | 294 | 62 | 62 |
| ENA24, negativas de noche | 551 | 120 | 111 |

Los porcentajes no caen exactos en 70/15/15 porque la unidad que se reparte es el grupo y los grupos tienen tamaños muy distintos. El mayor reúne 401 imágenes nocturnas de ENA24, casi todas negativas, y cayó entero en entrenamiento, que es donde menos afecta.

## 7. Verificación de la división

El script `fugas_hamming.py` compara el hash perceptual de cada imagen de entrenamiento contra las de validación y prueba, dentro de cada fuente, y cuenta los pares que quedaron en conjuntos distintos a distancias bajas. Se corrió antes y después de corregir el criterio de agrupamiento.

| Distancia de Hamming | Pares cruzados antes | Imágenes de val/prueba afectadas antes | Pares cruzados después | Imágenes afectadas después |
|---|---|---|---|---|
| 0 | 0 | 0 | 0 | 0 |
| ≤ 4 | 617 | 240 | 0 | 0 |
| ≤ 8 | 1999 | 408 | 0 | 0 |
| ≤ 12 | 4162 | 500 | 0 | 0 |
| ≤ 16 | 9342 | 690 | 1973 | 664 |

La columna "antes" corresponde a la división construida agrupando solo por hash exacto, sobre 963 imágenes de validación y prueba. Es decir, cuatro de cada diez imágenes con las que se iba a evaluar tenían una casi gemela en el conjunto de entrenamiento.

Que esos pares fueran fugas reales y no coincidencias del hash se comprobó abriendo las imágenes y leyendo la marca de tiempo que la cámara graba sobre la fotografía. Dos ejemplos del umbral 8: la cámara PR6 disparó a las 06:35:42 y 06:35:43 del 11 de enero de 2015 sobre el mismo venado, con una imagen en entrenamiento y otra en prueba; la cámara Cam1 disparó a las 13:29:26 y 13:29:35 del 2 de agosto de 2015 sobre la misma cabalgata, otra vez repartidas. Con umbral 12 ambos casos desaparecen.

No se subió a 16 porque a esa distancia empiezan a emparejarse 216 pares de fotografías de iNaturalist que solo se parecen de lejos, señal de que el hash deja de discriminar y de que la unión arrastraría imágenes sin relación real.

## 8. Preparación de la evaluación

Aunque la evaluación pertenece a una fase posterior, dos decisiones se tomaron aquí porque condicionan cómo se construye el dataset y no se pueden improvisar después.

La primera es que cada modelo se evaluará tres veces: sobre el conjunto de prueba completo, sobre su parte de ENA24 y sobre su parte de iNaturalist. Las cifras de ENA24 son el resultado principal, porque el dominio de la cámara trampa es el del trabajo; las de iNaturalist se presentan como complementarias y optimistas, dado que la correlación entre fuente y etiqueta las favorece. Las listas de rutas para cada partición salen directamente de `division.csv`.

La segunda es que, además de las métricas habituales de detección, se calculará el recall a nivel de imagen: la proporción de imágenes con oso donde el modelo pone al menos una detección por encima del umbral, sin importar cuán ajustada quede la caja. Es la métrica que corresponde al uso real de la aplicación, donde el investigador quiere saber qué archivos revisar.

## 9. Limitaciones del dataset resultante

Conviene enumerarlas aquí para que la lectura del capítulo 3 sea honesta.

- **La misma cámara aparece en entrenamiento y en prueba.** ENA24 no trae identificador de ubicación, y aunque lo trajera, con solo 893 imágenes de oso reservar cámaras completas para prueba dejaría muy poco material. Las métricas de la iteración 1 no miden generalización a cámaras nuevas, que es exactamente lo que se le pedirá al modelo en Angochagua. La referencia [15] del marco teórico documenta esa caída de rendimiento.
- **La correlación entre fuente y etiqueta está atenuada, no eliminada.** Dentro de iNaturalist siguen pesando más los positivos que los negativos (1128 frente a 611).
- **No hay imágenes vacías de cámara trampa.** ENA24 anota todas sus imágenes, así que ninguna está vacía, y en el material real serán la mayoría. No se puede medir la tasa de falsos positivos sobre escena vacía, que es el número que más le importará al investigador. Se decidió posponerlo a la iteración 2 porque una imagen vacía es muy específica de su cámara y su fondo.
- **Ningún oso andino nocturno.** Las 1368 imágenes nocturnas del dataset son todas de oso negro americano. La aproximación es razonable (ambos son úrsidos grandes de pelaje oscuro y en infrarrojo el color importa poco) pero es una aproximación.
- **El proxy no incluye la especie objetivo en el dominio objetivo.** Ninguna imagen del dataset es un oso andino visto por una cámara trampa. Esa combinación solo llega con el material de Angochagua, y es la razón de que toda esta iteración sea preliminar.

## 10. Reproducibilidad

El dataset se puede reconstruir desde `data/raw` ejecutando en orden `consolidar_inat.py`, `seleccionar_ena24.py`, `preetiquetar_inat.py`, `verificar_etiquetado_inat.py` y `dividir_dataset.py`, más las listas de descarte y las etiquetas revisadas, que están versionadas en `docs/preparacion_datos/`. El único paso que no es reproducible de forma automática es la corrección manual de las cajas, cuyo resultado se conserva como artefacto.

Los scripts de esta fase están escritos para reutilizarse en la iteración 2 cambiando las rutas, y en el caso del pre-etiquetado, cambiando también los pesos: en lugar de los de COCO se usarán los del modelo entrenado en esta iteración, que deberían proponer cajas bastante mejores sobre imágenes de cámara trampa.

## 11. Entregables de la fase

- `data/processed/` con el dataset completo y su `dataset.yaml`.
- `docs/preparacion_datos/division.csv`, la partición imagen por imagen.
- `docs/preparacion_datos/inat_etiquetado.csv`, el resultado del etiquetado con las cajas finales, ajustadas, eliminadas y nuevas.
- `docs/preparacion_datos/etiquetas_inat/`, copia versionable de las 1248 cajas trazadas a mano.
- `docs/preparacion_datos/fugas_antes.txt` y `fugas_despues.txt`, la evidencia de la verificación de la división.
- `docs/preparacion_datos/muestra_cajas/`, las imágenes con las cajas dibujadas encima.
- `protocolo_anotacion.md`, las reglas con las que se trazaron las cajas.
- Este informe.
