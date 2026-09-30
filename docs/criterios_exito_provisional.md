# Criterios de Éxito Provisionales

Definición preliminar de métricas de desempeño, requerimientos funcionales y objetivos de evaluación para el sistema de detección del Oso Andino (*Tremarctos ornatus*).

Versión del 4 de septiembre de 2026, actualizada el 10 de septiembre al cerrar la fase de preparación de los datos. Redactada sin posibilidad de consulta con el director (periodo de vacaciones). Todos los umbrales son supuestos de trabajo y quedan sujetos a revisión al retomar Titulación 2. CRISP-DM pide fijar estos criterios antes de modelar, y por eso se escriben ahora aunque no puedan validarse todavía.

---

## 0. Principio rector

En monitoreo de fauna para conservación, los dos tipos de error no cuestan lo mismo. Si el sistema pasa por alto un oso, ese registro se pierde y con él la información de presencia que justifica todo el proyecto. Si en cambio marca una imagen que no tiene oso, el costo es que el investigador la descarte en un par de segundos, que es exactamente lo que hoy hace con todo el material.

De ahí que el recall sea la métrica principal y la precisión una restricción: el sistema debe encontrar casi todos los osos, y a cambio se acepta un volumen razonable de falsos positivos. Esta jerarquía gobierna la elección del umbral de confianza y la lectura de los resultados del Capítulo 3.

Conviene distinguir dos niveles, como plantea CRISP-DM.

**Éxito del negocio.** El sistema reduce de forma apreciable el tiempo de revisión manual del material de las cámaras trampa, sin que el investigador pierda registros de oso que habría encontrado revisando a mano.

**Éxito de la minería de datos.** Los umbrales numéricos de las secciones siguientes.

## 1. Métricas de Rendimiento del Modelo

Los valores se expresan por separado para las dos iteraciones, porque no son comparables entre sí. La iteración 1 se entrena y evalúa con el dataset proxy (iNaturalist y ENA24), y la iteración 2 con el material real de Angochagua.

### 1.0 Sobre qué conjunto se miden

Esto se fija antes de entrenar, porque cambia la lectura de todos los números que siguen. El conjunto de prueba de la iteración 1 mezcla 260 imágenes de iNaturalist con 318 de ENA24, que son dos problemas distintos: fotografía tomada con cámara en mano, a plena luz y con el animal razonablemente centrado, frente a cámara trampa fija, con infrarrojo nocturno y el animal donde caiga. Un promedio de ambos no describe bien ninguno de los dos.

Cada modelo se evalúa tres veces, con listas de rutas separadas:

- **ENA24.** Es el resultado principal de la iteración 1, porque es el dominio del trabajo. Los criterios numéricos de este documento se leen sobre esta partición.
- **iNaturalist.** Complementaria, y hay que presentarla como optimista. Dentro de esa fuente los positivos superan a los negativos, de manera que el modelo puede apoyarse en el estilo de la fotografía y no solo en el animal. La construcción del dataset atenúa el problema, no lo elimina.
- **Global.** Se reporta por comparabilidad con la literatura, que suele publicar una sola cifra.

Advertencia de tamaño muestral: el conjunto de prueba de ENA24 con oso tiene 145 imágenes. Un recall estimado sobre esa cantidad lleva un margen de varios puntos porcentuales, así que las cifras por fuente son indicativas y no se deben presentar con dos decimales como si fueran precisas.

> **Nota 2026-09-30.** En validación, iNaturalist resultó más difícil que ENA24 para los seis modelos: con el mismo umbral, el recall por imagen fue de 0.88 a 0.93 en iNaturalist y de 0.955 en ENA24. La previsión de este apartado, que presentaba iNaturalist como la partición optimista, no se cumplió. Se revisa con el conjunto de prueba antes de redactar el capítulo 3.

### 1.1 Métricas

- **Precisión (Precision) / Recall / F1-Score**:
  - Recall en el conjunto de prueba: meta de 0.90 o superior en ambas iteraciones. Es el criterio que no debería negociarse.
  - Precisión: mínimo aceptable de 0.70 en la iteración 1. En la iteración 2, una vez conocido el volumen real de imágenes vacías, habrá que revisarlo, porque en cámaras trampa la mayor parte del material no tiene animal y una precisión baja se traduce en muchas imágenes falsas que revisar.
  - F1 se reporta como resumen, pero no se optimiza contra él. Maximizar F1 trata ambos errores como equivalentes y ya se explicó por qué aquí no lo son.

- **Recall a nivel de imagen**: proporción de imágenes con oso en las que el modelo pone al menos una detección por encima del umbral, sin importar cuán ajustada quede la caja. Se calcula con un script propio sobre las predicciones de prueba, porque Ultralytics no lo reporta. Es la métrica que corresponde al uso real de la aplicación: el investigador quiere saber qué archivos revisar, y una caja algo desplazada no le cuesta nada mientras el aviso exista. Meta de 0.95 o superior, más exigente que el recall de detección porque la tarea es más fácil.

- **mAP (Mean Average Precision)**:
  - mAP@0.5 en la iteración 1: meta de 0.85 o superior. Es una meta de orden de magnitud, no una promesa.
  - mAP@0.5 en la iteración 2: el anteproyecto toma el 92.6 % reportado por [4] como referencia de rendimiento mínimo esperado. Hay que sostener esa referencia con cautela y explicarlo en la redacción: ese resultado se obtuvo sobre CCT, con 8500 imágenes y diez especies, y la nota de la Tabla 3 del marco teórico advierte que las cifras de mAP de los tres estudios no son comparables entre sí. Alcanzar o no ese número dependerá en buena medida de cuántas imágenes entregue Angochagua, dato que hoy no se conoce. Si al evaluar queda por debajo, el trabajo no fracasa: lo que corresponde es analizar la brecha y atribuirla a las condiciones del dataset.
  - mAP@0.5:0.95 se reporta siempre junto a mAP@0.5, porque es más exigente con el ajuste de la caja y evita una lectura optimista.
  - Estas cifras se comparan entre YOLO26 y RT-DETR bajo el protocolo de comparación justa descrito en `PLAN_FASES_SIGUIENTES.md`, sección 2.1.

- **Tasa de falsos positivos en especies similares o fondos complejos**:
  - Se mide por separado sobre los negativos del conjunto de prueba. En la iteración 1 hay dos grupos con carácter distinto: los de ENA24 (perro, coyote, lince rojo, venado, gato doméstico y caballo, en el mismo dominio de cámara trampa) y los de iNaturalist (lobo de páramo, tapir de montaña, puma, venado, perro, vaca, caballo y oveja, más rastros y paisajes). En la iteración 2, lo que aparezca en Angochagua, en particular ganado y perros ferales.
  - Meta provisional: menos del 10 % de esos negativos genera una detección por encima del umbral elegido.
  - Se reporta también el desempeño separando día y noche, porque el infrarrojo es la condición donde la literatura reporta más caídas y donde el dataset proxy es más débil para el oso andino.
  

- **Elección del umbral de confianza**:
  - No se usa el valor por defecto de la librería. Se elige recorriendo la curva de precisión contra recall sobre el conjunto de validación y tomando el umbral más alto que aún conserve un recall de 0.90.
  - El valor elegido se documenta, se usa en la evaluación de prueba y se fija como valor por defecto de la aplicación, donde el usuario podrá ajustarlo.
  > **Revisión 2026-09-30.** La regla anterior (el umbral más alto que conserve un recall de cajas de 0.90 en validación) se aplicó a los seis modelos de YOLO26 de la iteración 1. En ENA24 dejó el recall por imagen entre 0.91 y 0.94, por debajo de la meta de 0.95 de este mismo apartado, con falsas alarmas de apenas 1 a 4 %. Es decir, la regla fijaba el umbral con una métrica secundaria y dejaba sin cumplir la principal, aunque había margen para cumplirla. Se sustituye por esta: sobre ENA24 de validación, el umbral más alto que cumpla a la vez un recall por imagen de 0.95 o más y un recall de cajas de 0.90 o más, siempre que las falsas alarmas queden por debajo del 10 % en los negativos de ENA24 y en los de iNaturalist, medidos por separado. Si esa condición no se cumple, se sube el umbral hasta el valor más bajo que la cumpla y se informa que el modelo no alcanza la meta de recall. El cambio se decidió sobre validación, sin haber tocado el conjunto de prueba.


## 2. Métricas de Eficiencia y Despliegue

El alcance del anteproyecto descarta el procesamiento dentro de las cámaras y el tiempo real. La eficiencia importa por otra razón: la aplicación se ejecuta localmente en el equipo del investigador, que puede no tener GPU.

- **Latencia de inferencia por imagen/frame**:
  - Referencia medida: YOLO26n preentrenado en COCO procesó 1424 imágenes en unos 15 minutos en el equipo de desarrollo sin GPU, alrededor de 0.6 segundos por imagen. Es una medición aproximada, tomada del tiempo total de un script que además copiaba archivos, y sirve solo como punto de partida.
  - Meta provisional: menos de 1 segundo por imagen en CPU con el modelo elegido para la aplicación. Se mide de forma limpia, sin contar carga del modelo ni escritura de resultados.
  - Para MP4 se mide el tiempo por minuto de video, que depende del intervalo de extracción de fotogramas.

- **Consumo de recursos (RAM / GPU / CPU)**:
  - La aplicación debe funcionar en un equipo de gama media sin GPU dedicada. Meta provisional: menos de 4 GB de RAM durante el procesamiento.
  - El entrenamiento sí usa GPU en la nube (Kaggle) y no entra en este criterio.
  - Si el modelo con mejor mAP resulta demasiado pesado para la aplicación, se usan dos: el mejor para reportar resultados en el Capítulo 3 y una variante liviana para el despliegue. En ese caso hay que reportar las métricas de ambos y justificar la elección.

- **Requisitos funcionales de la aplicación** (criterio de aceptación del despliegue):
  - Procesa una carpeta completa con JPG y MP4 sin intervención del usuario.
  - Muestra cada detección con su miniatura, confianza, código de cámara, coordenadas, fecha y hora.
  - Exporta los resultados a CSV.
  - Funciona sin conexión a internet.
  - Degrada de forma controlada cuando falta un metadato: si una imagen no trae fecha en EXIF, la detección igual aparece y el campo queda marcado como no disponible, en lugar de que la aplicación falle.

## 3. Calidad de Datos

Los tres criterios de este apartado ya se pueden contrastar con el dataset construido, así que junto a cada uno va el valor medido. El detalle está en `informe_preparacion_datos.md` y en `division.csv`.

- **Diversidad de datasets**:
  - Iteración 1, cifras finales: 3832 imágenes con 2207 cajas. De iNaturalist, 1128 positivas con 1248 cajas revisadas a mano y 611 negativas. De ENA24, 893 positivas con 959 cajas y 1200 negativas. Reparto: 2656 en entrenamiento, 598 en validación y 578 en prueba.
  - Toda imagen del dataset tiene su licencia y atribución registradas, incluidas las negativas descargadas después. Este criterio se cumple y no es negociable, porque sostiene la legitimidad del dataset ante el tribunal.
  - Iteración 2: el material de Angochagua debe cubrir varias cámaras y ambas condiciones de luz. Si llega concentrado en una sola cámara o solo en capturas diurnas, hay que decirlo como limitación en lugar de presentar las métricas como generalizables.

- **Balance de clases y variaciones de iluminación/hábitat**:
  - Proporción de negativos frente a positivos en entrenamiento: al menos 0.9 a 1 y como mucho 2 a 1. El valor medido es 1274 negativos frente a 1382 positivos, o sea 0.92 a 1.
  - Ese mínimo se rebajó desde el 1 a 1 que fijaba la versión anterior de este documento, y conviene explicar por qué. Al construir el dataset se tomaron 1200 negativos de ENA24 sobre unos 5600 candidatos disponibles, cantidad decidida antes de saber cuántos negativos aportaría iNaturalist. Subir la proporción exige volver a generar el dataset completo, y el margen que separa 0.92 de 1.0 no justifica rehacer un conjunto ya verificado, con la división comprobada sin fugas. Si al evaluar la precisión queda por debajo del mínimo del apartado 1.1, la primera medida correctiva es justamente esa: subir los negativos de ENA24 a unos 1600, regenerar y volver a entrenar. Los candidatos están ahí sin usar.
  - Presencia de nocturnas en el conjunto de prueba: al menos un 30 %. El valor medido es 197 de 578, un 34 %. En la iteración 1 esas nocturnas son de ENA24 y no de oso andino, lo que es una limitación conocida que debe declararse.
  - Ninguna imagen del conjunto de prueba puede pertenecer al mismo grupo que una de entrenamiento (observación en iNaturalist, pseudosecuencia en ENA24, cámara y secuencia en Angochagua). Si esto se incumple, las métricas no valen y hay que rehacer la división. Verificado con `fugas_hamming.py`: cero pares de imágenes casi idénticas entre conjuntos distintos hasta una distancia de Hamming de 12. Antes de corregir el criterio de agrupamiento había 1999 pares que afectaban a 408 imágenes de validación y prueba.
  - Queda una fuga que ninguna agrupación por hash resuelve y que este criterio no cubre: la misma cámara de ENA24 aparece en entrenamiento y en prueba con el mismo fondo fijo. Las métricas de la iteración 1 no miden generalización a cámaras nuevas, y eso hay que escribirlo en el Capítulo 3. En la iteración 2, con material de las 15 cámaras de Angochagua, sí se puede reservar alguna cámara completa para prueba, que es la forma correcta de medirlo.

## 4. Qué no cuenta como criterio de éxito

- Superar a YOLO26 con RT-DETRv3 o al revés. La comparación es un resultado a reportar, no una meta a alcanzar. Que una arquitectura gane o pierda es igual de publicable mientras el protocolo sea justo.
- Las métricas de la iteración 1. Son preliminares, se obtienen sobre datos proxy y no son los resultados del trabajo. Sirven para verificar que el pipeline funciona y para tener un punto de partida del fine-tuning.
- El baseline zero-shot del modelo COCO (761 de 1424 imágenes con detección de confianza alta). Es un punto de referencia de partida, no un objetivo.
- El resultado del experimento de ablación, si llega a correrse. Entrenar solo con iNaturalist, solo con ENA24 y con ambas fuentes, y evaluar las tres sobre el conjunto de prueba de ENA24, responde con números si la combinación de fuentes ayudó. Es un resultado interesante para el Capítulo 3 en cualquier dirección que salga, y por eso no se le pone meta.

## 5. Pendiente de validar con el director y el asesor

1. ¿Se acepta el recall como métrica principal por encima de la precisión, con la justificación del apartado 0?
2. ¿Se mantiene el 92.6 % de [4] como referencia mínima esperada, sabiendo que proviene de un dataset mucho mayor y de otra especie?
3. ¿Qué umbral de precisión mínimo es tolerable para el investigador que va a usar la aplicación? Esta respuesta debería salir de quien revise el material a mano, no de la literatura.
4. ¿Cuántas imágenes de Angochagua se esperan y en qué proporción de día y noche? Sin ese dato, las metas de la iteración 2 son una conjetura.
5. ¿Hay un requisito de tiempo de procesamiento por parte del usuario final (por ejemplo, revisar el material de una campaña completa en una jornada)?
6. ¿Se acepta que los criterios numéricos se lean sobre la partición de ENA24 y no sobre la cifra global, con la justificación del apartado 1.0?
7. ¿Se acepta la revisión del 30 de septiembre de la regla del umbral, que lo fija por el recall por imagen (0.95) y deja las falsas alarmas como restricción?
