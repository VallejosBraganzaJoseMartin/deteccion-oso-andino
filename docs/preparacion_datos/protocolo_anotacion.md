# Protocolo de anotación

Reglas con las que se trazaron las cajas delimitadoras del dataset de la iteración 1, escritas para que puedan aplicarse igual en la iteración 2 con el material de Angochagua y para que cualquiera pueda comprobar con qué criterio se construyó el *ground truth*.

Versión del 10 de septiembre de 2026. Aplicada por un solo anotador (el autor del trabajo) sobre las 1128 imágenes de iNaturalist que forman los positivos del dataset proxy. Las cajas de ENA24 vienen anotadas por sus autores y no pasaron por este protocolo, solo por una conversión de formato y una verificación visual por muestreo.

## 1. Herramienta y flujo

X-AnyLabeling 4.0.6, instalado en local en un entorno virtual aparte y ejecutado sobre Python 3.14.2. Se trabajó con guardado automático activado.

El flujo fue el mismo para todas las imágenes:

1. Un modelo YOLO26n preentrenado en COCO propuso cajas de la clase "bear" con umbral de confianza 0.2. El umbral bajo es deliberado: borrar una caja sobrante cuesta menos tiempo que dibujar una que falta.
2. Esas cajas se cargaron en la herramienta y se recorrieron las 1128 imágenes en orden, una por una, corrigiendo lo que hiciera falta.
3. Las cajas propuestas por el modelo se conservaron en una carpeta separada de las revisadas, de modo que el aporte del pre-etiquetado se puede medir y las cajas manuales tienen respaldo.

Ninguna caja entró al dataset sin haber pasado por la pantalla del anotador. El pre-etiquetado es una ayuda al trazado, no una fuente de etiquetas.

## 2. Clase

Una sola clase, `oso`, que agrupa al oso andino (*Tremarctos ornatus*) y al oso negro americano (*Ursus americanus*). El motivo está en la ficha del 7 de septiembre del registro de decisiones.

## 3. Reglas de trazado

- **Una caja por animal.** Si hay varios osos, cada uno lleva la suya. Las crías cuentan como osos y llevan caja propia.
- **La caja ciñe el cuerpo visible**, incluyendo cabeza y patas. No se deja margen de cortesía alrededor ni se recorta por dentro.
- **Oclusión parcial.** Si la vegetación o una roca tapan parte del animal pero la silueta se infiere con claridad, la caja cubre el animal completo. Si no se puede inferir dónde termina, la caja cubre solo lo visible.
- **Se anota todo oso**, por lejano, borroso, oscuro o parcialmente tapado que esté. Esas son justamente las imágenes que el modelo necesita aprender y las que el modelo preentrenado no supo detectar.
- **Cajas duplicadas.** Cuando el modelo propuso dos cajas sobre el mismo animal, se conservó una y se borró la otra. Es un caso frecuente cuando se trabaja con umbral bajo.
- **Corregir en lugar de rehacer.** Si la caja propuesta contiene al oso pero está mal ajustada, se mueven sus vértices. Solo se dibuja de cero cuando no hay caja o cuando la propuesta no corresponde al animal.

## 4. Criterios de descarte

Una imagen se saca del dataset, en lugar de anotarse, cuando ocurre alguna de estas cosas. El descarte se registra en un archivo de texto con el nombre del archivo y el motivo, y un script se encarga después de moverla. No se borran archivos a mano ni se toca `data/raw`.

| Motivo | Qué incluye |
|---|---|
| `rastro` | Huellas, excrementos, arañazos en troncos, restos. Hay oso en el sentido de que estuvo ahí, pero no hay animal que detectar |
| `cautiverio` | Oso en zoológico o recinto evidente, con rejas, cemento o comederos |
| `pantalla` | Fotografías de la pantalla de una cámara o de un monitor, impresiones y montajes tipo collage. Hay un oso en la imagen, pero el dominio es una pantalla |
| `duda` | El anotador no puede decidir si esa mancha oscura es un oso, una sombra, una roca o un tronco |
| `sin_oso` | Paisajes o escenas donde no aparece ningún oso |

La regla sobre las dudas merece subrayarse: si quien anota no puede decidir, la imagen no sirve ni para entrenar ni para evaluar, y forzar una decisión solo introduce ruido. En la iteración 1 quedaron 111 imágenes en esa categoría.

Las imágenes descartadas por `rastro` y `sin_oso` no se pierden: entran al dataset como negativos, con archivo de etiquetas vacío. Las de `cautiverio`, `pantalla` y `duda` quedan fuera por completo, porque todas contienen un oso de alguna forma y usarlas como negativos enseñaría lo contrario de lo que se busca.

## 5. Formato de salida

Una línea por caja en un archivo de texto con el mismo nombre que la imagen:

```
0 cx cy w h
```

donde `0` es la clase (`oso`, la única), y las cuatro coordenadas son el centro, el ancho y el alto de la caja, normalizados entre 0 y 1 respecto al tamaño de la imagen. Las imágenes sin ningún oso llevan un archivo vacío, que Ultralytics interpreta como imagen de fondo.

## 6. Verificación

Un archivo de etiquetas vacío es válido para Ultralytics, así que un olvido del anotador no produce ningún error: convierte un positivo en negativo sin avisar. Por eso, al terminar cada sesión se pasó un script de verificación que comprueba que cada imagen tenga su archivo de etiquetas, que ninguno esté vacío entre los positivos, que todas las líneas estén bien formadas y que ninguna coordenada se salga del rango válido.

Para las cajas que no se anotaron a mano, como las de ENA24, la verificación es visual: se dibujan las cajas sobre una muestra de imágenes y se revisan a ojo. Una conversión de coordenadas mal hecha produce cajas perfectamente válidas en formato pero situadas en el lugar equivocado, y ninguna comprobación automática lo detecta.

## 7. Limitaciones conocidas de este protocolo

- **Un solo anotador.** No hay forma de medir el acuerdo entre anotadores ni la consistencia del criterio a lo largo de las horas de trabajo. Si el tiempo lo permite, una segunda pasada sobre una muestra aleatoria daría una idea de cuánto varía el propio criterio.
- **Criterio de oclusión subjetivo.** Decidir si una silueta "se infiere con claridad" depende de quién mire. Es la regla más difícil de aplicar de forma consistente.
- **Sin categoría de dificultad.** Otros datasets marcan las cajas difíciles (animal muy pequeño, muy tapado) para poder excluirlas del cálculo de métricas. Aquí no se hizo, así que todas las cajas pesan igual al evaluar.

## 8. Cambios previstos para la iteración 2

- Añadir la categoría `vacia` a los criterios de clasificación, que en el material real será la más frecuente.
- El pre-etiquetado se hará con los pesos del modelo entrenado en la iteración 1, no con los de COCO.
- Las reglas de trazado y de descarte se mantienen sin cambios, salvo que la clase pasa a ser exclusivamente oso andino, porque en Angochagua no hay otros úrsidos.
