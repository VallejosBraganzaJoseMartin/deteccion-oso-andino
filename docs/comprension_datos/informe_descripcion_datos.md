# Informe de descripcion y calidad de los datos (iteracion 1, dataset proxy)

Generado automaticamente por describir_dataset.py. Revisar y completar a mano.

### iNaturalist (Tremarctos ornatus)

- Archivos de imagen encontrados: 1424
- Archivos corruptos o ilegibles: 0
- Resolucion (megapixeles): mediana 0.70, minimo 0.06, maximo 1.05
- Ancho x alto mas frecuente: 1024x683
- Orientacion: {'horizontal': 982, 'vertical': 423, 'cuadrada': 19}
- Imagenes en escala de grises (posible infrarrojo): 4 (0.3 %)
- Brillo medio (0-255): mediana 122; imagenes oscuras (< 60): 26
- Tamano en disco: 0.88 GB

### iNaturalist: licencias y procedencia

- Registros en el CSV de licencias: 1424 (observaciones distintas: 693)
- Licencias: {'cc-by-nc': 1287, 'cc-by': 101, 'cc0': 36}
- Con 'Ecuador' en el campo lugar (texto libre): 631
- Con coordenadas dentro de la caja aproximada de Ecuador: 1064 (sin coordenadas: 17)
- Marcadas como cautivas: {'False': 1424}
- Fotos por observacion: media 2.05, maximo 20; observaciones con mas de una foto: 253
- Registros del CSV sin archivo en disco: 0

### ENA24 (LILA BC)

- Archivos de imagen encontrados: 8789
- Archivos corruptos o ilegibles: 0
- Resolucion (megapixeles): mediana 3.15, minimo 2.07, maximo 7.99
- Ancho x alto mas frecuente: 2048x1536
- Orientacion: {'horizontal': 8789}
- Imagenes en escala de grises (posible infrarrojo): 4686 (53.3 %)
- Brillo medio (0-255): mediana 95; imagenes oscuras (< 60): 1733
- Tamano en disco: 3.89 GB

### ENA24: anotaciones (JSON COCO Camera Traps)

- Imagenes en el JSON: 9676; anotaciones: 11596; categorias: 23
- Campos disponibles por imagen: ['file_name', 'height', 'id', 'width']
  (si hay 'location', 'seq_id' o 'datetime', usalos para dividir train/val/test sin fugas)
- Imagenes (y anotaciones) por categoria:
  - American Crow: 947 imagenes, 1278 cajas
  - American Black Bear: 893 imagenes, 959 cajas
  - Human: 887 imagenes, 1161 cajas
  - Dog: 751 imagenes, 774 cajas
  - Virginia Opossum: 725 imagenes, 725 cajas
  - Chicken: 542 imagenes, 749 cajas
  - Domestic Cat: 491 imagenes, 491 cajas
  - Grey Fox: 422 imagenes, 438 cajas
  - Red Fox: 413 imagenes, 413 cajas
  - White_Tailed_Deer: 350 imagenes, 381 cajas
  - Eastern Fox Squirrel: 340 imagenes, 343 cajas
  - Horse: 338 imagenes, 495 cajas
  - Coyote: 334 imagenes, 344 cajas
  - Eastern Cottontail: 331 imagenes, 340 cajas
  - Bobcat: 328 imagenes, 333 cajas
  - Eastern Chipmunk: 311 imagenes, 311 cajas
  - Eastern Gray Squirrel: 305 imagenes, 319 cajas
  - Striped Skunk: 297 imagenes, 297 cajas
  - Vehicle: 293 imagenes, 295 cajas
  - Northern Raccoon: 291 imagenes, 292 cajas
  - Wild Turkey: 289 imagenes, 427 cajas
  - Woodchuck: 206 imagenes, 206 cajas
  - Bird: 200 imagenes, 225 cajas
- Imagenes sin ninguna anotacion (posibles vacias): 0
- Categorias que contienen 'bear': ['American Black Bear']
- Cajas de oso: 959 en 893 imagenes; formato de bbox en pixeles
- Area relativa de la caja de oso (fraccion de la imagen): mediana 0.126; cajas pequenas (< 1 % de la imagen): 56; cajas grandes (> 25 %): 312
- Imagenes del JSON sin archivo en disco: 887; archivos en disco que no aparecen en el JSON: 0

### Duplicados (mismo hash perceptual)

- Grupos de duplicados: 1009; archivos implicados: 2902 (detalle en duplicados.csv)

### Pendiente de revision manual

- iNaturalist: separar huellas, excrementos, restos y fotos de cautiverio evidentes.
- Definir criterio de division train/val/test: por observacion_id (iNaturalist) y por location/seq_id (ENA24).
