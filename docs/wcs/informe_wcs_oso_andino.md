# Imágenes de oso andino en WCS Camera Traps (LILA BC)

Revisión hecha el 2 de octubre de 2026, sin tocar el dataset proxy. Su uso en el trabajo está pendiente de decidir con el director (pregunta 29). Scripts: `explorar_wcs.py`, `descargar_wcs.py` y `registrar_revision.py`. Tablas en `docs/wcs/`.

## 1. Fuente y licencia

- **Dataset:** WCS Camera Traps, publicado en LILA BC con datos de la Wildlife Conservation Society. Según su ficha, tiene unas 1.4 millones de imágenes de unas 675 especies de 12 países, y cerca de la mitad están vacías.
- **Licencia:** Community Data License Agreement, variante permisiva (CDLA-Permissive-1.0), la misma de ENA24. Verificado en la ficha de lila.science el 2 de octubre de 2026.
- **Advertencias de la propia ficha:**
  - Las secuencias se infieren por la hora de las fotos.
  - Las etiquetas se pusieron en parte por imagen y en parte por secuencia, así que una foto marcada con un animal puede estar vacía.
  - Las fotos con personas aparecen en los metadatos, pero no están publicadas.

## 2. Archivos usados

En `data/raw/wcs/`, tal como bajaron de lila.science (D8):

| Archivo | Tamaño | Contenido |
|---|---|---|
| `wcs_camera_traps.json.zip` | 45.3 MB (JSON de 838.4 MB) | 1 369 991 imágenes, una anotación por imagen y 676 categorías, en formato COCO Camera Traps |
| `wcs_20220205_bboxes_with_classes.zip` | 24.0 MB (JSON de 231.6 MB) | 298 725 imágenes, 429 482 anotaciones (374 270 con caja) y 678 categorías |

`explorar_wcs.py` los lee sin descomprimirlos y necesitó unos 3 GB de RAM en el equipo local. El servidor de LILA está bloqueado en el entorno de trabajo de Claude, así que estos archivos solo los puede descargar el estudiante.

## 3. Qué dicen los metadatos

- **Categoría y número de fotos:** la 470, `tremarctos ornatus`, tiene 462 anotaciones en 462 imágenes. Existe además `ursus thibetanus` (474), que no se usa.
- **Campos de cada foto:**
  - `country_code`: `bol` (Bolivia) en las 462.
  - `wcs_id`, `match_level` (0 en 451 fotos y 1 en 11; no se sabe qué significa), `status` (vacío) y `corrupt` (falso en todas).
  - En las anotaciones, `count`, `sex` y `age` valen 1, "unknown" y "unknown" en todas, aunque hay fotos con 2 a 4 osos.
- **Numeración dentro de la secuencia:** WCS numera las fotos de cada secuencia desde 1, aunque el formato COCO Camera Traps dice desde 0.
- **Carpetas:** las 462 fotos están repartidas en 13 carpetas, de `animals/0470/` a `animals/0524/`. El número de la carpeta no es el de la categoría.
- **Cámaras y fechas:**
  - 8 ubicaciones (cámaras): 3429, 3430, 3439, 3902, 3905, 3906, 3907 y 3913. Tres de ellas reúnen 345 fotos.
  - Del 6 de abril al 19 de junio de 2012, en 13 combinaciones de cámara y día.
  - Según el reloj de la cámara, 445 fotos son de 06:00 a 17:59 y 17 de las 18:xx.
  - Todas miden 2048×1536.
- **Secuencias:** 40, de 10, 20, 30 o 40 fotos. En 32 todas las fotos están etiquetadas como oso. En las otras 8 hay además 48 fotos etiquetadas como vacías.
- **Otras especies en esas cámaras:** son de los Andes altos de Perú y Bolivia, entre ellas *Didelphis pernigra*, *Cuniculus taczanowskii*, *Lagidium viscacia*, *Mazama chunyi*, *Lycalopex culpaeus* e *Hippocamelus antisensis*. Es coherente con Bolivia (inferencia).

## 4. Cajas

| Estado en el archivo de cajas | Fotos |
|---|---|
| Con caja de oso andino (123 cajas: 103 fotos con 1, 5 con 2, 2 con 3 y 1 con 4) | 111 |
| Revisadas y marcadas como vacías, sin caja | 32 |
| No están en el archivo de cajas | 319 |

- **Tamaño:** la caja mediana ocupa el 9.1 % de la foto; 3 cajas ocupan menos del 1 % y 22 más del 25 %.
- **Ningún oso pequeño:** a 640 px, el lado menor de la caja más pequeña de cada foto mide entre 32 y 64 px en 16 fotos y 64 px o más en 95. No hay osos más pequeños, así que D20 se mantiene.
- **Cobertura:** solo 19 de las 40 secuencias tienen alguna caja.

## 5. Descarga

`descargar_wcs.py` bajó por HTTP solo las 462 fotos, con la URL de cada una (`https://storage.googleapis.com/public-datasets-lila/wcs-unzipped/<file_name>`), en 70 segundos.

- **Destino:** `data/raw/wcs/imagenes/`, conservando la ruta de LILA.
- **Comprobación:** 226 MB, todas JPEG por dentro y del tamaño que dice el JSON.
- **Escala de grises sin ser de noche:** 290 fotos están en escala de grises, pero el reloj de la cámara marca entre las 10:00 y las 18:00. En la secuencia revisada, la hora grabada sobre la foto ("10:56:41 AM") coincide con el JSON, así que el reloj no está mal. Son fotos de día en el modo monocromo de la cámara; probablemente la cámara entra en ese modo con poca luz (no verificado). Consecuencia: en WCS, escala de grises no significa noche, a diferencia de ENA24. Si se evalúa con WCS, el día y la noche se separan por la hora.
- **Marca de agua:** las fotos llevan hora, temperatura, número de cámara y "M 1/10", y un logo que parece de Reconyx (no verificado).

## 6. Revisión a ojo

Hecha por el estudiante el 2 de octubre con las categorías del protocolo de anotación, sobre copias con las cajas de LILA dibujadas (`data/revisar/wcs/`), y registrada con `registrar_revision.py` en `docs/wcs/revision_manual_wcs.csv`.

| Estado en LILA | Oso | Sin oso | Duda | Total |
|---|---|---|---|---|
| Caja de oso | 107 | 0 | 4 | 111 |
| Vacía según sus cajas | 0 | 13 | 19 | 32 |
| Sin revisar | 255 | 28 | 36 | 319 |
| **Total** | **362** | **41** | **59** | **462** |

- **Criterios coincidentes:** ninguna foto con caja de LILA quedó como "sin oso" y ninguna "vacía según LILA" quedó como "oso". Las 111 cajas de LILA se revisaron a ojo y ninguna está fuera del oso.
- **Cobertura:** 36 de las 40 secuencias y las 8 cámaras tienen al menos una foto con oso.
- **Dos cámaras por estación:** cada ubicación tiene dos códigos de despliegue (por ejemplo, d0021 y d0022 en la 3430). Marcan temperaturas distintas y llevan números de cámara distintos (315 y 370).
- **Etiquetas por evento de estación:** las 4 secuencias sin ningún oso son de la cámara d0021 de la estación 3430. A esas mismas horas, la otra cámara (d0022) fotografió osos. En una de esas secuencias, la d0021 aparece movida e inclinada hacia el cielo, probablemente por uno de los osos. Lo más probable es que la etiqueta se pusiera al evento de la estación y no a cada cámara.
- **Incidencia:** al clasificar se crearon 37 copias "- Copy.jpg" en `oso`. Se borraron antes del registro final. `registrar_revision.py` no detecta duplicados.

## 7. Usos posibles (a decidir con el director)

| Uso | Qué hace falta | Salvedad |
|---|---|---|
| Evaluación externa por imagen del modelo de la iteración 1 | Nada más: 362 fotos con oso y 41 sin oso; las 59 dudosas quedan fuera, como dice el protocolo | Son 13 combinaciones de cámara y día, así que la incertidumbre es grande; conviene reportar también por evento |
| Evaluación con métricas de cajas | Dibujar 255 cajas (con pre-etiquetado, como en iNaturalist) | |
| Plan B de la iteración 2 | Lo mismo, más una división por cámara | Bolivia, no Ecuador; pocas visitas |
| Fotos vacías reales de cámaras andinas | Otra descarga: hay miles en esas 8 cámaras (por ejemplo, 13 980 fotos en la 3907, el 98 % vacías) | Cubriría el hueco principal del proxy (pregunta 13) |

## 8. Archivos

En `docs/wcs/`:
- `wcs_oso_andino.csv`: una fila por foto, con URL, campos de WCS y cajas.
- `wcs_secuencias.csv`: una fila por secuencia.
- `wcs_resumen.txt`: la salida de `explorar_wcs.py`.
- `wcs_descarga.csv`: formato, tamaño, escala de grises y nombre de la copia de revisión de cada foto.
- `revision_manual_wcs.csv`: la clasificación a ojo.

Fuera de git: `data/raw/wcs/`, `data/revisar/wcs/` y `data/wcs_mosaicos/` (50 hojas, una por secuencia).
