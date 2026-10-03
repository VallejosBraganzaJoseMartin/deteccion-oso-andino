# Fragmentos para preguntas_pendientes.md (cierre del 2 de octubre de 2026)

Cada bloque dice dónde va. Lo que no se menciona queda igual.

---

## 1. Línea de "Última actualización": reemplazar por

Última actualización: 2 de octubre de 2026, al cerrar la conversación de WCS, evaluación del modelo COCO y de la ablación, y resumen para el director.

---

## 2. Tabla "Abiertas"

**2a.** Fila 13: reemplazar las tres últimas columnas (Estado, Responsable y Resolución) por:

| Pendiente | Estudiante / Director | Novedad del 2 de octubre: bajar un subconjunto de LILA resultó fácil, por HTTP foto a foto (462 fotos de WCS en 70 s). WCS tiene miles de fotos vacías y de otras especies de las mismas 8 cámaras andinas donde aparece el oso, más parecidas a Angochagua que las de Norteamérica. Se decide junto con la pregunta 29 |

**2b.** Fila 22: quitarla de "Abiertas" (pasa a "Resueltas", bloque 3).

**2c.** Fila 29: reemplazar la columna Resolución por este texto, en una sola línea porque va dentro de la tabla:

Revisadas el 2 de octubre (`docs/wcs/informe_wcs_oso_andino.md`): 362 fotos muestran el oso, 41 no y 59 son dudosas; 107 tienen caja de LILA, verificadas a ojo; Bolivia, 8 cámaras, 13 combinaciones de cámara y día; licencia CDLA-Permissive-1.0. Opciones: evaluación externa por imagen (ya), evaluación con cajas (dibujando 255), plan B de la iteración 2, y fotos vacías de esas cámaras (pregunta 13)

**2d.** Añadir una fila nueva:

| 30 | En el control de mAP50 del 2 de octubre, los modelos de la ablación difieren del mAP50 de Ultralytics más que los de YOLO26 (hasta 0.035 en iNaturalist con A1b, frente a 0.015 en septiembre). ¿Por qué? | Técnica | Pendiente | Estudiante | No afecta al recall por imagen ni a las falsas alarmas. Hipótesis sin verificar: en particiones donde el modelo es malo, cambios pequeños de caja mueven más el AP, y las referencias de la ablación tienen 3 decimales |

---

## 3. Tabla "Resueltas"

**3a.** Reemplazar la fila 12, que tiene las columnas corridas, por:

| 12 | ¿Bajo qué licencia se distribuye ENA24? Falta anotarla en el informe de comprensión de los datos y en el anexo de licencias | Técnica | 30 sep | CDLA-Permissive-1.0, verificada en lila.science. Ver la revisión del 30 de septiembre de la ficha de fuentes con licencia rastreable en `registro_decisiones.md`. Falta reescribir con voz propia el párrafo del informe de comprensión |

**3b.** Añadir:

| 22 | ¿Conviene correr el experimento de ablación (solo iNaturalist, solo ENA24 y ambas fuentes)? | Técnica | 2 oct | Sí, y se hizo (`experimentos.md`). Sin ENA24, el modelo no alcanza la meta en cámara trampa (como máximo 0.85, con 36 % de falsas alarmas). Sin iNaturalist rinde parecido en cámara trampa, pero con fotos de oso andino y de otras especies falla: detecta 0.53–0.59 y da 20–23 % de falsas alarmas. Se conservan ambas fuentes en la iteración 2 (D21). La pregunta hablaba de evaluar sobre prueba: se hizo sobre validación y se repetirá en la evaluación final |

---

## 4. Sección "Notas sobre preguntas que cambiaron de forma": añadir al final

La pregunta 13 cambió el 2 de octubre. La nota anterior decía que la descarga selectiva desde LILA BC no se había investigado. Con WCS se resolvió: cada foto se baja por HTTP con su ruta. Lo que queda es decidir si vale la pena, junto con la pregunta 29.
