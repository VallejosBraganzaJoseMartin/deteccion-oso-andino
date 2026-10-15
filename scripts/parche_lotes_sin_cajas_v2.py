"""Parche del proyecto oso andino (versión 2, 2026-10-05) al código oficial de RT-DETRv3.

Problema: si ninguna imagen de un lote de entrenamiento tiene cajas, la función de denoising devuelve None
para todos los grupos de consultas y el código de RT-DETRv3 falla al unir piezas (paddle.concat con None).
El RT-DETR original sí contempla ese caso (comprueba None); la versión 3, con varios grupos, no.

Este parche hace que esos lotes se entrenen como en el RT-DETR original: sin consultas de denoising, solo con
las consultas normales, cuya pérdida de clasificación enseña al modelo que en esas fotos no hay oso.
Reemplaza al parche de la versión 1, que saltaba esos lotes (y que con 2 GPU congelaría el entrenamiento).

Toca tres sitios:
  1. rtdetr_transformerv3.py, _get_decoder_input: no concatena el denoising de un grupo si es None.
  2. rtdetr_transformerv3.py, forward: arma la máscara de atención de un grupo sin denoising y le pone un
     dn_meta de marcador con el número de consultas del grupo, para que la cabeza pueda separar los grupos.
  3. detr_head.py, DINOv3Head.forward: en los grupos marcados calcula la pérdida sin denoising y repite la
     reducción entre GPUs que la pérdida de denoising habría hecho, para que todas las GPUs hagan las mismas.

Uso: python parche_lotes_sin_cajas_v2.py RUTA_DEL_REPOSITORIO   (se puede correr varias veces)"""
import sys
from pathlib import Path

repo = Path(sys.argv[1])
MARCA = "Parche del proyecto oso andino v2"

CAMBIOS = {
    repo / "ppdet/modeling/transformers/rtdetr_transformerv3.py": [
        (  # 1a. referencia de las cajas de denoising
            "            if denoising_bbox_unacts is not None and not (self.o2m_branch and g_id == self.num_groups - 1):\n",
            "            if denoising_bbox_unacts is not None and not (self.o2m_branch and g_id == self.num_groups - 1) \\\n"
            "                    and denoising_bbox_unacts[g_id] is not None:  # " + MARCA + "\n",
        ),
        (  # 1b. consultas de denoising
            "            if denoising_classes is not None and not (self.o2m_branch and g_id == self.num_groups - 1):\n",
            "            if denoising_classes is not None and not (self.o2m_branch and g_id == self.num_groups - 1) \\\n"
            "                    and denoising_classes[g_id] is not None:  # " + MARCA + "\n",
        ),
        (  # 2a. máscara de atención de un grupo sin denoising
            "                else:\n"
            "                    end = end + attn_masks[g_id].shape[1]\n",
            "                elif attn_masks[g_id] is None:  # " + MARCA + ": lote sin cajas, grupo sin denoising\n"
            "                    end = end + self.num_queries[g_id]\n"
            "                    new_mask = new_mask > 0.1 if g_id > 0 else new_mask >= 0.0\n"
            "                    new_attn_mask[begin: end, begin: end] = new_mask\n"
            "                else:\n"
            "                    end = end + attn_masks[g_id].shape[1]\n",
        ),
        (  # 2b. marcador para la cabeza
            "            attn_masks = new_attn_mask\n",
            "            attn_masks = new_attn_mask\n"
            "            # " + MARCA + ": los grupos sin denoising llevan un dn_meta de marcador con su número de consultas\n"
            "            dn_metas = [dm if dm is not None else {'dn_num_split': [0, self.num_queries[i]], 'sin_denoising': True}\n"
            "                        for i, dm in enumerate(dn_metas)]\n",
        ),
    ],
    repo / "ppdet/modeling/heads/detr_head.py": [
        (  # 3a. separación por grupo
            "                for g_id in range(num_groups):\n"
            "                    dn_out_bboxes_gid, dec_out_bboxes_gid = paddle.split(\n"
            "                        dec_out_bboxes[g_id], dn_meta[g_id]['dn_num_split'], axis=2)\n"
            "                    dn_out_logits_gid, dec_out_logits_gid = paddle.split(\n"
            "                        dec_out_logits[g_id], dn_meta[g_id]['dn_num_split'], axis=2)\n",
            "                for g_id in range(num_groups):\n"
            "                    sin_dn = dn_meta[g_id].get('sin_denoising', False)  # " + MARCA + "\n"
            "                    if sin_dn:\n"
            "                        dn_out_bboxes_gid = dn_out_logits_gid = None\n"
            "                        dec_out_bboxes_gid, dec_out_logits_gid = dec_out_bboxes[g_id], dec_out_logits[g_id]\n"
            "                    else:\n"
            "                        dn_out_bboxes_gid, dec_out_bboxes_gid = paddle.split(\n"
            "                            dec_out_bboxes[g_id], dn_meta[g_id]['dn_num_split'], axis=2)\n"
            "                        dn_out_logits_gid, dec_out_logits_gid = paddle.split(\n"
            "                            dec_out_logits[g_id], dn_meta[g_id]['dn_num_split'], axis=2)\n",
        ),
        (  # 3b. pérdida sin denoising y reducción entre GPUs equivalente
            "                        dn_out_logits=dn_out_logits_gid,\n"
            "                        dn_meta=dn_meta[g_id])\n",
            "                        dn_out_logits=dn_out_logits_gid,\n"
            "                        dn_meta=None if sin_dn else dn_meta[g_id])\n"
            "                    if sin_dn:  # " + MARCA + ": misma cantidad de reducciones entre GPUs que con denoising\n"
            "                        self.loss._get_num_gts(inputs['gt_class'])\n",
        ),
    ],
}

for ruta, cambios in CAMBIOS.items():
    texto = ruta.read_text()
    if MARCA in texto:
        print("Ya estaba aplicado en", ruta.name)
        continue
    for viejo, nuevo in cambios:
        n = texto.count(viejo)
        if n != 1:
            raise SystemExit(f"{ruta.name}: el texto de anclaje aparece {n} veces (se esperaba 1); no se parcha nada.")
        texto = texto.replace(viejo, nuevo)
    ruta.write_text(texto)
    print("Parche aplicado en", ruta.name)
