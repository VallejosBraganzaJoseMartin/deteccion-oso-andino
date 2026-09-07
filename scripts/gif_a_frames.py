import argparse
import os
from PIL import Image, ImageSequence


def main():
    parser = argparse.ArgumentParser(
        description="Extrae frames de una imagen animada (GIF) y los guarda en formato JPEG."
    )
    parser.add_argument(
        "-i", "--input",
        required=True,
        help="Ruta del archivo de entrada (GIF o imagen animada)."
    )
    parser.add_argument(
        "-o", "--output",
        default="data/frames_gif",
        help="Carpeta de destino para los frames extraídos (por defecto: data/frames_gif)."
    )
    parser.add_argument(
        "-s", "--salto",
        type=int,
        default=2,
        help="Guarda 1 de cada N frames para evitar duplicados excesivos (por defecto: 2)."
    )

    args = parser.parse_args()

    archivo_entrada = args.input
    carpeta_salida = args.output
    salto_frames = args.salto

    if not os.path.exists(archivo_entrada):
        raise FileNotFoundError(f"No se encontró el archivo: {archivo_entrada}")

    nombre_base = os.path.splitext(os.path.basename(archivo_entrada))[0]
    carpeta_destino = os.path.join(carpeta_salida, nombre_base)
    os.makedirs(carpeta_destino, exist_ok=True)

    with Image.open(archivo_entrada) as img:
        contador_guardados = 0
        for i, frame in enumerate(ImageSequence.Iterator(img)):
            if i % salto_frames == 0:
                # Conversión obligatoria de paleta indexada a canales RGB
                frame_rgb = frame.convert("RGB")
                nombre_salida = os.path.join(
                    carpeta_destino, f"{nombre_base}_frame_{contador_guardados:03d}.jpg"
                )
                frame_rgb.save(nombre_salida, format="JPEG", quality=95)
                contador_guardados += 1

    print(f"Se extrajeron {contador_guardados} frames válidos en '{carpeta_destino}'.")


if __name__ == "__main__":
    main()