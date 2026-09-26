#!/usr/bin/env python3
"""Extrae cuadros espaciados de un video para preparar un detector de paquetes."""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2


def rotar(frame, grados: int):
    giros = {
        0: None,
        90: cv2.ROTATE_90_CLOCKWISE,
        180: cv2.ROTATE_180,
        270: cv2.ROTATE_90_COUNTERCLOCKWISE,
    }
    giro = giros[grados]
    return frame if giro is None else cv2.rotate(frame, giro)


def main() -> None:
    parser = argparse.ArgumentParser(description="Extrae cuadros para etiquetar paquetes.")
    parser.add_argument("video", help="ruta al video de la selladora")
    parser.add_argument("--salida", default="dataset/imagenes", help="carpeta para los cuadros")
    parser.add_argument("--intervalo", type=float, default=0.5,
                        help="segundos entre cuadros extraidos")
    parser.add_argument("--rotacion", type=int, choices=[0, 90, 180, 270], default=0)
    args = parser.parse_args()
    if args.intervalo <= 0:
        parser.error("--intervalo debe ser mayor que cero")

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise SystemExit(f"No se pudo abrir el video: {args.video}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    duracion = frames / fps if frames else 0
    carpeta = Path(args.salida)
    carpeta.mkdir(parents=True, exist_ok=True)

    guardados = 0
    segundo = 0.0
    while segundo < duracion:
        cap.set(cv2.CAP_PROP_POS_MSEC, segundo * 1000)
        ok, frame = cap.read()
        if not ok:
            break
        frame = rotar(frame, args.rotacion)
        destino = carpeta / f"muestra_{guardados:05d}.jpg"
        if not cv2.imwrite(str(destino), frame):
            cap.release()
            raise SystemExit(f"No se pudo guardar: {destino}")
        guardados += 1
        segundo += args.intervalo

    cap.release()
    print(f"Cuadros guardados: {guardados}")
    print(f"Carpeta: {carpeta.resolve()}")


if __name__ == "__main__":
    main()