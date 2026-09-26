#!/usr/bin/env python3
"""Etiqueta cajas de paquetes en formato YOLO para entrenamiento."""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2


EXTENSIONES = {".jpg", ".jpeg", ".png", ".bmp"}


def guardar_etiquetas(ruta: Path, cajas: list[tuple[int, int, int, int]],
                      ancho: int, alto: int) -> None:
    lineas = []
    for x1, y1, x2, y2 in cajas:
        centro_x = ((x1 + x2) / 2) / ancho
        centro_y = ((y1 + y2) / 2) / alto
        caja_ancho = (x2 - x1) / ancho
        caja_alto = (y2 - y1) / alto
        lineas.append(f"0 {centro_x:.6f} {centro_y:.6f} {caja_ancho:.6f} {caja_alto:.6f}")
    ruta.write_text("\n".join(lineas), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Dibuja cajas sobre los empaques completos.")
    parser.add_argument("--imagenes", default="dataset/imagenes")
    parser.add_argument("--etiquetas", default="dataset/etiquetas")
    args = parser.parse_args()
    carpeta_imagenes = Path(args.imagenes)
    carpeta_etiquetas = Path(args.etiquetas)
    if not carpeta_imagenes.is_dir():
        raise SystemExit(f"No existe la carpeta de imagenes: {carpeta_imagenes}")
    carpeta_etiquetas.mkdir(parents=True, exist_ok=True)
    imagenes = sorted(p for p in carpeta_imagenes.iterdir() if p.suffix.lower() in EXTENSIONES)
    pendientes = [p for p in imagenes if not (carpeta_etiquetas / f"{p.stem}.txt").exists()]
    if not pendientes:
        raise SystemExit("No hay imagenes pendientes de etiquetar.")

    estado: dict = {"inicio": None, "actual": None, "cajas": []}

    def mouse(event, x, y, _flags, _param):
        if event == cv2.EVENT_LBUTTONDOWN:
            estado["inicio"] = (x, y)
            estado["actual"] = (x, y)
        elif event == cv2.EVENT_MOUSEMOVE and estado["inicio"] is not None:
            estado["actual"] = (x, y)
        elif event == cv2.EVENT_LBUTTONUP and estado["inicio"] is not None:
            x0, y0 = estado["inicio"]
            caja = (min(x0, x), min(y0, y), max(x0, x), max(y0, y))
            if caja[2] - caja[0] >= 8 and caja[3] - caja[1] >= 8:
                estado["cajas"].append(caja)
            estado["inicio"] = None
            estado["actual"] = None
        elif event == cv2.EVENT_RBUTTONDOWN and estado["cajas"]:
            estado["cajas"].pop()

    nombre_ventana = "Etiquetar paquetes"
    cv2.namedWindow(nombre_ventana, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(nombre_ventana, mouse)
    guardadas = 0
    try:
        for indice, ruta_imagen in enumerate(pendientes, start=1):
            imagen = cv2.imread(str(ruta_imagen))
            if imagen is None:
                print(f"No se pudo leer: {ruta_imagen}")
                continue
            alto, ancho = imagen.shape[:2]
            estado["cajas"] = []
            while True:
                vista = imagen.copy()
                for x1, y1, x2, y2 in estado["cajas"]:
                    cv2.rectangle(vista, (x1, y1), (x2, y2), (0, 255, 0), 2)
                if estado["inicio"] and estado["actual"]:
                    cv2.rectangle(vista, estado["inicio"], estado["actual"], (0, 220, 255), 2)
                cv2.putText(vista, f"{indice}/{len(pendientes)}  cajas: {len(estado['cajas'])}",
                            (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 220, 255), 2)
                cv2.putText(vista, "Arrastra caja completa | clic derecho deshace | s guarda | n vacia | q sale",
                            (10, alto - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 220, 255), 1)
                cv2.imshow(nombre_ventana, vista)
                tecla = cv2.waitKey(20) & 0xFF
                if tecla == ord("u") and estado["cajas"]:
                    estado["cajas"].pop()
                elif tecla == ord("s"):
                    ruta_txt = carpeta_etiquetas / f"{ruta_imagen.stem}.txt"
                    guardar_etiquetas(ruta_txt, estado["cajas"], ancho, alto)
                    guardadas += 1
                    break
                elif tecla == ord("n"):
                    ruta_txt = carpeta_etiquetas / f"{ruta_imagen.stem}.txt"
                    ruta_txt.write_text("", encoding="utf-8")
                    guardadas += 1
                    break
                elif tecla == ord("q") or tecla == 27:
                    return
    finally:
        cv2.destroyAllWindows()

    print(f"Imagenes etiquetadas en esta sesion: {guardadas}")
    print(f"Etiquetas: {carpeta_etiquetas.resolve()}")


if __name__ == "__main__":
    main()