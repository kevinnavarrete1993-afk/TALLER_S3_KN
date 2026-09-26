#!/usr/bin/env python3
"""Divide imagenes y etiquetas en train/val para Ultralytics YOLO."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path


EXTENSIONES = {".jpg", ".jpeg", ".png", ".bmp"}


def validar_etiqueta(ruta: Path) -> None:
    for numero, linea in enumerate(ruta.read_text(encoding="utf-8").splitlines(), start=1):
        partes = linea.split()
        if len(partes) != 5 or partes[0] != "0":
            raise SystemExit(f"Etiqueta invalida en {ruta}, linea {numero}")
        try:
            valores = [float(valor) for valor in partes[1:]]
        except ValueError as exc:
            raise SystemExit(f"Coordenadas invalidas en {ruta}, linea {numero}") from exc
        if any(not 0 <= valor <= 1 for valor in valores):
            raise SystemExit(f"Coordenadas fuera de 0..1 en {ruta}, linea {numero}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepara imagenes etiquetadas para entrenar YOLO.")
    parser.add_argument("--imagenes", default="dataset/imagenes")
    parser.add_argument("--etiquetas", default="dataset/etiquetas")
    parser.add_argument("--salida", default="dataset/yolo")
    parser.add_argument("--validacion", type=float, default=0.2)
    args = parser.parse_args()
    if not 0 < args.validacion < 1:
        parser.error("--validacion debe estar entre 0 y 1")

    carpeta_imagenes = Path(args.imagenes)
    carpeta_etiquetas = Path(args.etiquetas)
    destino = Path(args.salida)
    imagenes = sorted(p for p in carpeta_imagenes.iterdir()
                      if p.suffix.lower() in EXTENSIONES) if carpeta_imagenes.is_dir() else []
    if len(imagenes) < 5:
        raise SystemExit("Se necesitan al menos 5 imagenes etiquetadas.")
    if (destino / "data.yaml").exists():
        raise SystemExit(f"Ya existe {destino / 'data.yaml'}; elige otra salida para no mezclar divisiones.")

    pares = []
    for imagen in imagenes:
        etiqueta = carpeta_etiquetas / f"{imagen.stem}.txt"
        if not etiqueta.is_file():
            raise SystemExit(f"Falta etiqueta para {imagen.name}")
        validar_etiqueta(etiqueta)
        pares.append((imagen, etiqueta))

    cantidad_val = max(1, round(len(pares) * args.validacion))
    grupos = {"train": pares[:-cantidad_val], "val": pares[-cantidad_val:]}
    for grupo, elementos in grupos.items():
        carpeta_img = destino / "images" / grupo
        carpeta_lbl = destino / "labels" / grupo
        carpeta_img.mkdir(parents=True, exist_ok=True)
        carpeta_lbl.mkdir(parents=True, exist_ok=True)
        for imagen, etiqueta in elementos:
            shutil.copy2(imagen, carpeta_img / imagen.name)
            shutil.copy2(etiqueta, carpeta_lbl / etiqueta.name)

    manifiesto = {
        "path": str(destino.resolve()),
        "train": "images/train",
        "val": "images/val",
        "names": {"0": "paquete"},
    }
    (destino / "data.yaml").write_text(json.dumps(manifiesto, indent=2), encoding="utf-8")
    print(f"Entrenamiento: {len(grupos['train'])} imagenes")
    print(f"Validacion: {len(grupos['val'])} imagenes")
    print(f"Configuracion YOLO: {(destino / 'data.yaml').resolve()}")


if __name__ == "__main__":
    main()
