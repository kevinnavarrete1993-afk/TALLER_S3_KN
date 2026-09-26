#!/usr/bin/env python3
"""Detecta la palabra QUALITY con OCR y registra cada lectura como SALIDA."""

from __future__ import annotations

import argparse
import csv
from difflib import SequenceMatcher
import json
import os
import re
import shutil
import sys
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np


def rotar(frame: np.ndarray, grados: int) -> np.ndarray:
    """Rota un cuadro 0, 90, 180 o 270 grados segun la orientacion del video."""
    giros = {
        0: None,
        90: cv2.ROTATE_90_CLOCKWISE,
        180: cv2.ROTATE_180,
        270: cv2.ROTATE_90_COUNTERCLOCKWISE,
    }
    giro = giros[grados]
    return frame if giro is None else cv2.rotate(frame, giro)


def seleccionar_linea(frame: np.ndarray) -> list[list[int]]:
    """Permite marcar dos extremos con el mouse y devuelve la linea en pixeles."""
    puntos: list[tuple[int, int]] = []

    def mouse(event, x, y, _flags, _param):
        """Guarda un punto con clic izquierdo o quita el ultimo con clic derecho."""
        if event == cv2.EVENT_LBUTTONDOWN and len(puntos) < 2:
            puntos.append((x, y))
        elif event == cv2.EVENT_RBUTTONDOWN and puntos:
            puntos.pop()

    cv2.namedWindow("Linea de salida", cv2.WINDOW_NORMAL)
    cv2.setMouseCallback("Linea de salida", mouse)
    while True:
        vista = frame.copy()
        for punto in puntos:
            cv2.circle(vista, punto, 6, (0, 220, 255), -1)
        if len(puntos) == 2:
            cv2.line(vista, puntos[0], puntos[1], (0, 220, 255), 2)
        cv2.putText(vista, "Dos clics en la salida; ENTER confirma; clic derecho deshace",
                    (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 220, 255), 1)
        cv2.imshow("Linea de salida", vista)
        tecla = cv2.waitKey(20) & 0xFF
        if tecla == 27:
            cv2.destroyWindow("Linea de salida")
            raise SystemExit("Calibracion cancelada.")
        if tecla in (10, 13) and len(puntos) == 2:
            cv2.destroyWindow("Linea de salida")
            return [list(puntos[0]), list(puntos[1])]


def calibrar(frame: np.ndarray, ruta_config: Path, rotacion: int) -> dict:
    """Pide la linea de seguimiento y guarda linea y rotacion en JSON."""
    print("Dibuja UNA linea cruzando la trayectoria de la palabra QUALITY.")
    linea = seleccionar_linea(frame)
    config = {
        "rotacion": rotacion,
        "linea": linea,
    }
    ruta_config.write_text(json.dumps(config, indent=2), encoding="utf-8")
    print(f"Calibracion guardada en: {ruta_config}")
    return config


def detectar_quality(frame: np.ndarray, pytesseract, linea: list[list[int]],
                     confianza_min: int = 25, margen: int = 180
                     ) -> list[tuple[int, int, int, int]]:
    """Busca QUALITY con Tesseract cerca de la linea y devuelve sus cajas.

    El margen limita la region analizada para reducir el costo del OCR; la
    similitud de texto admite errores pequenos causados por desenfoque.
    """
    alto_frame, ancho_frame = frame.shape[:2]
    # Analizar solo el rectangulo alrededor de la zona de conteo acelera OCR.
    x0 = max(0, min(p[0] for p in linea) - margen)
    y0 = max(0, min(p[1] for p in linea) - margen)
    x1 = min(ancho_frame, max(p[0] for p in linea) + margen)
    y1 = min(alto_frame, max(p[1] for p in linea) + margen)
    recorte = frame[y0:y1, x0:x1]
    if recorte.size == 0:
        return []
    escala = 2
    gris = cv2.cvtColor(recorte, cv2.COLOR_BGR2GRAY)
    # Ampliar el texto ayuda a Tesseract a leer la impresion pequena del empaque.
    ampliado = cv2.resize(gris, None, fx=escala, fy=escala,
                          interpolation=cv2.INTER_CUBIC)
    datos = pytesseract.image_to_data(
        ampliado,
        config="--oem 3 --psm 11",
        output_type=pytesseract.Output.DICT,
    )
    cajas = []
    for texto, confianza, x, y, ancho, alto in zip(
        datos["text"], datos["conf"], datos["left"], datos["top"],
        datos["width"], datos["height"],
    ):
        palabra = re.sub(r"[^a-z]", "", texto.casefold())
        try:
            confianza_num = float(confianza)
        except (TypeError, ValueError):
            continue
        # Se toleran faltas pequenas del OCR, pero se exige que parezca QUALITY.
        parecido = (len(palabra) >= 5
                and SequenceMatcher(None, palabra, "quality").ratio() >= 0.75)
        if not parecido or confianza_num < confianza_min:
            continue
        cajas.append((x0 + int(x / escala), y0 + int(y / escala),
                      max(1, int(ancho / escala)), max(1, int(alto / escala))))
    return cajas


def lado(linea: list[list[int]], punto: tuple[float, float]) -> int:
    """Devuelve en que lado de la linea esta el punto: -1, 0 o 1."""
    (x1, y1), (x2, y2) = linea
    cruz = (x2 - x1) * (punto[1] - y1) - (y2 - y1) * (punto[0] - x1)
    return 0 if abs(cruz) < 1e-6 else (1 if cruz > 0 else -1)


def sobre_segmento(linea: list[list[int]], punto: tuple[float, float]) -> bool:
    """Indica si la proyeccion del punto cae sobre el segmento dibujado."""
    inicio = np.asarray(linea[0], dtype=float)
    vector = np.asarray(linea[1], dtype=float) - inicio
    largo2 = float(vector @ vector)
    if largo2 < 1:
        return False
    proyeccion = float(vector @ (np.asarray(punto) - inicio)) / largo2
    return -0.08 <= proyeccion <= 1.08


class Unidad:
    """Representa una marca QUALITY detectada y su estado de seguimiento."""

    def __init__(self, identificador: int, caja: tuple[int, int, int, int], lado_actual: int):
        """Inicializa el ID, caja, centro y estado de esta marca."""
        self.id = identificador
        self.caja = caja
        self.centro = self._centro(caja)
        self.lado = lado_actual
        self.perdida = 0
        self.contada = False

    @staticmethod
    def _centro(caja: tuple[int, int, int, int]) -> tuple[float, float]:
        """Calcula el centro de una caja (x, y, ancho, alto)."""
        x, y, ancho, alto = caja
        return x + ancho / 2, y + alto / 2


class Rastreador:
    """Asocia lecturas OCR cercanas para conservar el ID entre cuadros."""

    def __init__(self, distancia_max: float = 100, cuadros_perdida: int = 12):
        """Configura distancia de asociacion y tolerancia a lecturas ausentes."""
        self.distancia_max = distancia_max
        self.cuadros_perdida = cuadros_perdida
        self.fichas: list[Unidad] = []
        self.siguiente_id = 1

    def actualizar(self, cajas: list[tuple[int, int, int, int]], linea: list[list[int]],
                   invertir: bool) -> list[Unidad]:
        """Actualiza seguimientos y devuelve una SALIDA por cada lectura OCR.

        Una marca que reaparece cerca conserva su ID, pero cada lectura vuelve
        a producir un evento. Por tanto, el total representa lecturas OCR, no
        necesariamente empaques fisicos unicos.
        """
        # Las marcas pueden faltar temporalmente si Tesseract no lee un cuadro.
        for ficha in self.fichas:
            ficha.perdida += 1
        self.fichas = [ficha for ficha in self.fichas
                       if ficha.perdida <= self.cuadros_perdida]

        centros = [Unidad._centro(caja) for caja in cajas]
        pares = []
        for indice_ficha, ficha in enumerate(self.fichas):
            for indice_caja, centro in enumerate(centros):
                distancia = float(np.hypot(centro[0] - ficha.centro[0],
                                           centro[1] - ficha.centro[1]))
                if distancia < self.distancia_max:
                    pares.append((distancia, indice_ficha, indice_caja))
        pares.sort()
        usados_fichas: set[int] = set()
        usados_cajas: set[int] = set()
        for _distancia, indice_ficha, indice_caja in pares:
            if indice_ficha in usados_fichas or indice_caja in usados_cajas:
                continue
            ficha = self.fichas[indice_ficha]
            centro = centros[indice_caja]
            lado_nuevo = lado(linea, centro)
            # Cada lectura asociada a una marca existente tambien cuenta salida.
            ficha.caja = cajas[indice_caja]
            ficha.centro = centro
            ficha.perdida = 0
            ficha.evento = True
            if lado_nuevo:
                ficha.lado = lado_nuevo
            usados_fichas.add(indice_ficha)
            usados_cajas.add(indice_caja)

        for indice, caja in enumerate(cajas):
            if indice not in usados_cajas:
                # Una caja sin seguimiento cercano inicia un ID y evento nuevo.
                ficha = Unidad(self.siguiente_id, caja, lado(linea, centros[indice]))
                ficha.contada = True
                ficha.evento = True
                self.fichas.append(ficha)
                self.siguiente_id += 1

        self.fichas = [f for f in self.fichas if f.perdida <= self.cuadros_perdida]
        eventos = []
        for ficha in self.fichas:
            if getattr(ficha, "evento", False):
                eventos.append(ficha)
                # Consumir el evento evita volver a emitirlo fuera de este cuadro.
                ficha.evento = False
        return eventos

    def visibles(self) -> list[Unidad]:
        """Devuelve marcas vistas en el cuadro actual, no las temporalmente perdidas."""
        return [f for f in self.fichas if f.perdida == 0]


def dibujar(frame: np.ndarray, tracker: Rastreador, linea: list[list[int]],
            salidas: int) -> np.ndarray:
    """Dibuja la linea, cajas e ID de QUALITY y el total acumulado."""
    vista = frame.copy()
    cv2.line(vista, tuple(linea[0]), tuple(linea[1]), (0, 220, 255), 2, cv2.LINE_AA)
    for ficha in tracker.visibles():
        x, y, ancho, alto = ficha.caja
        color = (0, 255, 0) if ficha.contada else (255, 180, 0)
        cv2.rectangle(vista, (x, y), (x + ancho, y + alto), color, 2)
        cv2.putText(vista, f"QUALITY #{ficha.id}", (x, max(y - 6, 16)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)
    cv2.rectangle(vista, (8, 8), (230, 55), (30, 30, 30), -1)
    cv2.putText(vista, f"SALIDAS: {salidas}", (18, 31),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(vista, "Marca seguida: QUALITY", (18, 49),
                cv2.FONT_HERSHEY_SIMPLEX, 0.38, (220, 220, 220), 1)
    return vista


def main() -> None:
    """Lee argumentos, prepara OCR y procesa el video cuadro por cuadro."""
    parser = argparse.ArgumentParser(description="Cuenta empaques siguiendo la palabra QUALITY por una linea.")
    parser.add_argument("video", help="archivo de video")
    parser.add_argument("--config", default=str(Path(__file__).with_name("config_fichas.json")))
    parser.add_argument("--calibrar", action="store_true", help="dibujar la linea de salida de nuevo")
    parser.add_argument("--rotacion", type=int, choices=[0, 90, 180, 270], default=None)
    parser.add_argument("--ocr-confianza", type=int, default=25,
                        help="confianza minima de Tesseract para aceptar QUALITY")
    parser.add_argument("--distancia-max", type=float, default=100)
    parser.add_argument("--perdida-max", type=int, default=12)
    parser.add_argument("--stride", type=int, default=8, help="analizar cada N cuadros")
    parser.add_argument("--velocidad", type=float, default=1.0,
                        help="velocidad de vista previa; 0 procesa a maxima velocidad")
    parser.add_argument("--invertir", action="store_true", help="invertir el sentido de salida")
    parser.add_argument("--csv", default=None, help="ruta CSV; por defecto crea un nombre con fecha")
    parser.add_argument("--record", default=None, help="guardar video anotado")
    parser.add_argument("--headless", action="store_true", help="procesar sin abrir ventana")
    args = parser.parse_args()

    if not 0 <= args.ocr_confianza <= 100 or args.stride < 1 or args.velocidad < 0:
        parser.error("--ocr-confianza debe estar entre 0 y 100; --stride positivo; --velocidad no negativa")
    if not os.path.isfile(args.video):
        parser.error(f"No existe el video: {args.video}")

    try:
        import pytesseract
    except ImportError as exc:
        raise SystemExit(
            "Falta el paquete pytesseract. Ejecuta 'python -m pip install pytesseract'."
        ) from exc
    # Buscar la instalacion usual de Windows si Tesseract no esta en PATH.
    if not shutil.which("tesseract"):
        rutas_tesseract = (
            Path(os.environ.get("ProgramFiles", r"C:\Program Files"))
            / "Tesseract-OCR" / "tesseract.exe",
            Path(os.environ.get("ProgramFiles(x86)", r"C:\Program Files (x86)"))
            / "Tesseract-OCR" / "tesseract.exe",
        )
        ruta_encontrada = next((ruta for ruta in rutas_tesseract if ruta.is_file()), None)
        if ruta_encontrada:
            pytesseract.pytesseract.tesseract_cmd = str(ruta_encontrada)
    try:
        pytesseract.get_tesseract_version()
    except pytesseract.TesseractNotFoundError as exc:
        raise SystemExit(
            "Falta el motor Tesseract OCR de Windows. Instálalo y vuelve a ejecutar."
        ) from exc

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise SystemExit(f"OpenCV no pudo abrir el video: {args.video}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    ok, primer_frame = cap.read()
    if not ok:
        raise SystemExit("No se pudo leer el primer cuadro.")

    ruta_config = Path(args.config)
    if args.calibrar or not ruta_config.exists():
        # La calibracion guarda la linea para poder reutilizarla despues.
        rotacion = args.rotacion if args.rotacion is not None else 0
        primer_frame = rotar(primer_frame, rotacion)
        config = calibrar(primer_frame, ruta_config, rotacion)
    else:
        config = json.loads(ruta_config.read_text(encoding="utf-8"))
        rotacion = args.rotacion if args.rotacion is not None else int(config["rotacion"])
        primer_frame = rotar(primer_frame, rotacion)

    linea = config["linea"]
    tracker = Rastreador(args.distancia_max, args.perdida_max)
    salidas = 0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    ruta_csv = (Path(args.csv).expanduser().resolve() if args.csv
                else Path(__file__).with_name(f"salidas_{timestamp}.csv").resolve())
    ruta_csv.parent.mkdir(parents=True, exist_ok=True)
    ruta_video = Path(args.record).expanduser().resolve() if args.record else None
    escritor = None
    if ruta_video:
        ruta_video.parent.mkdir(parents=True, exist_ok=True)

    frame = primer_frame
    indice = 0
    fin_natural = False
    print(f"[video] {Path(args.video).resolve()}")
    print(f"[csv] {ruta_csv}")
    if ruta_video:
        print(f"[mp4] {ruta_video}")
    try:
        with ruta_csv.open("w", newline="", encoding="utf-8") as archivo_csv:
            csv_writer = csv.writer(archivo_csv)
            csv_writer.writerow(["segundo_video", "evento", "marca_id", "salidas_total"])
            while True:
                if indice:
                    ok, frame = cap.read()
                    if not ok:
                        fin_natural = True
                        break
                    frame = rotar(frame, rotacion)
                if indice % args.stride == 0:
                    # OCR encuentra QUALITY; el rastreador convierte cajas en eventos.
                    cajas = detectar_quality(frame, pytesseract, linea, args.ocr_confianza)
                    eventos = tracker.actualizar(cajas, linea, args.invertir)
                    for ficha in eventos:
                        salidas += 1
                        segundo = indice / fps
                        # Una fila por lectura aceptada, con total acumulado.
                        csv_writer.writerow([round(segundo, 3), "SALIDA", ficha.id, salidas])
                        archivo_csv.flush()
                        print(f"SALIDA #{salidas}: QUALITY #{ficha.id}, t={segundo:.2f}s")

                    vista = dibujar(frame, tracker, linea, salidas)
                if ruta_video:
                    if escritor is None:
                        alto, ancho = vista.shape[:2]
                        escritor = cv2.VideoWriter(str(ruta_video), cv2.VideoWriter_fourcc(*"mp4v"),
                                                   fps / args.stride, (ancho, alto))
                        if not escritor.isOpened():
                            escritor.release()
                            escritor = None
                            raise SystemExit(f"No pude crear el video de salida: {ruta_video}")
                    if escritor.isOpened() and indice % args.stride == 0:
                        escritor.write(vista)
                if not args.headless:
                    cv2.imshow("Conteo de fichas", vista)
                    espera_ms = max(1, round(1000 / fps / args.velocidad)) if args.velocidad else 1
                    tecla = cv2.waitKey(espera_ms) & 0xFF
                    if tecla == ord("q"):
                        break
                    if tecla == ord(" "):
                        cv2.waitKey(0)
                indice += 1
    finally:
        cap.release()
        if escritor:
            escritor.release()
        if not args.headless:
            if fin_natural:
                print("Fin del video. Pulsa cualquier tecla para cerrar la vista.")
                cv2.waitKey(0)
            cv2.destroyAllWindows()

    print(f"\nCuadros leidos: {indice} / {total_frames}")
    print(f"Total de salidas: {salidas}")
    print(f"CSV: {ruta_csv}")
    if ruta_video:
        print(f"Video anotado: {ruta_video}")


if __name__ == "__main__":
    main()