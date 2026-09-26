# Conteo de unidades de la selladora

# Conteo por la marca "Quality"

El programa usa OCR para buscar la palabra impresa `Quality`. Cada lectura OCR
genera un evento `SALIDA`, aunque sea la misma marca que ya aparecio en cuadros
anteriores. No hace falta dibujar cajas ni entrenar un modelo.

## Instalacion

Instala Tesseract OCR para Windows una sola vez (si ya aparece en Program Files,
ese paso esta hecho):

```powershell
winget install --id UB-Mannheim.TesseractOCR --exact
python -m pip install pytesseract
```

El programa detecta automaticamente Tesseract en `C:\Program Files\Tesseract-OCR`.

## Ejecucion

Desde esta carpeta, ejecuta en PowerShell:

```powershell
python unidades.py "..\..\WhatsApp Video 2026-09-25 at 7.39.33 PM.mp4" --rotacion 180 --calibrar --record quality_anotado.mp4
```

En la ventana, haz dos clics para dibujar la zona del recorrido donde se lee la
palabra `Quality` y pulsa Enter. Se guarda en `config_fichas.json`. El conteo de
salida se genera al detectar una marca nueva dentro de esa zona; no depende del
sentido de cruce de la linea.

- `--ocr-confianza 25` ajusta cuan estricta es la lectura OCR.
- `--invertir` cambia el sentido que se considera salida.
- `--stride 8` analiza uno de cada ocho cuadros; usa `--stride 4` si el empaque
  se mueve rapido y el equipo puede procesarlo.
- `--velocidad 0.5` reproduce la vista previa a media velocidad; `--velocidad 0`
  procesa tan rapido como sea posible.
- `--headless` procesa sin ventana. Requiere haber calibrado antes.
- `--csv resultado.csv` y `--record anotado.mp4` permiten elegir las salidas.
- `--calibrar` vuelve a dibujar la linea.

El contador ahora suma lecturas OCR, no empaques unicos: una misma bolsa visible
en varios cuadros puede generar varias salidas. Para contar unidades fisicas,
se debe volver al conteo unico por seguimiento o usar un cruce de linea validado.