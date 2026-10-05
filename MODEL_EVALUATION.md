# Model Evaluation — TactiVision AI

Regla: **solo se registran resultados medidos realmente**. `n/a` = todavía no medido.

## Ronda 1 — prueba de integración (no es una evaluación de fútbol)

- Fecha: 2026-10-04 · Script: `python evaluate_models.py --videos ... --models YOLO26n YOLO11n RT-DETR-l --runtime onnx --max-frames 62`
- Máquina: contenedor Linux x86_64, **1 núcleo de CPU**, sin GPU. Runtime: onnxruntime (CPU).
- Videos: clips de ejemplo públicos de Ultralytics (`solutions_ci_demo.mp4`, escena urbana con
  muchas personas, 640×360; `decelera_landscape_min.mov`, 2 personas, 1920×1080).
  **No son partidos de fútbol**: sirven para medir velocidad y estabilidad del tracking,
  no la calidad táctica.

| Model | Runtime | Video | Frames | Inference (ms/frame) | Processed FPS | Players/frame | Mean player conf. | Ball detection | Tracks | Fragmentation ratio* | Mean track length | Precision | Recall | mAP |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| YOLO26n | onnx | solutions_ci_demo.mp4 | 62 | 102.86 | 7.72 | 16.90 | 0.579 | 0.0 | 24 | 1.420 | 43.67 | n/a | n/a | n/a |
| YOLO26n | onnx | decelera_landscape_min.mov | 11 | 97.99 | 7.78 | 2.00 | 0.873 | 0.0 | 2 | 1.000 | 11.00 | n/a | n/a | n/a |
| YOLO11n | onnx | solutions_ci_demo.mp4 | 62 | 136.27 | 6.68 | 18.29 | 0.535 | 0.0 | 24 | 1.312 | 47.25 | n/a | n/a | n/a |
| YOLO11n | onnx | decelera_landscape_min.mov | 11 | 126.65 | 6.42 | 2.00 | 0.851 | 0.0 | 2 | 1.000 | 11.00 | n/a | n/a | n/a |
| RT-DETR-l | onnx | — | — | no ejecutado | — | — | — | — | — | — | — | — | — | — |

\* `tracks únicos / jugadores promedio por frame`. 1.0 = una identidad por persona; valores
mayores indican cambios de ID o tracks fragmentados.

Verificación de detección (imágenes de referencia de Ultralytics, umbral 0.25):
YOLO26n y YOLO11n detectan 4 personas en `bus.jpg` y 2 en `zidane.jpg`.

### Lectura de la ronda 1

- En esta CPU, YOLO26n fue ~25 % más rápido que YOLO11n (≈100 ms vs ≈130 ms por frame) y tuvo
  confianza media algo mayor. YOLO11n detectó algo más de personas por frame en la escena densa.
- El balón no aparece en estos clips, así que `ball detection = 0.0` no evalúa nada todavía.
- RT-DETR-l no tiene exportación ONNX oficial; requiere `MODEL_RUNTIME=ultralytics` (PyTorch),
  que no pudo instalarse en el entorno de prueba (los wheels CUDA de PyTorch superan el disco
  disponible). **Pendiente**.
- Precision / Recall / mAP requieren un dataset de fútbol etiquetado. **Pendiente**.

Configuración inicial confirmada: **YOLO26n + ByteTrack** (se cambia con `MODEL_NAME=YOLO11n`).

## Ronda 2 — pendiente (videos de fútbol reales)

1. Conseguir 3–5 clips de transmisión (15–30 s, cámara principal) con derechos de uso académico.
2. `python evaluate_models.py --videos clip1.mp4 clip2.mp4 clip3.mp4 --models YOLO26n YOLO11n`
3. Para precision/recall/mAP: etiquetar ~100 frames (clases `person`, `sports ball`) en formato
   YOLO, crear `soccer.yaml` y ejecutar con `--dataset soccer.yaml` (requiere `requirements-eval.txt`).
4. Copiar aquí la tabla Markdown que imprime el script.
