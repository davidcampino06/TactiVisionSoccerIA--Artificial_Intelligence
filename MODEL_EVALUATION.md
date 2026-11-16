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

## Unión de trayectorias cortadas (`track_stitching.py`)

ByteTrack le da un ID nuevo a un jugador que estuvo tapado o fuera de cuadro más de lo que su
memoria aguanta. El módulo sugiere qué pares de IDs son el mismo jugador; aquí se midió cuántas de
esas sugerencias son correctas, mirando cada par con `tools/review_track_links.py` (el jugador en el
último cuadro del ID que termina y en el primero del ID que empieza, recortados del video).

Clips de transmisión, YOLO26n ONNX en CPU, `frame_stride=3`, `max_frames=300`:

| Clip | IDs (tracks) | Enlaces sugeridos | Correctos a la vista | Dudosos | Errores claros | IDs si se aceptan todos |
|---|---|---|---|---|---|---|
| Ataque.mp4 (se usó para ajustar) | 98 | 22 | 20 | 2 | 0 | 76 |
| contra_ataque_y_mala_defensa.mp4 (no usado para ajustar) | 56 | 14 | 12 | 2 | 0 | 42 |

"Dudoso" = el recorte no deja ver con certeza si es el mismo jugador (el jugador está tapado por
otro en el último cuadro, o la cámara cambió de plano).

### Lectura

- Con el primer radio (0.75 + 2.0 alturas por segundo de hueco) Ataque.mp4 daba 29 enlaces y 4 eran
  errores claros (por ejemplo el n.º 5 enlazado con el n.º 33 del mismo equipo, 1.5 alturas de
  distancia). Todos los enlaces correctos quedaban a menos de 0.85 alturas del punto previsto y los
  errores por encima de 1.0, así que el radio quedó en 0.6 + 0.4 alturas por segundo.
- Ese ajuste también quitó 3 enlaces correctos (2 del arquero, que corre en diagonal y cambia de
  dirección): se prefiere perder un enlace a mezclar a dos jugadores.
- El color de la camiseta no distingue a dos compañeros del mismo equipo; por eso el dorsal lo
  decide una persona, y los enlaces son sugerencias que se aceptan o rechazan, nunca se unen solos.
- Es una revisión a ojo de 36 pares en 2 clips, no un dataset etiquetado: sirve para saber que el
  orden de magnitud es razonable, no para dar un porcentaje de precisión con decimales.

## Sugerencias de árbitro y arquero (`role_suggestions.py`)

Revisión a ojo con `tools/review_roles.py` (foto de cada trayectoria agrupada por camiseta), mismos
ajustes que arriba:

| Clip | Trayectorias sugeridas | Grupos de camiseta | Bien agrupadas y son árbitro o arquero | Otra persona |
|---|---|---|---|---|
| Ataque.mp4 | 11 | 3 | 10 (6 del arquero naranja, 4 del árbitro) | 1 (alguien del banquillo junto a la línea) |
| contra_ataque_y_mala_defensa.mp4 | 5 | 2 | 5 (4 del árbitro, 1 del arquero) | 0 |
| tiro_de_esquina_a_gol.mp4 | 2 | 1 | 2 (el árbitro) | 0 |

### Lectura

- La utilidad real es el grupo: el arquero de Ataque.mp4 quedó partido en 6 IDs y el árbitro en 4;
  al confirmar uno, la persona ya sabe quiénes son los demás de la misma camiseta.
- En los tres clips todas salieron como "arquero o árbitro": nunca hubo dos personas con la misma
  camiseta a la vez (el juez de línea no sale en plano). La sugerencia "cuerpo arbitral" solo aparece
  con dos a la vez durante 3 cuadros analizados o más; está probada con video sintético, no con uno
  real.
- Primer intento sin filtros: el arquero salía como "cuerpo arbitral" porque ByteTrack abrió una
  segunda caja sobre él durante 3 cuadros, y un jugador blanco con una caja pequeña dio una
  "camiseta" verde oscuro (era el césped). Por eso dos cajas a menos de una altura cuentan como una
  persona y un color de camiseta verde césped no se tiene en cuenta.
- Una persona fuera del juego (banquillo) también tiene una camiseta que no es de ningún equipo; la
  sugerencia no la distingue: por eso lo confirma una persona (rol "Otro" en la app).
