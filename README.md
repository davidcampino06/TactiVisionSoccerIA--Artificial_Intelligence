# TactivisionSoccerIA-ai

AI Service de **TactiVision IA**. Analiza videos de fútbol y devuelve datos estructurados
(detecciones, tracking, indicadores tácticos, posibles problemas tácticos con evidencia y
confianza, y recomendaciones).

> Solo el **Backend** llama a este servicio. El Frontend **nunca** se comunica con él.
> Si `AI_SERVICE_API_KEY` está definido, cada análisis exige el header `X-API-Key`.

## Pipeline

```
Video → extracción de frames (cada FRAME_STRIDE) → YOLO (person / sports ball)
      → filtro de cancha (césped bajo los pies) → ByteTrack (Track ID)
      → equipos por color de camiseta (k-means) → posiciones normalizadas
      → indicadores tácticos (Strategy) → reglas de posibles problemas
      → evidencia + confianza → recomendaciones
```

Cada capa está separada y es explicable:

| Capa | Archivo | Responsabilidad |
|---|---|---|
| Visual Detection | `detectors.py` | YOLO26n / YOLO11n (ONNX) o Ultralytics (.pt). Factory Method |
| Tracking | `tracking.py` | ByteTrack → `track_id` temporal (sin biometría) |
| Equipos | `team_assignment.py` | Clustering de color de camiseta → TEAM_A / TEAM_B / UNASSIGNED |
| Tactical Metrics | `tactical_analysis.py` | Indicadores (patrón Strategy) |
| Issues + Recommendations | `tactical_insights.py` | Reglas con umbral, persistencia, evidencia y confianza |
| Facade | `pipeline.py` | `VideoAnalysisPipeline.analyze_video()` orquesta todo |
| SIMULATION MODE | `simulation.py` | Posiciones sintéticas, siempre etiquetadas `SIMULATION_MODE` |
| Evaluación | `evaluate_models.py` | Comparación de modelos con métricas medidas |

## Endpoints

| Método | Ruta | Descripción |
|---|---|---|
| GET | `/api/health` | `{"service": "TactiVision AI", "status": "OK"}` |
| GET | `/api/model` | Modelo configurado, runtime, tracker, versión, si cargó y errores |
| POST | `/api/analyze` | `multipart/form-data`: `video` + parámetros opcionales (ver abajo) |
| POST | `/api/analyze/simulation` | SIMULATION MODE (sin video). Parámetros `frames`, `seed` |

Parámetros de `/api/analyze`: `confidence_threshold` (0.25), `frame_stride` (3), `max_frames` (300),
`team_a_attack_direction` (`unknown` | `left_to_right` | `right_to_left`), `pitch_filter` (true),
`include_detections` (true).

Respuesta (resumen):

```json
{
  "status": "COMPLETED",
  "mode": "REAL_VIDEO_ANALYSIS",
  "model": "YOLO26n",
  "model_version": "ultralytics-8.4.38",
  "frames_processed": 100,
  "players_detected": 18,
  "ball_detected": true,
  "detections": [...],
  "tracking": [{"track_id": 3, "label": "Player Track 3", "team": "TEAM_A", "trajectory": [[frame, x, y]]}],
  "tactical_indicators": [{"name": "team_a.team_width", "value": 0.41, "unit": "frame_height_ratio", "threshold": 0.3}],
  "possible_issues": [{"issue": "Possible tactical issue: ...", "evidence": "...", "confidence": 0.62, "related_indicators": [...]}],
  "recommendations": [...],
  "warnings": [...]
}
```

## Ejecutar localmente

```bash
cd TactivisionSoccerIA-ai
python -m venv .venv
# Windows: .venv\Scripts\activate    Linux/Mac: source .venv/bin/activate
pip install -r requirements-dev.txt
copy .env.example .env               # Linux/Mac: cp .env.example .env
uvicorn main:app --reload --port 8001
```

- Health: http://localhost:8001/api/health
- Modelo: http://localhost:8001/api/model
- Swagger: http://localhost:8001/docs

La primera vez descarga `yolo26n.onnx` (~10 MB) desde los releases oficiales de Ultralytics
a la carpeta `models/` (ignorada por git).

Tests: `python -m pytest -q`
(opcional: `PEOPLE_IMAGE_PATH=ruta/a/foto_con_personas.jpg` para el test de detección real).

## ¿Por qué ONNX y no PyTorch por defecto?

Los pesos `.pt` requieren PyTorch (+2 GB instalado). Render Free tiene 512 MB de RAM.
Ultralytics publica exportaciones ONNX oficiales de YOLO26n y YOLO11n, que se ejecutan con
`onnxruntime` (~20 MB). Verificado en este proyecto con las imágenes de ejemplo de
Ultralytics (`bus.jpg` → 4 personas, `zidane.jpg` → 2 personas). No se comparó todavía
contra la ejecución `.pt` en PyTorch (ver `MODEL_EVALUATION.md`). Para RT-DETR o para calcular mAP con un dataset etiquetado se usa
`MODEL_RUNTIME=ultralytics` + `requirements-eval.txt`.

## Limitaciones conocidas (honestas)

- Modelos COCO pre-entrenados: detectan `person` (jugadores, árbitros, suplentes) y `sports ball`.
  El balón es pequeño y su detección en planos abiertos es baja; se reporta `ball_detection_rate`.
- No hay calibración de cancha (homografía): las métricas son relativas a la imagen y asumen la
  cámara principal de transmisión (largo de cancha = eje X).
- Equipos por color: árbitros/porteros quedan `UNASSIGNED`; kits parecidos bajan `separation_score`.
- `defensive_line_position` depende del encuadre: con cámara que se mueve es solo orientativo
  (por eso su regla tiene `reliability=0.6`).
- Si el video no muestra césped en la mayoría de frames, **no se generan problemas tácticos**.
- Los umbrales en `DEFAULT_THRESHOLDS` son heurísticos: deben calibrarse con videos reales.

## Render (más adelante)

Web Service · Build: `pip install -r requirements.txt` ·
Start: `uvicorn main:app --host 0.0.0.0 --port $PORT` · Variables: las de `.env.example`.
