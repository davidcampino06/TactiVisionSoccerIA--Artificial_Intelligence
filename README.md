# TactivisionSoccerIA-ai

[![CI](https://github.com/davidcampino06/TactiVisionSoccerIA--Artificial_Intelligence/actions/workflows/ci.yml/badge.svg)](https://github.com/davidcampino06/TactiVisionSoccerIA--Artificial_Intelligence/actions/workflows/ci.yml)

Integración continua (GitHub Actions, `.github/workflows/ci.yml`): en cada push a `main` y en cada
pull request se ejecutan pruebas del servicio de IA, incluido el modelo de detección.

AI Service de **TactiVision IA**. Analiza videos de fútbol y devuelve datos estructurados
(detecciones, tracking, indicadores tácticos, posibles problemas tácticos con evidencia y
confianza, y recomendaciones).

> Solo el **Backend** llama a este servicio. El Frontend **nunca** se comunica con él.
> Si `AI_SERVICE_API_KEY` está definido, cada análisis exige el header `X-API-Key`.

## Seguridad del servicio

Solo el backend llama a este servicio. Las protecciones están en `main.py` y `request_guard.py`:

| Protección | Cómo funciona |
|---|---|
| Clave compartida | Cada análisis exige `X-API-Key`; se compara con `secrets.compare_digest` (tiempo constante, no revela cuántos caracteres coinciden). |
| Límite de peticiones | Ventana deslizante por cliente (cola FIFO con las horas de sus últimas peticiones): más de `RATE_LIMIT_PER_MINUTE` en 60 s → 429 con `Retry-After`. Va antes de la clave, así que adivinarla también queda frenado. |
| Tipo real del archivo | Además de la extensión se revisan los primeros bytes: MP4/MOV (caja ISO-BMFF), AVI (`RIFF…AVI `), MKV/WebM (firma EBML). Un `.mp4` que en realidad es texto o una imagen se rechaza con 400 antes de guardarse o llegar a OpenCV. |
| Tamaño máximo | `MAX_UPLOAD_SIZE_MB`; se corta mientras se recibe, sin cargar el archivo entero en memoria. |
| Cabeceras HTTP | `X-Content-Type-Options: nosniff`, `X-Frame-Options: DENY`, `Referrer-Policy: no-referrer`, `Cache-Control: no-store`. |

Logs (`log_setup.py`): cada análisis registra su inicio y fin (tamaño, cuadros, segundos, posibles
problemas), y quedan avisos de las peticiones rechazadas (clave inválida, archivo falso, límite). Un
filtro oculta claves, tokens y contraseñas antes de escribir cualquier línea.

Pruebas: `tests/test_security.py` y `tests/test_logging.py`.

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
| Red de cercanía | `proximity_graph.py` | Grafo de apoyos entre compañeros recorrido con BFS (ver abajo) |
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
  "proximity_network": {"frame_number": 42, "link_distance": 0.2, "teams": {"TEAM_A": {"nodes": [...], "links": [[3, 7]], "groups": 2, "isolated": [9], "diameter": 3, "hops": {"3": {"7": 1}}}}},
  "possible_issues": [{"issue": "Possible tactical issue: ...", "evidence": "...", "confidence": 0.62, "related_indicators": [...]}],
  "recommendations": [...],
  "warnings": [...]
}
```

## Fases con y sin balón (`PossessionPhaseStrategy`)

En cada cuadro con el balón visible, lo tiene el equipo del jugador más cercano si está a menos de
`BALL_CONTROL_DISTANCE` (0,06 del ancho); si nadie está tan cerca, el balón está suelto y el cuadro
no cuenta. Con eso se miden tres indicadores nuevos:

| Indicador | Equipo | Qué mide | Posible problema |
|---|---|---|---|
| `pressure_on_ball` | Sin balón | Jugadores a menos de `PRESSURE_RADIUS` (0,1) del balón | `LOW_BALL_PRESSURE`: nadie presiona |
| `attacking_width` | Con balón | Ancho que ocupa el equipo en la imagen | `NARROW_ATTACK`: menos del 35 % |
| `players_ahead_of_ball` | Con balón | Jugadores por delante del balón hacia el arco rival (necesita la dirección de ataque) | `NO_ATTACKING_DEPTH`: ninguno por delante |

Cada indicador tiene su prueba con posiciones hechas a mano en `tests/test_indicators.py`, igual que
los anteriores (forma, compactación, líneas, zonas, balón y red de cercanía).

## Dirección de ataque automática

Si el usuario no indica hacia dónde ataca el Equipo A, `detect_attack_direction` la estima: cada
equipo se coloca entre el balón y su propio arco, así que en promedio el que defiende el arco
izquierdo está más a la izquierda que su rival. Cada cuadro con ambos equipos visibles y separados
(más de 0,03 del ancho entre sus posiciones medias) vota; se acepta con al menos 8 votos y un 70 %
de acuerdo, y si no, queda desconocida. El resultado lo explica en `warnings` y en `parameters`
(`attack_direction_source`: `user`, `detected` o `unknown`). La dirección del usuario siempre manda.

## Red de cercanía: grafo + BFS

Cada jugador visible de un equipo es un **nodo**. Dos compañeros quedan unidos por una **arista**
cuando están a menos de `LINK_DISTANCE` (0,2 del ancho de la imagen): cerca para un pase corto o una
cobertura. Sobre ese grafo, la búsqueda en anchura (BFS) responde preguntas tácticas:

| Indicador | Qué mide | Cómo se calcula |
|---|---|---|
| `network_components` | En cuántos grupos sin conexión quedó el equipo (1 = todo conectado) | Componentes conexas: un BFS desde cada nodo aún no visitado |
| `isolated_players` | Jugadores sin ningún compañero cerca | Nodos de grado 0 |
| `network_average_links` | Compañeros cercanos por jugador | Grado medio = 2·aristas / nodos |
| `network_diameter` | La ruta más larga, en pases cortos, entre dos compañeros conectados | BFS desde cada nodo, máximo de las distancias |

Complejidad con n jugadores y m aristas: construir el grafo O(n²), un BFS O(n + m), todas las
componentes O(n + m), todas las distancias O(n·(n + m)). La cola del BFS es propia (arreglo + índice
del frente, encolar y desencolar en O(1)).

Las reglas `DISCONNECTED_TEAM` y `ISOLATED_PLAYERS` generan posibles problemas cuando el equipo se
parte en 2 o más grupos, o deja 2 o más jugadores sin apoyo, en la mayoría de los cuadros. La
respuesta incluye `proximity_network`: la red del cuadro con más jugadores, lista para dibujar, con
las distancias BFS entre cada par de jugadores conectados.

## Ejecutar localmente

Versión de Python: **3.13**, fijada en `.python-version`. Render la usa al desplegar (si el servicio no
tiene la variable `PYTHON_VERSION`, que tendría prioridad) y es la misma de GitHub Actions.

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

## Render

Web Service · Build: `pip install -r requirements.txt` ·
Start: `uvicorn main:app --host 0.0.0.0 --port $PORT` · Variables: las de `.env.example`.
Python: `.python-version` (3.13); la variable `PYTHON_VERSION` del servicio, si existe, tiene prioridad.
