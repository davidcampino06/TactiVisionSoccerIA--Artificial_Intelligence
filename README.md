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

## Color de tu equipo (opcional)

`POST /api/analyze` acepta `own_kit_color` (`#rrggbb`). Los dos equipos se separan **siempre** por el
color de las camisetas del video; el color indicado solo dice cuál de los dos es el del usuario
(`teams.own_team`). Como el color medido es un promedio con sombras (más apagado que el uniforme
real), no se exige una distancia absoluta: el uniforme tiene que estar **claramente** más cerca de un
equipo que del otro (distancia ≤ 0,65 de la del otro, `OWN_TEAM_MAX_RATIO`). Con uniformes reales en
los clips de calibración el equipo correcto siempre quedó por debajo de 0,57; un uniforme verde
contra un equipo rojo y uno azul da 0,73 y no se fuerza: el análisis lo avisa y no marca ninguno.

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

**La forma se juzga en su fase.** Como el balón no se ve en todos los cuadros, la posesión se
recuerda `POSSESSION_MEMORY_SECONDS` (2 s) después de verlo. Los indicadores de forma defensiva
(líneas, compactación, distancia al compañero, aislados, grupos, zona más ocupada) se miden solo
mientras el rival tiene el balón, y el ancho del equipo solo mientras no defiende
(`INDICATOR_PHASE`). Un equipo que casi no defendió no se evalúa, y el análisis lo avisa. Con los
clips reales esto bajó las falsas alarmas de 13 a 3 ([CALIBRATION.md](CALIBRATION.md)); las pruebas
están en `tests/test_phases.py`.

## Cada alerta explicada y mostrada en la cancha

Cada recomendación trae `explanation` con tres textos en lenguaje de fútbol y una jugada real:

| Campo | Qué es |
|---|---|
| `seen` | Qué vio la IA, con los números traducidos: "Mientras el rival tenía el balón (unos 2 segundos de juego), en el Equipo A la distancia de cada jugador a su compañero más cercano fue en promedio el 17,7 % del ancho de la imagen…" |
| `meaning` | Qué significa en la cancha |
| `coach` | Qué podría trabajar el cuerpo técnico |
| `example` | Un cuadro real donde pasó: jugadores, equipos, balón, quién tenía el balón y un dibujo de la regla (`overlay`) |

El cuadro de ejemplo (`issue_examples.py`) se elige entre los cuadros de la fase correcta donde el
umbral se cruzó de verdad, con casi todo el equipo visible, y es el de valor **más cercano al
promedio** reportado (representativo, no el más exagerado). El dibujo depende de la regla: compañero
más cercano, área del bloque, grupos de la red, hueco entre líneas, zona más ocupada, ancho de banda
a banda, radio de presión alrededor del balón o jugadores por delante del balón. Las posiciones son
las de la imagen (no hay calibración de cancha): la app lo dice al mostrarla.

## Lo que el equipo hizo bien (`strengths`)

La respuesta trae también `strengths`: lo que cada equipo hizo bien, con el mismo formato que las
alertas (evidencia, confianza, indicador) y una explicación en lenguaje de fútbol:

| Campo | Qué es |
|---|---|
| `seen` | Qué vio la IA, con los números: "Mientras el rival tenía el balón (unos 4 segundos de juego), en el Equipo A el espacio entre la línea de defensas y la de mediocampistas fue menor al 15 % del ancho de la imagen en el 80 % de esos momentos…" |
| `meaning` | Qué significa en la cancha |
| `keep` | Cómo mantenerlo |
| `example` | Un cuadro real del lado bueno, el más cercano al promedio, con el mismo dibujo que las alertas |

Hoy son cuatro: **líneas juntas al defender**, **siempre con un compañero cerca**, **presión sobre
el balón** y **profundidad en ataque**. Se muestran solo si el dato es claro y sostenido (un límite
más estricto que el del problema, en la mayoría de los momentos) y nunca junto al problema
contrario del mismo equipo. Cómo se eligieron y por qué no hay "bloque compacto":
[CALIBRATION.md](CALIBRATION.md), etapa 6.

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
- Los umbrales en `DEFAULT_THRESHOLDS` se calibraron con 7 clips reales de TV (ver
  [CALIBRATION.md](CALIBRATION.md)); es una calibración orientativa, no una precisión garantizada.

## Render

Web Service · Build: `pip install -r requirements.txt` ·
Start: `uvicorn main:app --host 0.0.0.0 --port $PORT` · Variables: las de `.env.example`.
Python: `.python-version` (3.13); la variable `PYTHON_VERSION` del servicio, si existe, tiene prioridad.
