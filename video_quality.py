"""Clip check before the analysis: is this video fit for the AI, and how much of it will be analysed?

Every check says whether it is MEASURED (read from the file or counted by the model) or ESTIMATED
(an image heuristic), so the app never presents a guess as a fact. Thresholds come from the 13
labelled calibration clips (CALIBRATION.md) and from degraded copies of them:

    duration / coverage   measured   the analysis takes frames from the start until its frame cap:
                                     with a longer clip the end is never analysed
    resolution, fps       measured   from the file
    pitch visible         measured   share of sampled frames with enough grass (same mask as the analysis)
    players on the pitch  measured   YOLO on 8 frames; team indicators need 4 per team, lines 6
    ball                  measured   YOLO on the same 8 frames (it is small: seeing it in few is normal)
    camera cuts           estimated  abrupt image changes between frames 0.2 s apart (replays, close-ups,
                                     TV graphics): tracking restarts there
    camera distance       estimated  median height of the players in the image
    lighting              measured   mean brightness; only extreme values are flagged

What it does not check: sharpness. In the tests YOLO kept finding the same players even with a
strong blur, darker images and 426 px, so a sharpness score would warn about clips that work.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from detectors import OBJECT_TYPE_BALL, OBJECT_TYPE_PLAYER, ObjectDetector
from pipeline import MIN_PITCH_GREEN_RATIO, _green_mask, _is_on_pitch

SAMPLES = 8                  # frames given to YOLO
CUT_SAMPLING_HZ = 5          # frames per second compared to find cuts
CUT_MIN_CORRELATION = 0.85   # colour histograms of two frames 0.2 s apart; a pan stays above 0.97
CUT_MAX_DIFFERENCE = 25.0    # mean grey-level change (64x36); pans stayed below 18 in the clips
PLAYERS_FOR_TEAMS = 8        # 4 per team: below this the team indicators cannot be computed
PLAYERS_FOR_LINES = 12       # 6 per team for the line indicators
MIN_WIDTH = 480
MIN_FPS = 20
FAR_HEIGHT, CLOSE_HEIGHT = 0.03, 0.25   # player height / image height
DARK, BRIGHT = 40.0, 220.0              # mean grey level 0..255 (the clips were 95-111)


@dataclass(frozen=True)
class Mode:
    stride: int
    max_frames: int

    def seconds(self, fps: float) -> float:
        return self.stride * self.max_frames / fps


def _check(code: str, status: str, measured: bool, title: str, detail: str, value: object = None) -> dict:
    return {"code": code, "status": status, "measured": measured, "title": title, "detail": detail, "value": value}


def _fmt(seconds: float) -> str:
    return f"{seconds:.0f} s" if seconds >= 10 else f"{seconds:.1f} s".replace(".", ",")


def _histogram(frame: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(cv2.resize(frame, (160, 90)), cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1], None, [30, 32], [0, 180, 0, 256])
    return cv2.normalize(hist, hist).flatten()


def coverage_check(duration: float | None, fps: float, fast: Mode, full: Mode) -> dict:
    if not duration:
        return _check("coverage", "warn", True, "Duración desconocida",
                      "El archivo no informa su duración: no se puede saber cuánto se analizará.")
    fast_s, full_s = fast.seconds(fps), full.seconds(fps)
    value = {"duration": round(duration, 2), "fast_seconds": round(fast_s, 2), "full_seconds": round(full_s, 2)}
    if duration <= fast_s + 0.5:
        return _check("coverage", "ok", True, f"Dura {_fmt(duration)}: se analiza completo",
                      "Cabe entero en el modo rápido y en el completo.", value)
    if duration <= full_s + 0.5:
        return _check("coverage", "warn", True, f"Dura {_fmt(duration)}: usa el modo completo",
                      f"El modo rápido solo analiza los primeros {_fmt(fast_s)}; el completo lo cubre entero.", value)
    return _check("coverage", "bad", True, f"Dura {_fmt(duration)}: se analizaría solo el comienzo",
                  f"El análisis toma los cuadros desde el inicio: el modo rápido llega a {_fmt(fast_s)} y el "
                  f"completo a {_fmt(full_s)}. Recorta el clip a la jugada que quieres estudiar.", value)


def assess(video_path: str, detector: ObjectDetector, fast: Mode, full: Mode,
           confidence: float = 0.25, ball_confidence: float = 0.10) -> dict:
    capture = cv2.VideoCapture(str(video_path))
    try:
        if not capture.isOpened():
            raise ValueError("Could not open the video.")
        raw_fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
        fps = raw_fps if 1.0 <= raw_fps <= 240.0 else 25.0
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        if width <= 0 or height <= 0:
            raise ValueError("Video has invalid dimensions.")
        duration = frame_count / fps if frame_count else None
        # Only the part the longest mode would analyse matters (and keeps the check fast).
        window = min(duration or full.seconds(fps), full.seconds(fps))
        step = max(1, round(fps / CUT_SAMPLING_HZ))
        last_frame = int(window * fps)
        frames: list[tuple[float, np.ndarray]] = []
        index = 0
        while index <= last_frame and capture.grab():
            if index % step == 0:
                ok, frame = capture.retrieve()
                if ok and frame is not None:
                    frames.append((index / fps, frame))
            index += 1
    finally:
        capture.release()
    if not frames:
        raise ValueError("No frames could be decoded from the video.")

    checks = [coverage_check(duration, fps, fast, full)]
    resolution = {"width": width, "height": height}
    checks.append(_check("resolution", "warn" if width < MIN_WIDTH else "ok", True,
                         f"Resolución {width} × {height}",
                         "Muy baja: los jugadores lejanos y el balón se pierden." if width < MIN_WIDTH
                         else "Suficiente para la IA (trabaja la imagen a 640 px).", resolution))
    checks.append(_check("fps", "warn" if fps < MIN_FPS else "ok", True, f"{fps:.0f} cuadros por segundo",
                         "Pocos cuadros por segundo: el seguimiento pierde jugadores entre un cuadro y otro."
                         if fps < MIN_FPS else "Suficientes para seguir a los jugadores.", round(fps, 2)))

    # Cuts: compare each sampled frame with the previous one.
    cuts: list[float] = []
    for (_, previous), (time_b, current) in zip(frames, frames[1:]):
        grey_a = cv2.cvtColor(cv2.resize(previous, (64, 36)), cv2.COLOR_BGR2GRAY).astype(np.float32)
        grey_b = cv2.cvtColor(cv2.resize(current, (64, 36)), cv2.COLOR_BGR2GRAY).astype(np.float32)
        correlation = cv2.compareHist(_histogram(previous), _histogram(current), cv2.HISTCMP_CORREL)
        if correlation < CUT_MIN_CORRELATION or float(np.abs(grey_a - grey_b).mean()) > CUT_MAX_DIFFERENCE:
            if not cuts or time_b - cuts[-1] > 1.0:  # one cut, not every frame of a transition
                cuts.append(round(time_b, 1))
    if cuts:
        moments = ", ".join(_fmt(t) for t in cuts[:5])
        checks.append(_check("cuts", "warn", False, f"{len(cuts)} cambio{'s' if len(cuts) > 1 else ''} brusco{'s' if len(cuts) > 1 else ''} de imagen",
                             f"Hacia {moments}: parece una repetición, un primer plano o un gráfico de TV. Ahí el "
                             "seguimiento se reinicia y esos cuadros mezclan otra toma; recorta el clip a una sola toma.",
                             cuts))
    else:
        checks.append(_check("cuts", "ok", False, "Una sola toma", "No se ven cambios bruscos de plano.", []))

    # Model on evenly spaced frames.
    sample = [frames[i][1] for i in np.linspace(0, len(frames) - 1, min(SAMPLES, len(frames))).astype(int)]
    with_pitch, players, heights, balls, brightness = 0, [], [], 0, []
    for frame in sample:
        brightness.append(float(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).mean()))
        green = _green_mask(frame)
        pitch = float((green > 0).mean()) >= MIN_PITCH_GREEN_RATIO
        with_pitch += pitch
        detections = detector.detect(frame, min(confidence, ball_confidence))
        people = [d for d in detections if d.object_type == OBJECT_TYPE_PLAYER and d.confidence >= confidence
                  and (not pitch or _is_on_pitch(d, green))]
        players.append(len(people))
        heights += [d.height / height for d in people]
        balls += any(d.object_type == OBJECT_TYPE_BALL and d.confidence >= ball_confidence for d in detections)
    n = len(sample)

    share = with_pitch / n
    checks.append(_check("pitch", "ok" if share >= 0.75 else "warn" if share >= 0.5 else "bad", True,
                         f"Cancha visible en {with_pitch} de {n} cuadros",
                         "Se reconoce el césped." if share >= 0.75 else
                         "En varios cuadros no se ve la cancha: esos momentos no cuentan para los indicadores."
                         if share >= 0.5 else "Casi no se ve una cancha de fútbol: la IA no generará problemas ni fortalezas.",
                         round(share, 2)))

    median_players = float(np.median(players)) if players else 0.0
    label = f"{median_players:.0f} jugadores en la cancha por cuadro (mediana)"
    if median_players >= PLAYERS_FOR_LINES:
        checks.append(_check("players", "ok", True, label, "Suficientes para todos los indicadores.", median_players))
    elif median_players >= PLAYERS_FOR_TEAMS:
        checks.append(_check("players", "warn", True, label,
                             "Alcanzan para los indicadores de equipo, pero los de líneas necesitan 6 por equipo y "
                             "pueden quedar sin medir.", median_players))
    else:
        checks.append(_check("players", "bad", True, label,
                             "Muy pocos: los indicadores de equipo necesitan al menos 4 jugadores por equipo.",
                             median_players))

    checks.append(_check("ball", "ok" if balls else "warn", True, f"Balón visto en {balls} de {n} cuadros",
                         "Es pequeño y es normal verlo en pocos cuadros." if balls else
                         "No se vio en la muestra: la presión sobre el balón y la fase de juego pueden no juzgarse.",
                         balls))

    if heights:
        median_height = float(np.median(heights))
        far, close = median_height < FAR_HEIGHT, median_height > CLOSE_HEIGHT
        checks.append(_check("distance", "warn" if far or close else "ok", False,
                             "Cámara muy lejana" if far else "Cámara muy cerrada" if close else "Distancia de cámara adecuada",
                             "Los jugadores se ven muy pequeños: la IA puede no detectarlos." if far else
                             "Se ven pocos jugadores muy grandes (primer plano): no hay vista táctica." if close else
                             f"Los jugadores miden cerca del {median_height * 100:.0f} % de la altura de la imagen.",
                             round(median_height, 3)))

    light = float(np.median(brightness))
    checks.append(_check("light", "warn" if light < DARK or light > BRIGHT else "ok", True,
                         "Imagen muy oscura" if light < DARK else "Imagen muy clara" if light > BRIGHT else "Iluminación adecuada",
                         "Puede costar separar a los equipos por el color de la camiseta." if light < DARK or light > BRIGHT
                         else "Brillo normal para separar los equipos por color.", round(light, 1)))

    worst = "bad" if any(c["status"] == "bad" for c in checks) else "warn" if any(c["status"] == "warn" for c in checks) else "ok"
    return {"verdict": worst, "video": {"fps": round(fps, 3), "frame_count": frame_count, "width": width,
                                        "height": height, "duration_seconds": round(duration, 2) if duration else None},
            "sampled_frames": n, "checks": checks}
