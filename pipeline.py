"""Video analysis pipeline.

Design pattern - Facade: ``VideoAnalysisPipeline.analyze_video`` hides the
whole process behind one call:

    Video -> frame extraction -> YOLO detection -> ByteTrack tracking
          -> team assignment -> positions -> tactical indicators
          -> possible issues (evidence + confidence) -> recommendations

Each step is implemented by a separate collaborator (Single Responsibility).
"""

from __future__ import annotations

import logging
import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

from config import settings
from detectors import OBJECT_TYPE_BALL, OBJECT_TYPE_PLAYER, Detection, ObjectDetector
from tactical_analysis import (
    ATTACK_UNKNOWN,
    PHASE_DEFENDING,
    AnalysisContext,
    FrameSnapshot,
    PlayerPosition,
    TacticalAnalyzer,
    in_phase,
    network_snapshot,
    possession_by_frame,
    resolve_attack_direction,
)
from tactical_insights import MIN_SAMPLES, TacticalInsightGenerator
from team_assignment import TEAM_A, TEAM_B, UNASSIGNED, JerseyColorTeamClassifier, TeamAssignmentResult
from tracking import TrackedDetection, TrackerFactory

logger = logging.getLogger("tactivision.pipeline")

MODE_REAL = "REAL_VIDEO_ANALYSIS"
MODE_SIMULATION = "SIMULATION_MODE"

MIN_PITCH_GREEN_RATIO = 0.15
MAX_TRAJECTORY_POINTS = 120


# Overlap between the two halves, as a share of the frame width, so a ball on the centre line is
# fully inside at least one of them.
HALF_OVERLAP = 0.10


def find_ball_in_halves(detector: ObjectDetector, frame: np.ndarray, threshold: float) -> list[Detection]:
    """Looks for the ball in the left and right halves of the frame (with overlap).

    The model works on 640 px images: a 1920 px frame is shrunk 3x and a ball of 15 px becomes 5 px.
    Each half is shrunk only 1.5x. Returned boxes are in full-frame pixels."""
    width = frame.shape[1]
    half, overlap = width // 2, int(width * HALF_OVERLAP)
    found = []
    for start, end in ((0, min(half + overlap, width)), (max(half - overlap, 0), width)):
        for d in detector.detect(frame[:, start:end], threshold):
            if d.object_type == OBJECT_TYPE_BALL and d.confidence >= threshold:
                found.append(Detection(d.x1 + start, d.y1, d.x2 + start, d.y2, d.confidence, d.class_id, d.object_type))
    return found


def phase_warnings(context: AnalysisContext, team_colors: dict[str, str]) -> list[str]:
    """Says which team could not be judged on its defensive shape (too few frames defending)."""
    if not team_colors:
        return []
    owners = possession_by_frame(context)
    warnings = []
    for team, label in ((TEAM_A, "Equipo A"), (TEAM_B, "Equipo B")):
        defending = sum(1 for owner in owners if in_phase(PHASE_DEFENDING, owner, team))
        if defending < MIN_SAMPLES:
            warnings.append(
                f"El {label} defendió solo {defending} de {len(owners)} cuadros analizados (o no se vio quién "
                "tenía el balón): su forma defensiva (líneas, compactación, apoyos) no se evaluó."
            )
    return warnings


def untracked_id(record_index: int, index: int) -> int:
    """Negative id for a player seen without a track: unique inside the video, never a real track."""
    return -(record_index * 1000 + index + 1)

class VideoProcessingError(ValueError):
    """The video could not be read or produced no usable frames."""

@dataclass
class AnalysisOptions:
    """Parameters of one analysis request; defaults come from the service settings."""
    confidence_threshold: float = settings.confidence_threshold
    ball_confidence_threshold: float = settings.ball_confidence_threshold
    frame_stride: int = settings.default_frame_stride
    max_frames: int = settings.default_max_frames
    team_a_attack_direction: str = ATTACK_UNKNOWN
    pitch_filter: bool = True
    include_detections: bool = True
    require_pitch_for_insights: bool = True
    # Experiment, off: when the ball is not found in the whole frame, look again in the two halves,
    # where it is twice as big for the 640 px model. The ball was seen twice as often, but the
    # results agreed less with the labelled clips (CALIBRATION.md), so the app does not use it.
    precise_ball: bool = False

@dataclass
class _FrameRecord:
    frame_number: int
    timestamp_seconds: float
    tracked_players: list[TrackedDetection]
    ball: Detection | None
    # Players detected in this frame that the tracker did not return (a new track is only
    # confirmed when it is matched again in the next frame, and broadcast cameras pan a lot
    # between sampled frames). They still count for the tactical picture of this frame.
    untracked_players: list[Detection] = field(default_factory=list)

    def player_count(self) -> int:
        return len(self.tracked_players) + len(self.untracked_players)

def _green_mask(frame_bgr: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
    return cv2.inRange(hsv, (35, 40, 40), (85, 255, 255))

def _is_on_pitch(detection: Detection, green_mask: np.ndarray) -> bool:
    """A person is on the pitch if there is grass around the feet."""
    foot_x, foot_y = detection.foot_point
    half_width = max(int(detection.width * 0.6), 4)
    mask_height, mask_width = green_mask.shape
    y1 = int(min(max(foot_y - detection.height * 0.05, 0), mask_height - 1))
    y2 = int(min(foot_y + detection.height * 0.10 + 2, mask_height))
    x1 = int(min(max(foot_x - half_width, 0), mask_width - 1))
    x2 = int(min(foot_x + half_width, mask_width))
    region = green_mask[y1:y2, x1:x2]
    return region.size > 0 and float((region > 0).mean()) >= 0.25

def _ball_on_pitch(ball: Detection, green_mask: np.ndarray) -> bool:
    """The ball is on the pitch if there is grass around it (a box three times its size)."""
    size = max(ball.width, ball.height, 6.0)
    center_x, center_y = ball.center
    mask_height, mask_width = green_mask.shape
    x1, x2 = int(max(center_x - 1.5 * size, 0)), int(min(center_x + 1.5 * size, mask_width))
    y1, y2 = int(max(center_y - 1.5 * size, 0)), int(min(center_y + 1.5 * size, mask_height))
    region = green_mask[y1:y2, x1:x2]
    return region.size > 0 and float((region > 0).mean()) >= 0.25


class VideoAnalysisPipeline:
    """Facade over detector, tracker, team classifier, tactical analyzer and insights."""
    def __init__(
        self,
        detector: ObjectDetector,
        tactical_analyzer: TacticalAnalyzer | None = None,
        insight_generator: TacticalInsightGenerator | None = None,
        tracker_name: str = settings.tracker_name,
    ) -> None:
        self._detector = detector
        self._tactical_analyzer = tactical_analyzer or TacticalAnalyzer()
        self._insight_generator = insight_generator or TacticalInsightGenerator()
        self._tracker_name = tracker_name

    # ------------------------------------------------------------------ public
    def analyze_video(self, video_path: Path, options: AnalysisOptions) -> dict:
        """Facade of the whole use case: frames, detection, tracking, teams, indicators and insights."""
        started = time.perf_counter()
        capture = cv2.VideoCapture(str(video_path))
        if not capture.isOpened():
            raise VideoProcessingError("The file could not be opened as a video.")

        try:
            video_info = self._video_info(capture)
            records, extraction_stats = self._extract_and_track(capture, video_info, options)
        finally:
            capture.release()

        if not records:
            raise VideoProcessingError("No frames could be decoded from the video.")

        # Warnings are shown to the user, so they are written in Spanish.
        warnings = list(extraction_stats["warnings"])
        team_result = extraction_stats["team_classifier"].assign()
        warnings.extend(team_result.warnings)

        snapshots = self._build_snapshots(records, team_result, video_info)
        direction, direction_source, direction_agreement, direction_warning = resolve_attack_direction(
            options.team_a_attack_direction, snapshots)
        if direction_warning:
            warnings.append(direction_warning)
        context = AnalysisContext(
            snapshots=snapshots,
            aspect_ratio=video_info["height"] / video_info["width"],
            team_a_attack_direction=direction,
        )

        indicators = self._tactical_analyzer.analyze(context)
        warnings.extend(phase_warnings(context, team_result.team_colors_hex))
        possible_issues, recommendations = [], []
        if extraction_stats["pitch_visible"] or not options.require_pitch_for_insights:
            possible_issues, recommendations = self._insight_generator.generate(
                indicators, team_result.separation_score
            )
        else:
            warnings.append(
                "NO se generaron posibles problemas tácticos ni recomendaciones porque el video "
                "no parece mostrar una cancha de fútbol."
            )

        processing_seconds = time.perf_counter() - started
        frames_processed = len(records)
        player_counts = [record.player_count() for record in records]
        ball_frames = sum(1 for record in records if record.ball is not None)

        return {
            "status": "COMPLETED",
            "mode": MODE_REAL,
            "model": self._detector.model_name,
            "model_version": self._detector.model_version,
            "runtime": self._detector.runtime,
            "tracker": self._tracker_name,
            "video": video_info,
            "parameters": {
                "confidence_threshold": options.confidence_threshold,
                "ball_confidence_threshold": options.ball_confidence_threshold,
                "frame_stride": options.frame_stride,
                "max_frames": options.max_frames,
                "team_a_attack_direction": direction,
                "attack_direction_source": direction_source,
                "attack_direction_agreement": direction_agreement,
                "pitch_filter": options.pitch_filter,
                "precise_ball": options.precise_ball,
            },
            "pitch_visible": extraction_stats["pitch_visible"],
            "frames_processed": frames_processed,
            "players_detected": int(max(player_counts) if player_counts else 0),
            "average_players_per_frame": round(float(np.mean(player_counts)), 2) if player_counts else 0.0,
            "unique_player_tracks": len(team_result.track_teams) or self._unique_tracks(records),
            "ball_detected": ball_frames > 0,
            "ball_detection_rate": round(ball_frames / frames_processed, 4),
            "teams": {
                "colors": team_result.team_colors_hex,
                "separation_score": team_result.separation_score,
                "track_teams": {str(k): v for k, v in team_result.track_teams.items()},
            },
            "performance": {
                "processing_seconds": round(processing_seconds, 2),
                "mean_inference_ms": round(float(np.mean(extraction_stats["inference_ms"])), 2),
                "processed_fps": round(frames_processed / processing_seconds, 2) if processing_seconds else 0.0,
            },
            "detection_stability": self._stability(records),
            "detections": self._detections_payload(records, team_result) if options.include_detections else [],
            "tracking": self._tracking_payload(records, team_result.track_teams, video_info),
            "tactical_indicators": [indicator.to_dict() for indicator in indicators],
            "proximity_network": network_snapshot(context),
            "possible_issues": possible_issues,
            "recommendations": recommendations,
            "warnings": warnings,
        }

    # ----------------------------------------------------------------- helpers
    @staticmethod
    def _video_info(capture: cv2.VideoCapture) -> dict:
        fps = capture.get(cv2.CAP_PROP_FPS) or 0.0
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        if width <= 0 or height <= 0:
            raise VideoProcessingError("Video has invalid dimensions.")
        fps = fps if 1.0 <= fps <= 240.0 else 25.0
        return {
            "fps": round(fps, 3),
            "frame_count": frame_count,
            "width": width,
            "height": height,
            "duration_seconds": round(frame_count / fps, 2) if frame_count else None,
        }

    def _extract_and_track(self, capture: cv2.VideoCapture, video_info: dict, options: AnalysisOptions
                           ) -> tuple[list[_FrameRecord], dict]:
        stride = max(1, options.frame_stride)
        tracker = TrackerFactory.create(self._tracker_name, frame_rate=video_info["fps"] / stride)
        team_classifier = JerseyColorTeamClassifier()
        records: list[_FrameRecord] = []
        inference_ms: list[float] = []
        frames_without_pitch = 0
        filtered_out_people = 0

        frame_number = -1
        while len(records) < options.max_frames:
            if not capture.grab():
                break
            frame_number += 1
            if frame_number % stride:
                continue
            ok, frame = capture.retrieve()
            if not ok or frame is None:
                break

            detections = self._detector.detect(
                frame, min(options.confidence_threshold, options.ball_confidence_threshold)
            )
            inference_ms.append(self._detector.last_inference_ms)

            players = [
                d for d in detections
                if d.object_type == OBJECT_TYPE_PLAYER and d.confidence >= options.confidence_threshold
            ]
            balls = [
                d for d in detections
                if d.object_type == OBJECT_TYPE_BALL and d.confidence >= options.ball_confidence_threshold
            ]
            green = _green_mask(frame)
            pitch_visible = float((green > 0).mean()) >= MIN_PITCH_GREEN_RATIO
            if pitch_visible:
                # A "ball" with no grass around it is a TV graphic (a crest on the scoreboard) or
                # something in the stands, never the match ball.
                balls = [ball for ball in balls if _ball_on_pitch(ball, green)]
            if not balls and options.precise_ball:
                started_halves = time.perf_counter()
                balls = find_ball_in_halves(self._detector, frame, options.ball_confidence_threshold)
                if pitch_visible:
                    balls = [ball for ball in balls if _ball_on_pitch(ball, green)]
                inference_ms[-1] += (time.perf_counter() - started_halves) * 1000.0
            if not pitch_visible:
                frames_without_pitch += 1
            elif options.pitch_filter:
                on_pitch = [d for d in players if _is_on_pitch(d, green)]
                filtered_out_people += len(players) - len(on_pitch)
                players = on_pitch

            tracked_players = tracker.update(players)
            team_classifier.collect(frame, tracked_players)
            tracked_boxes = {id(tracked.detection) for tracked in tracked_players}
            untracked = [d for d in players if id(d) not in tracked_boxes]
            for index, detection in enumerate(untracked):
                team_classifier.collect_untracked(frame, (len(records), index), detection)
            best_ball = max(balls, key=lambda d: d.confidence) if balls else None
            records.append(
                _FrameRecord(frame_number, frame_number / video_info["fps"], tracked_players, best_ball, untracked)
            )

        warnings: list[str] = []
        if records and frames_without_pitch > len(records) / 2:
            warnings.append(
                f"{frames_without_pitch} de {len(records)} cuadros muestran poco césped; el video podría "
                "no ser de fútbol y los indicadores tácticos podrían no ser confiables."
            )
        if filtered_out_people:
            warnings.append(
                f"Se ignoraron {filtered_out_people} detecciones de personas fuera de la cancha (público, banca)."
            )
        return records, {
            "pitch_visible": bool(records) and frames_without_pitch <= len(records) / 2,
            "team_classifier": team_classifier,
            "inference_ms": inference_ms or [0.0],
            "warnings": warnings,
        }

    @staticmethod
    def _build_snapshots(records: list[_FrameRecord], team_result: TeamAssignmentResult,
                         video_info: dict) -> list[FrameSnapshot]:
        width, height = video_info["width"], video_info["height"]
        snapshots = []
        for record_index, record in enumerate(records):
            players = []
            for tracked in record.tracked_players:
                team = team_result.track_teams.get(tracked.track_id, UNASSIGNED)
                foot_x, foot_y = tracked.detection.foot_point
                players.append(
                    PlayerPosition(tracked.track_id, team, foot_x / width, foot_y / height, tracked.detection.confidence)
                )
            for index, detection in enumerate(record.untracked_players):
                team = team_result.detection_teams.get((record_index, index), UNASSIGNED)
                foot_x, foot_y = detection.foot_point
                players.append(PlayerPosition(untracked_id(record_index, index), team, foot_x / width,
                                              foot_y / height, detection.confidence))
            ball = None
            if record.ball is not None:
                ball_x, ball_y = record.ball.center
                ball = (ball_x / width, ball_y / height)
            snapshots.append(FrameSnapshot(record.frame_number, record.timestamp_seconds, players, ball))
        return snapshots

    @staticmethod
    def _unique_tracks(records: list[_FrameRecord]) -> int:
        return len({tracked.track_id for record in records for tracked in record.tracked_players})

    @staticmethod
    def _stability(records: list[_FrameRecord]) -> dict:
        track_frames: dict[int, int] = defaultdict(int)
        for record in records:
            for tracked in record.tracked_players:
                track_frames[tracked.track_id] += 1
        counts = [record.player_count() for record in records]
        lengths = list(track_frames.values())
        untracked = sum(len(record.untracked_players) for record in records)
        tracked_mean = float(np.mean([len(record.tracked_players) for record in records])) if records else 0.0
        return {
            "player_count_std": round(float(np.std(counts)), 3) if counts else 0.0,
            "mean_track_length_frames": round(float(np.mean(lengths)), 2) if lengths else 0.0,
            "tracks_longer_than_10_frames": sum(1 for length in lengths if length >= 10),
            # 1.0 means one track per visible player; higher values mean identity switches/fragments.
            "track_fragmentation_ratio": round(len(lengths) / tracked_mean, 3) if tracked_mean else 0.0,
            # Players used in the tactical picture without a track id (seen in one sampled frame only).
            "untracked_player_detections": untracked,
        }

    @staticmethod
    def _detections_payload(records: list[_FrameRecord], team_result: TeamAssignmentResult) -> list[dict]:
        payload = []
        for record_index, record in enumerate(records):
            items = [(tracked.detection, tracked.track_id,
                      team_result.track_teams.get(tracked.track_id, UNASSIGNED))
                     for tracked in record.tracked_players]
            items += [(detection, None, team_result.detection_teams.get((record_index, index), UNASSIGNED))
                      for index, detection in enumerate(record.untracked_players)]
            if record.ball is not None:
                items.append((record.ball, None, None))
            for detection, track_id, team in items:
                payload.append({
                    "frame_number": record.frame_number,
                    "timestamp": round(record.timestamp_seconds, 4),
                    "object_type": detection.object_type,
                    "confidence": round(detection.confidence, 4),
                    "bounding_box_x": round(detection.x1, 2),
                    "bounding_box_y": round(detection.y1, 2),
                    "bounding_box_width": round(detection.width, 2),
                    "bounding_box_height": round(detection.height, 2),
                    "track_id": track_id,
                    "team": team,
                })
        return payload

    @staticmethod
    def _tracking_payload(records: list[_FrameRecord], track_teams: dict[int, str], video_info: dict) -> list[dict]:
        width, height = video_info["width"], video_info["height"]
        trajectories: dict[int, list] = defaultdict(list)
        for record in records:
            for tracked in record.tracked_players:
                foot_x, foot_y = tracked.detection.foot_point
                trajectories[tracked.track_id].append(
                    [record.frame_number, round(foot_x / width, 4), round(foot_y / height, 4)]
                )

        payload = []
        for track_id, points in sorted(trajectories.items()):
            step = max(1, len(points) // MAX_TRAJECTORY_POINTS)
            payload.append({
                "track_id": track_id,
                "label": f"Player Track {track_id}",
                "team": track_teams.get(track_id, UNASSIGNED),
                "first_frame": points[0][0],
                "last_frame": points[-1][0],
                "frames_seen": len(points),
                "trajectory": points[::step],
            })
        return payload
