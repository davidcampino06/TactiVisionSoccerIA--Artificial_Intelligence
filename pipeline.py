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
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from config import settings
from detectors import OBJECT_TYPE_BALL, OBJECT_TYPE_PLAYER, Detection, ObjectDetector
from tactical_analysis import (
    ATTACK_UNKNOWN,
    AnalysisContext,
    FrameSnapshot,
    PlayerPosition,
    TacticalAnalyzer,
)
from tactical_insights import TacticalInsightGenerator
from team_assignment import UNASSIGNED, JerseyColorTeamClassifier
from tracking import TrackedDetection, TrackerFactory

logger = logging.getLogger("tactivision.pipeline")

MODE_REAL = "REAL_VIDEO_ANALYSIS"
MODE_SIMULATION = "SIMULATION_MODE"

MIN_PITCH_GREEN_RATIO = 0.15
MAX_TRAJECTORY_POINTS = 120

class VideoProcessingError(ValueError):
    """The video could not be read or produced no usable frames."""

@dataclass
class AnalysisOptions:
    confidence_threshold: float = settings.confidence_threshold
    ball_confidence_threshold: float = settings.ball_confidence_threshold
    frame_stride: int = settings.default_frame_stride
    max_frames: int = settings.default_max_frames
    team_a_attack_direction: str = ATTACK_UNKNOWN
    pitch_filter: bool = True
    include_detections: bool = True
    require_pitch_for_insights: bool = True

@dataclass
class _FrameRecord:
    frame_number: int
    timestamp_seconds: float
    tracked_players: list[TrackedDetection]
    ball: Detection | None

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

        warnings = list(extraction_stats["warnings"])
        team_result = extraction_stats["team_classifier"].assign()
        warnings.extend(team_result.warnings)

        snapshots = self._build_snapshots(records, team_result.track_teams, video_info)
        context = AnalysisContext(
            snapshots=snapshots,
            aspect_ratio=video_info["height"] / video_info["width"],
            team_a_attack_direction=options.team_a_attack_direction,
        )
        if options.team_a_attack_direction == ATTACK_UNKNOWN:
            warnings.append(
                "Attack direction not provided: defensive line indicators were replaced by "
                "'largest_line_gap'."
            )

        indicators = self._tactical_analyzer.analyze(context)
        possible_issues, recommendations = [], []
        if extraction_stats["pitch_visible"] or not options.require_pitch_for_insights:
            possible_issues, recommendations = self._insight_generator.generate(
                indicators, team_result.separation_score
            )
        else:
            warnings.append(
                "Possible tactical issues and recommendations were NOT generated because the video "
                "does not appear to show a football pitch."
            )

        processing_seconds = time.perf_counter() - started
        frames_processed = len(records)
        player_counts = [len(record.tracked_players) for record in records]
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
                "team_a_attack_direction": options.team_a_attack_direction,
                "pitch_filter": options.pitch_filter,
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
            "detections": self._detections_payload(records, team_result.track_teams) if options.include_detections else [],
            "tracking": self._tracking_payload(records, team_result.track_teams, video_info),
            "tactical_indicators": [indicator.to_dict() for indicator in indicators],
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

    def _extract_and_track(self, capture, video_info: dict, options: AnalysisOptions):
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
            if not pitch_visible:
                frames_without_pitch += 1
            elif options.pitch_filter:
                on_pitch = [d for d in players if _is_on_pitch(d, green)]
                filtered_out_people += len(players) - len(on_pitch)
                players = on_pitch

            tracked_players = tracker.update(players)
            team_classifier.collect(frame, tracked_players)
            best_ball = max(balls, key=lambda d: d.confidence) if balls else None
            records.append(
                _FrameRecord(frame_number, frame_number / video_info["fps"], tracked_players, best_ball)
            )

        warnings: list[str] = []
        if records and frames_without_pitch > len(records) / 2:
            warnings.append(
                f"{frames_without_pitch} of {len(records)} frames show little grass; the video may "
                "not be a football broadcast and tactical indicators may not be meaningful."
            )
        if filtered_out_people:
            warnings.append(
                f"{filtered_out_people} person detections outside the pitch were ignored (crowd, bench)."
            )
        return records, {
            "pitch_visible": bool(records) and frames_without_pitch <= len(records) / 2,
            "team_classifier": team_classifier,
            "inference_ms": inference_ms or [0.0],
            "warnings": warnings,
        }

    @staticmethod
    def _build_snapshots(records, track_teams: dict[int, str], video_info: dict) -> list[FrameSnapshot]:
        width, height = video_info["width"], video_info["height"]
        snapshots = []
        for record in records:
            players = []
            for tracked in record.tracked_players:
                team = track_teams.get(tracked.track_id, UNASSIGNED)
                foot_x, foot_y = tracked.detection.foot_point
                players.append(
                    PlayerPosition(tracked.track_id, team, foot_x / width, foot_y / height, tracked.detection.confidence)
                )
            ball = None
            if record.ball is not None:
                ball_x, ball_y = record.ball.center
                ball = (ball_x / width, ball_y / height)
            snapshots.append(FrameSnapshot(record.frame_number, record.timestamp_seconds, players, ball))
        return snapshots

    @staticmethod
    def _unique_tracks(records) -> int:
        return len({tracked.track_id for record in records for tracked in record.tracked_players})

    @staticmethod
    def _stability(records) -> dict:
        track_frames: dict[int, int] = defaultdict(int)
        for record in records:
            for tracked in record.tracked_players:
                track_frames[tracked.track_id] += 1
        counts = [len(record.tracked_players) for record in records]
        mean_players = float(np.mean(counts)) if counts else 0.0
        lengths = list(track_frames.values())
        return {
            "player_count_std": round(float(np.std(counts)), 3) if counts else 0.0,
            "mean_track_length_frames": round(float(np.mean(lengths)), 2) if lengths else 0.0,
            "tracks_longer_than_10_frames": sum(1 for length in lengths if length >= 10),
            # 1.0 means one track per visible player; higher values mean identity switches/fragments.
            "track_fragmentation_ratio": round(len(lengths) / mean_players, 3) if mean_players else 0.0,
        }

    @staticmethod
    def _detections_payload(records, track_teams: dict[int, str]) -> list[dict]:
        payload = []
        for record in records:
            items = [(tracked.detection, tracked.track_id) for tracked in record.tracked_players]
            if record.ball is not None:
                items.append((record.ball, None))
            for detection, track_id in items:
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
                    "team": track_teams.get(track_id, UNASSIGNED) if track_id is not None else None,
                })
        return payload

    @staticmethod
    def _tracking_payload(records, track_teams: dict[int, str], video_info: dict) -> list[dict]:
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
