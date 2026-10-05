"""SIMULATION MODE.

Generates synthetic player positions (no video, no YOLO) and runs them
through the SAME tactical analysis and insight rules used for real videos.

Purpose: let the Backend and Frontend be developed and tested without a
video or a heavy model. Every response is labelled ``SIMULATION_MODE`` and
must never be presented as a real video analysis.
"""

from __future__ import annotations
import numpy as np

from pipeline import MODE_SIMULATION
from tactical_analysis import ATTACK_LEFT_TO_RIGHT, AnalysisContext, FrameSnapshot, PlayerPosition, TacticalAnalyzer
from tactical_insights import TacticalInsightGenerator
from team_assignment import TEAM_A, TEAM_B

# Outfield shapes in image coordinates (x along pitch length, y along pitch width).
TEAM_A_SHAPE = [  # 4-4-2 attacking left -> right, deliberately stretched between lines
    (0.12, 0.2), (0.12, 0.4), (0.12, 0.6), (0.12, 0.8),
    (0.42, 0.2), (0.42, 0.4), (0.42, 0.6), (0.42, 0.8),
    (0.58, 0.4), (0.58, 0.6),
]
TEAM_B_SHAPE = [  # 4-3-3 compact block
    (0.82, 0.25), (0.82, 0.42), (0.82, 0.58), (0.82, 0.75),
    (0.70, 0.35), (0.70, 0.5), (0.70, 0.65),
    (0.60, 0.3), (0.60, 0.5), (0.60, 0.7),
]

def run_simulation(frames: int = 120, seed: int = 7) -> dict:
    random = np.random.default_rng(seed)
    snapshots: list[FrameSnapshot] = []
    tracking: dict[int, list] = {}

    for frame_index in range(frames):
        drift_x = 0.03 * np.sin(frame_index / 15.0)
        players = []
        for offset, (team, shape) in enumerate(((TEAM_A, TEAM_A_SHAPE), (TEAM_B, TEAM_B_SHAPE))):
            for index, (base_x, base_y) in enumerate(shape):
                track_id = offset * 100 + index + 1
                x = float(np.clip(base_x + drift_x + random.normal(0, 0.01), 0, 1))
                y = float(np.clip(base_y + random.normal(0, 0.015), 0, 1))
                players.append(PlayerPosition(track_id, team, x, y, 0.8))
                tracking.setdefault(track_id, []).append([frame_index, round(x, 4), round(y, 4)])
        ball = (float(np.clip(0.5 + drift_x * 3, 0, 1)), 0.5)
        snapshots.append(FrameSnapshot(frame_index, frame_index / 25.0, players, ball))

    context = AnalysisContext(snapshots=snapshots, aspect_ratio=9 / 16, team_a_attack_direction=ATTACK_LEFT_TO_RIGHT)
    indicators = TacticalAnalyzer().analyze(context)
    possible_issues, recommendations = TacticalInsightGenerator().generate(indicators, team_separation=1.0)

    return {
        "status": "COMPLETED",
        "mode": MODE_SIMULATION,
        "model": "SIMULATION",
        "model_version": "synthetic-positions-v1",
        "runtime": "none",
        "tracker": "none",
        "video": {"fps": 25.0, "frame_count": frames, "width": 1280, "height": 720, "duration_seconds": frames / 25.0},
        "parameters": {"seed": seed, "frames": frames, "team_a_attack_direction": ATTACK_LEFT_TO_RIGHT},
        "frames_processed": frames,
        "players_detected": len(TEAM_A_SHAPE) + len(TEAM_B_SHAPE),
        "average_players_per_frame": float(len(TEAM_A_SHAPE) + len(TEAM_B_SHAPE)),
        "unique_player_tracks": len(tracking),
        "ball_detected": True,
        "ball_detection_rate": 1.0,
        "teams": {"colors": {}, "separation_score": 1.0, "track_teams": {}},
        "performance": {"processing_seconds": 0.0, "mean_inference_ms": 0.0, "processed_fps": 0.0},
        "detection_stability": {},
        "detections": [],
        "tracking": [
            {
                "track_id": track_id,
                "label": f"Player Track {track_id}",
                "team": TEAM_A if track_id < 100 else TEAM_B,
                "first_frame": points[0][0],
                "last_frame": points[-1][0],
                "frames_seen": len(points),
                "trajectory": points[::4],
            }
            for track_id, points in tracking.items()
        ],
        "tactical_indicators": [indicator.to_dict() for indicator in indicators],
        "possible_issues": possible_issues,
        "recommendations": recommendations,
        "warnings": ["SIMULATION MODE: synthetic positions, not a real video analysis."],
    }
