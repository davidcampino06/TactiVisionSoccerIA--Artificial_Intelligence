"""Model evaluation script (Phase 2/3 comparison).

Compares detectors on the SAME videos and records only measured values.

Measured without labels (any video):
    inference time, processed FPS, players per frame, ball detection rate,
    mean detection confidence, tracking stability (ByteTrack).

Measured with a labelled dataset (optional, requires ultralytics + PyTorch):
    precision, recall, mAP50, mAP50-95  ->  --dataset path/to/data.yaml

Usage:
    python evaluate_models.py --videos clip1.mp4 clip2.mp4 --models YOLO26n YOLO11n
    python evaluate_models.py --videos clip.mp4 --models YOLO26n --dataset soccer.yaml
"""

from __future__ import annotations

import argparse
import json
import platform
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from config import settings
from detectors import OBJECT_TYPE_BALL, OBJECT_TYPE_PLAYER, DetectorFactory, ModelLoadError
from pipeline import AnalysisOptions, VideoAnalysisPipeline

def evaluate_on_video(detector, video_path: Path, frame_stride: int, max_frames: int) -> dict:
    result = VideoAnalysisPipeline(detector).analyze_video(
        video_path,
        AnalysisOptions(frame_stride=frame_stride, max_frames=max_frames, include_detections=True),
    )
    player_confidences = [d["confidence"] for d in result["detections"] if d["object_type"] == OBJECT_TYPE_PLAYER]
    ball_confidences = [d["confidence"] for d in result["detections"] if d["object_type"] == OBJECT_TYPE_BALL]
    return {
        "video": video_path.name,
        "frames": result["frames_processed"],
        "mean_inference_ms": result["performance"]["mean_inference_ms"],
        "processed_fps": result["performance"]["processed_fps"],
        "avg_players_per_frame": result["average_players_per_frame"],
        "mean_player_confidence": round(float(np.mean(player_confidences)), 4) if player_confidences else None,
        "ball_detection_rate": result["ball_detection_rate"],
        "mean_ball_confidence": round(float(np.mean(ball_confidences)), 4) if ball_confidences else None,
        "unique_tracks": result["unique_player_tracks"],
        **{f"tracking_{key}": value for key, value in result["detection_stability"].items()},
    }

def evaluate_with_dataset(model_name: str, dataset_yaml: Path) -> dict:
    """Precision/recall/mAP need ground-truth labels; uses Ultralytics validation."""
    try:
        from ultralytics import RTDETR, YOLO
    except ImportError:
        return {"error": "ultralytics not installed (pip install -r requirements-eval.txt)"}
    detector = DetectorFactory.create(model_name, "ultralytics", settings.model_dir, settings.model_auto_download)
    weights = settings.model_dir / detector.spec.weights_file_name
    model = RTDETR(str(weights)) if detector.spec.family == "RT-DETR" else YOLO(str(weights))
    metrics = model.val(data=str(dataset_yaml), classes=[0, 32], verbose=False)
    return {
        "precision": round(float(metrics.box.mp), 4),
        "recall": round(float(metrics.box.mr), 4),
        "mAP50": round(float(metrics.box.map50), 4),
        "mAP50_95": round(float(metrics.box.map), 4),
    }

def to_markdown(rows: list[dict]) -> str:
    columns = [
        "model", "runtime", "video", "frames", "mean_inference_ms", "processed_fps", "avg_players_per_frame",
        "mean_player_confidence", "ball_detection_rate", "unique_tracks",
        "tracking_track_fragmentation_ratio", "tracking_mean_track_length_frames",
        "precision", "recall", "mAP50", "mAP50_95",
    ]
    lines = ["| " + " | ".join(columns) + " |", "|" + "---|" * len(columns)]
    for row in rows:
        lines.append("| " + " | ".join("n/a" if row.get(c) is None else str(row.get(c)) for c in columns) + " |")
    return "\n".join(lines)

def main() -> None:
    parser = argparse.ArgumentParser(description="Compare TactiVision detection models.")
    parser.add_argument("--videos", nargs="+", type=Path, required=True)
    parser.add_argument("--models", nargs="+", default=["YOLO26n", "YOLO11n"])
    parser.add_argument("--runtime", default=settings.model_runtime)
    parser.add_argument("--frame-stride", type=int, default=1)
    parser.add_argument("--max-frames", type=int, default=150)
    parser.add_argument("--dataset", type=Path, default=None, help="Ultralytics data.yaml with labels")
    parser.add_argument("--output", type=Path, default=Path("evaluation_results.json"))
    args = parser.parse_args()

    rows: list[dict] = []
    for model_name in args.models:
        try:
            detector = DetectorFactory.create(model_name, args.runtime, settings.model_dir, settings.model_auto_download)
        except ModelLoadError as error:
            rows.append({"model": model_name, "runtime": args.runtime, "error": str(error)})
            print(f"[{model_name}] could not be loaded: {error}")
            continue
        dataset_metrics = evaluate_with_dataset(model_name, args.dataset) if args.dataset else {}
        for video_path in args.videos:
            row = {"model": model_name, "runtime": args.runtime}
            row.update(evaluate_on_video(detector, video_path, args.frame_stride, args.max_frames))
            row.update(dataset_metrics)
            rows.append(row)
            print(f"[{model_name}] {video_path.name}: {row['mean_inference_ms']} ms/frame")

    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "machine": {"platform": platform.platform(), "processor": platform.processor() or "unknown"},
        "results": rows,
    }
    args.output.write_text(json.dumps(report, indent=2))
    print()
    print(to_markdown([row for row in rows if "error" not in row]))

if __name__ == "__main__":
    main()
