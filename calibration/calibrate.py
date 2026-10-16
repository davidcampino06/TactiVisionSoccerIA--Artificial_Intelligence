"""Threshold calibration with real broadcast clips.

Step 1, measure: run the real pipeline (the same "fast" settings the app uses) on every clip and
keep, for each team, the value of every indicator in every analysed frame.

    python -m calibration.calibrate measure calibration/clips/*.mp4

Step 2, evaluate: compare what the rules say with what a person saw in each clip
(calibration/labels.json) and, for every rule, look for the threshold that agrees best.

    python -m calibration.calibrate evaluate

The clips are not committed (broadcast footage, and large): only the measurements and the labels.
A rule "fires" exactly as in tactical_insights: at least MIN_SAMPLES frames and the threshold
crossed in at least MIN_PERSISTENCE of them.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np

from detectors import DetectorFactory
from config import settings
from pipeline import AnalysisOptions, VideoAnalysisPipeline
from tactical_analysis import DEFAULT_THRESHOLDS, AnalysisContext, IndicatorResult, TacticalAnalyzer
from tactical_insights import MIN_PERSISTENCE, MIN_SAMPLES

HERE = Path(__file__).resolve().parent
MEASUREMENTS = HERE / "measurements.json"
LABELS = HERE / "labels.json"

# The app's "fast" detail (MatchDetailPage: frame_stride 5, max_frames 120).
FAST = {"frame_stride": 5, "max_frames": 120}

# Question asked about each team in each clip -> the rule(s) it checks.
# A label True means "the person saw this problem", False "the person saw it was fine".
QUESTION_RULES: dict[str, list[str]] = {
    "lines_far_apart": ["defensive_midfield_distance", "largest_line_gap"],
    "not_compact": ["team_compactness_area", "average_player_distance"],
    "players_far_from_teammates": ["nearest_teammate_distance", "isolated_players"],
    "split_in_groups": ["network_components"],
    "no_pressure_on_ball": ["pressure_on_ball"],
    "narrow_attack": ["attacking_width", "team_width"],
    "crowded_zone": ["max_zone_concentration"],
}


class RecordingAnalyzer(TacticalAnalyzer):
    """The normal analyzer that also keeps the indicators it computed (Decorator idea)."""

    def __init__(self) -> None:
        super().__init__()
        self.last: list[IndicatorResult] = []

    def analyze(self, context: AnalysisContext) -> list[IndicatorResult]:
        self.last = super().analyze(context)
        return self.last


def measure(clips: list[Path], output: Path = MEASUREMENTS, precise_ball: bool = False) -> dict:
    """Runs the pipeline on every clip and returns the per-frame values of every indicator."""
    detector = DetectorFactory.create(settings.model_name, settings.model_runtime, settings.model_dir,
                                      settings.model_auto_download)
    results = json.loads(output.read_text(encoding="utf-8")) if output.exists() else {}
    for clip in clips:
        recorder = RecordingAnalyzer()
        response = VideoAnalysisPipeline(detector, tactical_analyzer=recorder).analyze_video(
            clip, AnalysisOptions(**FAST, include_detections=False, precise_ball=precise_ball))
        results[clip.stem] = {
            "frames": response["frames_processed"],
            "seconds": response["performance"]["processing_seconds"],
            "ball_detection_rate": response["ball_detection_rate"],
            "separation": response.get("teams", {}).get("separation_score"),
            "colors": response.get("teams", {}).get("colors"),
            "attack_direction": response["parameters"].get("team_a_attack_direction"),
            "issues": sorted({f"{issue['team']}:{issue['code']}" for issue in response["possible_issues"]}),
            "indicators": {
                indicator.full_name: {
                    "team": indicator.team,
                    "indicator": indicator.name,
                    "direction": indicator.threshold_direction,
                    "phase": indicator.phase,
                    "values": [round(float(value), 4) for value in indicator.values],
                }
                for indicator in recorder.last
            },
        }
        print(f"{clip.stem}: {response['frames_processed']} frames, {len(recorder.last)} indicators, "
              f"issues {results[clip.stem]['issues']}")
    output.write_text(_compact_json(results), encoding="utf-8")
    return results


def _compact_json(data: dict) -> str:
    """Indented JSON with every list of numbers on one line (readable diffs, small file)."""
    text = json.dumps(data, indent=1, ensure_ascii=False)
    return re.sub(r"\[\s*([-0-9.eE,\s]+?)\s*\]",
                  lambda match: "[" + ", ".join(match.group(1).split()).replace(",,", ",") + "]", text) + "\n"


def fires(values: list[float], threshold: float, direction: str) -> bool:
    """Same decision as TacticalInsightGenerator: enough frames and a persistent crossing."""
    if len(values) < MIN_SAMPLES:
        return False
    array = np.array(values)
    crossed = array >= threshold if direction == "above" else array <= threshold
    return float(crossed.mean()) >= MIN_PERSISTENCE


def cases(measurements: dict, labels: dict, indicator: str) -> list[tuple[str, list[float], str, bool]]:
    """(clip:team, values, direction, person saw a problem) for every labelled team and this indicator."""
    found = []
    for clip, teams in labels.items():
        for team, answers in teams.items():
            for question, rules in QUESTION_RULES.items():
                if indicator not in rules or answers.get(question) is None:
                    continue
                data = measurements.get(clip, {}).get("indicators", {}).get(f"{team.lower()}.{indicator}")
                if data and len(data["values"]) >= MIN_SAMPLES:
                    found.append((f"{clip}:{team}", data["values"], data["direction"], bool(answers[question])))
    return found


def score(items: list[tuple[str, list[float], str, bool]], threshold: float) -> dict:
    """Hits, false alarms and misses of one threshold against the labels."""
    tp = fp = fn = tn = 0
    for _, values, direction, problem in items:
        alarm = fires(values, threshold, direction)
        tp += alarm and problem
        fp += alarm and not problem
        fn += problem and not alarm
        tn += not problem and not alarm
    return {"tp": tp, "fp": fp, "fn": fn, "tn": tn, "agree": tp + tn, "total": len(items)}


def best_threshold(items: list[tuple[str, list[float], str, bool]], current: float) -> float:
    """The threshold that agrees with most labels, changed conservatively.

    With only a few clips, a candidate is considered only if it keeps every problem the current
    threshold already finds and adds no false alarm (a problem shown that the person did not
    see). Among the best candidates, the one closest to the current value wins, rounded to two
    decimals when that gives the same result."""
    base = score(items, current)
    candidates = sorted({round(v, 3) for _, values, _, _ in items for v in values} | {current})
    allowed = [t for t in candidates if score(items, t)["fp"] <= base["fp"] and score(items, t)["tp"] >= base["tp"]]
    best = min(allowed, key=lambda t: (-score(items, t)["agree"], score(items, t)["fp"], abs(t - current)))
    rounded = round(best, 2)
    return rounded if score(items, rounded) == score(items, best) else best


def evaluate(measurements: dict, labels: dict) -> list[dict]:
    """Per indicator: agreement with the current threshold and with the best one found."""
    rows = []
    for indicator in sorted({rule for rules in QUESTION_RULES.values() for rule in rules}):
        items = cases(measurements, labels, indicator)
        if not items:
            continue
        current = DEFAULT_THRESHOLDS[indicator]
        best = best_threshold(items, current)
        rows.append({"indicator": indicator, "cases": len(items), "current": current,
                     "current_score": score(items, current), "best": best, "best_score": score(items, best)})
    return rows


def question_scores(measurements: dict, labels: dict, thresholds: dict[str, float]) -> dict[str, dict]:
    """Per question: does the app show the problem (any of its rules fires) when the person saw it?"""
    result = {}
    for question, rules in QUESTION_RULES.items():
        tp = fp = fn = tn = 0
        for clip, teams in labels.items():
            for team, answers in teams.items():
                if answers.get(question) is None:
                    continue
                indicators = measurements.get(clip, {}).get("indicators", {})
                alarm = any(
                    fires(data["values"], thresholds[rule], data["direction"])
                    for rule in rules if (data := indicators.get(f"{team.lower()}.{rule}"))
                )
                problem = bool(answers[question])
                tp += alarm and problem
                fp += alarm and not problem
                fn += problem and not alarm
                tn += not problem and not alarm
        result[question] = {"tp": tp, "fp": fp, "fn": fn, "tn": tn, "agree": tp + tn, "total": tp + fp + fn + tn}
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)
    run = sub.add_parser("measure", help="analyse clips and store the indicator values")
    run.add_argument("clips", nargs="+", type=Path)
    run.add_argument("--precise-ball", action="store_true", help="look for the ball in both halves (full detail)")
    run.add_argument("--output", type=Path, default=MEASUREMENTS)
    check = sub.add_parser("evaluate", help="compare the rules with labels.json")
    check.add_argument("--measurements", type=Path, default=MEASUREMENTS)
    args = parser.parse_args()
    if args.command == "measure":
        measure(args.clips, args.output, args.precise_ball)
        return 0
    measurements = json.loads(args.measurements.read_text(encoding="utf-8"))
    labels = json.loads(LABELS.read_text(encoding="utf-8"))
    print(json.dumps({"indicators": evaluate(measurements, labels),
                      "questions": question_scores(measurements, labels, DEFAULT_THRESHOLDS)}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
