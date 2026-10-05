"""Pattern detection, possible tactical issues and recommendations (steps 4-7).

Every issue is a POSSIBLE tactical issue. It is derived from explicit rules
over the indicators, so every output can be explained:

    issue  <-  rule(indicator, threshold, persistence)
    evidence    = the numbers that triggered the rule
    confidence  = how much we trust those numbers (see ``_confidence``)
    recommendation = a review suggestion linked to the same indicators

Confidence formula (all factors in 0..1):

    persistence  = fraction of analysed frames where the threshold was crossed
    sample       = min(1, samples / 20)
    detection    = mean YOLO confidence of the players used
    team         = jersey-colour separation score (1.0 for match-level metrics)
    reliability  = fixed factor of the rule (lower for image-relative metrics)

    confidence = persistence * (0.5 + 0.5*sample) * (0.5 + 0.5*detection)
                 * (0.5 + 0.5*team) * reliability
"""

from __future__ import annotations
from dataclasses import dataclass
from tactical_analysis import IndicatorResult

MIN_PERSISTENCE = 0.5
MIN_SAMPLES = 8

@dataclass(frozen=True)
class IssueRule:
    code: str
    issue: str
    indicator: str
    reliability: float
    evidence_template: str
    recommendation: str

ISSUE_RULES: list[IssueRule] = [
    IssueRule(
        code="EXCESSIVE_DISTANCE_BETWEEN_LINES",
        issue="Possible tactical issue: excessive distance between defensive and midfield lines",
        indicator="defensive_midfield_distance",
        reliability=0.9,
        evidence_template="Defence-midfield distance averaged {value} (threshold {threshold}) "
                            "and exceeded it in {persistence}% of {samples} frames.",
        recommendation="Review the distance between defensive and midfield lines; consider "
                        "stepping the back line up or dropping a midfielder to close the gap.",
    ),
    IssueRule(
        code="EXCESSIVE_DISTANCE_BETWEEN_LINES",
        issue="Possible tactical issue: excessive distance between team lines",
        indicator="largest_line_gap",
        reliability=0.75,
        evidence_template="Largest gap between consecutive lines averaged {value} (threshold "
                            "{threshold}) in {persistence}% of {samples} frames. Attack direction unknown.",
        recommendation="Review the vertical distances between lines in this sequence; set the "
                        "attack direction to identify which lines are separated.",
    ),
    IssueRule(
        code="LOW_TEAM_COMPACTNESS",
        issue="Possible tactical issue: low team compactness",
        indicator="team_compactness_area",
        reliability=0.85,
        evidence_template="Convex-hull area of the team averaged {value} of the image "
                            "(threshold {threshold}) in {persistence}% of {samples} frames.",
        recommendation="Review team compactness when out of possession; reduce the occupied "
                        "area by shortening distances between lines and between channels.",
    ),
    IssueRule(
        code="LOW_TEAM_COMPACTNESS",
        issue="Possible tactical issue: large average distance between team-mates",
        indicator="average_player_distance",
        reliability=0.8,
        evidence_template="Mean distance between team-mates averaged {value} (threshold "
                            "{threshold}) in {persistence}% of {samples} frames.",
        recommendation="Review collective positioning; players appear far from each other, "
                        "which can reduce support and pressing coverage.",
    ),
    IssueRule(
        code="EXCESSIVE_SPACING",
        issue="Possible tactical issue: excessive spacing between nearby players",
        indicator="nearest_teammate_distance",
        reliability=0.8,
        evidence_template="Distance to the closest team-mate averaged {value} (threshold "
                            "{threshold}) in {persistence}% of {samples} frames.",
        recommendation="Review support distances; isolated players may lack passing options "
                        "or defensive cover.",
    ),
    IssueRule(
        code="DEFENSIVE_LINE_TOO_DEEP",
        issue="Possible tactical issue: defensive line too deep",
        indicator="defensive_line_position",
        reliability=0.6,
        evidence_template="Defensive line stayed at {value} from the own-goal image edge "
                            "(threshold {threshold}) in {persistence}% of {samples} frames. "
                            "Image-relative metric.",
        recommendation="Review the height of the defensive line; a very deep line can leave "
                        "space between defence and midfield. Confirm with a wide camera view.",
    ),
    IssueRule(
        code="HIGH_ZONE_CONCENTRATION",
        issue="Possible tactical issue: high player concentration in one zone",
        indicator="max_zone_concentration",
        reliability=0.85,
        evidence_template="The busiest 3x3 zone held {value_percent}% of visible players "
                            "(threshold {threshold_percent}%) in {persistence}% of {samples} frames.",
        recommendation="Review occupation of the pitch; too many players in one zone can leave "
                        "other zones uncovered for switches of play.",
    ),
    IssueRule(
        code="INSUFFICIENT_WIDTH",
        issue="Possible tactical issue: insufficient width",
        indicator="team_width",
        reliability=0.75,
        evidence_template="Team width averaged {value} of the image height (threshold "
                            "{threshold}) in {persistence}% of {samples} frames.",
        recommendation="Review width in possession; consider wide players holding the "
                        "touchline to stretch the opponent's block.",
    ),
    IssueRule(
        code="NUMERICAL_IMBALANCE",
        issue="Possible tactical issue: numerical imbalance in a zone",
        indicator="max_zone_numerical_difference",
        reliability=0.7,
        evidence_template="One image third showed a difference of {value} visible players "
                        "between teams (threshold {threshold}) in {persistence}% of {samples} frames.",
        recommendation="Review numerical balance in the affected third; check whether "
                        "supporting players arrive on time to equalise the zone.",
    ),
]

def _severity(indicator: IndicatorResult) -> str:
    if not indicator.threshold:
        return "LOW"
    relative_excess = abs(indicator.value - indicator.threshold) / abs(indicator.threshold)
    beyond = (
        indicator.value >= indicator.threshold
        if indicator.threshold_direction == "above"
        else indicator.value <= indicator.threshold
    )
    if not beyond or relative_excess < 0.15:
        return "LOW"
    return "MEDIUM" if relative_excess < 0.35 else "HIGH"

def _confidence(indicator: IndicatorResult, rule: IssueRule, team_separation: float) -> tuple[float, dict]:
    persistence = indicator.exceed_ratio
    sample_factor = min(1.0, indicator.samples / 20.0)
    detection_factor = indicator.mean_detection_confidence
    team_factor = 1.0 if indicator.team == "MATCH" else team_separation
    confidence = (
        persistence
        * (0.5 + 0.5 * sample_factor)
        * (0.5 + 0.5 * detection_factor)
        * (0.5 + 0.5 * team_factor)
        * rule.reliability
    )
    breakdown = {
        "persistence": round(persistence, 4),
        "sample_factor": round(sample_factor, 4),
        "detection_factor": round(detection_factor, 4),
        "team_separation_factor": round(team_factor, 4),
        "rule_reliability": rule.reliability,
    }
    return round(min(max(confidence, 0.0), 1.0), 4), breakdown

class TacticalInsightGenerator:
    """Applies the issue rules and builds explainable recommendations."""

    def __init__(self, rules: list[IssueRule] | None = None) -> None:
        self._rules = rules if rules is not None else ISSUE_RULES

    def generate(self, indicators: list[IndicatorResult], team_separation: float) -> tuple[list[dict], list[dict]]:
        possible_issues: list[dict] = []
        recommendations: list[dict] = []

        for rule in self._rules:
            for indicator in indicators:
                if indicator.name != rule.indicator:
                    continue
                if indicator.samples < MIN_SAMPLES or indicator.exceed_ratio < MIN_PERSISTENCE:
                    continue

                confidence, breakdown = _confidence(indicator, rule, team_separation)
                evidence = rule.evidence_template.format(
                    value=round(indicator.value, 3),
                    threshold=indicator.threshold,
                    value_percent=round(indicator.value * 100, 1),
                    threshold_percent=round((indicator.threshold or 0) * 100, 1),
                    persistence=round(indicator.exceed_ratio * 100, 1),
                    samples=indicator.samples,
                )
                issue_id = f"{rule.code}:{indicator.team}:{indicator.name}"
                severity = _severity(indicator)
                possible_issues.append({
                    "id": issue_id,
                    "code": rule.code,
                    "issue": rule.issue,
                    "team": indicator.team,
                    "evidence": evidence,
                    "confidence": confidence,
                    "confidence_breakdown": breakdown,
                    "severity": severity,
                    "related_indicators": [indicator.full_name],
                })
                recommendations.append({
                    "issue_id": issue_id,
                    "title": rule.issue,
                    "team": indicator.team,
                    "recommendation": rule.recommendation,
                    "evidence": evidence,
                    "confidence": confidence,
                    "severity": severity,
                    "related_indicators": [indicator.full_name],
                })

        possible_issues.sort(key=lambda item: item["confidence"], reverse=True)
        recommendations.sort(key=lambda item: item["confidence"], reverse=True)
        return possible_issues, recommendations
