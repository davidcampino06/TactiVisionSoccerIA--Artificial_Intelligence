"""What the team did well (strengths), told in football words with the numbers behind them.

The mirror image of the possible issues (tactical_insights.py): a strength is reported only when
the indicator stayed clearly on the GOOD side of a stricter limit for most of the moments that
matter (the same phase of play as the issue), never when the issue of the same indicator fired.

    strength  <-  share of frames with value on the good side of ``limit`` >= ``min_share``
    confidence = share * (0.5 + 0.5*sample) * (0.5 + 0.5*detection) * (0.5 + 0.5*team) * reliability

Only indicators that agreed with the coach's labels of the calibration clips are used (see
CALIBRATION.md, stage 6): with zero strengths reported where the coach saw that problem. Compactness
area and distance to the nearest team-mate are left out on purpose: with a broadcast camera they
depend so much on the zoom that a "compact" reading appeared in clips the coach judged open.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from tactical_analysis import IndicatorResult
from tactical_insights import DEFENDING_MOMENT, TEAM_NAMES, _plain_duration, format_number

MIN_SAMPLES = 8


@dataclass(frozen=True)
class StrengthRule:
    code: str
    title: str
    indicator: str
    direction: str          # "below": good when value <= limit; "above": good when value >= limit
    limit: float
    min_share: float        # share of frames on the good side needed to report it
    reliability: float
    moment: str             # when it is measured, as in the issue explanations
    good: str               # what was good, completes "en el Equipo A ..."
    meaning: str            # what it means on the pitch
    keep: str               # how to keep doing it
    conflicts: tuple[str, ...] = ()  # issue codes of the same team that rule this strength out


STRENGTH_RULES: list[StrengthRule] = [
    StrengthRule(
        code="LINES_TOGETHER",
        title="Líneas juntas al defender",
        indicator="defensive_midfield_distance",
        direction="below", limit=0.15, min_share=0.5, reliability=0.8,
        moment=DEFENDING_MOMENT,
        good="el espacio entre la línea de defensas y la de mediocampistas fue menor al 15 % del ancho de la imagen",
        meaning="Defensa y mediocampo defendieron cerca uno del otro: no quedó un hueco entre líneas donde un rival "
                "pudiera recibir solo y girarse de frente al arco.",
        keep="Mantener esa distancia corta: que la defensa y el mediocampo suban y bajen juntos, como un solo bloque.",
        conflicts=("EXCESSIVE_DISTANCE_BETWEEN_LINES",),
    ),
    StrengthRule(
        code="NOBODY_ISOLATED",
        title="Siempre con un compañero cerca",
        indicator="isolated_players",
        direction="below", limit=0.0, min_share=0.9, reliability=0.75,
        moment=DEFENDING_MOMENT,
        good="ningún jugador quedó sin un compañero cerca",
        meaning="Al defender nadie quedó solo: cada jugador tuvo una ayuda a pocos metros para hacer la cobertura si el "
                "rival lo superaba, y un pase corto a mano al recuperar el balón.",
        keep="Seguir moviéndose como bloque hacia el lado del balón, sin dejar jugadores sueltos lejos de la jugada.",
        conflicts=("ISOLATED_PLAYERS", "DISCONNECTED_TEAM"),
    ),
    StrengthRule(
        code="PRESSURE_ON_BALL",
        title="Presión sobre el balón",
        indicator="pressure_on_ball",
        direction="above", limit=1.0, min_share=0.75, reliability=0.6,
        moment="Cuando el rival tenía el balón",
        good="hubo al menos un jugador encima del rival que llevaba el balón",
        meaning="El rival casi nunca pudo jugar tranquilo: siempre hubo alguien apurando al que tenía el balón, sin "
                "tiempo para levantar la cabeza y elegir el pase.",
        keep="Mantener la reacción del jugador más cercano y que los compañeros acompañen para tapar los pases cortos.",
        conflicts=("LOW_BALL_PRESSURE",),
    ),
    StrengthRule(
        code="ATTACKING_DEPTH",
        title="Profundidad en ataque",
        indicator="players_ahead_of_ball",
        direction="above", limit=2.0, min_share=0.75, reliability=0.6,
        moment="Cuando tenía el balón",
        good="hubo al menos dos compañeros por delante del balón",
        meaning="Con el balón siempre hubo jugadores por delante ofreciendo el pase hacia adelante y obligando a la "
                "defensa rival a retroceder.",
        keep="Seguir atacando el espacio a la espalda de la defensa: esos desmarques abren pasillos por dentro.",
        conflicts=("NO_ATTACKING_DEPTH",),
    ),
]


def good_share(indicator: IndicatorResult, rule: StrengthRule) -> float:
    """Fraction of frames on the good side of the rule's limit."""
    if not indicator.values:
        return 0.0
    values = np.array(indicator.values)
    good = values <= rule.limit if rule.direction == "below" else values >= rule.limit
    return round(float(good.mean()), 4)


def is_good(value: float, rule: StrengthRule) -> bool:
    return value <= rule.limit if rule.direction == "below" else value >= rule.limit


def _confidence(indicator: IndicatorResult, rule: StrengthRule, share: float, team_separation: float) -> tuple[float, dict]:
    sample_factor = min(1.0, indicator.samples / 20.0)
    detection_factor = indicator.mean_detection_confidence
    team_factor = 1.0 if indicator.team == "MATCH" else team_separation
    confidence = share * (0.5 + 0.5 * sample_factor) * (0.5 + 0.5 * detection_factor) \
        * (0.5 + 0.5 * team_factor) * rule.reliability
    breakdown = {
        "persistence": round(share, 4),
        "sample_factor": round(sample_factor, 4),
        "detection_factor": round(detection_factor, 4),
        "team_separation_factor": round(team_factor, 4),
        "rule_reliability": rule.reliability,
    }
    return round(min(max(confidence, 0.0), 1.0), 4), breakdown


def plain_strength(indicator: IndicatorResult, rule: StrengthRule, share: float,
                   seconds_per_frame: float | None = None) -> dict:
    """What the AI saw (with the real numbers), what it means and how to keep it, in plain Spanish.

    "80 %" keeps a non-breaking space so the number and the sign never end up on different lines."""
    team = TEAM_NAMES.get(indicator.team, indicator.team)
    seen = (f"{rule.moment} ({_plain_duration(indicator.samples, seconds_per_frame)}), en {team} {rule.good} "
            f"en el {format_number(share * 100, 0)}\u00a0% de esos momentos.")
    if indicator.unit == "frame_width_ratio":
        seen = seen[:-1] + f" (en promedio, el {format_number(indicator.value * 100, 1)}\u00a0%)."
    return {"seen": seen, "meaning": rule.meaning, "keep": rule.keep}


class StrengthGenerator:
    """Applies the strength rules; returns the strengths, most confident first."""

    def __init__(self, rules: list[StrengthRule] | None = None) -> None:
        self._rules = rules if rules is not None else STRENGTH_RULES

    def generate(self, indicators: list[IndicatorResult], team_separation: float,
                 possible_issues: list[dict] | None = None, seconds_per_frame: float | None = None) -> list[dict]:
        flagged = {(issue["team"], issue["code"]) for issue in possible_issues or []}
        strengths = []
        for rule in self._rules:
            for indicator in indicators:
                if indicator.name != rule.indicator or indicator.samples < MIN_SAMPLES:
                    continue
                if any((indicator.team, code) in flagged for code in rule.conflicts):
                    continue
                share = good_share(indicator, rule)
                if share < rule.min_share:
                    continue
                confidence, breakdown = _confidence(indicator, rule, share, team_separation)
                strengths.append({
                    "id": f"{rule.code}:{indicator.team}:{indicator.name}",
                    "code": rule.code,
                    "title": rule.title,
                    "team": indicator.team,
                    "evidence": f"En el {format_number(share * 100, 1)} % de {indicator.samples} cuadros el valor quedó "
                                f"del lado bueno de {format_number(rule.limit, 3)} (promedio "
                                f"{format_number(indicator.value, 3)}).",
                    "confidence": confidence,
                    "confidence_breakdown": breakdown,
                    "related_indicators": [indicator.full_name],
                    "explanation": plain_strength(indicator, rule, share, seconds_per_frame),
                })
        strengths.sort(key=lambda item: item["confidence"], reverse=True)
        return strengths


RULE_BY_CODE = {rule.code: rule for rule in STRENGTH_RULES}
