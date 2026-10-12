"""Pattern detection, possible tactical issues and recommendations (steps 4-7).

User-facing texts (issue, evidence, recommendation) are in Spanish because the app is shown in
Spanish; rule codes and indicator names stay in English.

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
        issue="Posible problema táctico: distancia excesiva entre la línea defensiva y el mediocampo",
        indicator="defensive_midfield_distance",
        reliability=0.9,
        evidence_template="La distancia entre defensa y mediocampo promedió {value} (umbral {threshold}) "
                            "y lo superó en el {persistence} % de {samples} cuadros.",
        recommendation="Revisar la distancia entre la línea defensiva y el mediocampo; considerar "
                        "adelantar la línea de fondo o retrasar un mediocampista para cerrar el espacio.",
    ),
    IssueRule(
        code="EXCESSIVE_DISTANCE_BETWEEN_LINES",
        issue="Posible problema táctico: distancia excesiva entre las líneas del equipo",
        indicator="largest_line_gap",
        reliability=0.75,
        evidence_template="El mayor espacio entre líneas consecutivas promedió {value} (umbral "
                            "{threshold}) en el {persistence} % de {samples} cuadros. Dirección de ataque desconocida.",
        recommendation="Revisar las distancias verticales entre líneas en esta secuencia; indicar la "
                        "dirección de ataque para identificar qué líneas están separadas.",
    ),
    IssueRule(
        code="LOW_TEAM_COMPACTNESS",
        issue="Posible problema táctico: baja compactación del equipo",
        indicator="team_compactness_area",
        reliability=0.85,
        evidence_template="El área ocupada por el equipo (envolvente convexa) promedió {value} de la imagen "
                            "(umbral {threshold}) en el {persistence} % de {samples} cuadros.",
        recommendation="Revisar la compactación sin balón; reducir el área ocupada acortando "
                        "las distancias entre líneas y entre carriles.",
    ),
    IssueRule(
        code="LOW_TEAM_COMPACTNESS",
        issue="Posible problema táctico: distancia promedio grande entre compañeros",
        indicator="average_player_distance",
        reliability=0.8,
        evidence_template="La distancia media entre compañeros promedió {value} (umbral "
                            "{threshold}) en el {persistence} % de {samples} cuadros.",
        recommendation="Revisar el posicionamiento colectivo; los jugadores aparecen lejos entre sí, "
                        "lo que puede reducir los apoyos y la cobertura en la presión.",
    ),
    IssueRule(
        code="EXCESSIVE_SPACING",
        issue="Posible problema táctico: separación excesiva entre jugadores cercanos",
        indicator="nearest_teammate_distance",
        reliability=0.8,
        evidence_template="La distancia al compañero más cercano promedió {value} (umbral "
                            "{threshold}) en el {persistence} % de {samples} cuadros.",
        recommendation="Revisar las distancias de apoyo; los jugadores aislados pueden quedarse sin "
                        "opciones de pase o sin cobertura defensiva.",
    ),
    IssueRule(
        code="DEFENSIVE_LINE_TOO_DEEP",
        issue="Posible problema táctico: línea defensiva demasiado retrasada",
        indicator="defensive_line_position",
        reliability=0.6,
        evidence_template="La línea defensiva se mantuvo a {value} del borde de la imagen de su propio arco "
                            "(umbral {threshold}) en el {persistence} % de {samples} cuadros. "
                            "Métrica relativa a la imagen.",
        recommendation="Revisar la altura de la línea defensiva; una línea muy retrasada puede dejar "
                        "espacio entre defensa y mediocampo. Confirmar con un plano amplio de cámara.",
    ),
    IssueRule(
        code="HIGH_ZONE_CONCENTRATION",
        issue="Posible problema táctico: alta concentración de jugadores en una zona",
        indicator="max_zone_concentration",
        reliability=0.85,
        evidence_template="La zona más ocupada de la cuadrícula 3x3 reunió el {value_percent} % de los jugadores visibles "
                            "(umbral {threshold_percent} %) en el {persistence} % de {samples} cuadros.",
        recommendation="Revisar la ocupación de la cancha; demasiados jugadores en una zona pueden dejar "
                        "otras zonas sin cubrir ante cambios de orientación.",
    ),
    IssueRule(
        code="INSUFFICIENT_WIDTH",
        issue="Posible problema táctico: amplitud insuficiente",
        indicator="team_width",
        reliability=0.75,
        evidence_template="El ancho del equipo promedió {value} de la altura de la imagen (umbral "
                            "{threshold}) en el {persistence} % de {samples} cuadros.",
        recommendation="Revisar la amplitud con balón; considerar extremos pegados a la "
                        "banda para estirar el bloque rival.",
    ),
    IssueRule(
        code="NUMERICAL_IMBALANCE",
        issue="Posible problema táctico: desequilibrio numérico en una zona",
        indicator="max_zone_numerical_difference",
        reliability=0.7,
        evidence_template="Un tercio de la imagen mostró una diferencia de {value} jugadores visibles "
                        "entre equipos (umbral {threshold}) en el {persistence} % de {samples} cuadros.",
        recommendation="Revisar el equilibrio numérico en el tercio afectado; verificar si los "
                        "jugadores de apoyo llegan a tiempo para igualar la zona.",
    ),
    IssueRule(
        code="DISCONNECTED_TEAM",
        issue="Posible problema táctico: equipo partido en grupos sin conexión",
        indicator="network_components",
        reliability=0.7,
        evidence_template="En la red de cercanía (grafo recorrido con BFS) el equipo formó en promedio {value} "
                            "grupos sin compañeros cerca entre sí (umbral {threshold}) en el {persistence} % "
                            "de {samples} cuadros.",
        recommendation="Revisar los apoyos entre líneas; acercar un jugador que una los grupos para que el "
                        "balón pueda circular con pases cortos y la presión tenga cobertura.",
    ),
    IssueRule(
        code="ISOLATED_PLAYERS",
        issue="Posible problema táctico: jugadores sin apoyo cercano",
        indicator="isolated_players",
        reliability=0.65,
        evidence_template="En promedio {value} jugadores quedaron sin ningún compañero cerca en la red de "
                            "cercanía (umbral {threshold}) en el {persistence} % de {samples} cuadros.",
        recommendation="Revisar la posición de los jugadores aislados; un apoyo cercano les da una opción "
                        "de pase y una cobertura si pierden el balón.",
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

def format_number(value: float, decimals: int) -> str:
    """Formats a number the Spanish way: decimal comma, dot for thousands, no trailing zeros.

    The evidence texts are read by Spanish speakers, for whom "6.777" means six thousand.
    """
    text = f"{value:,.{decimals}f}"
    if decimals:
        text = text.rstrip("0").rstrip(".")
    return text.replace(",", "\u0000").replace(".", ",").replace("\u0000", ".")


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
                    value=format_number(indicator.value, 3),
                    threshold=format_number(indicator.threshold or 0, 3),
                    value_percent=format_number(indicator.value * 100, 1),
                    threshold_percent=format_number((indicator.threshold or 0) * 100, 1),
                    persistence=format_number(indicator.exceed_ratio * 100, 1),
                    samples=format_number(indicator.samples, 0),
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
