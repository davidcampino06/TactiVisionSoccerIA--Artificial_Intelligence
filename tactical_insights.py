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

# Added to the evidence so the coach knows which moments were judged (see INDICATOR_PHASE).
PHASE_NOTE = {
    "defending": " Medido solo en los cuadros en que el rival tenía el balón (el equipo defendía).",
    "not_defending": " Medido solo en los cuadros en que el equipo no estaba defendiendo.",
}

@dataclass(frozen=True)
class IssueRule:
    """Turns one indicator crossing its threshold into a possible issue and a recommendation."""
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
        # 0.85 before calibration; with real clips it had the most false alarms (2 of 7): lowered.
        reliability=0.6,
        evidence_template="La zona más ocupada de la cuadrícula 3x3 reunió el {value_percent} % de los jugadores visibles "
                            "(umbral {threshold_percent} %) en el {persistence} % de {samples} cuadros.",
        recommendation="Revisar la ocupación de la cancha; demasiados jugadores en una zona pueden dejar "
                        "otras zonas sin cubrir ante cambios de orientación.",
    ),
    IssueRule(
        code="INSUFFICIENT_WIDTH",
        issue="Posible problema táctico: amplitud insuficiente",
        indicator="team_width",
        # 0.75 before calibration; 1 false alarm of 3 with real clips: lowered.
        reliability=0.65,
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
    IssueRule(
        code="LOW_BALL_PRESSURE",
        issue="Posible problema táctico: poca presión sobre el balón",
        indicator="pressure_on_ball",
        reliability=0.6,
        evidence_template="Sin el balón, el equipo tuvo en promedio {value} jugadores cerca del balón "
                            "(umbral {threshold}) y nadie lo presionó en el {persistence} % de {samples} cuadros.",
        recommendation="Revisar la presión tras pérdida y el primer defensor; acercar al jugador más próximo "
                        "al balón para reducir el tiempo del rival.",
    ),
    IssueRule(
        code="NARROW_ATTACK",
        issue="Posible problema táctico: ataque demasiado estrecho",
        indicator="attacking_width",
        reliability=0.65,
        evidence_template="Con el balón, el equipo ocupó en promedio {value_percent} % del ancho visible "
                            "(umbral {threshold_percent} %) en el {persistence} % de {samples} cuadros.",
        recommendation="Revisar la amplitud en ataque; abrir a los extremos o laterales para estirar la "
                        "defensa rival y crear espacios por dentro.",
    ),
    IssueRule(
        code="NO_ATTACKING_DEPTH",
        issue="Posible problema táctico: falta de profundidad en ataque",
        indicator="players_ahead_of_ball",
        reliability=0.6,
        evidence_template="Con el balón, el equipo tuvo en promedio {value} jugadores por delante del balón "
                            "(umbral {threshold}) en el {persistence} % de {samples} cuadros.",
        recommendation="Revisar los desmarques de ruptura; al menos un jugador por delante del balón da una "
                        "opción de pase hacia adelante y fija a la defensa rival.",
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


@dataclass(frozen=True)
class PlainExplanation:
    """The same issue told like a coach would: what the AI saw, what it means, what to do."""
    measure: str   # what was measured, e.g. "la distancia de cada jugador a su compañero más cercano"
    moment: str    # when it is measured, e.g. "Mientras el rival tenía el balón"
    meaning: str   # what it means on the pitch
    coach: str     # what a coaching staff could work on


DEFENDING_MOMENT = "Mientras el rival tenía el balón"
PLAIN: dict[str, PlainExplanation] = {
    "defensive_midfield_distance": PlainExplanation(
        "el espacio entre la línea de defensas y la de mediocampistas", DEFENDING_MOMENT,
        "Quedó un hueco grande entre la defensa y el mediocampo. Un rival que reciba ahí puede girarse "
        "de frente al arco sin que nadie lo presione.",
        "Adelantar la línea defensiva o retrasar a un mediocampista para cerrar ese espacio entre líneas."),
    "largest_line_gap": PlainExplanation(
        "el mayor espacio entre dos líneas del equipo", DEFENDING_MOMENT,
        "El equipo quedó estirado: entre dos de sus líneas hubo un hueco grande donde el rival puede recibir solo.",
        "Acercar las líneas entre sí al defender; que la línea más cercana al hueco dé un paso para cerrarlo."),
    "team_compactness_area": PlainExplanation(
        "el área que ocupaban sus jugadores visibles", DEFENDING_MOMENT,
        "El bloque defensivo estaba abierto: los jugadores ocupaban mucho espacio y entre ellos quedaban pasillos "
        "para que el rival pase el balón.",
        "Defender más juntos: reducir distancias entre jugadores y entre líneas para cerrar los pasillos."),
    "average_player_distance": PlainExplanation(
        "la distancia media entre compañeros", DEFENDING_MOMENT,
        "Los jugadores estaban lejos unos de otros: si el rival supera a uno, el compañero está demasiado lejos para "
        "ayudar.",
        "Trabajar el bloque compacto: moverse juntos hacia el lado del balón y acortar distancias."),
    "nearest_teammate_distance": PlainExplanation(
        "la distancia de cada jugador a su compañero más cercano", DEFENDING_MOMENT,
        "Defendían con los jugadores muy separados entre sí. Si un rival supera a uno, no hay un compañero cerca para "
        "hacer la cobertura y quedan espacios para pasar entre ellos.",
        "Juntar más las líneas al defender, para que ningún jugador quede solo y siempre haya una cobertura cerca."),
    "defensive_line_position": PlainExplanation(
        "la distancia de la línea defensiva a su propio arco", "Durante el video",
        "La defensa estaba muy cerca de su arco: el equipo se hunde y deja mucho espacio delante para que el rival "
        "llegue y remate.",
        "Subir la línea defensiva unos metros, coordinada con el mediocampo, para alejar al rival del arco."),
    "max_zone_concentration": PlainExplanation(
        "la parte de sus jugadores visibles en la zona más ocupada", DEFENDING_MOMENT,
        "Muchos jugadores estaban en la misma zona. Eso deja otras zonas sin nadie y un cambio de orientación del "
        "rival las encuentra vacías.",
        "Repartir mejor la ocupación: que no vayan todos al balón y alguien vigile el lado contrario."),
    "team_width": PlainExplanation(
        "el ancho que ocupaba el equipo de banda a banda", "Mientras no estaba defendiendo",
        "El equipo jugaba por el centro sin abrir la cancha. Así el rival defiende cómodo, todo junto, porque no hay "
        "que cubrir las bandas.",
        "Abrir a los extremos o laterales pegados a la banda para estirar la defensa rival."),
    "max_zone_numerical_difference": PlainExplanation(
        "la diferencia de jugadores entre los dos equipos en un mismo tercio de la imagen", "En el partido",
        "En una zona un equipo tuvo varios jugadores más que el otro: quien está en inferioridad ahí pierde los duelos "
        "y las coberturas.",
        "Revisar en el video esa zona: equilibrar jugadores donde el rival se carga o aprovechar la superioridad."),
    "network_components": PlainExplanation(
        "el número de grupos de jugadores sin un pase corto entre ellos", DEFENDING_MOMENT,
        "El equipo estaba partido, por ejemplo la defensa por un lado y los delanteros por otro, sin nadie en medio. "
        "Al recuperar el balón no hay a quién dárselo cerca y al defender queda un hueco entre los grupos.",
        "Poner un jugador de enlace, normalmente un mediocampista, que una los grupos."),
    "isolated_players": PlainExplanation(
        "el número de jugadores sin ningún compañero cerca", DEFENDING_MOMENT,
        "Hubo jugadores solos, sin un compañero a distancia de ayuda. Un rival los puede encarar en uno contra uno sin "
        "que nadie haga la cobertura.",
        "Acercar un apoyo a esos jugadores o recolocarlos más cerca del bloque."),
    "pressure_on_ball": PlainExplanation(
        "el número de defensores cerca del jugador que llevaba el balón", "Cuando el rival tenía el balón",
        "Dejaron jugar al rival con tranquilidad: el que tenía el balón podía levantar la cabeza y elegir el pase sin "
        "que nadie lo apurara.",
        "Que el jugador más cercano salga a presionar apenas el rival recibe, sobre todo justo después de perder el "
        "balón."),
    "attacking_width": PlainExplanation(
        "el ancho del equipo mientras tenía el balón", "Cuando tenía el balón",
        "Atacaban amontonados por el centro, sin jugadores abiertos: el rival cierra el medio y el ataque se atasca.",
        "Abrir a los extremos o laterales en ataque para estirar la defensa rival y abrir espacios por dentro."),
    "players_ahead_of_ball": PlainExplanation(
        "el número de compañeros por delante del balón, hacia el arco rival", "Cuando tenía el balón",
        "Nadie estaba por delante del balón: el que lo llevaba no tenía un pase hacia adelante y el ataque solo podía "
        "ir para atrás o para los lados.",
        "Trabajar desmarques de ruptura: que al menos un jugador ataque el espacio a la espalda de la defensa."),
}

TEAM_NAMES = {"TEAM_A": "el Equipo A", "TEAM_B": "el Equipo B", "MATCH": "el partido"}
COUNT_UNITS = {"players": ("jugador", "jugadores"), "groups": ("grupo", "grupos")}


def _plain_amount(value: float, unit: str, with_reference: bool = True) -> str:
    """0.177 frame_width_ratio -> "el 17,7 % del ancho de la imagen"; 2.5 groups -> "2,5 grupos"."""
    percent = f"el {format_number(value * 100, 1)} %"
    reference = {"frame_width_ratio": " del ancho de la imagen", "frame_height_ratio": " del alto de la imagen",
                 "frame_area_ratio": " de la imagen", "player_ratio": ""}
    if unit in reference:
        return percent + (reference[unit] if with_reference else "")
    singular, plural = COUNT_UNITS.get(unit, ("", ""))
    number = format_number(value, 1)
    return f"{number} {singular if number == '1' else plural}".strip()


def _plain_limit(threshold: float, direction: str | None, unit: str, persistence: str) -> str:
    """Where the limit is and how often it was crossed, said the way a coach would say it."""
    if unit in COUNT_UNITS and direction == "below":
        least = int(threshold) + 1                      # problem at/below 0.5 -> normal: at least 1
        return f"lo normal es al menos {_plain_amount(least, unit)}, y hubo menos en el {persistence} % de esos momentos."
    amount = _plain_amount(threshold, unit, with_reference=False)
    if direction == "above":
        return f"se considera un problema desde {amount}, y lo alcanzó en el {persistence} % de esos momentos."
    below = f"del {amount.removeprefix('el ')}" if amount.startswith("el ") else f"de {amount}"
    return f"se considera un problema por debajo {below}, y estuvo así en el {persistence} % de esos momentos."


def _plain_duration(samples: int, seconds_per_frame: float | None) -> str:
    if not seconds_per_frame:
        return f"{samples} cuadros analizados"
    seconds = samples * seconds_per_frame
    if seconds < 1:
        return "menos de 1 segundo de juego"
    rounded = round(seconds)
    return f"unos {rounded} segundo{'s' if rounded != 1 else ''} de juego"


def plain_explanation(indicator: IndicatorResult, seconds_per_frame: float | None = None) -> dict | None:
    """What the AI saw (with the real numbers), what it means and what to work on, in plain Spanish."""
    plain = PLAIN.get(indicator.name)
    if plain is None or indicator.threshold is None:
        return None
    team = TEAM_NAMES.get(indicator.team, indicator.team)
    seen = (
        f"{plain.moment} ({_plain_duration(indicator.samples, seconds_per_frame)}), en {team} "
        f"{plain.measure} fue en promedio {_plain_amount(indicator.value, indicator.unit)}; "
        + _plain_limit(indicator.threshold, indicator.threshold_direction, indicator.unit,
                       format_number(indicator.exceed_ratio * 100, 0))
    )
    if indicator.team == "MATCH":
        seen = seen.replace(f"en {team} ", "")
    return {"seen": seen[0].upper() + seen[1:], "meaning": plain.meaning, "coach": plain.coach}


class TacticalInsightGenerator:
    """Applies the issue rules and builds explainable recommendations."""

    def __init__(self, rules: list[IssueRule] | None = None) -> None:
        self._rules = rules if rules is not None else ISSUE_RULES

    def generate(self, indicators: list[IndicatorResult], team_separation: float,
                 seconds_per_frame: float | None = None) -> tuple[list[dict], list[dict]]:
        """Applies every rule; returns (possible issues, recommendations), most confident first.

        ``seconds_per_frame`` (frame stride / fps) turns analysed frames into seconds of play in the
        plain-language explanation."""
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
                ) + PHASE_NOTE.get(indicator.phase or "", "")
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
                    "explanation": plain_explanation(indicator, seconds_per_frame),
                })

        possible_issues.sort(key=lambda item: item["confidence"], reverse=True)
        recommendations.sort(key=lambda item: item["confidence"], reverse=True)
        return possible_issues, recommendations
