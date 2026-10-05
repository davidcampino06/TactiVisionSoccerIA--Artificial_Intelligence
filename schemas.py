"""Pydantic models that document and validate the AI Service API contract."""

from typing import Any, Literal
from pydantic import BaseModel, Field

class HealthResponse(BaseModel):
    service: str
    status: str

class ModelInfoResponse(BaseModel):
    model: str
    runtime: str
    tracker: str
    loaded: bool
    model_version: str | None = None
    family: str | None = None
    description: str | None = None
    load_error: str | None = None
    supported_models: list[str]
    detected_classes: dict[str, str] = Field(default_factory=dict)

class TacticalIndicatorOut(BaseModel):
    name: str
    indicator: str
    team: str
    value: float
    unit: str
    threshold: float | None
    threshold_direction: str | None
    exceed_ratio: float
    min: float
    max: float
    std: float
    samples: int
    description: str

class PossibleIssueOut(BaseModel):
    id: str
    code: str
    issue: str
    team: str
    evidence: str
    confidence: float = Field(ge=0, le=1)
    confidence_breakdown: dict[str, float]
    severity: Literal["LOW", "MEDIUM", "HIGH"]
    related_indicators: list[str]

class RecommendationOut(BaseModel):
    issue_id: str
    title: str
    team: str
    recommendation: str
    evidence: str
    confidence: float = Field(ge=0, le=1)
    severity: Literal["LOW", "MEDIUM", "HIGH"]
    related_indicators: list[str]

class AnalysisResponse(BaseModel):
    status: Literal["COMPLETED"]
    mode: Literal["REAL_VIDEO_ANALYSIS", "SIMULATION_MODE"]
    model: str
    model_version: str
    runtime: str
    tracker: str
    video: dict[str, Any]
    parameters: dict[str, Any]
    pitch_visible: bool = True
    frames_processed: int
    players_detected: int
    average_players_per_frame: float
    unique_player_tracks: int
    ball_detected: bool
    ball_detection_rate: float
    teams: dict[str, Any]
    performance: dict[str, float]
    detection_stability: dict[str, float]
    detections: list[dict[str, Any]]
    tracking: list[dict[str, Any]]
    tactical_indicators: list[TacticalIndicatorOut]
    possible_issues: list[PossibleIssueOut]
    recommendations: list[RecommendationOut]
    warnings: list[str]
