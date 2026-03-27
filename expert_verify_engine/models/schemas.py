from enum import StrEnum

from pydantic import BaseModel, Field


class ActionSelectionMode(StrEnum):
    INFORMATION_GAIN_GREEDY = "information_gain_greedy"
    INFORMATION_GAIN_SAMPLED = "information_gain_sampled"


class Competence(BaseModel):
    name: str
    weight: float


class CompetenceModel(BaseModel):
    competences: list[Competence]


CandidateDescription = str


class CandidateSheet(BaseModel):
    summary: str
    experiences: list[str]
    claims: list[str]


class Action(BaseModel):
    question: str
    target_competences: list[str]
    type: str
    action_selection: ActionSelectionMode = ActionSelectionMode.INFORMATION_GAIN_GREEDY


class EvidenceCompetence(BaseModel):
    e_plus: float
    e_minus: float


class EvidencePacket(BaseModel):
    competence: dict[str, EvidenceCompetence]
    behavior: dict[str, float]
    notes: str


class DecisionResult(BaseModel):
    accepted: bool
    score: float
    threshold: float


class TerminationDecision(BaseModel):
    continue_: bool = Field(validation_alias="continue")
    reason: str
