from pydantic import BaseModel, Field


class Competence(BaseModel):
    name: str
    weight: float


class CompetenceModel(BaseModel):
    competences: list[Competence]


class CandidateProfile(BaseModel):
    competences: dict[str, int]
    fraud_strategy: str
    linguistic_profile: str


class CandidateSheet(BaseModel):
    summary: str
    experiences: list[str]
    claims: list[str]


class Action(BaseModel):
    question: str
    target_competences: list[str]
    type: str


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
