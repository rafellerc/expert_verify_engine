from pydantic import BaseModel


class Competence(BaseModel):
    name: str
    weight: float


class CompetenceModel(BaseModel):
    competences: list[Competence]


class CandidateModel(BaseModel):
    competences: dict[str, int]
    behavior: str
    persona: str


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
