from expert_verify_engine.models.schemas import Competence


def normalize_competences(competences: list[Competence]) -> list[Competence]:
    total = sum(c.weight for c in competences)
    if total == 0:
        return competences
    return [Competence(name=c.name, weight=c.weight / total) for c in competences]
