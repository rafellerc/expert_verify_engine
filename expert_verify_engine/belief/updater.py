from expert_verify_engine.belief.belief_state import BeliefState
from expert_verify_engine.models.schemas import EvidencePacket


def update_belief(belief: BeliefState, evidence: EvidencePacket) -> BeliefState:
    for comp_name, comp_evidence in evidence.competence.items():
        # e_plus = max(0.0, min(1.0, comp_evidence.e_plus))
        # e_minus = max(0.0, min(1.0, comp_evidence.e_minus))
        # TODO: Decide on Clipping strategy for evidence values
        e_plus = comp_evidence.e_plus
        e_minus = comp_evidence.e_minus
        belief.update(comp_name, e_plus, e_minus)
    return belief


def compute_decision(
    belief: BeliefState, weights: dict[str, float], threshold: float
) -> tuple[bool, float]:
    score = 0.0
    for comp, weight in weights.items():
        prob = belief.probability(comp)
        score += weight * prob
    accepted = score > threshold
    return accepted, score
