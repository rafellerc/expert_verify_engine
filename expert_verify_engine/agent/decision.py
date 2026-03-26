from expert_verify_engine.app.config import get_config
from expert_verify_engine.belief.belief_state import BeliefState
from expert_verify_engine.models.schemas import DecisionResult


def compute_decision(belief: BeliefState, weights: dict[str, float]) -> DecisionResult:
    threshold = get_config("threshold")
    score = 0.0
    for comp, weight in weights.items():
        prob = belief.probability(comp)
        score += weight * prob

    accepted = score > threshold
    return DecisionResult(accepted=accepted, score=score, threshold=threshold)
