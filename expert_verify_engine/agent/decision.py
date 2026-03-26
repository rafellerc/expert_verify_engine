from expert_verify_engine.app.config import get_config
from expert_verify_engine.belief.belief_state import BeliefState
from expert_verify_engine.belief.decision_stats import (
    DecisionStats,
    compute_decision_stats,
)
from expert_verify_engine.models.schemas import DecisionResult


def compute_decision(
    belief: BeliefState,
    weights: dict[str, float],
    use_stats: bool = True,
) -> tuple[DecisionResult, DecisionStats | None]:
    threshold = get_config("threshold")
    score = 0.0
    for comp, weight in weights.items():
        prob = belief.probability(comp)
        score += weight * prob

    if use_stats:
        alpha_beta = belief.alpha_beta
        mc_samples = get_config("mc_samples", 10000)
        e_plus = get_config("e_plus", 0.5)
        e_minus = get_config("e_minus", 0.5)

        stats = compute_decision_stats(
            alpha_beta,
            weights,
            threshold,
            e_plus=e_plus,
            e_minus=e_minus,
            mc_samples=mc_samples,
        )

        accepted = stats.p_accept > 0.5
        return DecisionResult(
            accepted=accepted, score=score, threshold=threshold
        ), stats

    accepted = score > threshold
    return DecisionResult(accepted=accepted, score=score, threshold=threshold), None


def compute_decision_legacy(
    belief: BeliefState,
    weights: dict[str, float],
) -> DecisionResult:
    threshold = get_config("threshold")
    score = 0.0
    for comp, weight in weights.items():
        prob = belief.probability(comp)
        score += weight * prob

    accepted = score > threshold
    return DecisionResult(accepted=accepted, score=score, threshold=threshold)
