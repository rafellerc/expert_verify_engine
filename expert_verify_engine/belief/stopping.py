from __future__ import annotations

from dataclasses import dataclass

from expert_verify_engine.belief.decision_stats import (
    compute_decision_stats,
    compute_entropy,
    compute_p_accept_gaussian,
    compute_z_score,
)


@dataclass
class StoppingCriteria:
    epsilon: float = 0.05
    tau: float = 0.1
    z_threshold: float = 2.0
    delta: float = 0.001


def should_stop(
    alpha_beta: dict[str, tuple[float, float]],
    weights: dict[str, float],
    threshold: float,
    criteria: StoppingCriteria | None = None,
) -> tuple[bool, str]:
    """Check all stopping criteria, return (should_stop, reason)."""
    if criteria is None:
        criteria = StoppingCriteria()

    p_accept = compute_p_accept_gaussian(alpha_beta, weights, threshold)
    z = compute_z_score(alpha_beta, weights, threshold)
    entropy = compute_entropy(p_accept)

    if p_accept > (1 - criteria.epsilon):
        return True, f"P(Accept)={p_accept:.3f} > {1 - criteria.epsilon:.2f}"
    if p_accept < criteria.epsilon:
        return True, f"P(Accept)={p_accept:.3f} < {criteria.epsilon:.2f}"

    if abs(z) > criteria.z_threshold:
        return True, f"|Z|={abs(z):.2f} > {criteria.z_threshold:.1f}"

    if entropy < criteria.tau:
        return True, f"Entropy={entropy:.3f} < {criteria.tau:.2f}"

    stats = compute_decision_stats(alpha_beta, weights, threshold)
    if stats.max_ig < criteria.delta:
        return True, f"max(IG)={stats.max_ig:.4f} < {criteria.delta:.4f}"

    return False, "Continue"


def get_stop_reasons(
    alpha_beta: dict[str, tuple[float, float]],
    weights: dict[str, float],
    threshold: float,
    criteria: StoppingCriteria | None = None,
) -> dict[str, bool]:
    """Get status of all stopping criteria."""
    if criteria is None:
        criteria = StoppingCriteria()

    p_accept = compute_p_accept_gaussian(alpha_beta, weights, threshold)
    z = compute_z_score(alpha_beta, weights, threshold)
    entropy = compute_entropy(p_accept)
    stats = compute_decision_stats(alpha_beta, weights, threshold)

    return {
        "p_accept_high": p_accept > (1 - criteria.epsilon),
        "p_accept_low": p_accept < criteria.epsilon,
        "z_threshold": abs(z) > criteria.z_threshold,
        "entropy_low": entropy < criteria.tau,
        "voi_low": stats.max_ig < criteria.delta,
    }
