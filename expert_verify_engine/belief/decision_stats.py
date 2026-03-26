from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.stats import norm


def compute_mean_var(alpha: float, beta: float) -> tuple[float, float]:
    """Compute mean and variance of Beta distribution."""
    if alpha + beta <= 0:
        return 0.5, 0.25
    mu = alpha / (alpha + beta)
    var = (alpha * beta) / ((alpha + beta) ** 2 * (alpha + beta + 1))
    return mu, var


def compute_p_accept_gaussian(
    alpha_beta: dict[str, tuple[float, float]],
    weights: dict[str, float],
    threshold: float,
) -> float:
    """Gaussian approximation: Φ((μ_S - T) / √var_S)."""
    if not alpha_beta:
        return 0.5

    mu_s = 0.0
    var_s = 0.0

    for comp, (alpha, beta) in alpha_beta.items():
        mu, var = compute_mean_var(alpha, beta)
        w = weights.get(comp, 0)
        mu_s += w * mu
        var_s += (w**2) * var

    z = (mu_s - threshold) / np.sqrt(var_s + 1e-8)
    return norm.cdf(z)


def compute_z_score(
    alpha_beta: dict[str, tuple[float, float]],
    weights: dict[str, float],
    threshold: float,
) -> float:
    """Z-score: (μ_S - T) / √var_S."""
    if not alpha_beta:
        return 0.0

    mu_s = 0.0
    var_s = 0.0

    for comp, (alpha, beta) in alpha_beta.items():
        mu, var = compute_mean_var(alpha, beta)
        w = weights.get(comp, 0)
        mu_s += w * mu
        var_s += (w**2) * var

    return (mu_s - threshold) / np.sqrt(var_s + 1e-8)


def compute_entropy(p: float) -> float:
    """Binary entropy: -p log p - (1-p) log(1-p)."""
    if p <= 0 or p >= 1:
        return 0.0
    return -p * np.log(p) - (1 - p) * np.log(1 - p)


def compute_p_accept_mc(
    alpha_beta: dict[str, tuple[float, float]],
    weights: dict[str, float],
    threshold: float,
    n_samples: int = 10000,
    seed: int | None = None,
) -> float:
    """Monte Carlo estimation via Beta sampling."""
    if not alpha_beta:
        return 0.5

    if seed is not None:
        np.random.seed(seed)

    competences = list(alpha_beta.keys())
    alphas = np.array([alpha_beta[c][0] for c in competences])
    betas = np.array([alpha_beta[c][1] for c in competences])

    samples = np.random.beta(alphas, betas, size=(n_samples, len(competences)))

    w = np.array([weights.get(c, 0) for c in competences])
    scores = samples @ w

    return np.mean(scores >= threshold)


@dataclass
class DecisionStats:
    p_accept: float
    p_accept_mc: float
    z_score: float
    entropy: float
    max_ig: float
    ig_per_competence: dict[str, float]
    mu_s: float
    var_s: float


def compute_decision_stats(
    alpha_beta: dict[str, tuple[float, float]],
    weights: dict[str, float],
    threshold: float,
    e_plus: float = 0.5,
    e_minus: float = 0.5,
    mc_samples: int = 10000,
) -> DecisionStats:
    """Compute all decision statistics."""
    p_accept = compute_p_accept_gaussian(alpha_beta, weights, threshold)
    z_score = compute_z_score(alpha_beta, weights, threshold)
    entropy = compute_entropy(p_accept)

    mu_s = 0.0
    var_s = 0.0
    for comp, (alpha, beta) in alpha_beta.items():
        mu, var = compute_mean_var(alpha, beta)
        w = weights.get(comp, 0)
        mu_s += w * mu
        var_s += (w**2) * var

    ig_per_competence = _compute_all_ig(alpha_beta, weights, threshold, e_plus, e_minus)
    max_ig = max(ig_per_competence.values()) if ig_per_competence else 0.0

    p_accept_mc = compute_p_accept_mc(alpha_beta, weights, threshold, mc_samples)

    return DecisionStats(
        p_accept=p_accept,
        p_accept_mc=p_accept_mc,
        z_score=z_score,
        entropy=entropy,
        max_ig=max_ig,
        ig_per_competence=ig_per_competence,
        mu_s=mu_s,
        var_s=var_s,
    )


def _compute_information_gain(
    alpha_beta: dict[str, tuple[float, float]],
    weights: dict[str, float],
    threshold: float,
    competence: str,
    e_plus: float,
    e_minus: float,
) -> float:
    """Compute expected information gain from testing a competence."""
    if competence not in alpha_beta:
        return 0.0

    alpha, beta = alpha_beta[competence]
    mu = alpha / (alpha + beta)

    p_accept = compute_p_accept_gaussian(alpha_beta, weights, threshold)
    h_before = compute_entropy(p_accept)

    alpha_pos = alpha + e_plus
    beta_pos = beta
    alpha_beta_pos = {**alpha_beta, competence: (alpha_pos, beta_pos)}
    p_accept_pos = compute_p_accept_gaussian(alpha_beta_pos, weights, threshold)
    h_pos = compute_entropy(p_accept_pos)

    alpha_neg = alpha
    beta_neg = beta + e_minus
    alpha_beta_neg = {**alpha_beta, competence: (alpha_neg, beta_neg)}
    p_accept_neg = compute_p_accept_gaussian(alpha_beta_neg, weights, threshold)
    h_neg = compute_entropy(p_accept_neg)

    h_after = mu * h_pos + (1 - mu) * h_neg

    return h_before - h_after


def _compute_all_ig(
    alpha_beta: dict[str, tuple[float, float]],
    weights: dict[str, float],
    threshold: float,
    e_plus: float,
    e_minus: float,
) -> dict[str, float]:
    """Compute IG for all competences."""
    return {
        comp: _compute_information_gain(
            alpha_beta, weights, threshold, comp, e_plus, e_minus
        )
        for comp in alpha_beta
    }


def select_best_competence(
    alpha_beta: dict[str, tuple[float, float]],
    weights: dict[str, float],
    threshold: float,
    e_plus: float = 0.5,
    e_minus: float = 0.5,
) -> str | None:
    """Select competence with highest information gain."""
    all_ig = _compute_all_ig(alpha_beta, weights, threshold, e_plus, e_minus)
    if not all_ig:
        return None
    return max(all_ig, key=all_ig.get)
