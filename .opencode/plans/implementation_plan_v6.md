# Implementation Plan - v6 Decision Engine

## Phases Overview

| Phase | Description | Files to Create/Modify |
|-------|-------------|------------------------|
| 1 | Decision Statistics (P(Accept), Z-score, Entropy) | New: `belief/decision_stats.py` |
| 2 | Information Gain for question selection | New: `belief/selector.py` |
| 3 | Statistical stopping criteria | New: `belief/stopping.py` |
| 4 | Monte Carlo P(Accept) estimation | Update: `belief/decision_stats.py` |

---

## Phase 1: Decision Statistics

### 1.1 Create `belief/decision_stats.py`

**New file** with functions:

```python
# belief/decision_stats.py

import numpy as np
from scipy.stats import norm

def compute_mean_var(alpha: float, beta: float) -> tuple[float, float]:
    """Compute mean and variance of Beta(alpha, beta)."""
    mu = alpha / (alpha + beta)
    var = (alpha * beta) / ((alpha + beta)**2 * (alpha + beta + 1))
    return mu, var

def compute_p_accept_gaussian(
    alpha_beta: dict[str, tuple[float, float]],
    weights: dict[str, float],
    threshold: float,
) -> float:
    """Gaussian approximation: Φ((μ_S - T) / √var_S)."""
    mu_s = 0.0
    var_s = 0.0
    for comp, (alpha, beta) in alpha_beta.items():
        mu, var = compute_mean_var(alpha, beta)
        w = weights.get(comp, 0)
        mu_s += w * mu
        var_s += (w ** 2) * var
    z = (mu_s - threshold) / np.sqrt(var_s + 1e-8)
    return norm.cdf(z)

def compute_z_score(
    alpha_beta: dict[str, tuple[float, float]],
    weights: dict[str, float],
    threshold: float,
) -> float:
    """Z-score: (μ_S - T) / √var_S."""
    mu_s = 0.0
    var_s = 0.0
    for comp, (alpha, beta) in alpha_beta.items():
        mu, var = compute_mean_var(alpha, beta)
        w = weights.get(comp, 0)
        mu_s += w * mu
        var_s += (w ** 2) * var
    return (mu_s - threshold) / np.sqrt(var_s + 1e-8)

def compute_entropy(p: float) -> float:
    """Binary entropy: -p log p - (1-p) log(1-p)."""
    if p <= 0 or p >= 1:
        return 0.0
    return -p * np.log(p) - (1 - p) * np.log(1 - p)
```

### 1.2 Update `belief/__init__.py`

Add exports for new functions.

### 1.3 Update `agent/decision.py`

Add `DecisionStats` dataclass and `compute_decision_stats()` function.

---

## Phase 2: Information Gain

### 2.1 Create `belief/selector.py`

**New file**:

```python
# belief/selector.py

import numpy as np
from scipy.stats import norm
from belief.decision_stats import (
    compute_mean_var,
    compute_p_accept_gaussian,
    compute_entropy,
)

def compute_information_gain(
    alpha_beta: dict[str, tuple[float, float]],
    weights: dict[str, float],
    threshold: float,
    competence: str,
    e_plus: float = 0.5,
    e_minus: float = 0.5,
) -> float:
    """Compute expected information gain from testing a competence."""
    if competence not in alpha_beta:
        return 0.0
    
    alpha, beta = alpha_beta[competence]
    mu = alpha / (alpha + beta)
    
    # Current P(Accept) and entropy
    p_accept = compute_p_accept_gaussian(alpha_beta, weights, threshold)
    h_before = compute_entropy(p_accept)
    
    # Simulate positive update
    alpha_pos = alpha + e_plus
    beta_pos = beta
    alpha_beta_pos = {**alpha_beta, competence: (alpha_pos, beta_pos)}
    p_accept_pos = compute_p_accept_gaussian(alpha_beta_pos, weights, threshold)
    h_pos = compute_entropy(p_accept_pos)
    
    # Simulate negative update
    alpha_neg = alpha
    beta_neg = beta + e_minus
    alpha_beta_neg = {**alpha_beta, competence: (alpha_neg, beta_neg)}
    p_accept_neg = compute_p_accept_gaussian(alpha_beta_neg, weights, threshold)
    h_neg = compute_entropy(p_accept_neg)
    
    # Expected entropy after observing this competence
    h_after = mu * h_pos + (1 - mu) * h_neg
    
    return h_before - h_after

def select_best_competence(
    alpha_beta: dict[str, tuple[float, float]],
    weights: dict[str, float],
    threshold: float,
    e_plus: float = 0.5,
    e_minus: float = 0.5,
) -> str | None:
    """Select competence with highest information gain."""
    best_comp = None
    best_ig = -1.0
    
    for comp in alpha_beta:
        ig = compute_information_gain(
            alpha_beta, weights, threshold, comp, e_plus, e_minus
        )
        if ig > best_ig:
            best_ig = ig
            best_comp = comp
    
    return best_comp

def get_all_ig(
    alpha_beta: dict[str, tuple[float, float]],
    weights: dict[str, float],
    threshold: float,
    e_plus: float = 0.5,
    e_minus: float = 0.5,
) -> dict[str, float]:
    """Get IG for all competences."""
    return {
        comp: compute_information_gain(
            alpha_beta, weights, threshold, comp, e_plus, e_minus
        )
        for comp in alpha_beta
    }
```

### 2.2 Update `belief/__init__.py`

Add exports for selector functions.

---

## Phase 3: Statistical Stopping

### 3.1 Create `belief/stopping.py`

**New file**:

```python
# belief/stopping.py

from dataclasses import dataclass

from belief.decision_stats import (
    compute_p_accept_gaussian,
    compute_z_score,
    compute_entropy,
    compute_mean_var,
)
from belief.selector import get_all_ig

@dataclass
class StoppingCriteria:
    epsilon: float = 0.05
    tau: float = 0.1
    z_threshold: float = 2.0
    delta: float = 0.01


def should_stop(
    alpha_beta: dict[str, tuple[float, float]],
    weights: dict[str, float],
    threshold: float,
    criteria: StoppingCriteria,
) -> tuple[bool, str]:
    """Check all stopping criteria, return (should_stop, reason)."""
    
    p_accept = compute_p_accept_gaussian(alpha_beta, weights, threshold)
    z = compute_z_score(alpha_beta, weights, threshold)
    entropy = compute_entropy(p_accept)
    
    # Criterion 1: Probability threshold
    if p_accept > (1 - criteria.epsilon):
        return True, f"P(Accept)={p_accept:.3f} > {1-criteria.epsilon}"
    if p_accept < criteria.epsilon:
        return True, f"P(Accept)={p_accept:.3f} < {criteria.epsilon}"
    
    # Criterion 2: Z-score threshold
    if abs(z) > criteria.z_threshold:
        return True, f"|Z|={abs(z):.2f} > {criteria.z_threshold}"
    
    # Criterion 3: Entropy threshold
    if entropy < criteria.tau:
        return True, f"Entropy={entropy:.3f} < {criteria.tau}"
    
    # Criterion 4: Value of information
    all_ig = get_all_ig(alpha_beta, weights, threshold)
    max_ig = max(all_ig.values()) if all_ig else 0.0
    if max_ig < criteria.delta:
        return True, f"max(IG)={max_ig:.4f} < {criteria.delta}"
    
    return False, "Continue"
```

### 3.2 Update `belief/__init__.py`

Add exports for stopping module.

---

## Phase 4: Monte Carlo Estimation

### 4.1 Add to `belief/decision_stats.py`

```python
def compute_p_accept_mc(
    alpha_beta: dict[str, tuple[float, float]],
    weights: dict[str, float],
    threshold: float,
    n_samples: int = 10000,
    seed: int | None = None,
) -> float:
    """Monte Carlo estimation via Beta sampling."""
    if seed is not None:
        np.random.seed(seed)
    
    competences = list(alpha_beta.keys())
    alphas = np.array([alpha_beta[c][0] for c in competences])
    betas = np.array([alpha_beta[c][1] for c in competences])
    
    samples = np.random.beta(alphas, betas, size=(n_samples, len(competences)))
    
    w = np.array([weights.get(c, 0) for c in competences])
    scores = samples @ w
    
    return np.mean(scores >= threshold)
```

### 4.2 Update config schema

Add `p_accept_method` and `mc_samples` to config.

---

## Integration Steps

### Update `app/main.py`

1. Import new modules
2. In `run_interview()` loop:
   - After belief update: compute decision stats
   - Display stats in console
   - Check statistical stopping before LLM termination
3. If `use_ig_selection=True`: use IG selector instead of LLM-generated targets

### Update `app/config.py`

Add new config defaults:

```python
CONFIG = {
    # ... existing ...
    "p_accept_method": "gaussian",
    "mc_samples": 10000,
    "epsilon": 0.05,
    "tau": 0.1,
    "z_threshold": 2.0,
    "delta": 0.01,
    "e_plus": 0.5,
    "e_minus": 0.5,
    "use_llm_termination": True,
    "use_ig_selection": False,  # Start with False, can enable
}
```

---

## Testing

### Unit Tests

Create `tests/test_decision_stats.py`:

- `test_compute_mean_var()` - Verify Beta moments
- `test_compute_p_accept_gaussian()` - Compare to known values
- `test_compute_p_accept_mc()` - Should match Gaussian within 5%
- `test_compute_entropy()` - Edge cases (0, 1, 0.5)
- `test_information_gain()` - IG >= 0 always
- `test_select_best_competence()` - Returns valid competence
- `test_should_stop_*()` - Each criterion

---

## File Changes Summary

| Action | File | Change |
|--------|------|--------|
| Create | `belief/decision_stats.py` | New - stats functions |
| Create | `belief/selector.py` | New - IG selection |
| Create | `belief/stopping.py` | New - stopping criteria |
| Modify | `belief/__init__.py` | Add exports |
| Modify | `agent/decision.py` | Add DecisionStats |
| Modify | `app/config.py` | Add new config keys |
| Modify | `app/main.py` | Integrate into loop |
| Create | `tests/test_decision_stats.py` | Unit tests |
