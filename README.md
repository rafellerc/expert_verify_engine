# Expert Verify Engine

An LLM-driven candidate evaluation system using Partially Observable Markov Decision Processes (POMDP) with structured latent state estimation.

## Core Concept

The system estimates **P(Accept | history)** — the probability that a candidate should be accepted based on interview interactions — through Bayesian belief accumulation.

### The Belief Update Equation

```
Belief_{t+1} = Belief_t + Evidence_t
```

More precisely, each competence is modeled as a **Beta distribution**:

- α: positive evidence count
- β: negative evidence count

Probability estimate:
```
p_i = α_i / (α_i + β_i)
```

Evidence updates:
```
α_i ← α_i + e⁺_i
β_i ← β_i + e⁻_i
```

Where `e⁺` and `e⁻` are extracted from the LLM observation model.

---

## Decision Engine (v6)

The system now computes **probabilistic decision metrics** for more robust decision-making.

### 1. Belief Statistics

For each competence i:
```
μ_i = α_i / (α_i + β_i)                              # Mean
var_i = (α_i * β_i) / ((α_i + β_i)² * (α_i + β_i + 1))  # Variance
```

### 2. P(Accept) - Gaussian Approximation

Weighted sum statistics:
```
μ_S = Σ w_i * μ_i                                    # Weighted mean
var_S = Σ w_i² * var_i                               # Weighted variance
Z = (μ_S - T) / √var_S                               # Z-score
P(Accept) ≈ Φ(Z)                                     # Gaussian CDF
```

### 3. P(Accept) - Monte Carlo Alternative

```
Sample p_i ~ Beta(α_i, β_i) for all competences
Compute S = Σ w_i * p_i
P(Accept) ≈ mean(S >= T)                            # Empirical
```

### 4. Decision Entropy

Uncertainty in the decision:
```
H(p) = -p log₂(p) - (1-p) log₂(1-p)
```

### 5. Information Gain (Question Selection)

For each competence, compute expected information gain:
```
IG_i = H_before - E[H_after | test competence i]

Where:
- H_before = entropy of current P(Accept)
- Simulate: α' = α + e⁺, β' = β + e⁻ for both outcomes
- E[H_after] = μ_i * H_pos + (1-μ_i) * H_neg
```

The system selects the competence with highest IG as the target for the next question.
Two modes are available:
- **Greedy** (default): Always selects the competence with maximum IG
- **Sampled**: Samples a competence proportionally to normalized IG weights

### 6. Statistical Stopping Criteria

Terminate when any criterion is met:
```
1. P(Accept) > 1 - ε     (certain accept)
2. P(Accept) < ε         (certain reject)
3. |Z| > Z_threshold     (confident decision)
4. H(p) < tau            (low uncertainty)
5. max(IG_i) < delta     (low value of information)
```

Default thresholds: ε=0.05, τ=0.1, Z_threshold=2.0, δ=0.01

### 7. The Key Insight

```
Z = (signal - threshold) / uncertainty
```

The goal is to **move Z away from 0** — increase decision confidence.

---

## System Architecture

### Components

| Component | Purpose |
|-----------|---------|
| `models/schemas.py` | Pydantic data models |
| `belief/` | Beta distribution belief state, decision stats |
| `agent/policy.py` | Question generation & termination |
| `agent/decision.py` | Acceptance logic with stats |
| `observation/evaluator.py` | LLM observation → evidence extraction |
| `llm/client.py` | OpenRouter API client |
| `audit_log/` | Trajectory persistence |

### Data Models

**CandidateProfile** (latent ground truth):
```python
{
    "competences": {"Python": 1, "Machine Learning": 1},  # 0/1 latent truth
    "fraud_strategy": "honest",  # or "cheater"
    "linguistic_profile": "simple"  # linguistic characteristics
}
```

**CandidateSheet** (observable profile):
```python
{
    "summary": "2-3 sentence description",
    "experiences": ["list of experiences"],
    "claims": ["specific claims"]
}
```

**CompetenceModel** (what we're evaluating):
```python
{
    "competences": [
        {"name": "Addition", "weight": 0.25},
        {"name": "Subtraction", "weight": 0.25},
        ...
    ]
}
```

**EvidencePacket** (LLM-extracted evidence):
```python
{
    "competence": {"Addition": {"e_plus": 0.8, "e_minus": 0.1}},
    "behavior": {"guessing": 0.2},  # or "cheating" for professional mode
    "notes": "partial understanding shown"
}
```

**DecisionStats** (v6 new):
```python
{
    "p_accept": 0.78,          # Gaussian P(Accept)
    "p_accept_mc": 0.76,       # Monte Carlo P(Accept)
    "z_score": 1.85,           # Z-score
    "entropy": 0.64,           # Decision entropy
    "max_ig": 0.12,            # Max information gain
    "ig_per_competence": {...} # IG per competence
}
```

---

## Decision Model

### v6 (Current)

Uses probabilistic P(Accept):
```
P(Accept) = Φ((μ_S - T) / √var_S)
accept = P(Accept) > 0.5
```

### v5 (Legacy)

Deterministic weighted sum:
```
score = Σ w_i * p_i
accept = score > threshold
```

---

## Running an Interview

### Prerequisites

```bash
# Install dependencies
pip install -e .

# Set OpenRouter API key
export OPENROUTER_API_KEY="your-key-here"
```

### Basic Usage

```bash
# Start an interview with a role description
python -m app.main start example_role_descriptions/arithmetic_operations.txt

# Provide answers interactively
# Use /quit to exit without decision
# Use /end to end and get decision
# Use /resample to get a new question
```

### Full Example

```bash
# Run 5 questions then end
echo -e "75\n47\n72\n7\n/end" | python -m app.main start example_role_descriptions/arithmetic_operations.txt
```

### Other Commands

```bash
# Fork from a specific turn
python -m app.main fork <run_id> <turn_idx>

# List all runs
python -m app.main list-runs
```

---

## Configuration

Edit `app/config.py` to customize:

| Option | Default | Description |
|--------|---------|-------------|
| `model` | `arcee-ai/trinity-large-preview:free` | LLM model |
| `threshold` | `0.7` | Acceptance threshold |
| `max_steps` | `10` | Maximum questions |
| `temperature` | `0.2` | LLM temperature |
| `evidence_mode` | `raw` | Evidence extraction mode |
| **v6 Options** | | |
| `p_accept_method` | `gaussian` | P(Accept) method |
| `mc_samples` | `10000` | Monte Carlo samples |
| `epsilon` | `0.05` | P(Accept) stop threshold |
| `tau` | `0.1` | Entropy stop threshold |
| `z_threshold` | `2.0` | Z-score stop threshold |
| `delta` | `0.001` | Max IG stop threshold |
| `e_plus` | `0.5` | Positive evidence weight |
| `e_minus` | `0.5` | Negative evidence weight |
| `use_llm_termination` | `False` | Fallback to LLM termination |
| `action_selection_mode` | `"information_gain_greedy"` | Question selection mode: `"information_gain_greedy"` or `"information_gain_sampled"` |
| `use_stats_decision` | `True` | Use P(Accept) for decision |

---

## Prompt Modes

The system supports two prompt modes based on `role_type` in the role description:

### Student Mode
- Lenient scoring (e_plus + e_minus ~ 0.2)
- Question types: `recall`, `application`, `practice`
- Behavior: `guessing` detection

### Professional Mode
- Strict scoring (e_plus + e_minus ~ 0.5)
- Question types: `technical`, `behavioral`, `probing`
- Behavior: `cheating` detection

---

## Output

Interview data is saved to `output_dir/<run_id>/`:

```
output_dir/<run_id>/
├── competence_model.json
├── candidate_sheet.json
└── trajectory/
    └── <run_id>_trajectory.json
```

The trajectory JSON contains:
- All questions and answers
- Evidence extracted per turn
- Belief state after each turn
- Decision statistics (v6)
- Final decision and explanation

---

## Design Choices

### Why Beta Distributions?
- Conjugate prior for Bernoulli likelihood
- Naturally handles sequential evidence accumulation
- Provides uncertainty quantification (variance decreases with more evidence)

### Why POMDP?
- Candidate's true competence is latent (unobserved)
- Only observable through Q&A behavior
- Belief state maintains probability distribution over latent variables

### Why LLM for Evidence Extraction?
- Can understand nuanced responses
- Can detect partial understanding
- Can identify behavioral signals (guessing, cheating)

### Why Statistical Decision Making?
- Provides calibrated probability estimates instead of arbitrary scores
- Quantifies decision uncertainty explicitly
- Enables principled stopping criteria
- Information Gain enables optimal question selection