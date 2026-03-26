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

## System Architecture

### Components

| Component | Purpose |
|-----------|---------|
| `models/schemas.py` | Pydantic data models |
| `belief/` | Beta distribution belief state |
| `agent/policy.py` | Question generation & termination |
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

---

## Decision Model

The acceptance decision is deterministic:

```
score = Σ w_i * p_i
accept = score > threshold
```

Default threshold: **0.7**

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
