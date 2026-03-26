# Implementation Plan - Expert Verify Engine

## Overview

LLM-driven candidate evaluation using POMDP with structured latent state.
Estimates P(Accept | history) through Bayesian belief accumulation.

## Design Decisions

| Decision | Choice |
|----------|--------|
| LLM Provider | OpenRouter (`os.getenv("OPENROUTER_API_KEY")`) |
| Role Description Input | Text file path (passed as CLI argument) |
| Candidate Input | Pre-generated profiles (JSON files) |
| Evidence Mode | Raw only (no factorized) |
| Testing | Unit logic tests (belief updates, schema validation) |

## CLI Usage

```bash
cd expert_verify_engine
python -m app.main role_description.txt
```

## File Inputs

```
role_description.txt    # Input: role description text file
candidates/            # Pre-generated candidate profiles (JSON)
```

---

## Phase 1: Core Infrastructure

### 1.1 LLM Client
- OpenRouter wrapper with retry logic
- Uses `httpx` for API calls
- API key from `os.getenv("OPENROUTER_API_KEY")`

### 1.2 Pydantic Schemas
- `CompetenceModel` - List of competences with weights
- `CandidateSheet` - Summary, experiences, claims
- `Action` - Question, target competences, type
- `EvidencePacket` - Competence evidence, behavior, notes

### 1.3 Config System
- Centralized in `app/config.py`
- Model, evidence_mode, threshold, etc.

### 1.4 Competence Generator
- LLM prompt to extract competences from role description file
- Output: CompetenceModel (JSON)

### 1.5 Candidate Generator
- LLM prompt to generate candidate sheet from ground truth
- Input: Pre-defined candidate model (JSON)
- Output: CandidateSheet

### 1.6 Belief State
- Beta distribution per competence
- Initial: α=1, β=1 (uninformative prior)
- Store as dict: `{competence_name: (alpha, beta)}`

### 1.7 Belief Updater
- Deterministic: α += e_plus, β += e_minus
- Calculate probability: p = α / (α + β)

---

## Phase 2: Agent Loop

### 2.1 Action Generator
- LLM prompt to generate next question
- Inputs: competence model, candidate sheet, belief state, conversation history
- Output: Action (question, target_competences, type)

### 2.2 Observation Model
- LLM prompt to extract evidence from (question, answer)
- Output: EvidencePacket (raw evidence mode)

### 2.3 CLI Interface
- Accept role_description.txt path as argument
- Load pre-generated candidate profile
- Run interactive interview loop

### 2.4 Termination Logic
- LLM outputs "I'm done" signal, OR
- Max steps reached (configurable)

### 2.5 Conversation Logging
- Plain text transcript: "Interviewer: ...\nCandidate: ..."

---

## Phase 3: Decision & Explanation

### 3.1 Decision Model
- Score calculation: `score = Σ(w_i * p_i) for each competence`
- Accept if: `score > threshold`
- Configurable threshold (default: 0.7)

### 3.2 Belief Trajectory Logging
- JSON persistence of belief state over time
- Format: `[{"step": 0, "belief": {...}}, {"step": 1, "belief": {...}}]`

### 3.3 Explanation Model
- Post-hoc LLM to explain the decision
- Inputs: conversation history, belief trajectory, final decision

### 3.4 Output Formatting
- Rich-formatted CLI output
- Display: questions, belief updates, final decision, explanation

---

## Project Structure

The project lives inside `expert_verify_engine/`:

```
expert_verify_engine/
├── app/
│   ├── __init__.py
│   ├── main.py                 # CLI entrypoint
│   ├── config.py               # Global configuration
├── llm/
│   ├── __init__.py
│   ├── client.py               # OpenRouter client
│   ├── prompts/
│   │   ├── __init__.py
│   │   ├── competence.py       # Competence generator prompts
│   │   ├── candidate.py       # Candidate generator prompts
│   │   ├── action.py          # Action generator prompts
│   │   ├── observation.py     # Observation model prompts
│   │   └── explanation.py     # Explanation generator prompts
├── models/
│   ├── __init__.py
│   ├── schemas.py              # Pydantic schemas
│   ├── competence.py           # Competence model
│   ├── candidate.py           # Candidate model & sheet
├── belief/
│   ├── __init__.py
│   ├── belief_state.py         # Beta distribution storage
│   └── updater.py             # Belief update logic
├── agent/
│   ├── __init__.py
│   ├── policy.py               # Action generation
│   └── decision.py             # Acceptance logic
├── observation/
│   ├── __init__.py
│   ├── evaluator.py            # LLM observation model
│   └── evidence.py            # Evidence transformation
├── audit_log/
│   ├── __init__.py
│   ├── conversation.py        # Transcript logging
│   └── belief_logger.py       # Belief trajectory persistence
├── utils/
│   ├── __init__.py
│   ├── parsing.py             # JSON extraction
│   └── retry.py               # Retry logic
├── data/
│   ├── examples/              # Sample role descriptions
│   └── candidates/            # Pre-generated candidate profiles
├── tests/
│   └── test_belief.py         # Unit tests
└── pyproject.toml

# Root contains:
# - specs/          # Specification documents
# - notebooks/      # Research notebooks
# - AGENTS.md      # Agent instructions
```

---

## Dependencies

- `httpx` - OpenRouter API
- `pydantic` - Schema validation
- `rich` - CLI formatting
- `typer` - CLI interface

---

## Testing Strategy

### Unit Tests
- Belief update logic: α += e_plus, β += e_minus
- Probability calculation: p = α / (α + β)
- Schema validation: ensure LLM outputs match Pydantic models
- Config loading and defaults

### Integration Tests (Future)
- Full CLI flow with mocked LLM
- End-to-end with real OpenRouter calls
