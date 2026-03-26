# AGENTS.md - Expert Verify Engine

## Project Overview

LLM-driven candidate evaluation using POMDP with structured latent state.
Estimates P(Accept | history) through Bayesian belief accumulation.

## Project Structure

```
.
├── app/
│   ├── main.py                 # CLI entrypoint
│   ├── config.py               # Global configuration
├── llm/
│   ├── client.py               # OpenRouter client
│   └── prompts/                # competence, candidate, action, observation, explanation
├── models/
│   ├── schemas.py              # Pydantic schemas (JSON contracts)
│   ├── competence.py           # Competence model structures
│   └── candidate.py           # Ground truth + candidate sheet
├── belief/
│   ├── belief_state.py         # Beta distributions (α, β)
│   └── updater.py             # Belief update logic
├── agent/
│   ├── policy.py              # Action generation
│   └── decision.py            # Acceptance logic
├── observation/
│   ├── evaluator.py           # LLM observation model
│   └── evidence.py            # Evidence transformation
├── logging/
│   ├── conversation.py        # Transcript logging
│   └── belief_logger.py       # Belief trajectory persistence
├── utils/
│   ├── parsing.py             # JSON extraction
│   └── retry.py               # Retry logic for LLM calls
├── data/examples/             # Sample role descriptions
├── specs/                     # Specification documents
└── tests/                     # Test suite
```

## Commands

```bash
python -m app.main              # Run CLI application
python -m app.main --help      # Show CLI options
pytest                         # Run all tests
pytest tests/test_file.py::test_function -v  # Run single test
ruff check .                   # Lint
ruff format .                 # Format
mypy .                        # Type check
```

## Dependencies

- `httpx` - OpenRouter API interaction
- `pydantic` - Strict JSON schema enforcement
- `rich` - Formatted CLI output
- `typer` - CLI interface

## Code Style Guidelines

### Imports

Order: 1) Standard library, 2) Third-party, 3) Local.

```python
import json
from typing import Any

import pydantic
import rich
import httpx

from llm.client import LLMClient
from models.schemas import EvidencePacket
```

### Formatting

- Max line length: 88 characters
- 4 spaces for indentation
- Trailing commas in multi-line

### Type Hints

Use type hints on all function signatures.

```python
def update_belief(alpha: float, beta: float, e_plus: float, e_minus: float) -> tuple[float, float]:
    alpha_new = alpha + e_plus
    beta_new = beta + e_minus
    return alpha_new, beta_new
```

### Naming

- **Functions/variables**: snake_case
- **Classes**: PascalCase
- **Constants**: UPPER_SNAKE_CASE
- **Private methods**: prefix with underscore

### Pydantic Models

All LLM outputs must conform to Pydantic schemas.

```python
from pydantic import BaseModel

class EvidencePacket(BaseModel):
    competence: dict[str, dict[str, float]]
    behavior: dict[str, float]
    notes: str

class Action(BaseModel):
    question: str
    target_competences: list[str]
    type: str  # "technical" | "behavioral" | "probing"
```

### Core Interfaces

```python
class LLMClient:
    def chat(self, prompt: str, temperature: float = 0.2) -> str:
        ...

def evaluate_answer(question: str, answer: str, target_competences: list[str]) -> EvidencePacket:
    ...

def update_belief(alpha: float, beta: float, evidence: EvidencePacket) -> tuple[float, float]:
    ...

def compute_decision(belief: dict[str, float], weights: dict[str, float], threshold: float) -> bool:
    score = sum(w * p for w, p in zip(weights.values(), belief.values()))
    return score > threshold
```

### Error Handling

Use specific exceptions, never bare `except:`.

```python
class LLMError(Exception):
    pass

class SchemaValidationError(Exception):
    pass
```

### Config

Centralize in `app/config.py`:

```python
CONFIG = {
    "model": "arcee-ai/trinity-large-preview:free",
    "evidence_mode": "raw",
    "enable_reasoning": False,
    "threshold": 0.7,
}
```

## Development Workflow

1. Create branch for each feature/fix
2. Run linting: `ruff check . && ruff format .`
3. Run tests: `pytest`
