from expert_verify_engine.llm.prompts.student.action import (
    ACTION_GENERATOR_PROMPT,
    TERMINATION_PROMPT,
)
from expert_verify_engine.llm.prompts.student.candidate import (
    CANDIDATE_ANSWER_PROMPT,
    CANDIDATE_GENERATOR_PROMPT,
)
from expert_verify_engine.llm.prompts.student.competence import (
    COMPETENCE_GENERATOR_PROMPT,
    COMPETENCE_VALIDATION_PROMPT,
)
from expert_verify_engine.llm.prompts.student.explanation import (
    EXPLANATION_PROMPT,
)
from expert_verify_engine.llm.prompts.student.observation import (
    OBSERVATION_PROMPT,
)

__all__ = [
    "ACTION_GENERATOR_PROMPT",
    "CANDIDATE_ANSWER_PROMPT",
    "CANDIDATE_GENERATOR_PROMPT",
    "COMPETENCE_GENERATOR_PROMPT",
    "COMPETENCE_VALIDATION_PROMPT",
    "EXPLANATION_PROMPT",
    "OBSERVATION_PROMPT",
    "TERMINATION_PROMPT",
]
