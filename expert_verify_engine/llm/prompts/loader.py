from typing import Any


class RoleDescriptionError(Exception):
    pass


def validate_role_description(content: str) -> dict[str, str]:
    lines = content.strip().split("\n")
    if len(lines) < 2:
        raise RoleDescriptionError(
            "Role description must have at least 2 lines. "
            "First two lines must be 'role_type: <type>' and 'name: <name>'"
        )

    role_type = None
    name = None

    for line in lines[:5]:
        line_stripped = line.strip()
        if line_stripped.startswith("role_type:"):
            role_type = line_stripped.split(":", 1)[1].strip()
        elif line_stripped.startswith("name:"):
            name = line_stripped.split(":", 1)[1].strip()

    if role_type is None:
        raise RoleDescriptionError(
            "First line must be 'role_type: <type>'. Valid types: professional, student"
        )

    if name is None:
        raise RoleDescriptionError("Second line must be 'name: <name>'")

    return {"role_type": role_type, "name": name}


def get_prompt_type(content: str) -> str:
    validated = validate_role_description(content)
    return validated["role_type"]


def load_prompts(prompt_type: str) -> dict[str, Any]:
    if prompt_type == "student":
        from expert_verify_engine.llm.prompts.student import (
            ACTION_GENERATOR_PROMPT,
            CANDIDATE_ANSWER_PROMPT,
            CANDIDATE_GENERATOR_PROMPT,
            COMPETENCE_GENERATOR_PROMPT,
            COMPETENCE_VALIDATION_PROMPT,
            EXPLANATION_PROMPT,
            OBSERVATION_PROMPT,
            TERMINATION_PROMPT,
        )
    elif prompt_type == "professional":
        from expert_verify_engine.llm.prompts.professional import (
            ACTION_GENERATOR_PROMPT,
            CANDIDATE_ANSWER_PROMPT,
            CANDIDATE_GENERATOR_PROMPT,
            COMPETENCE_GENERATOR_PROMPT,
            COMPETENCE_VALIDATION_PROMPT,
            EXPLANATION_PROMPT,
            OBSERVATION_PROMPT,
            TERMINATION_PROMPT,
        )
    else:
        raise RoleDescriptionError(
            f"Unknown prompt type: {prompt_type}. Valid types: professional, student"
        )

    return {
        "ACTION_GENERATOR_PROMPT": ACTION_GENERATOR_PROMPT,
        "TERMINATION_PROMPT": TERMINATION_PROMPT,
        "OBSERVATION_PROMPT": OBSERVATION_PROMPT,
        "COMPETENCE_GENERATOR_PROMPT": COMPETENCE_GENERATOR_PROMPT,
        "COMPETENCE_VALIDATION_PROMPT": COMPETENCE_VALIDATION_PROMPT,
        "CANDIDATE_GENERATOR_PROMPT": CANDIDATE_GENERATOR_PROMPT,
        "CANDIDATE_ANSWER_PROMPT": CANDIDATE_ANSWER_PROMPT,
        "EXPLANATION_PROMPT": EXPLANATION_PROMPT,
    }
