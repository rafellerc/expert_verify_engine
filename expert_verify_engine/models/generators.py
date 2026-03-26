from expert_verify_engine.llm.client import LLMClient
from expert_verify_engine.models.competence import normalize_competences
from expert_verify_engine.models.schemas import CompetenceModel
from expert_verify_engine.utils.parsing import parse_json


def generate_competences(
    role_description: str,
    client: LLMClient,
    competence_generator_prompt: str | None = None,
) -> CompetenceModel:
    from expert_verify_engine.llm.prompts.student.competence import (
        COMPETENCE_GENERATOR_PROMPT as STUDENT_PROMPT,
    )

    competence_prompt = competence_generator_prompt or STUDENT_PROMPT  # noqa: N806

    prompt = competence_prompt.format(role_description=role_description)
    response = client.chat(prompt, prompt_type="COMPETENCE_GENERATOR_PROMPT")
    model = parse_json(response, CompetenceModel)
    normalized = normalize_competences(model.competences)
    return CompetenceModel(competences=normalized)
