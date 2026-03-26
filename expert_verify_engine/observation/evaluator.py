from expert_verify_engine.llm.client import LLMClient
from expert_verify_engine.models.schemas import EvidencePacket
from expert_verify_engine.utils.parsing import parse_json


def evaluate_answer(
    question: str,
    answer: str,
    target_competences: list[str],
    client: LLMClient,
    observation_prompt: str | None = None,
) -> EvidencePacket:
    from expert_verify_engine.llm.prompts.student.observation import (
        OBSERVATION_PROMPT as STUDENT_PROMPT,
    )

    obs_prompt = observation_prompt or STUDENT_PROMPT  # noqa: N806

    prompt = obs_prompt.format(
        question=question,
        answer=answer,
        target_competences=", ".join(target_competences),
    )

    response = client.chat(prompt)
    return parse_json(response, EvidencePacket)
