from expert_verify_engine.llm.client import LLMClient
from expert_verify_engine.llm.prompts.observation import OBSERVATION_PROMPT
from expert_verify_engine.models.schemas import EvidencePacket
from expert_verify_engine.utils.parsing import parse_json


def evaluate_answer(
    question: str,
    answer: str,
    target_competences: list[str],
    client: LLMClient,
) -> EvidencePacket:
    prompt = OBSERVATION_PROMPT.format(
        question=question,
        answer=answer,
        target_competences=", ".join(target_competences),
    )

    response = client.chat(prompt)
    return parse_json(response, EvidencePacket)
