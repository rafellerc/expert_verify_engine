from expert_verify_engine.llm.client import LLMClient
from expert_verify_engine.models.schemas import CandidateProfile, CandidateSheet
from expert_verify_engine.utils.parsing import parse_json


def generate_candidate_sheet(
    ground_truth: CandidateProfile,
    client: LLMClient,
    candidate_generator_prompt: str | None = None,
) -> CandidateSheet:
    from expert_verify_engine.llm.prompts.student.candidate import (
        CANDIDATE_GENERATOR_PROMPT as STUDENT_PROMPT,
    )

    candidate_prompt = candidate_generator_prompt or STUDENT_PROMPT  # noqa: N806

    prompt = candidate_prompt.format(ground_truth=ground_truth.model_dump_json())
    response = client.chat(prompt)
    return parse_json(response, CandidateSheet)
