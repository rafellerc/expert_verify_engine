from expert_verify_engine.llm.client import LLMClient
from expert_verify_engine.llm.prompts.candidate import CANDIDATE_GENERATOR_PROMPT
from expert_verify_engine.models.schemas import CandidateModel, CandidateSheet
from expert_verify_engine.utils.parsing import parse_json


def generate_candidate_sheet(
    ground_truth: CandidateModel, client: LLMClient
) -> CandidateSheet:
    prompt = CANDIDATE_GENERATOR_PROMPT.format(
        ground_truth=ground_truth.model_dump_json()
    )
    response = client.chat(prompt)
    return parse_json(response, CandidateSheet)
