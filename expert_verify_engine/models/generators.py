from expert_verify_engine.llm.client import LLMClient
from expert_verify_engine.llm.prompts.competence import COMPETENCE_GENERATOR_PROMPT
from expert_verify_engine.models.competence import normalize_competences
from expert_verify_engine.models.schemas import CompetenceModel
from expert_verify_engine.utils.parsing import parse_json


def generate_competences(role_description: str, client: LLMClient) -> CompetenceModel:
    prompt = COMPETENCE_GENERATOR_PROMPT.format(role_description=role_description)
    response = client.chat(prompt)
    model = parse_json(response, CompetenceModel)
    normalized = normalize_competences(model.competences)
    return CompetenceModel(competences=normalized)
