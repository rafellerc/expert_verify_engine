import json

from expert_verify_engine.belief.belief_state import BeliefState
from expert_verify_engine.llm.client import LLMClient
from expert_verify_engine.llm.prompts.action import (
    ACTION_GENERATOR_PROMPT,
    TERMINATION_PROMPT,
)
from expert_verify_engine.llm.prompts.explanation import EXPLANATION_PROMPT
from expert_verify_engine.models.schemas import Action, CandidateSheet
from expert_verify_engine.utils.parsing import parse_json


def generate_question(
    belief: BeliefState,
    candidate_sheet: CandidateSheet,
    competence_model_json: str,
    history: str,
    client: LLMClient,
) -> Action:
    competence_model_str = competence_model_json
    belief_state_str = json.dumps(belief.get_all_probabilities(), indent=2)
    candidate_sheet_str = f"Summary: {candidate_sheet.summary}\nExperiences: {', '.join(candidate_sheet.experiences)}\nClaims: {', '.join(candidate_sheet.claims)}"

    prompt = ACTION_GENERATOR_PROMPT.format(
        competence_model=competence_model_str,
        candidate_sheet=candidate_sheet_str,
        belief_state=belief_state_str,
        history=history or "No previous questions.",
    )

    response = client.chat(prompt)
    return parse_json(response, Action)


def should_continue(
    belief: BeliefState,
    history: str,
    client: LLMClient,
) -> tuple[bool, str]:
    belief_state_str = json.dumps(belief.get_all_probabilities(), indent=2)

    prompt = TERMINATION_PROMPT.format(
        history=history or "No conversation yet.",
        belief_state=belief_state_str,
    )

    response = client.chat(prompt)
    result = parse_json(response, {"continue": bool, "reason": str})
    return result["continue"], result["reason"]


def generate_explanation(
    history: str,
    belief_trajectory: list[dict],
    final_belief: dict[str, float],
    decision: str,
    client: LLMClient,
) -> dict:
    belief_trajectory_str = json.dumps(belief_trajectory, indent=2)
    final_belief_str = json.dumps(final_belief, indent=2)

    prompt = EXPLANATION_PROMPT.format(
        history=history,
        belief_trajectory=belief_trajectory_str,
        final_belief=final_belief_str,
        decision=decision,
    )

    response = client.chat(prompt)
    return parse_json(response, dict)
