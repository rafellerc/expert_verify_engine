import json

from expert_verify_engine.belief.belief_state import BeliefState
from expert_verify_engine.llm.client import LLMClient
from expert_verify_engine.models.schemas import (
    Action,
    CandidateSheet,
    TerminationDecision,
)
from expert_verify_engine.utils.parsing import parse_json


def generate_question(
    belief: BeliefState,
    candidate_sheet: CandidateSheet,
    competence_model_json: str,
    history: str,
    client: LLMClient,
    action_generator_prompt: str | None = None,
) -> Action:
    from expert_verify_engine.llm.prompts.student.action import (
        ACTION_GENERATOR_PROMPT as STUDENT_ACTION,
    )

    action_prompt = action_generator_prompt or STUDENT_ACTION  # noqa: N806

    competence_model_str = competence_model_json
    belief_state_str = json.dumps(belief.get_all_probabilities(), indent=2)
    candidate_sheet_str = f"Summary: {candidate_sheet.summary}\nExperiences: {', '.join(candidate_sheet.experiences)}\nClaims: {', '.join(candidate_sheet.claims)}"

    prompt = action_prompt.format(
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
    termination_prompt: str | None = None,
) -> tuple[bool, str]:
    from expert_verify_engine.llm.prompts.student.action import (
        TERMINATION_PROMPT as STUDENT_TERMINATION,
    )

    term_prompt = termination_prompt or STUDENT_TERMINATION  # noqa: N806
    belief_state_str = json.dumps(belief.get_all_probabilities(), indent=2)

    prompt = term_prompt.format(
        history=history or "No conversation yet.",
        belief_state=belief_state_str,
    )

    response = client.chat(prompt)
    result = parse_json(response, TerminationDecision)
    return result.continue_, result.reason


def generate_explanation(
    history: str,
    belief_trajectory: list[dict],
    final_belief: dict[str, float],
    decision: str,
    client: LLMClient,
    explanation_prompt: str | None = None,
) -> dict:
    from expert_verify_engine.llm.prompts.student.explanation import (
        EXPLANATION_PROMPT as STUDENT_EXPLANATION,
    )

    explanation_p = explanation_prompt or STUDENT_EXPLANATION  # noqa: N806

    belief_trajectory_str = json.dumps(belief_trajectory, indent=2)
    final_belief_str = json.dumps(final_belief, indent=2)

    prompt = explanation_p.format(
        history=history,
        belief_trajectory=belief_trajectory_str,
        final_belief=final_belief_str,
        decision=decision,
    )

    from expert_verify_engine.utils.parsing import extract_json

    response = client.chat(prompt)
    json_str = extract_json(response)
    return json.loads(json_str)
