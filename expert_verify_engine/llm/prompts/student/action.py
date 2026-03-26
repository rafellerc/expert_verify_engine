ACTION_GENERATOR_PROMPT = """You are a teacher generating a quiz question for a student.

Competence Model:
{competence_model}

Student Profile:
{candidate_sheet}

Current Belief State (probability of competence 0-1):
{belief_state}

Conversation History:
{history}

Generate the next quiz question to evaluate the student's understanding.

Output ONLY JSON:
{{
  "question": "your quiz question",
  "target_competences": ["list of competencies this question tests"],
  "type": "recall | application | practice"
}}

Guidelines:
- Keep questions simple and direct
- Target competencies where belief is uncertain (around 0.5)
- Use "recall" for basic facts, "application" for using knowledge, "practice" for solving problems
- One question at a time, no multi-part questions
- Use concrete, specific questions
- CRITICAL: You MUST test a different competency than previous questions. Review the conversation history above - if a competency was already tested (even with different phrasing), test a NEW competency that hasn't been evaluated yet
- The belief state shows which competencies have been tested (higher probability = already tested). Prioritize competencies with probability around 0.5 that haven't been tested
- If the candidate points out ambiguity or confusion in your question, acknowledge it and ask a clarifying version instead
"""

TERMINATION_PROMPT = """Based on the conversation history and belief state, should the quiz continue or end?

Conversation History:
{history}

Belief State:
{belief_state}

Output ONLY JSON:
{{
  "continue": true/false,
  "reason": "brief explanation"
}}
"""
