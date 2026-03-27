ACTION_GENERATOR_PROMPT = """You are a teacher generating a quiz question for a student.

Competence Model:
{competence_model}

Student Profile:
{candidate_sheet}

Current Belief State (probability of competence 0-1):
{belief_state}

Conversation History:
{history}

{ig_target_competence}

Generate the next quiz question to evaluate the student's understanding.

Output ONLY JSON (no extra text):
{{
  "question": "your quiz question",
  "target_competences": ["list of competencies this question tests"],
  "type": "recall | application | practice"
}}

Guidelines:

- Keep questions simple and direct
- Use "recall" for basic facts, "application" for using knowledge, "practice" for solving problems
- One question at a time, no multi-part questions
- Use concrete, specific questions
- If the candidate points out ambiguity or confusion in your question, acknowledge it and ask a clarifying version instead
- Questions should require simple text answers, no multiple choice or complex formats
- The candidate should not be asked to answer multiple questions at once
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
