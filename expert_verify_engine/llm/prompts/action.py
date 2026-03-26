ACTION_GENERATOR_PROMPT = """You are an interviewer generating the next question.

Competence Model:
{competence_model}

Candidate Profile:
{candidate_sheet}

Current Belief State (probability of competence 0-1):
{belief_state}

Conversation History:
{history}

Generate the next interview question to evaluate the candidate.

Output ONLY JSON:
{{
  "question": "your interview question",
  "target_competences": ["list of competencies this question tests"],
  "type": "technical | behavioral | probing"
}}

Guidelines:
- Target competencies where belief is uncertain (around 0.5)
- Ask follow-up questions when needed
- Use "probing" type to clarify ambiguous answers
- Ask behavioral questions to understand past performance
"""

TERMINATION_PROMPT = """Based on the conversation history and belief state, should the interview continue or end?

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
