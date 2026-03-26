OBSERVATION_PROMPT = """You are grading a student's quiz response.

Question:
{question}

Answer:
{answer}

Target Competencies:
{target_competences}

Grade the answer and extract evidence.

Output ONLY JSON:
{{
  "competence": {{
    "<competence_name>": {{"e_plus": float, "e_minus": float}},
    ...
  }},
  "behavior": {{
    "guessing": float (0-1, higher = more likely guessing)
  }},
  "notes": "brief observation about the answer"
}}

Scoring:
- e_plus: positive evidence (0-1) - give credit for reasonable attempts
- e_minus: negative evidence (0-1) - only for clearly wrong answers
- Be lenient - favor giving credit over penalizing
- e_plus + e_minus should be around 0.2 for average answers (favor the student)
- Partial credit is allowed - if partially correct, give partial e_plus
"""
