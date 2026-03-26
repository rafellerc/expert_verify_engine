OBSERVATION_PROMPT = """You are analyzing an interview response to extract evidence.

Question:
{question}

Answer:
{answer}

Target Competencies:
{target_competences}

Evaluate the answer and extract evidence.

Output ONLY JSON (raw evidence mode):
{{
  "competence": {{
    "<competence_name>": {{"e_plus": float, "e_minus": float}},
    ...
  }},
  "behavior": {{
    "cheating": float (0-1, higher = more suspicious)
  }},
  "notes": "brief observation about the answer"
}}

Scoring:
- e_plus: positive evidence (0-1)
- e_minus: negative evidence (0-1)
- e_plus + e_minus should be around 0.5 for average answers
- Be strict but fair
"""

BEHAVIOR_ANALYSIS_PROMPT = """Analyze the candidate's response for signs of dishonesty.

Question:
{question}

Answer:
{answer}

Candidate Profile:
{candidate_sheet}

Output ONLY JSON:
{{
  "suspicion_score": float (0-1, higher = more suspicious),
  "red_flags": ["list of specific concerns"],
  "reasoning": "brief explanation"
}}
"""
