CANDIDATE_GENERATOR_PROMPT = """You are generating a candidate profile for interview simulation.

Ground Truth (hidden from interviewer):
{ground_truth}

Generate a realistic candidate sheet (what the interviewer would see):

Output ONLY a JSON object:
{{
  "summary": "2-3 sentence professional summary",
  "experiences": ["list of job experiences with company, title, duration"],
  "claims": ["specific technical claims, projects, achievements"]
}}

Rules:
- If behavior is "honest": profile reflects true competences accurately
- If behavior is "cheater": exaggerate experience, add inconsistencies, include plausible-sounding but fake projects
"""

CANDIDATE_ANSWER_PROMPT = """You are a job candidate answering interview questions.

Candidate Profile (your background):
{candidate_sheet}

Competencies being evaluated:
{competences}

Question:
{question}

Generate a realistic answer as this candidate would give.
- Be consistent with your background
- If you're a "cheater", your answers may contain subtle inconsistencies or gaps

Output ONLY JSON:
{{
  "answer": "your response to the question"
}}
"""
