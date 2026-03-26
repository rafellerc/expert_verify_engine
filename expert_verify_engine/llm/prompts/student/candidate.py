CANDIDATE_GENERATOR_PROMPT = """You are generating a student profile for a quiz.

Background (hidden from quiz):
{ground_truth}

Generate a student profile sheet:

Output ONLY a JSON object:
{{
  "summary": "2-3 sentence description of student's background",
  "experiences": ["list of relevant coursework, subjects, grade level"],
  "claims": ["specific knowledge or skills the student claims to have"]
}}

Rules:
- Profile reflects student's actual knowledge and abilities
"""

CANDIDATE_ANSWER_PROMPT = """You are a student answering quiz questions.

Student Profile (your background):
{candidate_sheet}

Competencies being evaluated:
{competences}

Question:
{question}

Generate an answer as this student would give.
- Be consistent with your background
- Students may not know everything

Output ONLY JSON:
{{
  "answer": "your response to the question"
}}
"""
