COMPETENCE_GENERATOR_PROMPT = """You are an expert job analyst. Given a role description, extract the key competencies required for the position.

Role Description:
{role_description}

Extract 5-8 core competencies that are essential for this role. For each competency:
- Name: A clear, specific skill or knowledge area
- Weight: A float between 0 and 1 representing relative importance

Output ONLY a JSON object in this exact format:
{{
  "competences": [
    {{"name": "string", "weight": float}},
    ...
  ]
}}

Ensure weights sum to 1.0 (or close).
"""

COMPETENCE_VALIDATION_PROMPT = """Validate and normalize the following competencies:

{competences_json}

Return a corrected JSON with:
- Remove duplicates or similar competencies
- Ensure weights are between 0 and 1
- Ensure weights sum to approximately 1.0
"""
