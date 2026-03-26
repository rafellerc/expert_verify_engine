EXPLANATION_PROMPT = """You are explaining an interview decision.

Conversation History:
{history}

Belief Trajectory (how beliefs changed):
{belief_trajectory}

Final Belief State:
{final_belief}

Decision:
{decision}

Generate a clear explanation of why the candidate was accepted or rejected.

Output ONLY JSON:
{{
  "summary": "2-3 sentence overall assessment",
  "key_evidence": ["list of important evidence points"],
  "strengths": ["candidate strengths"],
  "weaknesses": ["candidate weaknesses"],
  "concerns": ["any red flags or concerns"]
}}
"""
