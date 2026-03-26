EXPLANATION_PROMPT = """You are explaining a quiz evaluation result.

Conversation History:
{history}

Belief Trajectory (how beliefs changed):
{belief_trajectory}

Final Belief State:
{final_belief}

Decision:
{decision}

Generate a clear explanation of the student's performance.

Output ONLY JSON:
{{
  "summary": "2-3 sentence overall assessment",
  "key_evidence": ["list of important observations"],
  "strengths": ["areas where student performed well"],
  "weaknesses": ["areas where student needs improvement"],
  "concerns": ["any concerns about the answers"]
}}
"""
