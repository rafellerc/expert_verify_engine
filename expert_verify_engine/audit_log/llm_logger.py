import json
from datetime import datetime
from pathlib import Path
from typing import Any


class LLMLogger:
    PROMPT_TYPE_MAP = {
        "ACTION_GENERATOR_PROMPT": "action_generator",
        "CANDIDATE_GENERATOR_PROMPT": "candidate_generator",
        "CANDIDATE_ANSWER_PROMPT": "candidate_answer",
        "COMPETENCE_GENERATOR_PROMPT": "competence_generator",
        "COMPETENCE_VALIDATION_PROMPT": "competence_validation",
        "EXPLANATION_PROMPT": "explanation",
        "OBSERVATION_PROMPT": "observation",
        "TERMINATION_PROMPT": "termination",
    }

    def __init__(self, output_dir: Path) -> None:
        self.output_dir = output_dir
        self.run_dir: Path | None = None

    def set_run_id(self, run_id: str) -> None:
        self.run_dir = self.output_dir / run_id / "llm_calls"
        self.run_dir.mkdir(parents=True, exist_ok=True)

    def _get_folder_name(self, prompt_type: str | None) -> str:
        if prompt_type is None:
            return "unknown"
        return self.PROMPT_TYPE_MAP.get(prompt_type, prompt_type)

    def log(
        self,
        model_name: str,
        prompt: str,
        response: str,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        if not self.run_dir:
            return

        meta = metadata or {}
        prompt_type = meta.get("prompt_type")
        timing = meta.get("timing", {})

        folder_name = self._get_folder_name(prompt_type)
        call_dir = self.run_dir / folder_name
        call_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        filename = f"{timestamp}.json"

        data = {
            "timestamp": datetime.now().isoformat(),
            "model": model_name,
            "prompt_type": folder_name,
            "prompt": prompt,
            "response": response,
            "timing": timing,
            "metadata": {
                k: v for k, v in meta.items() if k not in ("prompt_type", "timing")
            },
        }

        (call_dir / filename).write_text(json.dumps(data, indent=2))
