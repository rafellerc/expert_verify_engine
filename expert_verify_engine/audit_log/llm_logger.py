import json
from datetime import datetime
from pathlib import Path
from typing import Any


class LLMLogger:
    def __init__(self, output_dir: Path) -> None:
        self.output_dir = output_dir
        self.run_dir: Path | None = None

    def set_run_id(self, run_id: str) -> None:
        self.run_dir = self.output_dir / run_id / "llm_calls"
        self.run_dir.mkdir(parents=True, exist_ok=True)

    def log(
        self,
        model_name: str,
        prompt: str,
        response: str,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        if not self.run_dir:
            return

        model_dir = self.run_dir / model_name
        model_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        filename = f"{timestamp}.json"

        data = {
            "timestamp": datetime.now().isoformat(),
            "model": model_name,
            "prompt": prompt,
            "response": response,
            "metadata": metadata or {},
        }

        (model_dir / filename).write_text(json.dumps(data, indent=2))
