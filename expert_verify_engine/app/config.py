import os
from datetime import datetime
from typing import Any

CONFIG: dict[str, Any] = {
    "model": os.getenv("OPENROUTER_MODEL", "arcee-ai/trinity-large-preview:free"),
    "api_key": os.getenv("OPENROUTER_API_KEY"),
    "evidence_mode": "raw",
    "enable_reasoning": False,
    "cheating_sensitivity": 1.0,
    "competence_count": "auto",
    "threshold": 0.7,
    "max_steps": 10,
    "temperature": 0.2,
    "save_belief_trajectory": True,
    "output_dir": "output",
    "config_name": "default",
}


def get_config(key: str) -> Any:
    return CONFIG.get(key)


def set_config(key: str, value: Any) -> None:
    CONFIG[key] = value


def generate_run_id(config_name: str | None = None) -> str:
    name = config_name or CONFIG.get("config_name", "default")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{name}_{timestamp}"


def get_timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")
