import os
from datetime import datetime
from typing import Any

CONFIG: dict[str, Any] = {
    # "model": os.getenv("OPENROUTER_MODEL", "arcee-ai/trinity-large-preview:free"),
    "model": os.getenv("OPENROUTER_MODEL", "nvidia/nemotron-3-super-120b-a12b:free"),
    # "model": os.getenv("OPENROUTER_MODEL", "stepfun/step-3.5-flash:free"),
    # "model": os.getenv("OPENROUTER_MODEL", "z-ai/glm-4.5-air:free"),
    # "model": os.getenv("OPENROUTER_MODEL", "mistralai/ministral-3b-2512"), # NOT FREE!!!!
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
    # v6 decision engine options
    "p_accept_method": "gaussian",
    "mc_samples": 10000,
    "epsilon": 0.05,
    "tau": 0.1,
    "z_threshold": 2.0,
    "delta": 0.001,
    "evidence_multiplier": 2.0,
    "use_llm_termination": False,
    "use_ig_selection": True,
    "use_stats_decision": True,
}


def get_config(key: str, default: Any = None) -> Any:
    return CONFIG.get(key, default)


def set_config(key: str, value: Any) -> None:
    CONFIG[key] = value


def generate_run_id(config_name: str | None = None) -> str:
    name = config_name or CONFIG.get("config_name", "default")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{name}_{timestamp}"


def get_timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")
