import json
from pathlib import Path

from expert_verify_engine.belief.belief_state import BeliefState


class BeliefLogger:
    def __init__(self) -> None:
        self.trajectory: list[dict] = []

    def log(self, step: int, belief: BeliefState) -> None:
        self.trajectory.append({"step": step, "belief": belief.to_dict()})

    def get_trajectory(self) -> list[dict]:
        return self.trajectory

    def save(self, path: Path) -> None:
        path.write_text(json.dumps(self.trajectory, indent=2))

    def load(self, path: Path) -> None:
        self.trajectory = json.loads(path.read_text())
