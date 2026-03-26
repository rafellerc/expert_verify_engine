import json
from pathlib import Path
from typing import Any

from expert_verify_engine.belief.belief_state import BeliefState


class Turn:
    def __init__(
        self,
        turn: int,
        action: dict[str, Any],
        answer: str,
        evidence: dict[str, Any],
        belief_after: dict[str, float],
        belief_alpha_beta: dict[str, dict[str, float]],
    ) -> None:
        self.turn = turn
        self.action = action
        self.answer = answer
        self.evidence = evidence
        self.belief_after = belief_after
        self.belief_alpha_beta = belief_alpha_beta

    def to_dict(self) -> dict[str, Any]:
        return {
            "turn": self.turn,
            "action": self.action,
            "answer": self.answer,
            "evidence": self.evidence,
            "belief_after": self.belief_after,
            "belief_alpha_beta": self.belief_alpha_beta,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Turn":
        return cls(
            turn=data["turn"],
            action=data["action"],
            answer=data["answer"],
            evidence=data["evidence"],
            belief_after=data["belief_after"],
            belief_alpha_beta=data.get("belief_alpha_beta", {}),
        )


class Trajectory:
    def __init__(
        self,
        run_id: str,
        config: dict[str, Any],
        competence_model: dict[str, Any],
        candidate_sheet: dict[str, Any],
    ) -> None:
        self.id = f"{run_id}_trajectory"
        self.run_id = run_id
        self.config = config
        self.competence_model = competence_model
        self.candidate_sheet = candidate_sheet
        self.turns: list[Turn] = []
        self.final_decision: dict[str, Any] | None = None
        self.explanation: dict[str, Any] | None = None
        self.forced_end: bool = False

    def add_turn(
        self,
        action: dict[str, Any],
        answer: str,
        evidence: dict[str, Any],
        belief: BeliefState,
    ) -> None:
        turn = Turn(
            turn=len(self.turns),
            action=action,
            answer=answer,
            evidence=evidence,
            belief_after=belief.get_all_probabilities(),
            belief_alpha_beta=belief.get_all_alpha_beta(),
        )
        self.turns.append(turn)

    def set_decision(self, decision: dict[str, Any]) -> None:
        self.final_decision = decision

    def set_explanation(self, explanation: dict[str, Any]) -> None:
        self.explanation = explanation

    def get_turn(self, idx: int) -> Turn | None:
        if 0 <= idx < len(self.turns):
            return self.turns[idx]
        return None

    def get_history(self) -> str:
        lines = []
        for turn in self.turns:
            lines.append(f"Interviewer: {turn.action.get('question', '')}")
            lines.append(f"Candidate: {turn.answer}")
        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "run_id": self.run_id,
            "config": self.config,
            "competence_model": self.competence_model,
            "candidate_sheet": self.candidate_sheet,
            "turns": [t.to_dict() for t in self.turns],
            "final_decision": self.final_decision,
            "explanation": self.explanation,
            "forced_end": self.forced_end,
        }

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2))

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Trajectory":
        traj = cls(
            run_id=data["run_id"],
            config=data["config"],
            competence_model=data["competence_model"],
            candidate_sheet=data["candidate_sheet"],
        )
        traj.turns = [Turn.from_dict(t) for t in data.get("turns", [])]
        traj.final_decision = data.get("final_decision")
        traj.explanation = data.get("explanation")
        traj.forced_end = data.get("forced_end", False)
        return traj

    @classmethod
    def load(cls, path: Path) -> "Trajectory":
        return cls.from_dict(json.loads(path.read_text()))


class TrajectoryManager:
    def __init__(self, output_dir: Path) -> None:
        self.output_dir = output_dir

    def get_run_dir(self, run_id: str) -> Path:
        return self.output_dir / run_id

    def save_competence_model(self, run_id: str, competence_model: dict) -> None:
        run_dir = self.get_run_dir(run_id)
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "competence_model.json").write_text(
            json.dumps(competence_model, indent=2)
        )

    def save_candidate_sheet(self, run_id: str, candidate_sheet: dict) -> None:
        run_dir = self.get_run_dir(run_id)
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "candidate_sheet.json").write_text(
            json.dumps(candidate_sheet, indent=2)
        )

    def save_trajectory(self, trajectory: Trajectory) -> None:
        run_dir = self.get_run_dir(trajectory.run_id)
        traj_dir = run_dir / "trajectory"
        traj_dir.mkdir(parents=True, exist_ok=True)
        (traj_dir / f"{trajectory.id}.json").write_text(
            json.dumps(trajectory.to_dict(), indent=2)
        )

    def load_trajectory(self, run_id: str) -> Trajectory:
        traj_file = (
            self.get_run_dir(run_id) / "trajectory" / f"{run_id}_trajectory.json"
        )
        return Trajectory.load(traj_file)

    def list_runs(self) -> list[str]:
        if not self.output_dir.exists():
            return []
        return [d.name for d in self.output_dir.iterdir() if d.is_dir()]
