import pytest

from expert_verify_engine.belief.belief_state import BeliefState
from expert_verify_engine.belief.updater import compute_decision, update_belief
from expert_verify_engine.models.schemas import EvidencePacket


def test_belief_state_initialization():
    competences = ["Python", "ML", "Data Analysis"]
    belief = BeliefState(competences)

    assert belief.probability("Python") == 0.5
    assert belief.probability("ML") == 0.5
    assert belief.probability("Data Analysis") == 0.5


def test_belief_update():
    belief = BeliefState(["Python"])

    evidence = EvidencePacket(
        competence={"Python": {"e_plus": 0.8, "e_minus": 0.1}},
        behavior={"cheating": 0.1},
        notes="Good answer",
    )

    update_belief(belief, evidence)

    assert belief.get_alpha("Python") == pytest.approx(1.8, rel=0.01)
    assert belief.get_beta("Python") == pytest.approx(1.1, rel=0.01)
    assert belief.probability("Python") == pytest.approx(1.8 / 2.9, rel=0.01)


def test_compute_decision():
    belief = BeliefState(["Python", "ML"])
    belief.update("Python", 2.0, 0.5)
    belief.update("ML", 1.5, 0.5)

    weights = {"Python": 0.6, "ML": 0.4}

    accepted, score = compute_decision(belief, weights, threshold=0.6)

    assert score > 0.6
    assert accepted is True


def test_belief_uncertainty():
    belief = BeliefState(["Python"])

    assert belief.probability("Python") == 0.5

    evidence = EvidencePacket(
        competence={"Python": {"e_plus": 0.5, "e_minus": 0.5}},
        behavior={"cheating": 0.0},
        notes="Neutral",
    )

    update_belief(belief, evidence)

    assert belief.probability("Python") == 0.5
