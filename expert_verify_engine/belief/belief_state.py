BeliefParams = tuple[float, float]


class BeliefState:
    def __init__(self, competences: list[str]) -> None:
        self._alpha_beta: dict[str, BeliefParams] = {}
        for comp in competences:
            self._alpha_beta[comp] = (1.0, 1.0)

    @property
    def alpha_beta(self) -> dict[str, BeliefParams]:
        return self._alpha_beta.copy()

    def get_alpha(self, competence: str) -> float:
        return self._alpha_beta.get(competence, (1.0, 1.0))[0]

    def get_beta(self, competence: str) -> float:
        return self._alpha_beta.get(competence, (1.0, 1.0))[1]

    def probability(self, competence: str) -> float:
        alpha, beta = self._alpha_beta.get(competence, (1.0, 1.0))
        return alpha / (alpha + beta) if (alpha + beta) > 0 else 0.5

    def get_all_probabilities(self) -> dict[str, float]:
        return {comp: self.probability(comp) for comp in self._alpha_beta}

    def get_all_alpha_beta(self) -> dict[str, dict[str, float]]:
        return {
            comp: {"alpha": alpha, "beta": beta}
            for comp, (alpha, beta) in self._alpha_beta.items()
        }

    def update(self, competence: str, e_plus: float, e_minus: float) -> None:
        if competence not in self._alpha_beta:
            self._alpha_beta[competence] = (1.0, 1.0)
        alpha, beta = self._alpha_beta[competence]
        self._alpha_beta[competence] = (alpha + e_plus, beta + e_minus)

    def to_dict(self) -> dict[str, dict[str, float]]:
        return {
            comp: {"alpha": alpha, "beta": beta, "probability": self.probability(comp)}
            for comp, (alpha, beta) in self._alpha_beta.items()
        }
