# Thesis: Knowledge Verification as Information Quantification

My central thesis is that candidate evaluation is fundamentally a problem of **information quantification under uncertainty**, rather than a pure classification or decision problem.

The objective is to determine whether a candidate truly possesses the competences required for a role. Fraud detection is not the primary goal, but an auxiliary task that helps assess **which observations are reliable indicators of competence**.

---

## Competence Modeling

Based on a role description we define a **Competence Model (CM)**:

$$
CM = [C_1, C_2, ..., C_n]
$$

where each $C_i$ is a **latent variable** representing the candidate’s ability to reliably execute competence $i$.

Crucially, these competences are **not directly observable**. Instead, they must be **inferred from evidence** gathered through interaction.

The system therefore maintains a **probability distribution** over each $C_i$, representing its current belief about the candidate’s competence, and updates this belief as new evidence is observed.

---

## Acceptance as a Probabilistic Decision

Based on the Competence Model we also define an Acceptance Function $Acc(C_1, ..., C_n)$ that defines what an acceptable candidate profile is as a function of which competences it has.

A simple way to define this function (though not the only one) is to define relative importances for each competence and compute the weighted score, which can be compared to a minimum threshold.

Because we have a probability model of each $C_i$ this gives is a real-time estimation of the probability that the candidate is acceptable. Therefore **our goal is to minimize the uncertainty about the probability of acceptance**:

$$
P(Acc(C_1, ..., C_n)) = 1
$$

---

## Genuineness of Observations

During the interview, at each turn $t$, the system asks a question and observes a response.

A key challenge is that a correct answer might come from genuine competence or by cheating such as consulting LLMs.

To model this, we introduce:

$$
G_t = P(\text{answer is genuine, not cheating})
$$

If we believe an answer was not genuinely generated **we must not use it to update our beliefs about the candidate's competences** (or we might use it as evidence of the candidate's ignorance if we assume the candidate only cheats when ignorant)

---

## Fraud Strategies and Perceptual cues

In order to reliably estimate the probability of fraud we define a new list of latent variables called Fraud Strategies:

$$
F = \{F_1, F_2, ..., F_k\}
$$

Each $F_j$ represents our belief that the candidate is employing fraud strategy $j$. Each strategy has to be characterized with its modus operandi and also what cues we expect it to generate.

As an example, for a strategy of reading answers off of ChatGPT we would expect that the candidate would display and reading eye movement during answers.

A big part of a full system would be to take information from all its channels (microphone, camera, UI) during screening and converting them into discrete perception cues observed during a turn. We denote these observation at time $t$ as:

$$
O_t
$$

The power of modelling fraud strategies as latent variables is that we are then able to accumulate evidence based on these cues over time and not be as sensitive to accidental cues that might just be genuine human behaviour.

---

## Integrating Dossier

Before the screening, we take the dossier from the candidate (CV, Linkedin) and try to determine:
1. **Initial Estimation of competences**: Based on listed experience we already can make updates to our competence estimates.
2. **Signals of fraud/exageration**: By cross-checking and applying critical judgement we might flag certain claims as potentially fraudulent, which allows us to already prepare questions with high information gain by confirming these disparities in the screening process (or possibly satisfarily resolving them)

---

## Sequential Inference

At the screening, on each step the system:

- chooses a question that maximizes expected Information Gain
- observes perceptual cues $O_t$  
- evaluates the quality and correctness of the response  
- estimates its genuineness $G_t$  
- updates beliefs over competences  
- updates beliefs over possible fraud strategies  

---

## Termination

The process terminates when the system reaches sufficient confidence in its estimate of $P(\text{Acc} = 1)$.
