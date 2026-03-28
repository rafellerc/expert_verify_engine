# Thesis: Knowledge Verification as Information Quantification

My central thesis is that candidate evaluation is fundamentally a problem of **information quantification under uncertainty**, rather than a pure classification or decision problem.

The objective is to determine whether a candidate truly possesses the competences required for a role. Fraud detection is not the primary goal, but an auxiliary mechanism that helps assess **which observations are reliable indicators of competence**.

---

## Competence Modeling

We define a **Competence Model (CM)**:

$$
CM = [C_1, C_2, ..., C_n]
$$

where each $ C_i \in [0,1] $ is a **latent variable** representing the candidate’s ability to reliably execute competence $ i $ under genuine, non-fraudulent conditions.

Crucially, these competences are **not directly observable**. There is no way to measure them in isolation. Instead, they must be **inferred from evidence** gathered through interaction.

The system therefore maintains a **probability distribution** over each $ C_i $, representing its current belief about the candidate’s competence, and updates this belief as new evidence is observed.

---

## Acceptance as a Probabilistic Decision

To define acceptance, we assign an importance weight to each competence and compute a weighted score:

$$
Score = \sum_i w_i C_i
$$

with a threshold $ T $ determining acceptance. Since competences are latent variables, the score itself is uncertain.

The system’s objective is therefore to estimate:

$$
P(\text{Acc} = 1) = P(Score > T)
$$

Rather than computing a fixed score, the system reasons in terms of **probabilities over uncertain quantities**.

---

## Genuineness of Observations

During the interview, at each turn $ t $, the system asks a question and observes a response.

A key challenge is that a correct answer might come from genuine competence or by a use of external aid such as consulting LLMs or more complex cheating tools like Cluely.

To model this, we introduce:

$$
G_t = P(\text{answer is genuine and thus reflects true competence})
$$

If we believe an answer was not genuinely generated we must not use it to update our beliefs about the candidate's competences (or we might use it as evidence of the candidate's ignorance if we assume the candidate only cheats when ignorant)

This reframes the problem from simply evaluating answers to evaluating **the reliability of observations as evidence**.

---

## Fraud Strategies and Perceptual cues

In order to reliably identify and probe fraud we extend our modelling of the candidate with a list of fraud strategies:

$$
F = \{f_1, f_2, ..., f_k\}
$$

Where each $ f_i $ describes a way to cheat on questions and the expected cues it would generate. 

For example a strategy of copying questions into ChatGPT and copying it back into the interview would generate the obvious cue of the text showing up all at once instead of being naturally typed over time.

A strategy of using Cluely to read reponses would generate cues of click sounds timed on the ends of questions and a constant reading gaze during answering.

A big part of a full system would be to take information from all its channels during screening and converting them to these discrete perception cues observed during a turn. We denote these observation at time $ t $ as:

$$
O_t
$$

The power of modelling fraud strategies as latent variables is that we are then able to accumulate evidence based on these cues over time and not be as sensitive to accidental cues that might just be genuine human behaviour.

---

## Information Sources

The system integrates two types of information:

1. **Pre-screening signals** (CV, experience, claims), used to initialize prior beliefs and pre-generate questions
2. **Interactive signals** from the interview process  

At screening time we also ask questions to discern the veracity of the candidate's claims on their dossier.

---

## Sequential Inference

The interview is modeled as a sequential process where each question is selected to maximize expected information gain.

At each step, the system:

- chooses a question
- observes perceptual cues $ O_t $  
- evaluates the quality and correctness of the response  
- estimates its genuineness $ G_t $  
- updates beliefs over competences  
- updates beliefs over possible fraud strategies  

This process explicitly separates:

- **what was answered** (outcome)  
- **how the answer was produced** (process)  

---

## Termination

The process terminates when the system reaches sufficient confidence in its estimate of $ P(\text{Acc} = 1) $, based on accumulated evidence and remaining uncertainty.
