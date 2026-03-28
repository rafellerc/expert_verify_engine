## Why This Is Not Primarily an RL Problem

Although the problem can be framed as a sequential decision process, a pure Reinforcement Learning formulation is not the most effective primary approach.

First, this is fundamentally a **Partially Observable Markov Decision Process (POMDP)** with a single latent state: the candidate’s true competence profile. The action space, however, is extremely large, as it includes all possible questions that could be asked in natural language. Learning an effective policy over such a space requires a highly capable model that has both language fluency as well as world-knowledge, typically a large language model.

Training such a model with policy gradient methods at the token level is **computationally expensive and inefficient**, especially when compared to leveraging pretrained LLMs directly for reasoning and generation.

Second, the reward signal in this setting is inherently tied to **information gain**: the value of a question lies in how much it reduces uncertainty about the candidate’s competences. This can be modeled explicitly and efficiently using a **Bayesian belief update framework**, without requiring trial-and-error learning over many episodes.

Finally, in a standard RL formulation, robustness to adversarial strategies emerges only through **exposure during training**. The agent must encounter each strategy repeatedly in order to learn how to handle it. This creates a fundamental limitation: when new strategies appear, the system must be retrained.

In contrast, by explicitly modeling candidate behavior, we can reason about new adversarial strategies **symbolically and compositionally**. Instead of retraining, we can incorporate new strategies by describing their expected cues and updating our inference process accordingly.

For these reasons, the core of the system is better framed as **explicit belief modeling and inference**, rather than end-to-end policy learning.

---

## Why This Is Still an RL Problem

Despite the limitations of a pure RL formulation, Reinforcement Learning remains an important component of the system.

The adversarial nature of the problem requires the agent to select actions that not only reduce uncertainty, but also **actively disrupt cheating strategies** and elicit genuine behavior. This introduces a strategic layer that goes beyond passive information gathering.

One key insight is that the **action space can be expanded beyond questions** to include interface-level decisions in the screening process. These actions can significantly increase the system’s ability to detect inconsistencies and fraud.

Examples include:

- **Modality of questions**: presenting questions as text or text-to-speech can expose candidates relying on modality-specific assistance tools.
- **Modality of answers**: requiring responses in text or voice can reveal discrepancies in fluency and depth of understanding.
- **Behavioral signals**: analyzing keystroke timing patterns can distinguish natural typing from copy-paste behavior, providing strong evidence of external assistance.

These actions are inherently **sequential and strategic**, as their effectiveness depends on the current belief state and prior observations.

In this context, RL becomes a powerful tool to optimize:
- **when to deploy specific strategies**
- **which modality to use**
- **how to balance exploration (probing) vs exploitation (decision)**

Therefore, while the core inference is best handled through explicit probabilistic modeling, RL plays a critical role in **learning effective interaction strategies in an adversarial environment**.