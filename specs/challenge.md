# Technical Project: Machine Learning Engineer

## Knowledge Integrity & Expert Verification Engine

### 1. The Problem

In an era of ubiquitous AI, “Expert Fraud” has evolved. While identity can be verified via traditional means, verifying depth of knowledge is the new frontier. Fraudulent actors now use AI to draft convincing screening answers and curate LinkedIn profiles that mirror high-demand roles they never actually held.

Your task is to design a system that evaluates the probability of “expertise fraud” by analyzing historical profile deltas, web signals, and screening responses. 

Please note: The fraud we are looking for here is a “ChatGPT expert”. We can assume that these people are real and verified using identity verification and SSN means and that type of fraud (identity) is not a concern. We are looking to identify when someone is overstating their experience and what they know (or products they have used).

You will have ~ 1 week to complete.

### 2. Part 1: Strategy Memo (Thesis)

**Requirement:** Maximum 1 page (under 500 words).

Outline your architectural thesis. Be concise and prioritize the following:

- **Feature Engineering:** What specific red flags in historical profile snapshots or screening answers suggest LLM-assisted fabrication versus natural career growth?
- **The RL Framework:** Define your Reinforcement Learning approach.

### 3. Part 2: Technical Implementation (Coding Exercise)

Create a Python-based proof-of-concept that embodies your theoretical approach.

**Requirements:**

- **Environment Design:** Build a simplified environment or framework that simulates the evaluation of an expert.
- **The Agent:** Implement an agent that learns to flag or pass candidates based on incoming data signals.
- **Logic Focus:** Use mock or synthetic data. You are free to choose the libraries or architectures you believe are best suited for this task.

### 4. Bonus Challenge: Multi-Modal “Live” Evaluator

**Scenario:** We are conducting video interviews where we record the candidate’s face, audio, and screen simultaneously.

Describe how you would evolve your RL model to detect fraud in a live setting. Briefly address:

- **The Cues:** What multi-modal signals would you extract? (Visual: eye-tracking. Audio: latency. Screen: tab-switching.)
- **The RL Training:** How would you train an RL model to detect a fraudulent answer in real-time?

### 5. Submission Guidelines & Evaluation

Please email your final report (the memo and the GitHub link to your project) to [**erica@pronexus.ai**](mailto:erica@pronexus.ai). Bonus: record a youtube video explaining your thought process and code.

- **Human-First Logic:** You may use AI to refine your writing, but the prioritization and problem-solving must be yours.
- **The “LLM Trap”:** We are specifically looking for human-led intuition. If your implementation mirrors a generic LLM recommendation without unique prioritization, it will result in a negative outcome.
- **Grading:** You will be graded on your ability to identify the most impactful signals, not the most numerous signals.