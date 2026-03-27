import json
from pathlib import Path

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from yaspin import yaspin

from expert_verify_engine.agent.decision import compute_decision
from expert_verify_engine.agent.policy import (
    generate_explanation,
    generate_question,
    should_continue,
)
from expert_verify_engine.app.config import (
    generate_run_id,
    get_config,
    get_timestamp,
)
from expert_verify_engine.audit_log.trajectory import Trajectory, TrajectoryManager
from expert_verify_engine.audit_log.llm_logger import LLMLogger
from expert_verify_engine.belief.belief_state import BeliefState
from expert_verify_engine.belief.decision_stats import (
    compute_decision_stats,
    select_competence_by_ig,
)
from expert_verify_engine.belief.stopping import StoppingCriteria, should_stop
from expert_verify_engine.belief.updater import update_belief
from expert_verify_engine.llm.client import LLMClient, LLMError
from expert_verify_engine.llm.prompts.loader import (
    RoleDescriptionError,
    get_prompt_type,
    load_prompts,
)
from expert_verify_engine.models.candidate import generate_candidate_sheet
from expert_verify_engine.models.generators import generate_competences
from expert_verify_engine.models.schemas import (
    ActionSelectionMode,
    CandidateDescription,
    CandidateSheet,
    CompetenceModel,
)
from expert_verify_engine.observation.evaluator import evaluate_answer
from expert_verify_engine.observation.evidence import transform_evidence

app = typer.Typer()
console = Console()

DEFAULT_OUTPUT_DIR = Path(__file__).parent.parent.parent / "output_dir"


def load_role_description(path: Path) -> str:
    return path.read_text()


def prompt_candidate_description() -> CandidateDescription:
    console.print("\n[bold cyan]Enter candidate description:[/bold cyan]")
    console.print(
        "[dim]Enter a summary of the candidate including background, experiences, and any notable traits[/dim]"
    )
    description = input("Description: ").strip()
    return description or "A candidate with standard background"


def display_decision_stats(
    belief: BeliefState, weights: dict[str, float], threshold: float
) -> None:
    """Display decision statistics in console."""
    alpha_beta = belief.alpha_beta
    mc_samples = get_config("mc_samples", 10000)
    evidence_multiplier = get_config("evidence_multiplier", 0.5)

    stats = compute_decision_stats(
        alpha_beta,
        weights,
        threshold,
        e_plus=evidence_multiplier,
        e_minus=evidence_multiplier,
        mc_samples=mc_samples,
    )

    console.print(
        f"[dim]Decision Stats: P(Accept)={stats.p_accept:.2f} | "
        f"P(Accept)MC={stats.p_accept_mc:.2f} | Z={stats.z_score:.2f} | "
        f"Entropy={stats.entropy:.3f} | Max IG={stats.max_ig:.3f}[/dim]"
    )


def run_interview(
    role_description: str,
    candidate: CandidateDescription,
    client: LLMClient,
    trajectory: Trajectory,
    output_dir: Path,
    traj_manager: TrajectoryManager,
    prompts: dict[str, str],
) -> tuple[BeliefState, CompetenceModel, CandidateSheet]:
    run_dir = output_dir / trajectory.run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    traj_dir = run_dir / "trajectory"
    traj_dir.mkdir(parents=True, exist_ok=True)

    console.print("\n[bold cyan]Step 1: Generating competence model...[/bold cyan]")
    console.print(
        f"        [dim]Will be saved to: {run_dir / 'competence_model.json'}[/dim]"
    )
    with yaspin(text="[competence model]", color="cyan"):
        competence_model = generate_competences(
            role_description, client, prompts.get("COMPETENCE_GENERATOR_PROMPT")
        )
    competence_json = {
        "competences": [
            {"name": c.name, "weight": c.weight} for c in competence_model.competences
        ]
    }
    traj_manager.save_competence_model(trajectory.run_id, competence_model.model_dump())

    console.print("[bold cyan]Step 2: Generating candidate sheet...[/bold cyan]")
    console.print(
        f"        [dim]Will be saved to: {run_dir / 'candidate_sheet.json'}[/dim]"
    )
    with yaspin(text="[candidate sheet]", color="cyan"):
        candidate_sheet = generate_candidate_sheet(
            candidate, client, prompts.get("CANDIDATE_GENERATOR_PROMPT")
        )
    traj_manager.save_candidate_sheet(trajectory.run_id, candidate_sheet.model_dump())

    console.print("[bold cyan]Step 3: Initializing belief state...[/bold cyan]")
    competence_names = [c.name for c in competence_model.competences]
    weights = {c.name: c.weight for c in competence_model.competences}
    belief = BeliefState(competence_names)
    threshold = get_config("threshold")

    max_steps = get_config("max_steps")
    step = 0

    action_selection_mode_str = get_config(
        "action_selection_mode", "information_gain_greedy"
    )
    action_selection_mode = ActionSelectionMode(action_selection_mode_str)
    use_llm_termination = get_config("use_llm_termination", True)
    epsilon = get_config("epsilon", 0.05)
    tau = get_config("tau", 0.1)
    z_threshold = get_config("z_threshold", 2.0)
    delta = get_config("delta", 0.001)
    evidence_multiplier = get_config("evidence_multiplier", 0.5)

    criteria = StoppingCriteria(
        epsilon=epsilon,
        tau=tau,
        z_threshold=z_threshold,
        delta=delta,
    )

    while step < max_steps:
        console.print(f"\n[bold yellow]--- Step {step + 1} ---[/bold yellow]")

        sampled = action_selection_mode == ActionSelectionMode.INFORMATION_GAIN_SAMPLED
        ig_target = None
        if belief.alpha_beta:
            ig_target = select_competence_by_ig(
                belief.alpha_beta,
                weights,
                threshold,
                evidence_multiplier,
                evidence_multiplier,
                sampled=sampled,
            )
            if ig_target:
                mode_str = "IG Sampled" if sampled else "IG Greedy"
                console.print(f"[dim]{mode_str} selected target: {ig_target}[/dim]")

        competence_json_str = json.dumps(competence_json)
        with yaspin(text="[action]", color="yellow"):
            action = generate_question(
                belief=belief,
                candidate_sheet=candidate_sheet,
                competence_model_json=competence_json_str,
                history=trajectory.get_history() if trajectory.turns else "",
                client=client,
                action_generator_prompt=prompts.get("ACTION_GENERATOR_PROMPT"),
                ig_target_competence=ig_target,
            )

        console.print(Panel(f"[bold]Question:[/bold] {action.question}"))
        console.print(
            f"[dim]Target: {', '.join(action.target_competences)} | Type: {action.type} | Selection: {action_selection_mode.value}[/dim]"
        )

        answer = input("Your answer: ")

        if answer.strip().startswith("/"):
            cmd = answer.strip().lower()
            if cmd in ("/quit", "/q"):
                console.print("[yellow]Interview ended by user.[/yellow]")
                trajectory.forced_end = True
                return belief, competence_model, candidate_sheet
            elif cmd in ("/end", "/e"):
                console.print("[yellow]Interview ended by user with /end.[/yellow]")
                trajectory.add_turn(
                    action=action.model_dump(),
                    answer=answer,
                    evidence={},
                    belief=belief,
                )
                return belief, competence_model, candidate_sheet
            elif cmd in ("/resample", "/r"):
                console.print("[dim]Resampling question...[/dim]")
                continue

        with yaspin(text="[observation]", color="green"):
            evidence = evaluate_answer(
                question=action.question,
                answer=answer,
                target_competences=action.target_competences,
                client=client,
                observation_prompt=prompts.get("OBSERVATION_PROMPT"),
            )

        sensitivity = get_config("evidence_multiplier", 0.5)

        alpha_beta_before = {}
        for comp in action.target_competences:
            alpha_beta_before[comp] = belief.alpha_beta.get(comp, (1.0, 1.0))

        console.print(
            Panel(
                f"[bold]Raw LLM evidence:[/bold]\n{evidence.notes}\n"
                f"e_plus: {list(evidence.competence.values())[0].e_plus:.3f}, "
                f"e_minus: {list(evidence.competence.values())[0].e_minus:.3f}",
                title="Observation",
                expand=False,
            )
        )

        adjusted_evidence = transform_evidence(evidence, sensitivity)

        console.print(
            Panel(
                f"[bold]Adjusted evidence (×{sensitivity}):[/bold]\n"
                f"e_plus: {list(adjusted_evidence.competence.values())[0].e_plus:.3f}, "
                f"e_minus: {list(adjusted_evidence.competence.values())[0].e_minus:.3f}",
                title="Evidence Multiplier",
                expand=False,
            )
        )

        update_belief(belief, adjusted_evidence)

        console.print("[bold]Alpha/Beta Updates:[/bold]")
        for comp in action.target_competences:
            alpha_before, beta_before = alpha_beta_before[comp]
            alpha_after, beta_after = belief.alpha_beta.get(comp, (1.0, 1.0))
            console.print(
                f"  {comp}: α {alpha_before:.2f}→{alpha_after:.2f}, "
                f"β {beta_before:.2f}→{beta_after:.2f}"
            )

        table = Table(title="Belief State")
        table.add_column("Competence", style="cyan")
        table.add_column("Probability", style="green")
        table.add_column("α", style="dim")
        table.add_column("β", style="dim")
        for comp in competence_names:
            alpha, beta = belief.alpha_beta.get(comp, (1.0, 1.0))
            table.add_row(
                comp,
                f"{belief.probability(comp):.2f}",
                f"{alpha:.2f}",
                f"{beta:.2f}",
            )
        console.print(table)

        display_decision_stats(belief, weights, threshold)

        trajectory.add_turn(
            action=action.model_dump(),
            answer=answer,
            evidence=evidence.model_dump(),
            belief=belief,
        )

        stop_reason = None
        if belief.alpha_beta:
            should_stop_flag, stop_reason = should_stop(
                belief.alpha_beta, weights, threshold, criteria
            )
            if should_stop_flag:
                console.print(
                    f"[bold yellow]Statistical stop: {stop_reason}[/bold yellow]"
                )
                break

        if use_llm_termination:
            with yaspin(text="[termination]", color="cyan"):
                continue_interview, term_reason = should_continue(
                    belief=belief,
                    history=trajectory.get_history(),
                    client=client,
                    termination_prompt=prompts.get("TERMINATION_PROMPT"),
                )
            reason_escaped = term_reason.replace("[", "[[").replace("]", "]]")
            console.print(f"[dim]LLM termination: {reason_escaped}[/dim]")

            step += 1
            if step >= max_steps:
                break
            if not continue_interview:
                console.print("[yellow]Interview ended early by model.[/yellow]")
                break
        else:
            step += 1

    return belief, competence_model, candidate_sheet


@app.command()
def start(
    role_description: Path,
    candidate: Path | None = None,
    output_dir: Path | None = None,
    config_name: str | None = None,
):
    if output_dir is None:
        output_dir = DEFAULT_OUTPUT_DIR

    role_desc = load_role_description(role_description)

    try:
        prompt_type = get_prompt_type(role_desc)
    except RoleDescriptionError as e:
        console.print(f"[bold red]Error:[/bold red] {e}")
        raise typer.Exit(1) from e

    console.print(f"[dim]Using prompt type: {prompt_type}[/dim]")
    prompts = load_prompts(prompt_type)

    if config_name is None:
        config_name = role_description.stem
        console.print(
            f"[dim]Using role description name as config: {config_name}[/dim]"
        )

    run_id = generate_run_id(config_name)
    console.print(f"[bold cyan]Using model: {get_config('model')}[/bold cyan]")
    console.print(f"[bold cyan]Starting interview with run_id: {run_id}[/bold cyan]")

    llm_logger = LLMLogger(output_dir)
    llm_logger.set_run_id(run_id)

    if candidate:
        candidate_description = candidate.read_text()
    else:
        candidate_description = prompt_candidate_description()

    traj_manager = TrajectoryManager(output_dir)
    config = {k: get_config(k) for k in ["threshold", "max_steps", "temperature"]}
    config["model"] = get_config("model")

    trajectory = Trajectory(
        run_id=run_id,
        config=config,
        competence_model={},
        candidate_sheet={},
    )

    run_dir = output_dir / run_id
    console.print(f"[bold cyan]Output directory:[/bold cyan] {run_dir}")

    try:
        client = LLMClient(logger=llm_logger.log)
    except LLMError as e:
        console.print(f"[bold red]Error:[/bold red] {e}")
        raise typer.Exit(1) from e

    belief, competence_model, candidate_sheet = run_interview(
        role_desc,
        candidate_description,
        client,
        trajectory,
        output_dir,
        traj_manager,
        prompts,
    )

    trajectory.competence_model = competence_model.model_dump()
    trajectory.candidate_sheet = candidate_sheet.model_dump()

    if trajectory.forced_end:
        console.print(
            "\n[yellow]Interview ended early - saving trajectory without decision.[/yellow]"
        )
        traj_manager.save_trajectory(trajectory)
        console.print(f"\n[dim]Trajectory saved to {output_dir / run_id}[/dim]")
        return

    use_stats = get_config("use_stats_decision", True)
    weights = {c.name: c.weight for c in competence_model.competences}
    decision, stats = compute_decision(belief, weights, use_stats=use_stats)

    console.print("\n[bold magenta]=== FINAL DECISION ===[/bold magenta]")
    if stats:
        console.print(
            f"[dim]P(Accept)={stats.p_accept:.3f} | P(Accept)MC={stats.p_accept_mc:.3f} | "
            f"Z={stats.z_score:.2f} | Entropy={stats.entropy:.3f}[/dim]"
        )
    if decision.accepted:
        console.print(
            f"[bold green]ACCEPTED[/bold green] (score: {decision.score:.2f})"
        )
    else:
        console.print(f"[bold red]REJECTED[/bold red] (score: {decision.score:.2f})")

    console.print("\n[bold cyan]Generating explanation...[/bold cyan]")
    final_belief = belief.get_all_probabilities()
    with yaspin(text="[explanation]", color="magenta"):
        explanation = generate_explanation(
            history=trajectory.get_history(),
            belief_trajectory=trajectory.to_dict()["turns"],
            final_belief=final_belief,
            decision=f"score: {decision.score:.2f}, p_accept: {stats.p_accept if stats else 'N/A'}",
            client=client,
            explanation_prompt=prompts.get("EXPLANATION_PROMPT"),
        )

    console.print(Panel(f"[bold]Summary:[/bold] {explanation.get('summary', 'N/A')}"))

    trajectory.set_decision(decision.model_dump())
    trajectory.set_explanation(explanation)

    traj_manager.save_trajectory(trajectory)
    traj_manager.save_competence_model(run_id, trajectory.competence_model)
    traj_manager.save_candidate_sheet(run_id, trajectory.candidate_sheet)

    console.print(f"\n[dim]Trajectory saved to {output_dir / run_id}[/dim]")


@app.command()
def fork(
    run_id: str,
    turn_idx: int,
    output_dir: Path | None = None,
):
    if output_dir is None:
        output_dir = DEFAULT_OUTPUT_DIR

    traj_manager = TrajectoryManager(output_dir)

    try:
        original_traj = traj_manager.load_trajectory(run_id)
    except FileNotFoundError as err:
        console.print(f"[bold red]Error:[/bold red] Trajectory {run_id} not found")
        raise typer.Exit(1) from err

    if turn_idx >= len(original_traj.turns):
        console.print(f"[bold red]Error:[/bold red] Turn {turn_idx} out of range")
        raise typer.Exit(1)

    new_run_id = f"{run_id}_fork_{turn_idx}_{get_timestamp()}"
    console.print(
        f"[bold cyan]Forking from {run_id} at turn {turn_idx} with new run_id: {new_run_id}[/bold cyan]"
    )

    llm_logger = LLMLogger(output_dir)
    llm_logger.set_run_id(new_run_id)

    try:
        client = LLMClient(logger=llm_logger.log)
    except LLMError as e:
        console.print(f"[bold red]Error:[/bold red] {e}")
        raise typer.Exit(1) from e

    new_traj = Trajectory(
        run_id=new_run_id,
        config=original_traj.config,
        competence_model=original_traj.competence_model,
        candidate_sheet=original_traj.candidate_sheet,
    )

    new_traj.turns = original_traj.turns[: turn_idx + 1]

    competence_names = [
        c["name"] for c in original_traj.competence_model.get("competences", [])
    ]
    weights = {
        c["name"]: c["weight"]
        for c in original_traj.competence_model.get("competences", [])
    }
    belief = BeliefState(competence_names)

    if new_traj.turns:
        last_belief_ab = new_traj.turns[-1].belief_alpha_beta
        for comp in competence_names:
            if comp in last_belief_ab:
                belief._alpha_beta[comp] = (
                    last_belief_ab[comp]["alpha"],
                    last_belief_ab[comp]["beta"],
                )

    max_steps = get_config("max_steps")
    step = turn_idx + 1

    while step < max_steps:
        console.print(f"\n[bold yellow]--- Step {step + 1} ---[/bold yellow]")

        competence_json_str = json.dumps(original_traj.competence_model)
        candidate_sheet = CandidateSheet(**original_traj.candidate_sheet)

        with yaspin(text="[action]", color="yellow"):
            action = generate_question(
                belief=belief,
                candidate_sheet=candidate_sheet,
                competence_model_json=competence_json_str,
                history=new_traj.get_history(),
                client=client,
            )

        console.print(Panel(f"[bold]Question:[/bold] {action.question}"))
        console.print(
            f"[dim]Target: {', '.join(action.target_competences)} | Type: {action.type}[/dim]"
        )

        answer = input("Your answer: ")

        if answer.strip().startswith("/"):
            cmd = answer.strip().lower()
            if cmd in ("/quit", "/q"):
                console.print("[yellow]Interview ended by user.[/yellow]")
                new_traj.forced_end = True
                break
            elif cmd in ("/end", "/e"):
                console.print("[yellow]Interview ended by user with /end.[/yellow]")
                new_traj.add_turn(
                    action=action.model_dump(),
                    answer=answer,
                    evidence={},
                    belief=belief,
                )
                break
            elif cmd in ("/resample", "/r"):
                console.print("[dim]Resampling question...[/dim]")
                continue

        with yaspin(text="[observation]", color="green"):
            evidence = evaluate_answer(
                question=action.question,
                answer=answer,
                target_competences=action.target_competences,
                client=client,
            )

        update_belief(belief, evidence)

        table = Table(title="Belief State")
        table.add_column("Competence", style="cyan")
        table.add_column("Probability", style="green")
        for comp in competence_names:
            table.add_row(comp, f"{belief.probability(comp):.2f}")
        console.print(table)

        new_traj.add_turn(
            action=action.model_dump(),
            answer=answer,
            evidence=evidence.model_dump(),
            belief=belief,
        )

        with yaspin(text="[termination]", color="cyan"):
            continue_interview, reason = should_continue(
                belief=belief,
                history=new_traj.get_history(),
                client=client,
            )
        reason_escaped = reason.replace("[", "[[").replace("]", "]]")
        console.print(f"[dim]Termination check: {reason_escaped}[/dim]")

        step += 1

        if step >= max_steps:
            break

        if not continue_interview:
            console.print("[bold yellow]Interview ended early by model.[/yellow]")
            break

    use_stats = get_config("use_stats_decision", True)
    decision, _ = compute_decision(belief, weights, use_stats=use_stats)

    console.print("\n[bold magenta]=== FINAL DECISION ===[/bold magenta]")
    if decision.accepted:
        console.print(
            f"[bold green]ACCEPTED[/bold green] (score: {decision.score:.2f})"
        )
    else:
        console.print(f"[bold red]REJECTED[/bold red] (score: {decision.score:.2f})")

    new_traj.set_decision(decision.model_dump())

    traj_manager.save_trajectory(new_traj)
    console.print(f"\n[dim]Trajectory saved to {output_dir / new_run_id}[/dim]")


@app.command()
def list_runs(output_dir: Path | None = None):
    if output_dir is None:
        output_dir = DEFAULT_OUTPUT_DIR
    traj_manager = TrajectoryManager(output_dir)
    runs = traj_manager.list_runs()
    if runs:
        console.print("[bold]Available runs:[/bold]")
        for run in runs:
            console.print(f"  • {run}")
    else:
        console.print("[dim]No runs found.[/dim]")


if __name__ == "__main__":
    app()
