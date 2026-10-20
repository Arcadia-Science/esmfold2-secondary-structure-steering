"""Explicitly launch and collect GPU design campaigns.

Generation reuses existing manifests and checks for model outputs before
submitting jobs. Collection preserves structures, scores and optimization traces;
composition and screening counts are calculated in memory. Publication execution
does not call this module. GPU reruns produce new samples, not exact replicas.
"""

from __future__ import annotations
import json
import time
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
import typer

from analysis import artifacts, composition, deposit, loss_trace, paths
from analysis.results import CampaignOutput
from analysis.targets import (
    ARM_SPECS,
    CAMPAIGN_ARMS,
    N_SEEDS,
    SGD_STEPS,
    TARGET_SPECS,
)
from analysis.targets import (
    arm as get_arm,
)
from analysis.targets import (
    target as get_target,
)

app = typer.Typer(pretty_exceptions_enable=False, add_completion=False)

REMOTE_CLASS = "ESMFold2DesignModal"
POLL_SECONDS = 60


def read_manifest(path: Path, *, key: str) -> pd.DataFrame:
    if path.is_file():
        frame = pd.read_csv(path)
    elif key == "run" and (path.parent / "calls.json").is_file():
        # Existing collected inputs retain their original provenance unchanged.
        calls = json.loads((path.parent / "calls.json").read_text())
        frame = pd.DataFrame(
            [{"run": name, "call_id": item["call_id"]} for name, item in calls.items()],
            columns=[key, "call_id"],
        )
    else:
        return pd.DataFrame(columns=[key, "call_id"])
    if (
        not {key, "call_id"}.issubset(frame.columns)
        or frame[key].duplicated().any()
        or frame[[key, "call_id"]].isna().any().any()
    ):
        raise ValueError(f"Invalid campaign manifest: {path}")
    return frame


def submit_jobs(path: Path, jobs: list[dict], *, key: str, submit: Callable) -> pd.DataFrame:
    """Resume submitted jobs and record each new call before submitting the next."""
    frame = read_manifest(path, key=key)
    records = frame.to_dict("records")
    submitted = set(frame[key])
    identifiers = [job[key] for job in jobs]
    if len(set(identifiers)) != len(identifiers) or not submitted.issubset(identifiers):
        raise ValueError("Prepared jobs do not match the existing manifest")
    for job in jobs:
        if job[key] in submitted:
            continue
        call = submit(job)
        records.append({key: job[key], "call_id": call.object_id})
        frame = pd.DataFrame(records)
        path.parent.mkdir(parents=True, exist_ok=True)
        partial = path.with_suffix(".tmp")
        frame.to_csv(partial, index=False)
        partial.replace(path)
        submitted.add(job[key])
    return frame


def completed_jobs(manifest: pd.DataFrame) -> Iterator[tuple[Any, Any, str]]:
    """Poll the same saved Modal call IDs for every experiment kind."""
    import modal

    for row in manifest.itertuples(index=False):
        try:
            value = modal.FunctionCall.from_id(row.call_id).get(timeout=0)
        except TimeoutError:
            yield row, None, "pending"
        except Exception as error:  # noqa: BLE001
            yield row, None, f"{type(error).__name__}: {error}"
        else:
            yield row, value, ""


@app.command()
def status(
    target: str = typer.Option("", help="Limit to one target key."),
    arm: str = typer.Option("", help="Limit to one arm key."),
) -> None:
    """Print which campaigns are present on disk. Touches no network."""
    keys = [target] if target else None
    arms = [arm] if arm else None
    frame = artifacts.status_table(keys, arms)
    typer.echo(frame.to_string(index=False))
    absent = artifacts.missing(keys, arms)
    if absent:
        typer.echo(f"\n{len(absent)} campaign(s) without results:")
        for key, name in absent:
            typer.echo(f"  {key} / {name}")
    else:
        typer.echo("\nEvery requested campaign has results. Nothing to run.")


def spawn_campaign(
    target: str,
    arm: str,
    n_seeds: int = N_SEEDS,
    steps: int = SGD_STEPS,
    sampling_steps: int = 0,
    log_step_structures: bool = False,
) -> pd.DataFrame:
    """Launch one campaign on Modal and write its call manifest.

    Reuses recorded call IDs and submits only missing seeds. Each call is saved
    immediately so an interrupted submission can resume from its manifest.

    ``sampling_steps`` and ``log_step_structures`` are diagnostic. They are
    recorded in ``protocol.json`` because they change what the run cost and what
    it produced, and a campaign carrying per-step structures is not the same
    artifact as one that does not.
    """
    spec = get_target(target)
    arm_spec = get_arm(arm)
    manifest_path = paths.campaign_manifest(target, arm)

    import modal

    from analysis.modal_app import APP_NAME

    directory = paths.campaign_dir(target, arm)
    directory.mkdir(parents=True, exist_ok=True)
    target_sequence = spec.sequence()

    protocol = {
        "target": spec.key,
        "target_uniprot": spec.uniprot,
        "target_construct_uniprot": spec.construct_uniprot,
        "target_construct_length_residues": spec.construct_length,
        "arm": arm_spec.key,
        "arm_label": arm_spec.label,
        "schedule": arm_spec.schedule.to_dict(),
        "n_seeds": n_seeds,
        "sgd_steps": steps,
        "sampling_steps": sampling_steps or None,
        "step_structures_logged": log_step_structures,
    }
    protocol_path = directory / "protocol.json"
    if protocol_path.is_file():
        if json.loads(protocol_path.read_text()) != protocol:
            raise ValueError(f"Campaign settings differ from the saved protocol: {protocol_path}")
    else:
        partial = protocol_path.with_suffix(".tmp")
        partial.write_text(json.dumps(protocol, indent=2) + "\n")
        partial.replace(protocol_path)

    remote = modal.Cls.from_name(APP_NAME, REMOTE_CLASS)()

    def submit(job):
        return remote.design.spawn(
            seed=job["seed"],
            steps=steps,
            target=spec.key,
            target_sequence=target_sequence,
            arm=arm_spec.key,
            sampling_steps=sampling_steps,
            log_step_structures=log_step_structures,
        )

    return submit_jobs(
        manifest_path, [{"seed": seed} for seed in range(n_seeds)], key="seed", submit=submit
    )


@app.command()
def spawn(
    target: str = typer.Option("", help="Target key from analysis.targets."),
    arm: str = typer.Option("", help="Arm key from analysis.targets."),
    n_seeds: int = typer.Option(N_SEEDS, help="Designs to generate."),
    steps: int = typer.Option(SGD_STEPS, help="Optimization steps per design."),
    sampling_steps: int = typer.Option(
        0,
        help=(
            "Structure-solver steps on every optimization step, confidence "
            "steps included. 0 keeps the cookbook's 1-then-50 split."
        ),
    ),
    log_step_structures: bool = typer.Option(
        False, help="Return every step's predicted complex as a PDB."
    ),
) -> None:
    """Launch one campaign on Modal and write its call manifest.

    ``--sampling-steps`` and ``--log-step-structures`` are for diagnostic runs.
    Raising the solver budget makes every step cost what a confidence step
    costs, and logging structures returns one complex per step, so the two
    together turn a design into a trajectory at a many-fold higher price.
    """
    if not target or not arm:
        raise typer.BadParameter("Supply --target and --arm")
    if artifacts.status(target, arm).has_results:
        typer.echo(
            f"Results already present for {target} / {arm}: "
            f"{paths.campaign_results(target, arm)}\nNothing to do."
        )
        raise typer.Exit(0)
    existed = paths.campaign_manifest(target, arm).is_file()
    manifest = spawn_campaign(target, arm, n_seeds, steps, sampling_steps, log_step_structures)
    verb = "Reusing manifest of" if existed else "Spawned"
    typer.echo(
        f"{verb} {len(manifest)} jobs for {target} / {arm}.\n"
        f"Manifest: {paths.campaign_manifest(target, arm)}\n"
        f"Collect when finished:\n"
        f"    python -m analysis.campaign collect --target {target} --arm {arm}"
    )


@dataclass(frozen=True)
class Collected:
    """What one collection produced."""

    target: str
    arm: str
    n_designs: int
    n_pending: int
    n_failed: int
    n_trace_rows: int
    n_viable: int | None


def _write_step_complexes(target: str, arm: str, seed: int, blob: bytes) -> None:
    """Write one design's per-step complexes, numbered by step.

    Does nothing when the design did not log them, which is every production
    campaign; the diagnostic runs that do are a handful. Step numbers are zero
    padded so the files sort in the order the optimizer produced them, which is
    what a viewer loading a directory as a trajectory needs.
    """
    from analysis.modal_app import unpack_structures

    structures = unpack_structures(blob)
    if not structures:
        return
    directory = paths.campaign_step_complexes(target, arm, seed)
    directory.mkdir(parents=True, exist_ok=True)
    for step, pdb_text in enumerate(structures):
        (directory / f"step{step:03d}.pdb").write_text(pdb_text)


def collect_campaign(target: str, arm: str) -> Collected:
    """Gather finished Modal calls into the tables, structures and loss log.

    Each returned payload carries four kinds of thing, which go four places:
    the predicted complex to ``all_complexes``, the per-step loss log to
    ``step_trace.parquet``, the weight schedule nowhere, and everything else to
    the per-design table via :class:`CampaignOutput`. The schedule is dropped
    rather than stored because it is one fact about the whole campaign, already
    written to ``protocol.json`` at spawn time.

    Raises if no call has finished, rather than writing a table that looks
    complete and is not. A call that failed remotely is recorded and skipped
    rather than allowed to abort the whole collection, because one bad design
    out of hundreds should not cost the rest.
    """
    manifest_path = paths.campaign_manifest(target, arm)
    if not manifest_path.is_file():
        raise FileNotFoundError(
            f"No manifest for {target} / {arm}. Spawn first:\n"
            f"    python -m analysis.campaign spawn --target {target} --arm {arm}"
        )

    manifest = read_manifest(manifest_path, key="seed")
    complexes = paths.campaign_complexes(target, arm)
    complexes.mkdir(parents=True, exist_ok=True)

    records = []
    trace_rows: list[dict[str, object]] = []
    failures: list[str] = []
    pending = 0
    for row, result, error in completed_jobs(manifest):
        if error == "pending":
            pending += 1
            continue
        if error:
            failures.append(f"seed {int(row.seed)}: {error}")
            continue
        output = (
            result
            if isinstance(result, CampaignOutput)
            else CampaignOutput.from_payload(result, seed=int(row.seed))
        )
        if output.complex_pdb:
            (complexes / f"seed{int(row.seed):02d}.pdb").write_text(output.complex_pdb)
        _write_step_complexes(target, arm, int(row.seed), output.step_complexes_gz)
        trace_rows.extend(output.trace)
        records.append(output.row)

    if not records:
        raise RuntimeError(f"No finished calls yet for {target} / {arm} ({pending} pending).")

    frame = pd.DataFrame(records).sort_values("seed")
    frame.to_parquet(paths.campaign_results(target, arm), index=False)

    trace = loss_trace.frame(trace_rows)
    trace.to_parquet(paths.campaign_trace(target, arm), index=False)

    table = composition.build(target, arm, complexes, paths.campaign_results(target, arm))

    summary = {
        "target": target,
        "arm": arm,
        "n_success": int(len(frame)),
        "n_pending": int(pending),
        "n_failed": len(failures),
        "failures": failures,
        "partial": pending > 0 or bool(failures),
        "trace_rows": int(len(trace)),
        "n_viable": int(composition.viable(table).sum()),
    }
    return Collected(
        target=target,
        arm=arm,
        n_designs=int(len(frame)),
        n_pending=int(pending),
        n_failed=len(failures),
        n_trace_rows=int(len(trace)),
        n_viable=summary["n_viable"],
    )


@app.command()
def collect(
    target: str = typer.Option("", help="Target key from analysis.targets."),
    arm: str = typer.Option("", help="Arm key from analysis.targets."),
    force: bool = typer.Option(False, help="Recollect even if results exist."),
) -> None:
    """Gather one campaign's finished calls into its tables and structures."""
    if not target or not arm:
        raise typer.BadParameter("Supply --target and --arm")
    if artifacts.status(target, arm).is_analysable and not force:
        typer.echo(
            f"Results already present: {paths.campaign_results(target, arm)}\n"
            f"Pass --force to recollect."
        )
        raise typer.Exit(0)
    done = collect_campaign(target, arm)
    typer.echo(
        f"Collected {done.n_designs} designs ({done.n_pending} pending) for "
        f"{target} / {arm}\n"
        f"  {paths.campaign_results(target, arm)}\n"
        f"  {paths.campaign_trace(target, arm)} ({done.n_trace_rows} step rows)"
    )
    if done.n_viable is not None:
        typer.echo(f"  {done.n_viable} screen passes of {done.n_designs}")


def _resolve(call: Any) -> str:
    """``ok``, ``pending``, or the remote error, without raising."""
    try:
        call.get(timeout=0)
    except TimeoutError:
        return "pending"
    except Exception as error:  # noqa: BLE001
        return f"{type(error).__name__}: {error}"
    return "ok"


def ensure(
    target_keys: list[str],
    arms: list[str],
    n_seeds: int = N_SEEDS,
    steps: int = SGD_STEPS,
    poll_seconds: int = POLL_SECONDS,
) -> pd.DataFrame:
    """Generate requested campaigns with missing model scores or structures.

    Derived measurements are never used to decide whether GPU work is needed.

    All missing campaigns are spawned up front so they queue against the whole
    container pool at once rather than one campaign at a time, but each is
    collected as soon as its own designs resolve rather than at the end. That
    matters for a long grid: an interruption halfway through leaves every
    finished campaign already on disk instead of losing the lot.
    """
    unknown_targets = sorted(set(target_keys) - set(TARGET_SPECS))
    unknown_arms = sorted(set(arms) - set(ARM_SPECS))
    if unknown_targets or unknown_arms:
        raise KeyError(
            f"unknown target(s) {unknown_targets} or arm(s) {unknown_arms}; "
            f"registered: {sorted(TARGET_SPECS)} and {sorted(ARM_SPECS)}"
        )

    pairs = [(target, arm) for target in target_keys for arm in arms]
    missing = [pair for pair in pairs if not composition.is_available(*pair)]
    if not missing:
        print(f"All {len(pairs)} campaigns already present. Nothing to run.")
        return artifacts.status_table(target_keys, arms)

    print(
        f"{len(missing)} of {len(pairs)} campaigns missing, "
        f"{len(missing) * n_seeds} designs at {steps} steps each."
    )

    import modal

    calls: dict[tuple[str, str], dict[int, Any]] = {}
    for target, arm in missing:
        manifest = spawn_campaign(target, arm, n_seeds, steps)
        calls[(target, arm)] = {
            int(row.seed): modal.FunctionCall.from_id(row.call_id) for row in manifest.itertuples()
        }
        print(f"  spawned {target} / {arm}: {len(manifest)} calls", flush=True)

    resolved: dict[tuple[str, str], dict[int, str]] = {pair: {} for pair in missing}
    remaining = list(missing)
    started = time.time()
    while remaining:
        for pair in list(remaining):
            for seed, call in calls[pair].items():
                if seed in resolved[pair]:
                    continue
                state = _resolve(call)
                if state != "pending":
                    resolved[pair][seed] = state
            if len(resolved[pair]) == len(calls[pair]):
                done = collect_campaign(*pair)
                print(
                    f"  [{(time.time() - started) / 60:5.1f} min] collected "
                    f"{done.target} / {done.arm}: {done.n_designs} designs, "
                    f"{done.n_viable} viable, {done.n_failed} failed",
                    flush=True,
                )
                remaining.remove(pair)
        if remaining:
            seen = sum(len(v) for v in resolved.values())
            total = sum(len(v) for v in calls.values())
            print(
                f"  [{(time.time() - started) / 60:5.1f} min] {seen}/{total} designs "
                f"resolved, {len(remaining)} campaign(s) still open",
                flush=True,
            )
            time.sleep(poll_seconds)

    failures = {
        (target, arm, seed): message
        for (target, arm), seeds in resolved.items()
        for seed, message in seeds.items()
        if message != "ok"
    }
    if failures:
        print(f"\n{len(failures)} design(s) failed:")
        for key, message in sorted(failures.items()):
            print(f"  {key}: {message}")
    return artifacts.status_table(target_keys, arms)


@app.command()
def run(
    targets: str = typer.Option("", help="Comma-separated target keys, blank for all."),
    arms: str = typer.Option("", help="Comma-separated arms, blank for the pub grid."),
    n_seeds: int = typer.Option(N_SEEDS, help="Designs per campaign."),
) -> None:
    """Generate every missing campaign, wait for it, and collect it.

    This is the entry point a reader would use to reproduce the analysis on
    their own Modal account. It launches nothing that already exists.
    """
    keys = [k.strip() for k in targets.split(",") if k.strip()] or list(TARGET_SPECS)
    names = [a.strip() for a in arms.split(",") if a.strip()] or list(CAMPAIGN_ARMS)
    frame = ensure(keys, names, n_seeds=n_seeds, steps=SGD_STEPS)
    typer.echo(frame.to_string(index=False))


@app.command()
def package(
    experiment: str = typer.Option("campaigns", help="Input archive: campaigns or reference."),
) -> None:
    """Package preserved inputs locally; does not upload or publish."""
    if experiment == "campaigns":
        artifact, members = deposit.DESIGN_TABLES, deposit.design_members()
    elif experiment == "reference":
        artifact, members = deposit.REFERENCE_INPUTS, deposit.reference_members()
    else:
        raise typer.BadParameter("Choose campaigns or reference")
    typer.echo(artifact.package(members))


if __name__ == "__main__":
    app()
