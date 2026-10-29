"""Explicitly launch GPU design campaigns and collect their model outputs."""

from __future__ import annotations
import json
import shutil
import time
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory
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
    absent = [
        (row.target, row.arm)
        for row in frame.itertuples(index=False)
        if not artifacts.is_complete(row.target, row.arm)
    ]
    if absent:
        typer.echo(f"\n{len(absent)} incomplete campaign(s):")
        for key, name in absent:
            typer.echo(f"  {key} / {name}")
    else:
        typer.echo("\nEvery requested campaign is complete. Nothing to run.")


def spawn_campaign(
    target: str,
    arm: str,
    n_seeds: int = N_SEEDS,
    steps: int = SGD_STEPS,
    sampling_steps: int = 0,
    log_step_structures: bool = False,
) -> pd.DataFrame:
    """Submit missing seeds and record each Modal call in a resumable manifest.

    Save the requested settings in the protocol; reject conflicting saved settings.
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
    """Submit missing seeds unless the campaign is complete."""
    if not target or not arm:
        raise typer.BadParameter("Supply --target and --arm")
    if artifacts.is_complete(target, arm):
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
    n_unsubmitted: int = 0


def _write_step_complexes(directory: Path, blob: bytes) -> None:
    """Unpack logged complexes into PDB files numbered by optimization step."""
    from analysis.modal_app import unpack_structures

    structures = unpack_structures(blob)
    if not structures:
        return
    directory.mkdir(parents=True, exist_ok=True)
    for step, pdb_text in enumerate(structures):
        (directory / f"step{step:03d}.pdb").write_text(pdb_text)


def collect_campaign(target: str, arm: str) -> Collected:
    """Save successful calls as score rows, structures, and a per-step loss log.

    Skip pending and failed calls. Raise if no successful results are available.
    Keep schedules out of score rows; they are recorded in the campaign protocol.
    """
    manifest_path = paths.campaign_manifest(target, arm)
    if not manifest_path.is_file():
        raise FileNotFoundError(
            f"No manifest for {target} / {arm}. Spawn first:\n"
            f"    python -m analysis.campaign spawn --target {target} --arm {arm}"
        )

    manifest = read_manifest(manifest_path, key="seed")
    expected = artifacts.expected_seeds(target, arm)
    if not set(manifest.seed).issubset(expected):
        raise ValueError("Manifest seeds differ from the campaign protocol")
    directory = paths.campaign_dir(target, arm)
    with TemporaryDirectory(prefix=".collection-", dir=directory) as temporary:
        staged = Path(temporary) / "outputs"
        staged.mkdir()
        done = _collect_staged(target, arm, manifest, expected, staged)
        _publish_collection(directory, staged)
    return done


def _collect_staged(
    target: str, arm: str, manifest: pd.DataFrame, expected: set[int], staged: Path
) -> Collected:
    """Build and validate a collection without changing published outputs."""
    complexes = staged / "all_complexes"
    complexes.mkdir()
    (staged / "step_complexes").mkdir()
    records = []
    trace_rows: list[dict[str, object]] = []
    failures: list[str] = []
    pending: list[int] = []
    for row, result, error in completed_jobs(manifest):
        seed = int(row.seed)
        if error == "pending":
            pending.append(seed)
            continue
        if error:
            failures.append(f"seed {seed}: {error}")
            continue
        output = (
            result
            if isinstance(result, CampaignOutput)
            else CampaignOutput.from_payload(result, seed=seed)
        )
        if output.row["seed"] != seed:
            raise ValueError(f"Result seed: expected {seed}, observed {output.row['seed']}")
        if not output.complex_pdb:
            raise ValueError(f"Missing complex structure for seed {seed}")
        (complexes / f"seed{seed:02d}.pdb").write_text(output.complex_pdb)
        _write_step_complexes(
            staged / "step_complexes" / f"seed{seed:02d}", output.step_complexes_gz
        )
        trace_rows.extend(output.trace)
        records.append(output.row)

    if not records:
        details = "; ".join(failures)
        raise RuntimeError(
            f"No successful calls for {target} / {arm} "
            f"({len(pending)} pending, {len(failures)} failed). {details}"
        )

    frame = pd.DataFrame(records).sort_values("seed")
    previous = paths.campaign_results(target, arm)
    if previous.is_file():
        old_seeds = set(pd.read_parquet(previous, columns=["seed"]).seed)
        if not old_seeds.issubset(set(frame.seed)):
            raise ValueError(
                "Recollection would discard previously collected seeds; keeping outputs"
            )
    results = staged / "all_results.parquet"
    frame.to_parquet(results, index=False)
    trace = loss_trace.frame(trace_rows)
    if set(trace.seed) != set(frame.seed):
        raise ValueError("Trace and score seeds do not match")
    trace.to_parquet(staged / "step_trace.parquet", index=False)
    table = composition.build(target, arm, complexes, results)
    unsubmitted = sorted(expected - set(manifest.seed))
    summary = {
        "target": target,
        "arm": arm,
        "n_success": int(len(frame)),
        "n_pending": len(pending),
        "pending_seeds": pending,
        "n_failed": len(failures),
        "failures": failures,
        "unsubmitted_seeds": unsubmitted,
        "partial": set(frame.seed) != expected,
        "trace_rows": int(len(trace)),
        "n_viable": int(composition.viable(table).sum()),
    }
    (staged / "collection.json").write_text(json.dumps(summary, indent=2) + "\n")
    return Collected(
        target=target,
        arm=arm,
        n_designs=int(len(frame)),
        n_pending=len(pending),
        n_failed=len(failures),
        n_trace_rows=int(len(trace)),
        n_viable=summary["n_viable"],
        n_unsubmitted=len(unsubmitted),
    )


def _publish_collection(directory: Path, staged: Path) -> None:
    """Replace validated outputs, restoring the old collection on write errors.

    The marker prevents an interrupted replacement from looking complete. The
    manifest and protocol stay in place; only collection outputs are replaced.
    """
    names = (
        "all_complexes",
        "step_complexes",
        "step_trace.parquet",
        "all_results.parquet",
        "collection.json",
    )
    backup = staged.parent / "previous"
    backup.mkdir()
    marker = directory / ".collecting"
    marker.touch()
    installed = []
    try:
        for name in names:
            destination = directory / name
            if destination.exists():
                destination.replace(backup / name)
            (staged / name).replace(destination)
            installed.append(name)
    except BaseException:
        for name in reversed(names):
            destination = directory / name
            if name in installed:
                if destination.is_dir():
                    shutil.rmtree(destination)
                else:
                    destination.unlink()
            if (backup / name).exists():
                (backup / name).replace(destination)
        marker.unlink()
        raise
    marker.unlink()


@app.command()
def collect(
    target: str = typer.Option("", help="Target key from analysis.targets."),
    arm: str = typer.Option("", help="Arm key from analysis.targets."),
    force: bool = typer.Option(False, help="Recollect even if results exist."),
) -> None:
    """Gather one campaign's finished calls into its tables and structures."""
    if not target or not arm:
        raise typer.BadParameter("Supply --target and --arm")
    if artifacts.is_complete(target, arm) and not force:
        typer.echo(
            f"Results already present: {paths.campaign_results(target, arm)}\n"
            f"Pass --force to recollect."
        )
        raise typer.Exit(0)
    done = collect_campaign(target, arm)
    typer.echo(
        f"Collected {done.n_designs} designs ({done.n_pending} pending, "
        f"{done.n_failed} failed, {done.n_unsubmitted} unsubmitted) for "
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
    """Submit missing seeds in incomplete campaigns, then poll and collect them.

    Collect each campaign once all its saved calls have resolved.
    """
    unknown_targets = sorted(set(target_keys) - set(TARGET_SPECS))
    unknown_arms = sorted(set(arms) - set(ARM_SPECS))
    if unknown_targets or unknown_arms:
        raise KeyError(
            f"unknown target(s) {unknown_targets} or arm(s) {unknown_arms}; "
            f"registered: {sorted(TARGET_SPECS)} and {sorted(ARM_SPECS)}"
        )

    pairs = [(target, arm) for target in target_keys for arm in arms]
    missing = [pair for pair in pairs if not artifacts.is_complete(*pair)]
    if not missing:
        print(f"All {len(pairs)} campaigns complete. Nothing to run.")
        return artifacts.status_table(target_keys, arms)

    print(
        f"{len(missing)} of {len(pairs)} campaigns incomplete, "
        f"{n_seeds} requested designs per campaign at {steps} steps each."
    )

    import modal

    calls: dict[tuple[str, str], dict[int, Any]] = {}
    for target, arm in missing:
        manifest = spawn_campaign(target, arm, n_seeds, steps)
        calls[(target, arm)] = {
            int(row.seed): modal.FunctionCall.from_id(row.call_id) for row in manifest.itertuples()
        }
        print(f"  tracking {target} / {arm}: {len(manifest)} calls", flush=True)

    resolved: dict[tuple[str, str], dict[int, str]] = {pair: {} for pair in missing}
    remaining = list(missing)
    collection_failures: dict[tuple[str, str], str] = {}
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
                try:
                    done = collect_campaign(*pair)
                except Exception as error:  # noqa: BLE001
                    collection_failures[pair] = f"{type(error).__name__}: {error}"
                else:
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
    if collection_failures:
        print(f"\n{len(collection_failures)} campaign(s) could not be collected:")
        for pair, message in sorted(collection_failures.items()):
            print(f"  {pair}: {message}")
    return artifacts.status_table(target_keys, arms)


@app.command()
def run(
    targets: str = typer.Option("", help="Comma-separated target keys, blank for all."),
    arms: str = typer.Option("", help="Comma-separated arms, blank for the pub grid."),
    n_seeds: int = typer.Option(N_SEEDS, help="Designs per campaign."),
) -> None:
    """Generate and collect campaigns missing model scores or structures."""
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
