"""Explicitly launch and collect GPU design campaigns.

Generation reuses existing manifests and checks for model outputs before
submitting jobs. Collection preserves structures, scores and optimization traces;
composition and screening counts are calculated in memory. Publication execution
does not call this module. GPU reruns produce new samples, not exact replicas.
"""

from __future__ import annotations
import io
import json
import shutil
import tempfile
import time
import zipfile
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Any

import numpy as np
import pandas as pd
import typer

from analysis import artifacts, composition, deposit, initialization, loss_trace, paths, priors
from analysis.targets import (
    ARM_SPECS,
    CAMPAIGN_ARMS,
    CRITIC_MODELS,
    INITIALIZATION_TARGET,
    N_SEEDS,
    PRIMARY_CRITIC,
    SGD_STEPS,
    TARGET_SPECS,
    full_initialization_arm,
    short_trajectory_arms,
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

ROUTED_KEYS = ("complex_pdb", "step_trace", "schedule")


def _empty_container_paths(value: object, path: str = "") -> list[str]:
    """Where an empty dict or list sits inside a nested payload.

    Parquet cannot store a struct with no fields, so an empty mapping anywhere
    inside a design record makes the whole table unwritable. The failure is
    worth naming precisely because of where it hides: only an arm that sets no
    steering weights produces one, so the vanilla arm breaks and every steered
    arm writes fine.
    """
    if isinstance(value, dict):
        if not value:
            return [path or "<root>"]
        found: list[str] = []
        for key, child in value.items():
            found.extend(_empty_container_paths(child, f"{path}.{key}" if path else str(key)))
        return found
    if isinstance(value, list | tuple):
        if not value:
            return [path or "<root>"]
        found = []
        for index, child in enumerate(value):
            found.extend(_empty_container_paths(child, f"{path}[{index}]"))
        return found
    return []


def design_record(seed: int, result: dict[str, Any]) -> dict[str, Any]:
    """One per-design row: the payload minus everything routed elsewhere."""
    record = {
        "seed": seed,
        **{key: value for key, value in result.items() if key not in ROUTED_KEYS},
    }
    empty = sorted(_empty_container_paths(record))
    if empty:
        raise ValueError(
            f"design payload has empty container(s) at {empty}, which parquet "
            f"cannot store. Either route the field in collect the way "
            f"{list(ROUTED_KEYS)} are, or give it a scalar form in "
            f"analysis.modal_app.design."
        )
    return record


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
    if arm_spec.initialization_prior is not None:
        raise ValueError(
            "Prior arms require prepared inputs; use --experiment short or --experiment full"
        )
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
        initialization.save(protocol_path, protocol)

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
    experiment: str = typer.Option(
        "", help="Prepared initialization experiment: softness, short, full."
    ),
    temperature: float = typer.Option(1.0, help="Prior temperature for short/full initialization."),
    root: Annotated[
        Path, typer.Option(help="Preserved initialization run directory.")
    ] = paths.INITIALIZATION,
) -> None:
    """Launch one campaign on Modal and write its call manifest.

    ``--sampling-steps`` and ``--log-step-structures`` are for diagnostic runs.
    Raising the solver budget makes every step cost what a confidence step
    costs, and logging structures returns one complex per step, so the two
    together turn a design into a trajectory at a many-fold higher price.
    """
    if experiment:
        if (
            target
            or arm
            or n_seeds != N_SEEDS
            or steps != SGD_STEPS
            or sampling_steps
            or log_step_structures
        ):
            raise typer.BadParameter(
                "Initialization runs use the prepared protocol; omit campaign overrides"
            )
        frame = spawn_initialization(root.expanduser().resolve(), experiment, temperature)
        typer.echo(f"Recorded {len(frame)} jobs for {experiment}")
        return
    if not target or not arm:
        raise typer.BadParameter("Supply --target and --arm, or --experiment")
    if temperature != 1 or root != paths.INITIALIZATION:
        raise typer.BadParameter("--temperature and --root require --experiment")
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


def _register_short_campaigns(protocol: dict, manifest: pd.DataFrame) -> dict[str, str]:
    """Represent a jointly submitted trio as three ordinary campaign arms."""
    arms = short_trajectory_arms(protocol["initialization_temperature"])
    seeds = manifest.run.str.extract(r"^runs/pair(\d+)$", expand=False)
    if seeds.isna().any() or seeds.astype(int).duplicated().any():
        raise ValueError("Invalid matched-run manifest")
    shared = manifest.assign(seed=seeds.astype(int)).drop(columns="run")
    if not set(shared.seed).issubset(range(protocol["pairs"])):
        raise ValueError("Unexpected matched-run seeds")
    for prior, arm in arms.items():
        settings = {
            "target": protocol["target"],
            "arm": arm,
            "output_kind": "trajectory",
            "initialization_prior": prior,
            "initialization_temperature": protocol["initialization_temperature"],
            "n_seeds": protocol["pairs"],
            "sgd_steps": protocol["updates"],
            "matched_input_set": protocol["protocol_id"],
            "source_protocol": protocol,
        }
        path = paths.campaign_protocol(protocol["target"], arm)
        if path.is_file() and initialization.read(path) != settings:
            raise ValueError(f"Campaign settings differ from the saved protocol: {path}")
        initialization.save(path, settings)
        saved = read_manifest(path.parent / "manifest.csv", key="seed")
        for row in saved.itertuples(index=False):
            found = shared.loc[shared.seed == row.seed, "call_id"]
            if len(found) != 1 or found.iloc[0] != row.call_id:
                raise ValueError(f"Campaign call IDs differ: {path.parent}")
        shared.to_csv(path.parent / "manifest.csv", index=False)
    return arms


def _copy_preserved(source: Path, destination: Path) -> None:
    """Copy original bytes, refusing to replace a different model output."""
    if destination.is_file():
        if source.read_bytes() != destination.read_bytes():
            raise ValueError(f"Different model output already exists: {destination}")
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)


def _register_full_campaign(directory: Path) -> str:
    """Record a prepared full initialization as an ordinary campaign."""
    original = initialization.read(directory / "protocol.json")
    arm = full_initialization_arm(original["initialization_temperature"])
    if original["arm"] != arm:
        raise ValueError("Prepared initialization arm does not match its temperature")
    target = original["target"]
    settings = {
        "target": target,
        "arm": arm,
        "arm_label": get_arm(arm).label,
        "experiment": "full",
        "initialization_prior": "sheet",
        "initialization_temperature": original["initialization_temperature"],
        "n_seeds": original["n_seeds"],
        "sgd_steps": original["steps"],
        "schedule": get_arm(arm).schedule.to_dict(),
        "source_protocol": original,
    }
    path = paths.campaign_protocol(target, arm)
    if path.is_file() and initialization.read(path) != settings:
        raise ValueError(f"Campaign settings differ from the saved protocol: {path}")
    initialization.save(path, settings)
    submission = path.parent / "submission"
    for file in [directory / "protocol.json", directory / "inputs.json"]:
        _copy_preserved(file, submission / file.name)
    for file in (directory / "inputs").glob("*.pt"):
        _copy_preserved(file, submission / "inputs" / file.name)
    return arm


def _full_design_result(source: Path, target: str, arm: str, seed: int, protocol: dict) -> dict:
    """Validate a full worker archive and expose the ordinary design payload."""
    original = protocol["source_protocol"]
    required = [
        source / name
        for name in (
            "protocol.json",
            "audit.json",
            "initialization.pt",
            "sequence-states.pt",
            "complete.json",
            "critics.json",
            "step_trace.csv",
            "step_trace.jsonl",
            *(f"{name}.pdb" for name in CRITIC_MODELS),
        )
    ]
    if missing := [file for file in required if not file.is_file()]:
        raise FileNotFoundError(f"Incomplete full design: {missing[:5]}")
    run_protocol = initialization.read(source / "protocol.json")
    if any(run_protocol.get(key) != value for key, value in original.items()):
        raise ValueError(f"Run protocol differs from submitted protocol: {source}")
    record = initialization.read(source / "complete.json")
    for key, expected in {
        "seed": seed,
        "arm": arm,
        "steps": protocol["sgd_steps"],
        "initialization_temperature": protocol["initialization_temperature"],
        "probability_margin": original["probability_margin"],
        "primary_critic": PRIMARY_CRITIC,
    }.items():
        if record[key] != expected:
            raise ValueError(f"Unexpected {key}: {source}")
    trace = initialization.trace(source / "step_trace.csv", protocol["sgd_steps"])
    if not trace.arm.eq(arm).all():
        raise ValueError(f"Unexpected full-design trace arm: {source}")
    if (
        not trace.target.eq(target).all()
        or not trace.seed.eq(seed).all()
        or not trace["steering_loss (dimensionless)"].eq(0).all()
        or not trace["base_structure_weight (dimensionless)"].eq(1).all()
    ):
        raise ValueError(f"Unexpected objective or run identity: {source}")
    live_trace = loss_trace.load(source / "step_trace.jsonl")
    pd.testing.assert_frame_equal(trace, live_trace, check_dtype=False, check_like=True)

    baseline_path = paths.campaign_trace(target, "vanilla")
    if initialization.digest(baseline_path) != original["baseline_trace_sha256"]:
        raise ValueError("Baseline trace changed since initialization preparation")
    baseline = loss_trace.load_campaign(target, "vanilla")
    matched = baseline[baseline.seed == seed].sort_values("step (index)").reset_index(drop=True)
    if (
        len(matched) != len(trace)
        or trace.loc[0, "binder sequence"] != matched.loc[0, "binder sequence"]
        or record["initial_sequence"] != trace.loc[0, "binder sequence"]
        or not trace["binder length (residues)"].eq(record["binder_length"]).all()
        or record["binder_length"] != len(record["initial_sequence"])
        or not trace["inversion model"].eq(matched["inversion model"]).all()
        or not np.allclose(trace["temperature (0-1)"], matched["temperature (0-1)"])
    ):
        raise ValueError(f"Unmatched initial sequence or optimization schedule: {source}")
    selected = trace.loc[trace["iptm (0-1)"].idxmax()]
    sequence = selected["binder sequence"]
    if record["selected_step"] != selected["step (index)"] or record["binder_sequence"] != sequence:
        raise ValueError(f"Completion record differs from selected sequence: {source}")
    scores = initialization.read(source / "critics.json")
    by_critic = {score["critic_name"]: score for score in scores}
    if len(scores) != len(CRITIC_MODELS) or set(by_critic) != set(CRITIC_MODELS):
        raise ValueError(f"Missing or duplicate critics: {source}")
    for name, score in by_critic.items():
        if score["designed_sequence"] != original["target_sequence"] + "|" + sequence:
            raise ValueError(f"Critic sequence differs from selected sequence: {source}")
        if not np.isfinite([score["iptm"], score["plddt_binder"]]).all():
            raise ValueError(f"Nonfinite critic scores: {source}")
        composition.secondary_structure.checked_backbone(source / f"{name}.pdb", sequence)
    primary = by_critic[PRIMARY_CRITIC]
    if not np.allclose(
        [record["iptm"], record["plddt"]], [primary["iptm"], primary["plddt_binder"]]
    ):
        raise ValueError(f"Completion record differs from primary critic: {source}")

    destination = paths.campaign_run(target, arm, seed)
    for file in source.rglob("*"):
        if file.is_file():
            _copy_preserved(file, destination / file.relative_to(source))
    return {
        "seed": seed,
        "target": target,
        "arm": arm,
        "arm_label": get_arm(arm).label,
        "steps": protocol["sgd_steps"],
        "initialization_temperature": protocol["initialization_temperature"],
        "initialization_prior": "sheet",
        "probability_margin": original["probability_margin"],
        "target_length_residues": len(original["target_sequence"]),
        "designed_sequence": primary["designed_sequence"],
        "binder_sequence": sequence,
        "binder_length_residues": len(sequence),
        "final_loss": primary["final_loss"],
        "critic_name": PRIMARY_CRITIC,
        "critic_iptm": primary["iptm"],
        "critic_plddt_binder": primary["plddt_binder"],
        "critic_plddt_target": primary["plddt_target"],
        "critic_plddt_complex": primary["plddt_complex"],
        "critic_distogram_iptm_proxy": primary["distogram_iptm_proxy"],
        "critic_iptm_by_model": {name: s["iptm"] for name, s in by_critic.items()},
        "critic_plddt_binder_by_model": {name: s["plddt_binder"] for name, s in by_critic.items()},
        "critic_distogram_iptm_proxy_by_model": {
            name: s["distogram_iptm_proxy"] for name, s in by_critic.items()
        },
        "selected_step": int(selected["step (index)"]),
        "design_steps_recorded": len(trace),
        "step_trace": trace.to_dict("records"),
        "complex_pdb": (source / f"{PRIMARY_CRITIC}.pdb").read_text(),
    }


def _full_campaign_jobs(manifest: pd.DataFrame, target: str, arm: str, protocol: dict) -> Iterator:
    """Reuse preserved full runs; poll only calls not yet collected."""
    remaining = []
    for row in manifest.itertuples(index=False):
        folder = paths.campaign_run(target, arm, int(row.seed))
        marker = folder / "collected.json"
        if marker.is_file():
            if initialization.read(marker)["call_id"] != row.call_id:
                raise ValueError(f"Collected call ID differs from campaign manifest: {folder}")
            yield row, _full_design_result(folder, target, arm, int(row.seed), protocol), ""
        else:
            remaining.append(row._asdict())
    for row, payload, error in completed_jobs(pd.DataFrame(remaining, columns=manifest.columns)):
        if error:
            yield row, None, error
            continue
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary).resolve()
            with zipfile.ZipFile(io.BytesIO(payload)) as archive:
                if any(
                    not (folder / n).resolve().is_relative_to(folder) for n in archive.namelist()
                ):
                    raise ValueError("Unsafe path in full-design archive")
                archive.extractall(folder)
            result = _full_design_result(folder, target, arm, int(row.seed), protocol)
        initialization.save(
            paths.campaign_run(target, arm, int(row.seed)) / "collected.json",
            {"call_id": row.call_id},
        )
        yield row, result, ""


def _write_trajectory_result(
    source: Path, target: str, arm: str, seed: int, protocol: dict
) -> tuple[dict, pd.DataFrame]:
    """Validate one saved trajectory, retain its inputs, and normalize its identity."""
    import torch

    prior = protocol["initialization_prior"]
    folder = source / prior
    record = initialization.read(folder / "complete.json")
    original = protocol["source_protocol"]
    for key, expected in {
        "pair": seed,
        "arm": prior,
        "updates": protocol["sgd_steps"],
        "annealing_horizon": original["annealing_horizon"],
        "initialization_temperature": protocol["initialization_temperature"],
        "probability_margin": original["probability_margin"],
    }.items():
        if record[key] != expected:
            raise ValueError(f"Unexpected {key}: {folder}")
    run_protocol = initialization.read(source / "protocol.json")
    if any(run_protocol.get(key) != value for key, value in original.items()):
        raise ValueError(f"Run protocol differs from submitted protocol: {source}")
    trace = initialization.trace(folder / "step_trace.csv", protocol["sgd_steps"])
    # The worker returns all arms together. Check their shared schedules before
    # assigning a matched-set identity to any one campaign's rows.
    paired = []
    campaign_arms = short_trajectory_arms(protocol["initialization_temperature"])
    for name in original["arms"]:
        other = initialization.trace(source / name / "step_trace.csv", protocol["sgd_steps"])
        if not other.arm.eq(campaign_arms[name]).all():
            raise ValueError(f"Unexpected trajectory arm: {source / name / 'step_trace.csv'}")
        paired.append(
            other.assign(
                seed=seed,
                **{
                    "matched input set": protocol["matched_input_set"],
                },
            )
        )
    loss_trace.validate_matching(pd.concat(paired, ignore_index=True))
    if (
        not trace.target.eq(target).all()
        or not trace.seed.eq(record["optimization_seed"]).all()
        or not trace["steering_loss (dimensionless)"].eq(0).all()
    ):
        raise ValueError(f"Unexpected objective or run identity: {folder}")
    initial = torch.load(source / "initialization.pt", map_location="cpu", weights_only=True)
    state = torch.load(folder / "state000.pt", map_location="cpu", weights_only=True)
    sequence = initial["bundle"]["sequence"]
    prepared = state["prepared"]
    if (
        initial["bundle"]["pair"] != seed
        or record["shared_initial_sequence"] != sequence
        or trace.iloc[0]["binder sequence"] != sequence
        or prepared["seq_list"] != [original["target_sequence"] + "|" + sequence]
        or initial["features"].keys() != prepared["inputs"].keys()
        or any(
            not torch.equal(value, prepared["inputs"][key])
            for key, value in initial["features"].items()
        )
        or not torch.equal(initial["logits"][prior], state["logits"])
    ):
        raise ValueError(f"Unmatched initial sequence, features or logits: {folder}")
    readouts = pd.read_csv(folder / "readouts.csv")
    if sorted(readouts.step.tolist()) != list(range(protocol["sgd_steps"])):
        raise ValueError(f"Incomplete coordinate readouts: {folder}")
    required = [source / name for name in ("initialization.pt", "audit.json", "protocol.json")]
    required += [
        folder / f"{prefix}{step:03d}.{suffix}"
        for step in range(protocol["sgd_steps"])
        for prefix, suffix in (("state", "pt"), ("step", "pdb"))
    ]
    if missing := [path for path in required if not path.is_file()]:
        raise FileNotFoundError(f"Incomplete trajectory: {missing[:5]}")
    destination = paths.campaign_run(target, arm, seed)
    for file in [*source.iterdir(), *folder.iterdir()]:
        if not file.is_file():
            continue
        output = (
            paths.campaign_step_complexes(target, arm, seed) / file.name
            if file.parent == folder and file.suffix == ".pdb"
            else destination / file.name
        )
        _copy_preserved(file, output)
    trace = trace.assign(
        target=target,
        seed=seed,
        **{
            "optimization seed": record["optimization_seed"],
            "matched input set": protocol["matched_input_set"],
        },
    )
    return {**record, "seed": seed, "arm": arm, "target": target}, trace


def _save_trajectories(target: str, arm: str, records: list, traces: list) -> None:
    pd.DataFrame(records).sort_values("seed").to_parquet(
        paths.campaign_results(target, arm), index=False
    )
    pd.concat(traces, ignore_index=True).sort_values(["seed", "step (index)"]).to_parquet(
        paths.campaign_trace(target, arm), index=False
    )


def import_short_trajectories(directory: Path) -> list[Path]:
    """Import already collected trios without contacting Modal or altering originals."""
    protocol = initialization.read(directory / "protocol.json")
    manifest = read_manifest(directory / "manifest.csv", key="run")
    if set(manifest.run) != {f"runs/pair{i:02d}" for i in range(protocol["pairs"])}:
        raise ValueError("Incomplete matched-run manifest")
    target = protocol["target"]
    arms = _register_short_campaigns(protocol, manifest)
    collected = {}
    for arm in arms.values():
        settings = initialization.read(paths.campaign_protocol(target, arm))
        records, traces = [], []
        for seed in range(protocol["pairs"]):
            record, trace = _write_trajectory_result(
                directory / f"runs/pair{seed:02d}", target, arm, seed, settings
            )
            records.append(record)
            traces.append(trace)
        collected[arm] = (records, traces)
        for name in ("inputs.json", "protocol.json", "manifest.csv"):
            _copy_preserved(directory / name, paths.campaign_dir(target, arm) / "submission" / name)
    loss_trace.validate_matching(
        pd.concat(
            [trace for _, traces in collected.values() for trace in traces], ignore_index=True
        )
    )
    for arm, (records, traces) in collected.items():
        _save_trajectories(target, arm, records, traces)
    return [paths.campaign_dir(target, arm) for arm in arms.values()]


def _collect_trajectory_campaign(target: str, arm: str, protocol: dict) -> Collected:
    manifest = read_manifest(paths.campaign_manifest(target, arm), key="seed")
    records, traces = [], []
    pending = failed = 0
    for row, payload, error in completed_jobs(manifest):
        if error:
            pending += error == "pending"
            failed += error != "pending"
            typer.echo(f"seed {row.seed}: {error}")
            continue
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary).resolve()
            with zipfile.ZipFile(io.BytesIO(payload)) as archive:
                if any(
                    not (folder / n).resolve().is_relative_to(folder) for n in archive.namelist()
                ):
                    raise ValueError("Unsafe path in trajectory archive")
                archive.extractall(folder)
            record, trace = _write_trajectory_result(folder, target, arm, int(row.seed), protocol)
        records.append(record)
        traces.append(trace)
    if not records:
        raise RuntimeError(f"No finished calls yet for {target} / {arm} ({pending} pending).")
    _save_trajectories(target, arm, records, traces)
    return Collected(target, arm, len(records), pending, failed, sum(map(len, traces)), None)


def collect_campaign(target: str, arm: str) -> Collected:
    """Gather finished Modal calls into the tables, structures and loss log.

    Each returned payload carries four kinds of thing, which go four places:
    the predicted complex to ``all_complexes``, the per-step loss log to
    ``step_trace.parquet``, the weight schedule nowhere, and everything else to
    the per-design table via :func:`design_record`. The schedule is dropped
    rather than stored because it is one fact about the whole campaign, already
    written to ``protocol.json`` at spawn time.

    Raises if no call has finished, rather than writing a table that looks
    complete and is not. A call that failed remotely is recorded and skipped
    rather than allowed to abort the whole collection, because one bad design
    out of hundreds should not cost the rest.
    """
    protocol_path = paths.campaign_protocol(target, arm)
    protocol = {}
    if protocol_path.is_file():
        protocol = initialization.read(protocol_path)
        if protocol.get("output_kind") == "trajectory":
            return _collect_trajectory_campaign(target, arm, protocol)
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
    jobs = (
        _full_campaign_jobs(manifest, target, arm, protocol)
        if protocol.get("experiment") == "full"
        else completed_jobs(manifest)
    )
    for row, result, error in jobs:
        if error == "pending":
            pending += 1
            continue
        if error:
            failures.append(f"seed {int(row.seed)}: {error}")
            continue
        pdb_text = result.pop("complex_pdb", "")
        if pdb_text:
            (complexes / f"seed{int(row.seed):02d}.pdb").write_text(pdb_text)
        _write_step_complexes(target, arm, int(row.seed), result.pop("step_complexes_gz", b""))
        trace_rows.extend(result.pop("step_trace"))
        records.append(design_record(int(row.seed), result))

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
    experiment: str = typer.Option("", help="Initialization experiment: softness, short, full."),
    temperature: float = typer.Option(1.0, help="Prior temperature for short/full initialization."),
    root: Annotated[
        Path, typer.Option(help="Preserved initialization run directory.")
    ] = paths.INITIALIZATION,
) -> None:
    """Gather one campaign's finished calls into its tables and structures."""
    if experiment:
        if target or arm:
            raise typer.BadParameter("Use either --experiment or --target/--arm")
        collect_initialization(root.expanduser().resolve(), experiment, temperature, force=force)
        return
    if not target or not arm:
        raise typer.BadParameter("Supply --target and --arm, or --experiment")
    if temperature != 1 or root != paths.INITIALIZATION:
        raise typer.BadParameter("--temperature and --root require --experiment")
    protocol_path = paths.campaign_protocol(target, arm)
    full = (
        protocol_path.is_file() and initialization.read(protocol_path).get("experiment") == "full"
    )
    if artifacts.status(target, arm).is_analysable and not force and not full:
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
def prepare(
    experiment: str = typer.Option(..., help="Initialization experiment: softness, short, full."),
    temperature: float = typer.Option(1.0, help="Prior temperature for short/full initialization."),
    root: Annotated[
        Path, typer.Option(help="Preserved initialization run directory.")
    ] = paths.INITIALIZATION,
) -> None:
    """Prepare matched initialization inputs before submitting GPU jobs."""
    typer.echo(initialization.prepare_inputs(root.expanduser().resolve(), experiment, temperature))


def spawn_initialization(root: Path, experiment: str, temperature: float) -> pd.DataFrame:
    import modal

    from analysis.modal_app import APP_NAME

    protocol, jobs = initialization.prepared_jobs(root, experiment, temperature)
    if protocol["app"] != APP_NAME:
        raise ValueError(
            "Prepared inputs name another Modal app; prepare a fresh run for this deployment"
        )
    if experiment == "softness":
        worker = modal.Cls.from_name(APP_NAME, "FixedInputModal")()
    else:
        worker = modal.Cls.from_name(APP_NAME, REMOTE_CLASS).with_options(
            timeout=7200, max_containers=10 if experiment == "full" else 5
        )(load_critics=experiment == "full")
    directory = initialization.experiment_dir(root, experiment, temperature)
    if experiment == "full":
        arm = _register_full_campaign(directory)
        path = paths.campaign_manifest(protocol["target"], arm)
    else:
        path = directory / "manifest.csv"
    manifest = submit_jobs(
        path,
        jobs,
        key="seed" if experiment == "full" else "run",
        submit=lambda job: getattr(worker, job["method"]).spawn(**job["kwargs"]),
    )
    if experiment == "short":
        _register_short_campaigns(initialization.read(path.parent / "protocol.json"), manifest)
    return manifest


def collect_initialization(
    root: Path, experiment: str, temperature: float, *, force: bool = False
) -> None:
    initialization.validate_experiment(experiment, temperature)
    if experiment == "full":
        arm = full_initialization_arm(temperature)
        done = collect_campaign(INITIALIZATION_TARGET, arm)
        typer.echo(f"Collected {done.n_designs} designs for {done.target} / {done.arm}")
        return
    directory = initialization.experiment_dir(root, experiment, temperature)
    manifest = read_manifest(directory / "manifest.csv", key="run")
    if manifest.empty:
        raise FileNotFoundError(f"No submitted jobs in {directory}")
    if experiment == "short":
        protocol = initialization.read(directory / "protocol.json")
        for arm in _register_short_campaigns(protocol, manifest).values():
            if artifacts.status(protocol["target"], arm).is_analysable and not force:
                continue
            done = collect_campaign(protocol["target"], arm)
            typer.echo(f"Collected {done.n_designs} trajectories for {done.target} / {done.arm}")
        return
    remaining = []
    for row in manifest.to_dict("records"):
        destination = directory / row["run"]
        if not destination.resolve().is_relative_to(directory.resolve()):
            raise ValueError("Unexpected run path in manifest")
        if force or not (destination / "collected.json").is_file():
            remaining.append(row)
    for row, raw, error in completed_jobs(pd.DataFrame(remaining, columns=manifest.columns)):
        if error:
            typer.echo(f"{row.run}: {error}")
            continue
        destination = directory / row.run
        with zipfile.ZipFile(io.BytesIO(raw)) as archive:
            for member in archive.infolist():
                path = destination / member.filename
                if not path.resolve().is_relative_to(destination.resolve()):
                    raise ValueError("Unexpected archive path")
                if member.is_dir():
                    path.mkdir(parents=True, exist_ok=True)
                else:
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.write_bytes(archive.read(member))
        initialization.save(destination / "collected.json", {"call_id": row.call_id})
        typer.echo(f"Collected {destination}")


@app.command()
def prior() -> None:
    """Calculate and print amino acid priors without saving a measurement cache."""
    typer.echo(json.dumps(priors.from_reference(), indent=2))


@app.command()
def package(
    experiment: str = typer.Option(
        "campaigns", help="Input archive: campaigns, initialization, reference."
    ),
    root: Annotated[
        Path, typer.Option(help="Initialization run directory, if selected.")
    ] = paths.INITIALIZATION,
) -> None:
    """Package preserved inputs locally; does not upload or publish."""
    if experiment == "initialization":
        root = root.expanduser().resolve()
        artifact = deposit.DataArtifact(
            deposit.INITIALIZATION_INPUTS.filename, root, "initialization model outputs"
        )
        members = deposit.initialization_members(root)
    elif experiment == "campaigns":
        artifact, members = deposit.DESIGN_TABLES, deposit.design_members()
    elif experiment == "reference":
        artifact, members = deposit.REFERENCE_INPUTS, deposit.reference_members()
    else:
        raise typer.BadParameter("Choose campaigns, initialization, or reference")
    typer.echo(artifact.package(members))


if __name__ == "__main__":
    app()
