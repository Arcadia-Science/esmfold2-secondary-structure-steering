"""Launch and collect design campaigns, gated on what already exists.

This is the only module that talks to a GPU. Every entry point checks the
filesystem first and refuses to redo work:

- :func:`ensure` generates only the campaigns whose composition table is
  missing, and is what the notebook calls under ``FORCE_RERUN=1``. Without
  that, the notebook fetches the deposited tables and never comes here.
- :func:`spawn_campaign` reuses an existing manifest rather than launching a
  second time, so an interrupted session resumes by collecting.
- :func:`collect_campaign` refuses to write a partial table.

The three plain functions hold the logic and the typer commands wrap them, so
the same code path serves a reader working from the notebook and one working
from a shell.

Regenerating a campaign will not reproduce the designs in the publication. The
design loop is not seed-reproducible: across two runs of an identically
configured arm, no seed reproduced its own result and per-seed interface pTM
moved 0.14 to 0.20. Use the deposited artifacts to reproduce figures, and use
this module to run the analysis on your own targets.
"""

from __future__ import annotations
import json
import time
from dataclasses import dataclass
from typing import Any

import pandas as pd
import typer

from analysis import artifacts, composition, loss_trace, paths
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
    target: str, arm: str, n_seeds: int = N_SEEDS, steps: int = SGD_STEPS
) -> pd.DataFrame:
    """Launch one campaign on Modal and write its call manifest.

    Returns the existing manifest untouched if there is one, so a rerun after
    an interruption resumes by collecting rather than by paying for the same
    designs twice.
    """
    spec = get_target(target)
    arm_spec = get_arm(arm)
    manifest_path = paths.campaign_manifest(target, arm)
    if manifest_path.is_file():
        return pd.read_csv(manifest_path)

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
        "arm_note": arm_spec.note,
        "schedule": arm_spec.schedule.to_dict(),
        "n_seeds": n_seeds,
        "sgd_steps": steps,
    }
    (directory / "protocol.json").write_text(json.dumps(protocol, indent=2) + "\n")

    remote = modal.Cls.from_name(APP_NAME, REMOTE_CLASS)()
    rows = []
    for seed in range(n_seeds):
        call = remote.design.spawn(
            seed=seed,
            steps=steps,
            target=spec.key,
            target_sequence=target_sequence,
            arm=arm_spec.key,
        )
        rows.append({"seed": seed, "call_id": call.object_id})

    manifest = pd.DataFrame(rows)
    manifest.to_csv(manifest_path, index=False)
    return manifest


@app.command()
def spawn(
    target: str = typer.Option(..., help="Target key from analysis.targets."),
    arm: str = typer.Option(..., help="Arm key from analysis.targets."),
    n_seeds: int = typer.Option(N_SEEDS, help="Designs to generate."),
    steps: int = typer.Option(SGD_STEPS, help="Optimization steps per design."),
) -> None:
    """Launch one campaign on Modal and write its call manifest."""
    if artifacts.status(target, arm).has_results:
        typer.echo(
            f"Results already present for {target} / {arm}: "
            f"{paths.campaign_results(target, arm)}\nNothing to do."
        )
        raise typer.Exit(0)
    existed = paths.campaign_manifest(target, arm).is_file()
    manifest = spawn_campaign(target, arm, n_seeds, steps)
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
    n_viable: int


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
    manifest_path = paths.campaign_manifest(target, arm)
    if not manifest_path.is_file():
        raise FileNotFoundError(
            f"No manifest for {target} / {arm}. Spawn first:\n"
            f"    python -m analysis.campaign spawn --target {target} --arm {arm}"
        )

    import modal

    manifest = pd.read_csv(manifest_path)
    complexes = paths.campaign_complexes(target, arm)
    complexes.mkdir(parents=True, exist_ok=True)

    records = []
    trace_rows: list[dict[str, object]] = []
    failures: list[str] = []
    pending = 0
    for row in manifest.itertuples():
        call = modal.FunctionCall.from_id(row.call_id)
        try:
            result = call.get(timeout=0)
        except TimeoutError:
            pending += 1
            continue
        except Exception as error:  # noqa: BLE001
            failures.append(f"seed {int(row.seed)}: {type(error).__name__}: {error}")
            continue
        pdb_text = result.pop("complex_pdb", "")
        if pdb_text:
            (complexes / f"seed{int(row.seed):02d}.pdb").write_text(pdb_text)
        trace_rows.extend(result.pop("step_trace"))
        records.append(design_record(int(row.seed), result))

    if not records:
        raise RuntimeError(
            f"No finished calls yet for {target} / {arm} ({pending} pending)."
        )

    frame = pd.DataFrame(records).sort_values("seed")
    frame.to_parquet(paths.campaign_results(target, arm), index=False)
    frame.to_csv(paths.campaign_dir(target, arm) / "all_results.csv", index=False)

    trace = loss_trace.frame(trace_rows)
    trace.to_parquet(paths.campaign_trace(target, arm), index=False)

    table = composition.build(
        target, arm, complexes, paths.campaign_results(target, arm)
    )
    composition.write(target, arm, table)

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
    (paths.campaign_summary(target, arm)).write_text(json.dumps(summary, indent=2) + "\n")
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
    target: str = typer.Option(..., help="Target key from analysis.targets."),
    arm: str = typer.Option(..., help="Arm key from analysis.targets."),
    force: bool = typer.Option(False, help="Recollect even if results exist."),
) -> None:
    """Gather one campaign's finished calls into its tables and structures."""
    if artifacts.status(target, arm).has_results and not force:
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
        f"  {paths.campaign_trace(target, arm)} ({done.n_trace_rows} step rows)\n"
        f"  {paths.campaign_composition(target, arm)} "
        f"({done.n_viable} viable of {done.n_designs})"
    )


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
    """Generate every requested campaign whose composition table is missing.

    This is the notebook's entry point. The gate is the composition table
    rather than the results table, because that is the artifact the notebook
    reads, and a campaign that has everything except it is not usable.

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
            int(row.seed): modal.FunctionCall.from_id(row.call_id)
            for row in manifest.itertuples()
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


if __name__ == "__main__":
    app()
