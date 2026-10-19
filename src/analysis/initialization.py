"""Prepare matched initialization inputs and measure their preserved model outputs."""

from __future__ import annotations
import copy
import hashlib
import io
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from analysis import composition, deposit, figures, loss_trace, paths
from analysis import secondary_structure as ss
from analysis.priors import (
    AA,
    from_reference,
    matched_probabilities,
    paired_inputs,
    validate_initialization_settings,
    validate_temperature,
)
from analysis.targets import (
    CRITIC_MODELS,
    FULL_INITIALIZATION_ARMS,
    INITIALIZATION_BASELINES,
    INITIALIZATION_SPECS,
    INITIALIZATION_TARGET,
    N_SEEDS,
    PRIMARY_CRITIC,
    SGD_STEPS,
    SHORT_TRAJECTORY_ARMS,
    full_initialization_arm,
    target,
)

ROOT = paths.INITIALIZATION
LABELS = {
    "vanilla": "Unsteered",
    "helix_mean_w10": "Loss weight 1.0",
    "helix_mean_w20": "Loss weight 2.0",
    **{arm: f"Sheet prior T={t}" for t, arm in FULL_INITIALIZATION_ARMS.items()},
}
COLORS = [
    figures.BASELINE_COLOR,
    figures.ARM_COLORS["helix_mean_w10"],
    figures.ARM_COLORS["helix_mean_w20"],
    "#244a78",
    "#258cc0",
    "#39a985",
]


def load(root: Path | None = None) -> dict[str, pd.DataFrame]:
    """Read model outputs and recompute measurements; never save derived tables."""
    if root is None:
        root = ROOT
        if not (root / "softness" / "protocol.json").is_file():
            deposit.INITIALIZATION_INPUTS.fetch()
    designs, hardening, critics = full_designs()
    softness, forwards = measure_softness(root / "softness")
    tables = {
        "designs": designs,
        "short-trajectories": pd.concat(
            [
                composition.measure_steps(INITIALIZATION_TARGET, arm)
                for arm in SHORT_TRAJECTORY_ARMS.values()
            ],
            ignore_index=True,
        ),
        "softness-inputs": softness,
        "softness-forwards": forwards,
        "hardening": hardening,
        "critic-scores": critics,
    }
    validate(tables)
    return tables


def validate(tables: dict[str, pd.DataFrame]) -> None:
    """Check completeness, matching and finite measurements across the follow-ups."""
    designs = tables["designs"]
    assert set(designs.arm) == set(LABELS)
    assert not designs.duplicated(["arm", "seed"]).any()
    for _, frame in designs.groupby("arm"):
        assert set(frame.seed) == set(range(N_SEEDS))
    assert np.allclose(designs[["helix", "sheet", "coil"]].sum(axis=1), 1)
    assert np.isfinite(designs[["helix", "sheet", "coil", "iptm", "plddt"]]).all().all()
    lengths = designs.pivot(index="seed", columns="arm", values="length")
    assert lengths.nunique(axis=1).eq(1).all()
    short = tables["short-trajectories"]
    loss_trace.validate_matching(short)
    for _, frame in short.groupby("arm"):
        loss_trace.validate_steps(
            frame,
            n_seeds=INITIALIZATION_SPECS["short"]["pairs"],
            steps=INITIALIZATION_SPECS["short"]["updates"],
        )
    softness = tables["softness-inputs"]
    assert len(softness) == (
        INITIALIZATION_SPECS["softness"]["inputs"]
        * len(INITIALIZATION_SPECS["softness"]["modes"])
        * len(INITIALIZATION_SPECS["softness"]["dropout"])
    )
    assert not softness.duplicated(["input_index", "mode", "dropout"]).any()
    for _, frame in softness.groupby(["mode", "dropout"]):
        assert set(frame.input_index) == set(range(INITIALIZATION_SPECS["softness"]["inputs"]))
    critics = tables["critic-scores"]
    assert (
        len(critics) == len(LABELS) * N_SEEDS * len(CRITICS)
        and not critics.duplicated(["arm", "seed", "critic"]).any()
    )
    assert np.isfinite(critics[["iptm", "plddt"]]).all().all()


def summary(designs: pd.DataFrame) -> pd.DataFrame:
    """One row per condition, with the same strict screen as the main paper."""
    rows = []
    for arm, label in LABELS.items():
        frame = designs[designs.arm == arm]
        passed = (
            frame.sheet.gt(composition.MIN_SHEET_FRACTION)
            & frame.iptm.gt(composition.MIN_IPTM)
            & frame.plddt.gt(composition.MIN_BINDER_PLDDT)
        )
        rows.append(
            {
                "Method": label,
                "Mean helix (%)": frame.helix.mean() * 100,
                "Mean sheet (%)": frame.sheet.mean() * 100,
                "Mean ipTM": frame.iptm.mean(),
                "Mean pLDDT": frame.plddt.mean(),
                "Screen passes / 30": int(passed.sum()),
            }
        )
    return pd.DataFrame(rows).set_index("Method")


def report() -> dict:
    """Numerical checks for the initialization claims made in the notebook."""
    data = load()
    short = data["short-trajectories"]
    hardening = data["hardening"]
    return {
        "full_optimization": summary(data["designs"]).to_dict(orient="index"),
        "short_step39_means": short[short["step (index)"] == 39]
        .groupby("arm")[
            [
                "helix (fraction 0-1)",
                "sheet (fraction 0-1)",
                "helix_mean (0-1)",
                "sequence entropy (bits)",
            ]
        ]
        .mean()
        .to_dict(orient="index"),
        "softness_means_at_dropout05": data["softness-inputs"]
        .query("dropout == 0.5")
        .groupby("mode")[["helix", "sheet"]]
        .mean()
        .to_dict(orient="index"),
        "median_first_step_below_2_bits": hardening.groupby("arm")["first_step_entropy_below_2.0"]
        .median()
        .to_dict(),
    }


def full_outcomes(frame: pd.DataFrame) -> plt.Figure:
    """Primary-critic outcomes after 150 updates, with means and screen cutoffs."""
    with plt.rc_context({"font.size": 10, "axes.labelsize": 10, "xtick.labelsize": 9}):
        fig, axes = plt.subplots(1, 3, figsize=(12, 3.8), layout="constrained")
        rng = np.random.default_rng(20260930)
        for ax, metric, title, threshold in zip(
            axes,
            ("sheet", "iptm", "plddt"),
            ("Final sheet fraction", "Final ipTM", "Final binder pLDDT"),
            (composition.MIN_SHEET_FRACTION, composition.MIN_IPTM, composition.MIN_BINDER_PLDDT),
            strict=True,
        ):
            for i, (arm, color) in enumerate(zip(LABELS, COLORS, strict=True)):
                values = frame.loc[frame.arm == arm, metric]
                ax.scatter(
                    i + rng.uniform(-0.14, 0.14, len(values)), values, color=color, alpha=0.75, s=20
                )
                ax.plot([i - 0.23, i + 0.23], [values.mean()] * 2, color="black", lw=2)
            ax.axhline(threshold, color="#999999", ls="--", lw=1)
            ax.set_xticks(
                range(6),
                ["Unsteered", "Loss\n1.0", "Loss\n2.0", "Prior\nT=1", "Prior\nT=2", "Prior\nT=4"],
            )
            ax.set(ylabel=title, ylim=(0, 100) if metric == "plddt" else (0, 1))
            ax.spines[["top", "right"]].set_visible(False)
        return fig


if __name__ == "__main__":
    print(json.dumps(report(), indent=2, allow_nan=False))


PRIMARY = PRIMARY_CRITIC
BASELINES = INITIALIZATION_BASELINES
CRITICS = set(CRITIC_MODELS)


def protocol(root: Path, experiment: str) -> dict:
    """Validate a run against the current experiment settings."""
    value = read(root / "protocol.json")
    for key, wanted in INITIALIZATION_SPECS[experiment].items():
        if key in {"controls", "torch_compile", "temperatures"}:
            continue
        if value.get(key) != wanted:
            raise ValueError(
                f"Protocol mismatch for {root}: {key}={value.get(key)!r}, expected {wanted!r}"
            )
    if experiment != "softness":
        validate_initialization_settings(value)
    return value


def read(path: Path):
    return json.loads(path.read_text())


def measure(file: Path, sequence: str, chain: str = "B") -> dict:
    """Recompute binder DSSP and backbone geometry from the saved coordinates."""
    backbone = ss.checked_backbone(file, sequence, chain)
    labels = ss.assign(backbone)
    return {
        "length": len(backbone),
        "helix": float((labels == "H").mean()),
        "sheet": float((labels == "E").mean()),
        "coil": float((labels == "-").mean()),
        "geometry_pass": ss.geometry_pass(backbone),
    }


def measure_softness(root: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Measure individual forwards before averaging the two replicates per input."""
    forwards = measure_softness_forwards(root)
    means = (
        forwards.groupby(["input_index", "mode", "dropout"])[["helix", "sheet", "coil"]]
        .mean()
        .reset_index()
    )
    return means, forwards


def measure_softness_forwards(root: Path) -> pd.DataFrame:
    """Measure each preserved prediction, retaining input and replicate identities."""
    settings = protocol(root, "softness")
    rows = []
    for index in range(settings["inputs"]):
        for mode in settings["modes"]:
            for dropout_value in settings["dropout"]:
                dropout = round(dropout_value * 10)
                for replicate in range(settings["replicates"]):
                    file = root / f"input{index:02d}/{mode}_drop{dropout}_rep{replicate}.json"
                    record = read(file)
                    if (
                        record["input_index"],
                        record["mode"],
                        record["dropout"],
                        record["replicate"],
                    ) != (index, mode, dropout / 10, replicate):
                        raise ValueError(f"Unexpected forward identity: {file}")
                    rows.append(
                        {
                            "input_index": index,
                            "mode": mode,
                            "dropout": dropout / 10,
                            "replicate": replicate,
                            **measure(
                                file.with_suffix(".pdb"),
                                record["binder_sequence"],
                                record["binder_chain"],
                            ),
                        }
                    )
    return pd.DataFrame(rows)


def trace(path: Path, steps: int) -> pd.DataFrame:
    frame = pd.read_csv(path)
    if frame["step (index)"].tolist() != list(range(steps)):
        raise ValueError(f"Missing or duplicate optimization steps: {path}")
    return frame


def hardening_summary(traces: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (arm, seed), frame in traces.groupby(["arm", "seed"]):
        frame = frame.sort_values("step (index)")
        if frame["step (index)"].tolist() != list(range(SGD_STEPS)):
            raise ValueError(f"Incomplete trace for {arm}, seed {seed}")
        row = {"arm": arm, "seed": seed}
        for cutoff in (3.0, 2.0, 1.0, 0.1):
            chosen = frame.loc[frame["sequence entropy (bits)"] < cutoff, "step (index)"]
            row[f"first_step_entropy_below_{cutoff}"] = (
                None if chosen.empty else int(chosen.iloc[0])
            )
        for step in (0, 40, 75, 100, SGD_STEPS - 1):
            row[f"entropy_step{step}"] = float(
                frame.loc[frame["step (index)"] == step, "sequence entropy (bits)"].iloc[0]
            )
        rows.append(row)
    return pd.DataFrame(rows)


def full_designs() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Summarize priors and controls through the same campaign readers."""
    designs, critics, traces = [], [], []
    columns = {
        "helix (fraction 0-1)": "helix",
        "sheet (fraction 0-1)": "sheet",
        "coil (fraction 0-1)": "coil",
        "iptm (0-1)": "iptm",
        "binder pLDDT (0-100)": "plddt",
        "binder length (residues)": "length",
    }
    for arm in (*BASELINES, *FULL_INITIALIZATION_ARMS.values()):
        designs.append(composition.load(INITIALIZATION_TARGET, arm).rename(columns=columns))
        traces.append(loss_trace.load_campaign(INITIALIZATION_TARGET, arm))
        for record in pd.read_parquet(paths.campaign_results(INITIALIZATION_TARGET, arm)).to_dict(
            "records"
        ):
            for name in CRITIC_MODELS:
                critics.append(
                    {
                        "arm": arm,
                        "seed": record["seed"],
                        "critic": name,
                        "iptm": record["critic_iptm_by_model"][name],
                        "plddt": record["critic_plddt_binder_by_model"][name],
                    }
                )
    return (
        pd.concat(designs, ignore_index=True),
        hardening_summary(pd.concat(traces, ignore_index=True)),
        pd.DataFrame(critics),
    )


def save(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def temperature_label(temperature: float) -> str:
    """Use the same label for integer and floating-point spellings of a temperature."""
    validate_temperature(temperature)
    value = float(temperature)
    return str(int(value)) if value.is_integer() else str(value)


def experiment_dir(root: Path, experiment: str, temperature: float) -> Path:
    validate_experiment(experiment, temperature)
    if experiment == "full" or (experiment == "short" and temperature != 1):
        return root / f"{experiment}-T{temperature_label(temperature)}"
    return root / experiment


def source_hashes() -> dict:
    return {
        str(p.relative_to(paths.REPO_ROOT)): digest(p)
        for p in sorted((paths.REPO_ROOT / "src/analysis").rglob("*.py"))
    }


def prepare_inputs(root: Path, experiment: str, temperature: float) -> Path:
    validate_experiment(experiment, temperature)
    directory = experiment_dir(root, experiment, temperature)
    if (directory / "protocol.json").exists():
        raise FileExistsError(f"Already prepared: {directory}. Reuse it or select a new --root.")
    protocol = copy.deepcopy(INITIALIZATION_SPECS[experiment])
    protocol["target"] = INITIALIZATION_TARGET
    protocol["target_sequence"] = target(INITIALIZATION_TARGET).sequence()
    if experiment == "full":
        protocol["seeds"] = list(range(protocol["n_seeds"]))
    from analysis.modal_app import APP_NAME

    protocol["app"] = APP_NAME
    protocol["experiment"] = experiment
    protocol["source_sha256"] = source_hashes()
    protocol["purpose"] = "Fresh reproduction; outcomes are not seed-reproducible"
    prior = from_reference() if experiment != "softness" else None
    if prior is not None:
        if prior["amino_acid_order"] != AA:
            raise ValueError("Unexpected prior amino-acid order")
        protocol["initialization_temperature"] = float(temperature)
        validate_initialization_settings(protocol)
        protocol["tempering"] = "q_T = normalize(q**(1/initialization_temperature))"
        protocol["initialization"] = "q_T + b_i*(onehot(chosen_aa)-uniform19)"
        protocol["boost_rule"] = "max(0, max_other(q_T) - q_T(chosen_aa) + probability_margin)"
        protocol["prior_sha256"] = hashlib.sha256(
            json.dumps(prior, sort_keys=True).encode()
        ).hexdigest()
    directory.mkdir(parents=True, exist_ok=True)
    if experiment == "softness":
        import modal

        bundles = modal.Function.from_name(protocol["app"], "prepare_screen_inputs").remote()
        for name, data in bundles.items():
            if Path(name).name != name:
                raise ValueError(f"Unexpected input name: {name}")
            destination = directory / "inputs" / name
            destination.parent.mkdir(exist_ok=True)
            destination.write_bytes(data)
        protocol["input_sha256"] = {
            name: hashlib.sha256(data).hexdigest() for name, data in bundles.items()
        }
    elif experiment == "short":
        bundles = paired_inputs()
        for item in bundles:
            for arm in protocol["arms"]:
                matched_probabilities(
                    prior[arm],
                    item["sequence"],
                    temperature=temperature,
                    margin=protocol["probability_margin"],
                )
        save(directory / "inputs.json", bundles)
        protocol["bundles_sha256"] = digest(directory / "inputs.json")
    else:
        import torch

        source = paths.campaign_trace(INITIALIZATION_TARGET, "vanilla")
        trace = pd.read_parquet(source)
        baseline_trace_sha256 = digest(source)
        initial = trace[trace["step (index)"] == 0].set_index("seed").sort_index()
        if list(initial.index) != list(range(protocol["n_seeds"])):
            raise ValueError(f"Expected {protocol['n_seeds']} baseline initial sequences")
        bundles = []
        for seed, row in initial.iterrows():
            sequence = row["binder sequence"]
            matched_probabilities(
                prior["sheet"],
                sequence,
                temperature=temperature,
                margin=protocol["probability_margin"],
            )
            item = {"seed": int(seed), "initial_sequence": sequence}
            if temperature != 1:
                t1 = torch.load(
                    paths.campaign_run(INITIALIZATION_TARGET, full_initialization_arm(1), seed)
                    / "initialization.pt",
                    weights_only=True,
                )
                if t1["bundle"]["initial_sequence"] != sequence:
                    raise ValueError(
                        "T=1 inputs differ from the current baseline; use its baseline"
                    )
                stream = io.BytesIO()
                torch.save(t1["features"], stream)
                destination = directory / "inputs" / f"seed{seed:02d}-features.pt"
                destination.parent.mkdir(exist_ok=True)
                destination.write_bytes(stream.getvalue())
                item["features_sha256"] = digest(destination)
            bundles.append(item)
        save(directory / "inputs.json", bundles)
        protocol.update(
            {
                "arm": full_initialization_arm(temperature),
                "aa_order": prior["amino_acid_order"],
                "inputs_sha256": digest(directory / "inputs.json"),
                "matching": (
                    "Baseline hard sequences and lengths; "
                    "other temperatures reuse T=1 atom features"
                ),
                "baseline_trace_sha256": baseline_trace_sha256,
            }
        )
    protocol["protocol_id"] = hashlib.sha256(
        json.dumps(protocol, sort_keys=True).encode()
    ).hexdigest()[:16]
    save(directory / "protocol.json", protocol)
    return directory


def validate_experiment(experiment: str, temperature: float) -> None:
    if experiment not in INITIALIZATION_SPECS:
        raise ValueError(f"Unknown initialization experiment: {experiment}")
    validate_temperature(temperature)
    if experiment == "softness" and temperature != 1:
        raise ValueError(
            "Temperature is only configurable for short and full initialization designs"
        )


def prepared_jobs(root: Path, experiment: str, temperature: float) -> tuple[dict, list[dict]]:
    """Validate preserved inputs and derive worker arguments before any submission."""
    validate_experiment(experiment, temperature)
    directory = experiment_dir(root, experiment, temperature)
    protocol = read(directory / "protocol.json")
    if protocol["source_sha256"] != source_hashes():
        raise RuntimeError("Source changed since preparation; deploy and prepare in a new --root")
    if protocol.get("experiment") != experiment or (
        experiment != "softness" and protocol["initialization_temperature"] != temperature
    ):
        raise ValueError("Prepared protocol does not match the requested experiment")
    recorded_id = protocol["protocol_id"]
    original = {k: v for k, v in protocol.items() if k != "protocol_id"}
    actual_id = hashlib.sha256(json.dumps(original, sort_keys=True).encode()).hexdigest()[:16]
    if recorded_id != actual_id:
        raise ValueError("Prepared protocol changed; prepare inputs in a new --root")
    if experiment != "softness":
        validate_initialization_settings(protocol)
    prior = from_reference() if experiment != "softness" else None
    if prior is not None:
        current = hashlib.sha256(json.dumps(prior, sort_keys=True).encode()).hexdigest()
        if current != protocol["prior_sha256"]:
            raise ValueError("Reference prior changed since preparation")
    jobs = []
    if experiment == "softness":
        for index in range(protocol["inputs"]):
            name = f"input{index:02d}"
            source = directory / "inputs" / f"{name}.pt"
            if digest(source) != protocol["input_sha256"][source.name]:
                raise ValueError(f"Changed input: {source}")
            jobs.append(
                {
                    "run": name,
                    "method": "screen",
                    "kwargs": {
                        "bundle": source.read_bytes(),
                        "target_sequence": protocol["target_sequence"],
                        "protocol_id": protocol["protocol_id"],
                    },
                }
            )
        jobs.append(
            {
                "run": "controls",
                "method": "controls",
                "kwargs": {
                    "controls": protocol["controls"],
                    "protocol_id": protocol["protocol_id"],
                },
            }
        )
        return protocol, jobs
    inputs = directory / "inputs.json"
    expected = protocol["bundles_sha256" if experiment == "short" else "inputs_sha256"]
    if digest(inputs) != expected:
        raise ValueError("Prepared inputs changed")
    if experiment == "full":
        protocol = {**protocol, "sheet_pdf": prior["sheet"]}
    else:
        protocol = {**protocol, "priors": {arm: prior[arm] for arm in protocol["arms"]}}
    for item in read(inputs):
        if experiment == "short":
            bundle = {
                **item,
                **{
                    arm: matched_probabilities(
                        prior[arm],
                        item["sequence"],
                        temperature=temperature,
                        margin=protocol["probability_margin"],
                    ).tolist()
                    for arm in protocol["arms"]
                },
            }
            name, method = f"runs/pair{item['pair']:02d}", "design_pair"
        else:
            bundle = item
            if temperature != 1:
                source = directory / "inputs" / f"seed{item['seed']:02d}-features.pt"
                if digest(source) != item["features_sha256"]:
                    raise ValueError(f"Changed features: {source}")
                bundle = {**item, "features_blob": source.read_bytes()}
            name, method = f"seed{item['seed']:02d}", "design_prior"
        identity = {"seed": item["seed"]} if experiment == "full" else {"run": name}
        jobs.append(
            {**identity, "method": method, "kwargs": {"bundle": bundle, "protocol": protocol}}
        )
    return protocol, jobs
