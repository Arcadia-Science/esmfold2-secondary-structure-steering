"""Matched-input forwards and prior-initialized designs using the shared design engine.

Deployment and model loading live in analysis.modal_app; these functions preserve
experiment-specific tensors, model outputs, and execution provenance.
"""

from __future__ import annotations
import hashlib
import io
import json
import math
import random
import time
import zipfile
from pathlib import Path

from analysis.targets import INITIALIZATION_SPECS, PRIMARY_CRITIC, SGD_STEPS

SETTINGS = INITIALIZATION_SPECS["softness"]


def tensor_bytes(value):
    import torch

    out = io.BytesIO()
    torch.save(value, out)
    return out.getvalue()


def prepare_screen_inputs() -> dict:
    """Generate CPU logits from the current settings; derive probabilities when used."""
    import torch

    from analysis.design.constants import CYS_IDX, TOKENS
    from analysis.design.sequence import build_initial_soft_sequence_logits

    bundles = {}
    for index in range(SETTINGS["inputs"]):
        with torch.random.fork_rng():
            torch.manual_seed(2026092500 + index)
            logits = build_initial_soft_sequence_logits("#" * SETTINGS["length"], 1).detach()
        temperature = 0.01 + 0.99 * 0.5 * (1 + math.cos(math.pi / SGD_STEPS))
        bundles[f"input{index:02d}.pt"] = tensor_bytes(
            {
                "input_index": index,
                "logits": logits,
                "temperature": temperature,
                "token_order": TOKENS[2:22],
                "cysteine_index": CYS_IDX,
                "initialization_device": "cpu",
            }
        )
    return bundles


def screen(
    model, audit, sources, bundle: bytes, target_sequence: str, protocol_id: str, archive
) -> bytes:
    import numpy as np
    import torch
    import torch.nn.functional as F

    from analysis.design.folding import fold_and_get_distogram, prepare_folding_inputs
    from analysis.design.losses import get_mid_points
    from analysis.design.sequence import sequence_to_one_hot
    from analysis.design.structure import build_complex

    value = torch.load(io.BytesIO(bundle), map_location="cpu", weights_only=True)
    index = value["input_index"]
    p0 = (value["logits"] / value["temperature"]).softmax(-1).cuda()
    hard = F.one_hot(p0.argmax(-1), 20).float()
    target = sequence_to_one_hot(target_sequence)
    prepared = prepare_folding_inputs(target_sequence, p0, seed=81000 + index)
    folder = Path("/study") / protocol_id / f"input{index:02d}"
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "initialization.pt").write_bytes(bundle)
    torch.save(
        {k: v.detach().cpu() for k, v in prepared["inputs"].items()},
        folder / "prepared-features.pt",
    )
    (folder / "audit.json").write_text(json.dumps(audit, indent=2))
    for name, source in sources.items():
        (folder / name).write_bytes(source)
    archive.commit()
    cases = [
        (mode, dropout, rep)
        for mode in SETTINGS["modes"]
        for dropout in SETTINGS["dropout"]
        for rep in range(SETTINGS["replicates"])
    ]
    random.Random(51200 + index).shuffle(cases)
    for order, (mode, dropout, rep) in enumerate(cases):
        label = f"{mode}_drop{int(dropout * 10)}_rep{rep}"
        marker = folder / f"{label}.json"
        if marker.exists():
            continue
        model.configure_lm_dropout(dropout, force_lm_dropout_during_inference=True)
        assert model.config.lm_dropout == dropout
        readout_seed = 900000 + index * 100 + order
        start = time.monotonic()
        # Clone to prevent any in-place forward operation from changing
        # the feature tensors used by subsequent conditions.
        fixed = {**prepared, "inputs": {k: v.clone() for k, v in prepared["inputs"].items()}}
        with torch.no_grad():
            result = fold_and_get_distogram(
                model,
                target_sequence,
                target,
                p0 if mode == "soft" else hard,
                num_loops=SETTINGS["num_loops"],
                num_sampling_steps=SETTINGS["sampling_steps"],
                calculate_confidence=False,
                seed=readout_seed,
                prepared=fixed,
                use_soft_sequence=mode != "discrete",
            )
        torch.cuda.synchronize()
        elapsed = time.monotonic() - start
        logits = result["distogram_logits"].detach().float()
        probabilities = logits.softmax(-1)
        midpoint = get_mid_points().to(logits.device)
        contacts = probabilities[..., midpoint < 7.0].sum(-1)
        binder = contacts[0, -SETTINGS["length"] :, -SETTINGS["length"] :]
        rung_probs = binder.diagonal(offset=4)
        rung_logits = logits[0, -SETTINGS["length"] :, -SETTINGS["length"] :, :][
            torch.arange(SETTINGS["length"] - 4), torch.arange(SETTINGS["length"] - 4) + 4
        ]
        np.savez_compressed(
            folder / f"{label}.npz",
            contact_probability_7A=contacts[0].cpu().numpy(),
            binder_i4_logits=rung_logits.cpu().numpy(),
            bin_midpoints=midpoint.cpu().numpy(),
            sample_atom_coords=result["output"]["sample_atom_coords"]
            .detach()
            .float()
            .cpu()
            .numpy(),
        )
        (folder / f"{label}.pdb").write_text(
            build_complex(result["inputs"], result["output"]).to_pdb_string()
        )
        marker.write_text(
            json.dumps(
                {
                    "input_index": index,
                    "mode": mode,
                    "dropout": dropout,
                    "replicate": rep,
                    "execution_order": order,
                    "readout_seed": readout_seed,
                    "seconds": elapsed,
                    "helix_proxy": rung_probs.mean().item(),
                    "binder_sequence": prepared["seq_list"][0].split("|")[-1],
                    "binder_chain": "B",
                    "target_sequence": target_sequence,
                    "target_chain": "A",
                    "input_sha256": hashlib.sha256(bundle).hexdigest(),
                    "num_loops": SETTINGS["num_loops"],
                    "sampling_steps": SETTINGS["sampling_steps"],
                    "sequence_updates": 0,
                    "lm_mask_pct": getattr(model.config, "lm_mask_pct", None),
                    "disable_msa_features": getattr(model.config, "disable_msa_features", None),
                },
                indent=2,
            )
        )
        archive.commit()
        print(
            f"input {index}, {label}: proxy {rung_probs.mean().item():.3f}, {elapsed:.1f}s",
            flush=True,
        )
        del result, logits, probabilities, contacts, binder, rung_probs, rung_logits, fixed
    data = io.BytesIO()
    with zipfile.ZipFile(data, "w", zipfile.ZIP_DEFLATED) as out:
        for file in sorted(folder.iterdir()):
            out.write(file, file.name)
    return data.getvalue()


def controls(model, audit, controls: list[dict], protocol_id: str, archive) -> bytes:
    """Native single-chain checks at the screen's inference settings."""
    import torch
    from esm.models.esmfold2 import ProteinInput, StructurePredictionInput

    from analysis.design.folding import prepare_esmfold2_tensors
    from analysis.design.structure import build_complex

    model.configure_lm_dropout(0.0, force_lm_dropout_during_inference=True)
    folder = Path("/study") / protocol_id / "controls"
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "audit.json").write_text(json.dumps(audit, indent=2))
    for index, item in enumerate(controls):
        raw = StructurePredictionInput(
            sequences=[ProteinInput(id=["0"], sequence=item["sequence"], msa=None)]
        )
        features = {
            k: v.unsqueeze(0).cuda()
            for k, v in prepare_esmfold2_tensors(raw, seed=72000 + index).items()
        }
        with torch.no_grad():
            output = model(
                **features,
                num_loops=SETTINGS["num_loops"],
                num_sampling_steps=SETTINGS["sampling_steps"],
                num_diffusion_samples=1,
                calculate_confidence=False,
                seed=73000 + index,
            )
        (folder / f"{item['name']}.pdb").write_text(build_complex(features, output).to_pdb_string())
        (folder / f"{item['name']}.json").write_text(json.dumps(item, indent=2))
        archive.commit()
    data = io.BytesIO()
    with zipfile.ZipFile(data, "w", zipfile.ZIP_DEFLATED) as out:
        for file in sorted(folder.iterdir()):
            out.write(file, file.name)
    return data.getvalue()


def design_pair(driver, audit, sources, bundle: dict, protocol: dict, archive) -> bytes:
    import pandas as pd
    import torch
    from esm.models.esmfold2.constants import PROTEIN_3TO1

    from analysis.design.constants import CYS_IDX, TOKENS
    from analysis.design.folding import fold_and_get_distogram, prepare_folding_inputs
    from analysis.design.loop import design_binder
    from analysis.design.losses import get_mid_points
    from analysis.design.sequence import sequence_to_one_hot
    from analysis.design.structure import build_complex
    from analysis.steering import helix_mean_loss

    index = bundle["pair"]
    folder = Path("/study") / protocol["protocol_id"] / "runs" / f"pair{index:02d}"
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "protocol.json").write_text(json.dumps(protocol, indent=2))
    (folder / "audit.json").write_text(json.dumps(audit, indent=2))
    for name, source in sources.items():
        (folder / name).write_bytes(source)
    order20 = [PROTEIN_3TO1[t] for t in TOKENS[2:22]]
    temperature0 = 0.01 + 0.99 * 0.5 * (1 + math.cos(math.pi / protocol["annealing_horizon"]))
    probabilities, logits = {}, {}
    for arm in ("helix", "sheet"):
        p19 = torch.tensor(bundle[arm], dtype=torch.float32)
        p20 = torch.zeros(1, protocol["binder_length"], 20)
        for j, aa in enumerate(bundle["aa_order"]):
            p20[0, :, order20.index(aa)] = p19[:, j]
        init = temperature0 * p20.clamp_min(1e-30).log()
        init[..., CYS_IDX] = -1e6
        recovered = (init / temperature0).softmax(-1)
        assert torch.allclose(recovered, p20, atol=1e-7, rtol=1e-6)
        probabilities[arm], logits[arm] = recovered, init
    assert torch.equal(probabilities["helix"].argmax(-1), probabilities["sheet"].argmax(-1))
    target = protocol["target_sequence"]
    prepared = prepare_folding_inputs(target, probabilities["helix"].cuda(), seed=72000 + index)
    assert prepared["seq_list"] == [target + "|" + bundle["sequence"]]
    cpu_features = {k: v.detach().cpu().clone() for k, v in prepared["inputs"].items()}
    torch.save(
        {
            "logits": logits,
            "features": cpu_features,
            "bundle": {k: v for k, v in bundle.items() if k not in {"helix", "sheet"}},
            "model_aa_order": order20,
            "temperature0": temperature0,
        },
        folder / "initialization.pt",
    )
    archive.commit()
    arm_order = ["helix", "sheet"] if index % 2 == 0 else ["sheet", "helix"]
    n_steps = protocol["updates"]
    for arm in arm_order:
        dest = folder / arm
        dest.mkdir(exist_ok=True)
        if (dest / "complete.json").exists():
            continue
        # Do not silently resume a partial optimizer with different RNG state.
        if (dest / "step_trace.jsonl").exists():
            raise RuntimeError(f"Partial trajectory requires explicit recovery: {dest}")

        def observe(state, dest=dest, arm=arm):
            state.pop("probabilities")  # Derived from saved logits and temperature.
            torch.save(state, dest / f"state{state['step']:03d}.pt")
            if (state["step"] + 1) % 5 == 0:
                archive.commit()
            print(f"pair={index} arm={arm} optimized={state['step'] + 1}/{n_steps}", flush=True)

        fixed = {
            "inputs": {k: v.cuda().clone() for k, v in cpu_features.items()},
            "inputs_list": [],
            "seq_list": list(prepared["seq_list"]),
        }
        result = design_binder(
            inversion_models=driver.inversion_models,
            hf_critic_models={},
            esmc_model=driver.esmc_model,
            target_name=protocol["target"],
            target_sequence=target,
            binder_name="soft_prior",
            binder_sequence="#" * protocol["binder_length"],
            is_antibody=False,
            seed=81000 + index,
            steps=protocol["annealing_horizon"],
            stop_after_steps=n_steps,
            sampling_steps=1,
            initial_logits=logits[arm],
            initial_prepared=fixed,
            step_observer=observe,
            trace_path=dest / "step_trace.jsonl",
        )
        trace = pd.DataFrame([r.row() for r in result[3]])
        trace.to_csv(dest / "step_trace.csv", index=False)
        del result
        archive.commit()
        target_hot = sequence_to_one_hot(target)
        readouts = []
        for step in range(n_steps):
            state = torch.load(dest / f"state{step:03d}.pt", weights_only=True)
            feature = state["prepared"]
            feature["inputs"] = {k: v.cuda() for k, v in feature["inputs"].items()}
            with torch.no_grad():
                folded = fold_and_get_distogram(
                    driver.inversion_models[state["inversion_model"]],
                    target,
                    target_hot,
                    (state["logits"].cuda() / state["temperature"]).softmax(-1),
                    num_loops=1,
                    num_sampling_steps=protocol["coordinate_solver_steps"],
                    calculate_confidence=False,
                    seed=state["seed"],
                    prepared=feature,
                )
                helix = helix_mean_loss(
                    folded["distogram_logits"], protocol["binder_length"], get_mid_points()
                ).item()
                (dest / f"step{step:03d}.pdb").write_text(
                    build_complex(folded["inputs"], folded["output"]).to_pdb_string()
                )
            readouts.append({"step": step, "helix_proxy_readout": helix})
            pd.DataFrame(readouts).to_csv(dest / "readouts.csv", index=False)
            del state, feature, folded
            if (step + 1) % 5 == 0:
                archive.commit()
            print(f"pair={index} arm={arm} readouts={step + 1}/{n_steps}", flush=True)
        metadata = {
            "pair": index,
            "arm": arm,
            "arm_order": arm_order,
            "optimization_seed": 81000 + index,
            "updates": n_steps,
            "annealing_horizon": protocol["annealing_horizon"],
            "optimization_solver_steps": 1,
            "readout_solver_steps": protocol["coordinate_solver_steps"],
            "num_loops": 1,
            "state_convention": "step k is before update k; post_update_logits also saved",
            "shared_initial_sequence": bundle["sequence"],
            "initial_entropy": float(
                -(probabilities[arm].clamp_min(1e-30).log2() * probabilities[arm]).sum(-1).mean()
            ),
        }
        (dest / "complete.json").write_text(json.dumps(metadata, indent=2))
        archive.commit()
    out = io.BytesIO()
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as zipped:
        for file in sorted(folder.rglob("*")):
            if file.is_file():
                zipped.write(file, str(file.relative_to(folder)))
    return out.getvalue()


def design_prior(driver, audit, sources, bundle: dict, protocol: dict, archive) -> bytes:
    import pandas as pd
    import torch
    from esm.models.esmfold2.constants import PROTEIN_3TO1

    from analysis.design.constants import CYS_IDX, TOKENS
    from analysis.design.folding import prepare_folding_inputs
    from analysis.design.loop import design_binder

    seed = bundle["seed"]
    directory = Path("/study") / protocol["protocol_id"] / f"seed{seed:02d}"
    directory.mkdir(parents=True, exist_ok=True)

    def pack():
        stream = io.BytesIO()
        with zipfile.ZipFile(stream, "w", zipfile.ZIP_DEFLATED) as z:
            for file in sorted(directory.iterdir()):
                if file.is_file():
                    z.write(file, file.name)
        return stream.getvalue()

    if (directory / "complete.json").exists():
        return pack()
    if (directory / "step_trace.jsonl").exists():
        raise RuntimeError(f"Partial optimizer run requires explicit recovery: {directory}")
    (directory / "protocol.json").write_text(json.dumps(protocol, indent=2))
    audit = dict(audit)
    audit["critics"] = {n: m.config.to_dict() for n, m in driver.hf_critic_models.items()}
    (directory / "audit.json").write_text(json.dumps(audit, indent=2))
    for name, source in sources.items():
        (directory / name).write_bytes(source)
    sequence = bundle["initial_sequence"]
    length = len(sequence)
    order = [PROTEIN_3TO1[t] for t in TOKENS[2:22]]
    p = torch.zeros(1, length, 20)
    q = dict(zip(protocol["aa_order"], protocol["sheet_pdf"], strict=True))
    for i, chosen in enumerate(sequence):
        for aa in protocol["aa_order"]:
            p[0, i, order.index(aa)] = q[aa] + protocol["boost"] * (float(aa == chosen) - 1 / 19)
    t0 = 0.01 + 0.99 * 0.5 * (1 + math.cos(math.pi / protocol["steps"]))
    logits = t0 * p.clamp_min(1e-30).log()
    logits[..., CYS_IDX] = -1e6
    recovered = (logits / t0).softmax(-1)
    assert torch.allclose(p, recovered, atol=1e-7, rtol=1e-6)
    assert "".join(order[i] for i in p.argmax(-1)[0]) == sequence
    target = protocol["target_sequence"]
    if protocol["initialization_temperature"] == 1:
        prepared = prepare_folding_inputs(target, p.cuda(), seed=seed)
    else:
        assert protocol["initialization_temperature"] in (2, 4)
        saved_features = torch.load(io.BytesIO(bundle["features_blob"]), weights_only=True)
        prepared = {
            "inputs": {k: v.cuda().clone() for k, v in saved_features.items()},
            "inputs_list": [],
            "seq_list": [target + "|" + sequence],
        }
    cpu_features = {k: v.detach().cpu().clone() for k, v in prepared["inputs"].items()}
    torch.save(
        {
            "logits": logits,
            "features": cpu_features,
            "bundle": {k: v for k, v in bundle.items() if k != "features_blob"},
            "model_aa_order": order,
            "temperature0": t0,
        },
        directory / "initialization.pt",
    )
    archive.commit()
    states = []

    def observe(state):
        state.pop("probabilities")  # Derived from saved logits and temperature.
        features = state.pop("prepared")
        state["sequences"] = features["seq_list"]
        if state["step"] == 0:
            for key, value in cpu_features.items():
                assert torch.equal(value, features["inputs"][key])
        states.append(state)
        if (state["step"] + 1) % 25 == 0:
            torch.save(states, directory / "sequence-states.pt")
            archive.commit()
            print(
                f"T={protocol['initialization_temperature']} seed={seed} "
                f"updates={state['step'] + 1}/{protocol['steps']}",
                flush=True,
            )

    _, _, critics, trace, _ = design_binder(
        inversion_models=driver.inversion_models,
        hf_critic_models=driver.hf_critic_models,
        esmc_model=driver.esmc_model,
        target_name=protocol["target"],
        target_sequence=target,
        binder_name="sheet_prior",
        binder_sequence="#" * length,
        is_antibody=False,
        seed=seed,
        steps=protocol["steps"],
        sampling_steps=protocol["optimization_solver_steps"],
        confidence_sampling_steps=protocol["confidence_selection_solver_steps"],
        initial_logits=logits,
        initial_prepared=prepared,
        step_observer=observe,
        trace_path=directory / "step_trace.jsonl",
    )
    pd.DataFrame([r.row() for r in trace]).to_csv(directory / "step_trace.csv", index=False)
    records = []
    for result in critics:
        name = result["critic_name"]
        (directory / f"{name}.pdb").write_text(result["complex"].to_pdb_string())
        records.append({k: v for k, v in result.items() if k not in ("complex", "logits")})
    (directory / "critics.json").write_text(json.dumps(records, indent=2))
    primary = next(r for r in records if r["critic_name"] == PRIMARY_CRITIC)
    metadata = {
        "seed": seed,
        "binder_length": length,
        "steps": len(trace),
        "arm": protocol["arm"],
        "initialization_temperature": protocol["initialization_temperature"],
        "boost": protocol["boost"],
        "initial_sequence": sequence,
        "binder_sequence": primary["designed_sequence"].split("|")[-1],
        "primary_critic": PRIMARY_CRITIC,
        "iptm": primary["iptm"],
        "plddt": primary["plddt_binder"],
        "selected_step": max((r for r in trace if r.iptm is not None), key=lambda r: r.iptm).step,
    }
    (directory / "complete.json").write_text(json.dumps(metadata, indent=2))
    archive.commit()
    print(
        f"T={protocol['initialization_temperature']} seed={seed} COMPLETE "
        f"iptm={metadata['iptm']:.3f} plddt={metadata['plddt']:.1f}",
        flush=True,
    )
    return pack()
