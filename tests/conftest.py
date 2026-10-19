"""Small preserved model payloads for collection tests; no GPU or network."""

import io
import json
import zipfile

import pandas as pd
import pytest
import torch
from analysis.targets import CRITIC_MODELS, PRIMARY_CRITIC, short_trajectory_arms


@pytest.fixture
def trajectory_archive():
    def build(protocol, seed=0, sequence="AI"):
        features = {"ref_pos": torch.tensor([[seed, 1.0, 2.0]])}
        logits = {prior: torch.tensor([[float(i)]]) for i, prior in enumerate(protocol["arms"])}
        initial = {
            "bundle": {"pair": seed, "sequence": sequence},
            "features": features,
            "logits": logits,
        }
        output = io.BytesIO()

        def tensor_bytes(value):
            buffer = io.BytesIO()
            torch.save(value, buffer)
            return buffer.getvalue()

        with zipfile.ZipFile(output, "w") as archive:
            archive.writestr("protocol.json", json.dumps(protocol))
            archive.writestr("audit.json", json.dumps({"model_revision": "original"}))
            archive.writestr("model.py", "# preserved model source\n")
            archive.writestr("initialization.pt", tensor_bytes(initial))
            for prior in protocol["arms"]:
                record = {
                    "pair": seed,
                    "arm": prior,
                    "optimization_seed": 81000 + seed,
                    "updates": protocol["updates"],
                    "annealing_horizon": protocol["annealing_horizon"],
                    "initialization_temperature": protocol["initialization_temperature"],
                    "probability_margin": protocol["probability_margin"],
                    "shared_initial_sequence": sequence,
                }
                archive.writestr(f"{prior}/complete.json", json.dumps(record))
                trace = pd.DataFrame(
                    {
                        "target": protocol["target"],
                        "arm": short_trajectory_arms(protocol["initialization_temperature"])[prior],
                        "seed": 81000 + seed,
                        "step (index)": range(protocol["updates"]),
                        "binder sequence": sequence,
                        "binder length (residues)": len(sequence),
                        "steering_loss (dimensionless)": 0.0,
                        "inversion model": "model1",
                        "temperature (0-1)": 0.999,
                        "helix_mean (0-1)": 0.2,
                        "sequence entropy (bits)": 4.0,
                    }
                )
                archive.writestr(f"{prior}/step_trace.csv", trace.to_csv(index=False))
                archive.writestr(
                    f"{prior}/step_trace.jsonl", trace.to_json(orient="records", lines=True)
                )
                readouts = pd.DataFrame(
                    {"step": range(protocol["updates"]), "helix_proxy_readout": 0.2}
                )
                archive.writestr(f"{prior}/readouts.csv", readouts.to_csv(index=False))
                for step in range(protocol["updates"]):
                    archive.writestr(
                        f"{prior}/step{step:03d}.pdb", f"coordinates {seed} {prior} {step}"
                    )
                    state = {
                        "logits": logits[prior],
                        "prepared": {
                            "inputs": features,
                            "seq_list": [protocol["target_sequence"] + "|" + sequence],
                        },
                    }
                    archive.writestr(f"{prior}/state{step:03d}.pt", tensor_bytes(state))
        return output.getvalue()

    return build


@pytest.fixture
def full_design_archive():
    """A full worker result with real, simple backbone coordinates and all critics."""
    from biotite.sequence import ProteinSequence

    def build(protocol, seed=0, sequence="AI"):
        def tensor_bytes(value):
            buffer = io.BytesIO()
            torch.save(value, buffer)
            return buffer.getvalue()

        atoms = []
        for index, aa in enumerate(sequence, start=1):
            for name, offset in (("N", -1.2), ("CA", 0.0), ("C", 1.3), ("O", 2.0)):
                atoms.append(
                    f"ATOM  {len(atoms) + 1:5d} {name:^4s} "
                    f"{ProteinSequence.convert_letter_1to3(aa):>3s} "
                    f"B{index:4d}    {3.8 * index + offset:8.3f}{0.0:8.3f}{0.0:8.3f}"
                    f"{1.0:6.2f}{85.0:6.2f}          {name[0]:>2s}\n"
                )
        pdb = "".join(atoms) + "END\n"
        steps = protocol["steps"]
        trace = pd.DataFrame(
            {
                "target": protocol["target"],
                "arm": protocol["arm"],
                "seed": seed,
                "step (index)": range(steps),
                "binder sequence": sequence,
                "binder length (residues)": len(sequence),
                "steering_loss (dimensionless)": 0.0,
                "base_structure_weight (dimensionless)": 1.0,
                "inversion model": "model1",
                "temperature (0-1)": 0.999,
                "sequence entropy (bits)": 4.0,
                "iptm (0-1)": [None] * (steps - 1) + [0.9],
            }
        )
        critics = [
            {
                "critic_name": name,
                "designed_sequence": protocol["target_sequence"] + "|" + sequence,
                "iptm": 0.9,
                "plddt_binder": 85.0,
                "plddt_target": 92.0,
                "plddt_complex": 89.0,
                "distogram_iptm_proxy": 0.8,
                "final_loss": 1.2,
            }
            for name in CRITIC_MODELS
        ]
        complete = {
            "seed": seed,
            "arm": protocol["arm"],
            "steps": steps,
            "initialization_temperature": protocol["initialization_temperature"],
            "probability_margin": protocol["probability_margin"],
            "initial_sequence": sequence,
            "binder_sequence": sequence,
            "binder_length": len(sequence),
            "selected_step": steps - 1,
            "primary_critic": PRIMARY_CRITIC,
            "iptm": 0.9,
            "plddt": 85.0,
        }
        output = io.BytesIO()
        with zipfile.ZipFile(output, "w") as archive:
            for name, data in (
                ("protocol.json", protocol),
                ("complete.json", complete),
                ("critics.json", critics),
                ("audit.json", {"model_revision": "original"}),
            ):
                archive.writestr(name, json.dumps(data))
            archive.writestr("model.py", "# preserved model source\n")
            archive.writestr(
                "initialization.pt",
                tensor_bytes(
                    {
                        "bundle": {"seed": seed, "initial_sequence": sequence},
                        "features": {"test": torch.tensor([seed])},
                    }
                ),
            )
            archive.writestr(
                "sequence-states.pt", tensor_bytes([{"step": i} for i in range(steps)])
            )
            archive.writestr("step_trace.csv", trace.to_csv(index=False))
            archive.writestr("step_trace.jsonl", trace.to_json(orient="records", lines=True))
            for name in CRITIC_MODELS:
                archive.writestr(f"{name}.pdb", pdb)
        return output.getvalue()

    return build
