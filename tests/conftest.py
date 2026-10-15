"""Small preserved model payloads for collection tests; no GPU or network."""

import io
import json
import zipfile

import pandas as pd
import pytest
import torch
from analysis.targets import short_trajectory_arms


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
