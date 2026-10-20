"""Shared result models preserve worker formats and validate archive metadata."""

import sys
from types import ModuleType, SimpleNamespace

import pandas as pd
import pytest
from analysis import modal_app
from analysis.results import (
    CampaignOutput,
    DesignResult,
    RunSpec,
)
from analysis.targets import CRITIC_MODELS, PRIMARY_CRITIC, arm


@pytest.fixture
def archived_result():
    return SimpleNamespace(
        protocol={"target": "pdl1_igv", "arm": "vanilla", "steps": 3},
        critics=[
            {
                "critic_name": name,
                "designed_sequence": "AAA|AI",
                "iptm": 0.9,
                "plddt_binder": 85.0,
                "plddt_target": 92.0,
                "plddt_complex": 89.0,
                "distogram_iptm_proxy": 0.8,
                "final_loss": 1.2,
            }
            for name in CRITIC_MODELS
        ],
        pdb="MODEL\nENDMDL\n",
    )


def test_worker_preserves_payload_and_table_schema(archived_result, monkeypatch, tmp_path):
    # Exercise the actual Modal method without loading esm or launching a GPU.
    design = ModuleType("analysis.design")
    design.loop = SimpleNamespace(SAMPLING_STEPS=1, CONFIDENCE_SAMPLING_STEPS=50)
    monkeypatch.setitem(sys.modules, "analysis.design", design)
    records = [
        {**record, "complex": SimpleNamespace(to_pdb_string=lambda: archived_result.pdb)}
        for record in archived_result.critics
    ]
    trace = [SimpleNamespace(row=lambda step=i: {"step (index)": step}) for i in range(3)]
    cls = modal_app.ESMFold2DesignModal._get_user_cls()
    worker = cls()
    worker.driver = SimpleNamespace(design=lambda **kwargs: ([], [{}, {}, {}], records, trace, []))
    payload = cls.design._get_raw_f()(
        worker, seed=0, steps=3, target="pdl1_igv", target_sequence="AAA", arm="vanilla"
    )
    expected = {
        "seed": 0,
        "steps": 3,
        "target": "pdl1_igv",
        "arm": "vanilla",
        "arm_label": "Vanilla",
        "target_length_residues": 3,
        "binder_name": modal_app.BINDER_NAME,
        "designed_sequence": "AAA|AI",
        "binder_sequence": "AI",
        "binder_length_residues": 2,
        "final_loss": 1.2,
        "critic_name": PRIMARY_CRITIC,
        "critic_iptm": 0.9,
        "critic_plddt_binder": 85.0,
        "critic_plddt_target": 92.0,
        "critic_plddt_complex": 89.0,
        "critic_distogram_iptm_proxy": 0.8,
        "critic_iptm_by_model": dict.fromkeys(CRITIC_MODELS, 0.9),
        "critic_plddt_binder_by_model": dict.fromkeys(CRITIC_MODELS, 85.0),
        "critic_distogram_iptm_proxy_by_model": dict.fromkeys(CRITIC_MODELS, 0.8),
        "design_steps_recorded": 3,
    }
    assert payload == {
        **expected,
        "schedule": arm("vanilla").schedule.to_dict(),
        "step_trace": [{"step (index)": i} for i in range(3)],
        "complex_pdb": archived_result.pdb,
        "step_complexes_gz": b"",
    }
    collected = CampaignOutput.from_payload(payload, seed=0)
    assert collected.row == expected
    path = tmp_path / "results.parquet"
    pd.DataFrame([collected.row]).to_parquet(path, index=False)
    assert pd.read_parquet(path).iloc[0].to_dict() == expected


def test_critic_records_must_agree_on_sequence_and_have_unique_names(archived_result):
    run = RunSpec.from_protocol(archived_result.protocol, seed=0)
    critics = archived_result.critics
    with pytest.raises(ValueError, match="duplicate critics"):
        DesignResult.from_critics(run, [*critics, critics[0]])
    with pytest.raises(ValueError, match="Missing primary critic"):
        DesignResult.from_critics(run, [r for r in critics if r["critic_name"] != PRIMARY_CRITIC])
    with pytest.raises(ValueError, match="Critic sequence differs"):
        DesignResult.from_critics(
            run, [{**critics[0], "designed_sequence": "AAA|II"}, *critics[1:]]
        )


def test_optional_critic_without_iptm_keeps_its_other_scores(archived_result):
    run = RunSpec(target="pdl1_igv", arm="vanilla", seed=0, steps=3)
    scaling = {**archived_result.critics[0], "critic_name": "scaling-critic", "iptm": None}
    result = DesignResult.from_critics(run, [*archived_result.critics, scaling])
    assert result.row()["critic_iptm_by_model"]["scaling-critic"] is None
    assert result.row()["critic_distogram_iptm_proxy_by_model"]["scaling-critic"] == 0.8


def test_incomplete_trace_and_wrong_manifest_seed_are_rejected(archived_result):
    run = RunSpec.from_protocol(archived_result.protocol, seed=0)
    result = DesignResult.from_critics(run, archived_result.critics, selected_step=2)
    with pytest.raises(ValueError, match="Expected 3 trace rows, observed 2"):
        CampaignOutput.from_design(result, [{}, {}], archived_result.pdb)
    with pytest.raises(ValueError, match="expected 1, observed 0"):
        CampaignOutput.from_payload({"seed": 0, "step_trace": []}, seed=1)


@pytest.mark.parametrize("sampling_steps,expected", [(0, (1, 50)), (1, (1, 1)), (7, (7, 7))])
def test_shared_worker_preserves_default_solver_split_and_explicit_overrides(
    monkeypatch, sampling_steps, expected
):
    import sys
    from types import ModuleType, SimpleNamespace

    from analysis import modal_app

    design = ModuleType("analysis.design")
    loop = ModuleType("analysis.design.loop")
    loop.SAMPLING_STEPS, loop.CONFIDENCE_SAMPLING_STEPS = 1, 50
    design.loop = loop
    monkeypatch.setitem(sys.modules, "analysis.design", design)
    monkeypatch.setitem(sys.modules, "analysis.design.loop", loop)
    received = {}

    def capture(**kwargs):
        received.update(kwargs)
        return None, None, [], [], ()

    cls = modal_app.ESMFold2DesignModal._get_user_cls()
    worker = cls()
    worker.driver = SimpleNamespace(design=capture)
    cls.design._get_raw_f()(
        worker, target_sequence="A", sampling_steps=sampling_steps, stop_after_steps=1
    )
    assert (received["sampling_steps"], received["confidence_sampling_steps"]) == expected
