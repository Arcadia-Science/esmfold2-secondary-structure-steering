"""Shared result models preserve worker formats and validate archive metadata."""

import io
import json
import sys
import zipfile
from dataclasses import replace
from types import ModuleType, SimpleNamespace

import pandas as pd
import pytest
from analysis import modal_app
from analysis.results import (
    CampaignOutput,
    DesignCompletion,
    DesignResult,
    RunSpec,
)
from analysis.targets import CRITIC_MODELS, PRIMARY_CRITIC, arm


@pytest.fixture
def archived_result(full_design_archive):
    protocol = {
        "target": "pdl1_igv",
        "target_sequence": "AAA",
        "arm": "init_sheet_t2",
        "steps": 3,
        "initialization_temperature": 2.0,
        "probability_margin": 0.001,
    }
    with zipfile.ZipFile(io.BytesIO(full_design_archive(protocol))) as archive:
        return SimpleNamespace(
            protocol=protocol,
            critics=json.loads(archive.read("critics.json")),
            completion=json.loads(archive.read("complete.json")),
            pdb=archive.read(f"{PRIMARY_CRITIC}.pdb").decode(),
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

    # The archive path adds initialization metadata to the same result schema.
    run = RunSpec.from_protocol(archived_result.protocol, seed=0)
    initialized = DesignResult.from_critics(run, archived_result.critics, selected_step=2)
    expected.pop("binder_name")
    expected.update(
        arm="init_sheet_t2",
        arm_label="Sheet prior T=2",
        initialization_prior="sheet",
        initialization_temperature=2.0,
        probability_margin=0.001,
        selected_step=2,
    )
    assert initialized.row() == expected


def test_completion_keeps_existing_disk_format(archived_result):
    run = RunSpec.from_protocol(archived_result.protocol, seed=0)
    result = DesignResult.from_critics(run, archived_result.critics, selected_step=2)
    written = DesignCompletion.from_result(result, initial_sequence="AI")
    assert written.to_record() == archived_result.completion
    read = DesignCompletion.from_record(written.to_record(), target=run.target, prior="sheet")
    assert read == written
    read.validate_against(result)


@pytest.mark.parametrize(
    "field, observed, message",
    [
        ("seed", 4, "seed: expected 0, observed 4"),
        ("arm", "vanilla", "arm: expected 'init_sheet_t2', observed 'vanilla'"),
        ("steps", 150, "steps: expected 3, observed 150"),
        ("initialization_temperature", 4.0, "temperature=4.0"),
        ("probability_margin", 0.01, "probability_margin=0.01"),
        ("primary_critic", "another-critic", "observed 'another-critic'"),
    ],
)
def test_completion_names_mismatched_settings(archived_result, field, observed, message):
    expected = RunSpec.from_protocol(archived_result.protocol, seed=0)
    record = {**archived_result.completion, field: observed}
    completion = DesignCompletion.from_record(record, target=expected.target, prior="sheet")
    with pytest.raises(ValueError, match=message):
        completion.run.validate_against(expected)


def test_completion_length_matches_sequence(archived_result):
    with pytest.raises(ValueError, match="Completion binder length differs from its sequence"):
        DesignCompletion.from_record(
            {**archived_result.completion, "binder_length": 99}, target="pdl1_igv", prior="sheet"
        )


@pytest.mark.parametrize(
    "field, value, message",
    [
        ("binder_sequence", "II", "selected sequence"),
        ("selected_step", 1, "selected sequence"),
        ("iptm", 0.8, "primary critic"),
        ("binder_plddt", 50.0, "primary critic"),
    ],
)
def test_completion_must_match_selected_result(archived_result, field, value, message):
    run = RunSpec.from_protocol(archived_result.protocol, seed=0)
    result = DesignResult.from_critics(run, archived_result.critics, selected_step=2)
    completion = DesignCompletion.from_result(result, initial_sequence="AI")
    with pytest.raises(ValueError, match=message):
        replace(completion, **{field: value}).validate_against(result)


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
