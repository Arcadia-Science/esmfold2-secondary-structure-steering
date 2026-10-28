"""Check payload routing and reject empty containers in campaign score rows."""

from __future__ import annotations
import re

import pandas as pd
import pytest
from analysis import campaign
from analysis.results import CampaignOutput


def a_payload() -> dict[str, object]:
    """A minimal campaign payload with scores, a schedule, trace, and PDB text."""
    return {
        "steps": 150,
        "target": "pdl1_igv",
        "arm": "vanilla",
        "arm_label": "Vanilla",
        "critic_iptm": 0.94,
        "critic_plddt_binder": 87.4,
        "critic_iptm_by_model": {"ESMFold2-Experimental-Fast": 0.94},
        "schedule": {"key": "vanilla", "waypoints": [{"step": 0, "weights": {}}]},
        "step_trace": [{"step (index)": 0}],
        "complex_pdb": "ATOM ...",
    }


def test_routed_fields_stay_out_of_the_row() -> None:
    payload = a_payload()
    payload["step_complexes_gz"] = b"compressed structures"
    output = CampaignOutput.from_payload(payload, seed=7)
    assert output.row["seed"] == 7
    for key in ("step_trace", "complex_pdb", "step_complexes_gz", "schedule"):
        assert key not in output.row
        assert key in payload, "Collection must not mutate the worker payload"
    assert output.trace == payload["step_trace"]
    assert output.complex_pdb == payload["complex_pdb"]
    assert output.step_complexes_gz == payload["step_complexes_gz"]


def test_a_populated_nested_field_is_allowed() -> None:
    """The per-critic dicts are nested and fine; only empty ones are not."""
    payload = a_payload()
    payload["critic_iptm_by_model"] = {"fast": 0.9, "cutoff": 0.8}
    output = CampaignOutput.from_payload(payload, seed=0)
    assert output.row["critic_iptm_by_model"] == {"fast": 0.9, "cutoff": 0.8}


def test_the_resulting_row_is_parquet_writable(tmp_path) -> None:
    frame = pd.DataFrame([CampaignOutput.from_payload(a_payload(), seed=0).row])
    path = tmp_path / "all_results.parquet"
    frame.to_parquet(path, index=False)
    assert pd.read_parquet(path).loc[0, "critic_plddt_binder"] == pytest.approx(87.4)


@pytest.mark.parametrize(
    ("field", "value", "where"),
    [
        ("weights", {}, "weights"),
        ("schedule_copy", {"waypoints": [{"weights": {}}]}, "schedule_copy.waypoints[0].weights"),
        ("critic_iptm_by_model", {}, "critic_iptm_by_model"),
        ("seeds_used", [], "seeds_used"),
    ],
)
def test_an_empty_container_is_named_rather_than_written(
    field: str, value: object, where: str
) -> None:
    with pytest.raises(ValueError, match=re.escape(where)):
        CampaignOutput.from_payload({**a_payload(), field: value}, seed=0)


def test_the_vanilla_schedule_is_exactly_what_broke_it() -> None:
    """Regression: the real payload, unrouted, has to be rejected."""
    from analysis import targets

    unrouted = {"schedule": targets.arm("vanilla").schedule.to_dict()}
    with pytest.raises(ValueError, match=r"schedule.waypoints\[0\].weights"):
        CampaignOutput(unrouted, [], "")


def test_the_archive_is_its_own_marker(tmp_path) -> None:
    """A half-finished download must not look like a complete one."""
    from analysis import deposit

    artifact = deposit.DataArtifact("thing.zip", tmp_path, "a thing")
    assert artifact.is_fetched() is False
    (tmp_path / "thing.partial").write_bytes(b"incomplete")
    assert artifact.is_fetched() is False, "a .partial must not satisfy the gate"
    artifact.archive_path.write_bytes(b"complete")
    assert artifact.is_fetched() is True


def test_packaging_round_trips_through_extraction(tmp_path) -> None:
    from analysis import deposit

    source = tmp_path / "nested"
    source.mkdir()
    (source / "table.parquet").write_bytes(b"pretend parquet")
    (tmp_path / "top.json").write_text("{}")

    artifact = deposit.DataArtifact("bundle.zip", tmp_path, "a bundle")
    artifact.package([source / "table.parquet", tmp_path / "top.json"])

    import zipfile

    with zipfile.ZipFile(artifact.archive_path) as archive:
        assert sorted(archive.namelist()) == ["nested/table.parquet", "top.json"]


def test_submission_resumes_after_interruption_without_duplicate_calls(tmp_path):
    from types import SimpleNamespace

    key, identifiers = "seed", [0, 1, 2]
    path = tmp_path / "manifest.csv"
    jobs = [{key: value} for value in identifiers]
    submitted = []

    def interrupted(job):
        if job[key] == identifiers[1]:
            raise ConnectionError("submission interrupted")
        submitted.append(job[key])
        return SimpleNamespace(object_id=f"call-{job[key]}")

    with pytest.raises(ConnectionError):
        campaign.submit_jobs(path, jobs, key=key, submit=interrupted)
    assert campaign.read_manifest(path, key=key)[key].tolist() == identifiers[:1]

    def resume(job):
        submitted.append(job[key])
        return SimpleNamespace(object_id=f"call-{job[key]}")

    frame = campaign.submit_jobs(path, jobs, key=key, submit=resume)
    campaign.submit_jobs(path, jobs, key=key, submit=resume)
    assert submitted == identifiers
    assert frame[key].tolist() == identifiers
    with pytest.raises(ValueError, match="match the existing manifest"):
        campaign.submit_jobs(path, jobs[1:], key=key, submit=resume)


def test_main_campaign_uses_incremental_submission_and_keeps_existing_calls(tmp_path, monkeypatch):
    from types import SimpleNamespace

    import modal
    from analysis import paths

    monkeypatch.setattr(paths, "CAMPAIGNS", tmp_path)
    calls = []

    def submit(**kwargs):
        calls.append(kwargs)
        return SimpleNamespace(object_id=f"call-{kwargs['seed']}")

    remote = SimpleNamespace(design=SimpleNamespace(spawn=submit))
    monkeypatch.setattr(modal.Cls, "from_name", lambda *args: lambda: remote)
    first = campaign.spawn_campaign("pdl1_igv", "vanilla", n_seeds=2)
    again = campaign.spawn_campaign("pdl1_igv", "vanilla", n_seeds=2)
    pd.testing.assert_frame_equal(first, again)
    assert [call["seed"] for call in calls] == [0, 1]
    assert all(call["arm"] == "vanilla" and call["target_sequence"] for call in calls)
    with pytest.raises(ValueError, match="settings differ"):
        campaign.spawn_campaign("pdl1_igv", "vanilla", n_seeds=3)


@pytest.fixture
def collection_setup(tmp_path, monkeypatch):
    import json
    from types import SimpleNamespace

    from analysis import composition, paths

    monkeypatch.setattr(paths, "CAMPAIGNS", tmp_path)
    monkeypatch.setattr(
        composition.secondary_structure,
        "binder_composition",
        lambda path: SimpleNamespace(
            n_residues=100, helix_fraction=0.2, sheet_fraction=0.5, coil_fraction=0.3
        ),
    )

    def prepare(target="pdl1_igv", arm="vanilla", count=2, submitted=None):
        directory = paths.campaign_dir(target, arm)
        directory.mkdir(parents=True, exist_ok=True)
        paths.campaign_protocol(target, arm).write_text(json.dumps({"n_seeds": count}))
        seeds = list(range(count)) if submitted is None else submitted
        frame = pd.DataFrame({"seed": seeds, "call_id": [f"call-{s}" for s in seeds]})
        frame.to_csv(paths.campaign_manifest(target, arm), index=False)
        return directory

    return prepare


def collection_calls(monkeypatch, states):
    from types import SimpleNamespace

    def completed(manifest):
        for row in manifest.itertuples():
            state = states[row.seed]
            if state:
                yield row, None, state
            else:
                payload = a_payload()
                payload["step_trace"] = [{"seed": row.seed, "step (index)": 0}]
                yield SimpleNamespace(seed=row.seed), payload, ""

    monkeypatch.setattr(campaign, "completed_jobs", completed)


def snapshot(directory):
    return {p.relative_to(directory): p.read_bytes() for p in directory.rglob("*") if p.is_file()}


def test_partial_collection_resumes_without_force(collection_setup, monkeypatch):
    import json

    from analysis import artifacts

    directory = collection_setup()
    collection_calls(monkeypatch, {0: "", 1: "pending"})
    campaign.collect(target="pdl1_igv", arm="vanilla", force=False)
    assert artifacts.status("pdl1_igv", "vanilla").is_analysable
    assert not artifacts.is_complete("pdl1_igv", "vanilla")
    summary = json.loads((directory / "collection.json").read_text())
    assert summary["partial"] and summary["pending_seeds"] == [1]

    collection_calls(monkeypatch, {0: "", 1: ""})
    campaign.collect(target="pdl1_igv", arm="vanilla", force=False)
    assert artifacts.is_complete("pdl1_igv", "vanilla")
    assert not list(directory.glob(".collection-*"))
    # Archived campaigns have protocols and outputs, but no new summary or manifest.
    (directory / "collection.json").unlink()
    (directory / "manifest.csv").unlink()
    assert artifacts.is_complete("pdl1_igv", "vanilla")
    (directory / ".collecting").touch()
    assert not artifacts.is_complete("pdl1_igv", "vanilla")


@pytest.mark.parametrize("state", ["pending", "RuntimeError: failed"])
def test_pending_and_failed_seeds_are_incomplete(collection_setup, monkeypatch, state):
    import json

    from analysis import artifacts

    directory = collection_setup()
    collection_calls(monkeypatch, {0: "", 1: state})
    done = campaign.collect_campaign("pdl1_igv", "vanilla")
    assert done.n_designs == 1
    assert done.n_pending + done.n_failed == 1
    assert not artifacts.is_complete("pdl1_igv", "vanilla")
    assert json.loads((directory / "collection.json").read_text())["partial"]


def test_unsubmitted_seeds_count_toward_completion(collection_setup, monkeypatch):
    from analysis import artifacts

    collection_setup(count=3, submitted=[0, 1])
    collection_calls(monkeypatch, {0: "", 1: ""})
    done = campaign.collect_campaign("pdl1_igv", "vanilla")
    assert done.n_unsubmitted == 1
    assert not artifacts.is_complete("pdl1_igv", "vanilla")


@pytest.mark.parametrize("existing", [False, True])
def test_validation_failure_preserves_outputs(collection_setup, monkeypatch, existing):
    from analysis import composition

    directory = collection_setup()
    collection_calls(monkeypatch, {0: "", 1: ""})
    if existing:
        campaign.collect_campaign("pdl1_igv", "vanilla")
    before = snapshot(directory)

    def invalid(path):
        raise ValueError("Invalid backbone")

    monkeypatch.setattr(composition.secondary_structure, "binder_composition", invalid)
    with pytest.raises(ValueError, match="Invalid backbone"):
        campaign.collect_campaign("pdl1_igv", "vanilla")
    assert snapshot(directory) == before
    assert not list(directory.glob(".collection-*"))


def test_recollection_cannot_drop_previously_saved_seeds(collection_setup, monkeypatch):
    directory = collection_setup()
    collection_calls(monkeypatch, {0: "", 1: ""})
    campaign.collect_campaign("pdl1_igv", "vanilla")
    before = snapshot(directory)
    collection_calls(monkeypatch, {0: "", 1: "results expired"})
    with pytest.raises(ValueError, match="discard previously collected seeds"):
        campaign.collect_campaign("pdl1_igv", "vanilla")
    assert snapshot(directory) == before


def test_publish_failure_restores_previous_collection(collection_setup, monkeypatch):
    from pathlib import Path

    directory = collection_setup()
    collection_calls(monkeypatch, {0: "", 1: "pending"})
    campaign.collect_campaign("pdl1_igv", "vanilla")
    before = snapshot(directory)
    collection_calls(monkeypatch, {0: "", 1: ""})
    replace = Path.replace

    def fail_install(path, target):
        if path.parent.name == "outputs" and path.name == "all_results.parquet":
            raise OSError("simulated disk error")
        return replace(path, target)

    monkeypatch.setattr(Path, "replace", fail_install)
    with pytest.raises(OSError, match="simulated disk error"):
        campaign.collect_campaign("pdl1_igv", "vanilla")
    assert snapshot(directory) == before


def test_ensure_collects_other_campaigns_after_failure(collection_setup, monkeypatch, capsys):
    from types import SimpleNamespace

    import modal
    from analysis import artifacts

    keys = list(campaign.TARGET_SPECS)[:2]
    for key in keys:
        collection_setup(target=key, count=1)
    manifests = {key: pd.DataFrame({"seed": [0], "call_id": [key]}) for key in keys}
    monkeypatch.setattr(campaign, "spawn_campaign", lambda target, *args: manifests[target])
    monkeypatch.setattr(
        modal.FunctionCall, "from_id", lambda call_id: SimpleNamespace(get=lambda **kw: None)
    )
    actual_collect = campaign.collect_campaign
    attempted = []

    def collect(target, arm):
        attempted.append(target)
        collection_calls(monkeypatch, {0: "failed" if target == keys[0] else ""})
        return actual_collect(target, arm)

    monkeypatch.setattr(campaign, "collect_campaign", collect)
    campaign.ensure(keys, ["vanilla"], n_seeds=1, poll_seconds=0)
    assert attempted == keys
    assert not artifacts.is_complete(keys[0], "vanilla")
    assert artifacts.is_complete(keys[1], "vanilla")
    assert "1 campaign(s) could not be collected" in capsys.readouterr().out


@pytest.mark.parametrize("damage", ["scores", "structure", "trace", "duplicates"])
def test_complete_requires_matching_seed_coverage(collection_setup, monkeypatch, damage):
    from analysis import artifacts

    directory = collection_setup()
    collection_calls(monkeypatch, {0: "", 1: ""})
    campaign.collect_campaign("pdl1_igv", "vanilla")
    assert artifacts.is_complete("pdl1_igv", "vanilla")
    if damage == "structure":
        (directory / "all_complexes" / "seed01.pdb").unlink()
    else:
        path = directory / ("step_trace.parquet" if damage == "trace" else "all_results.parquet")
        frame = pd.read_parquet(path)
        if damage == "duplicates":
            frame = pd.concat([frame, frame.iloc[:1]], ignore_index=True)
        else:
            frame = frame.iloc[:1]
        frame.to_parquet(path, index=False)
    assert not artifacts.is_complete("pdl1_igv", "vanilla")


def test_spawn_does_not_skip_partial_results(collection_setup, monkeypatch):
    from typer.testing import CliRunner

    collection_setup()
    collection_calls(monkeypatch, {0: "", 1: "pending"})
    campaign.collect_campaign("pdl1_igv", "vanilla")
    submitted = []

    def spawn(*args):
        submitted.append(args)
        return pd.DataFrame({"seed": [0, 1], "call_id": ["a", "b"]})

    monkeypatch.setattr(campaign, "spawn_campaign", spawn)
    result = CliRunner().invoke(
        campaign.app, ["spawn", "--target", "pdl1_igv", "--arm", "vanilla", "--n-seeds", "2"]
    )
    assert result.exit_code == 0, result.output
    assert len(submitted) == 1
