"""Routing a returned design payload into the per-design table.

The bug this guards against cost a collected campaign. A design payload carried
the whole weight schedule, and the vanilla schedule's waypoint holds an empty
weights mapping, which parquet cannot store: a struct needs at least one field.
The failure is nastier than it sounds because of where it hides. Only an arm
that sets no weights produces an empty mapping, so vanilla breaks and every
steered arm writes fine, which is the opposite of the order anyone would test
in.
"""

from __future__ import annotations
import re

import pandas as pd
import pytest
from analysis import campaign
from analysis.results import CampaignOutput


def a_payload() -> dict[str, object]:
    """The shape analysis.modal_app.design returns, trimmed to what matters."""
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


@pytest.mark.parametrize(
    ("key", "identifiers"), [("seed", [0, 1, 2]), ("run", ["input00", "input01", "controls"])]
)
def test_submission_resumes_after_interruption_without_duplicate_calls(tmp_path, key, identifiers):
    from types import SimpleNamespace

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
