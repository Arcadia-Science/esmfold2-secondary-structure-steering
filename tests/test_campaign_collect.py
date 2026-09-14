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

import pandas as pd
import pytest
from analysis import campaign


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
    record = campaign.design_record(7, {
        key: value for key, value in a_payload().items() if key != "step_trace"
    })
    assert record["seed"] == 7
    for key in campaign.ROUTED_KEYS:
        assert key not in record


def test_a_populated_nested_field_is_allowed() -> None:
    """The per-critic dicts are nested and fine; only empty ones are not."""
    record = campaign.design_record(0, {
        "critic_iptm_by_model": {"fast": 0.9, "cutoff": 0.8},
    })
    assert record["critic_iptm_by_model"] == {"fast": 0.9, "cutoff": 0.8}


def test_the_resulting_row_is_parquet_writable(tmp_path) -> None:
    payload = {k: v for k, v in a_payload().items() if k != "step_trace"}
    frame = pd.DataFrame([campaign.design_record(0, payload)])
    path = tmp_path / "all_results.parquet"
    frame.to_parquet(path, index=False)
    assert pd.read_parquet(path).loc[0, "critic_plddt_binder"] == pytest.approx(87.4)


@pytest.mark.parametrize(
    ("field", "value", "where"),
    [
        ("weights", {}, "weights"),
        ("schedule_copy", {"waypoints": [{"weights": {}}]},
         "schedule_copy.waypoints[0].weights"),
        ("critic_iptm_by_model", {}, "critic_iptm_by_model"),
        ("seeds_used", [], "seeds_used"),
    ],
)
def test_an_empty_container_is_named_rather_than_written(
    field: str, value: object, where: str
) -> None:
    with pytest.raises(ValueError, match="empty container"):
        campaign.design_record(0, {"critic_iptm": 0.9, field: value})
    assert campaign._empty_container_paths({field: value}) == [where]


def test_the_vanilla_schedule_is_exactly_what_broke_it() -> None:
    """Regression: the real payload, unrouted, has to be rejected."""
    from analysis import targets

    unrouted = {"schedule": targets.arm("vanilla").schedule.to_dict()}
    assert campaign._empty_container_paths(unrouted) == [
        "schedule.waypoints[0].weights"
    ]
