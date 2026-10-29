"""Check steering proxies, schedules, and trace records using synthetic inputs."""

from __future__ import annotations
import importlib.util
import json
import math
from dataclasses import replace

import numpy as np
import pandas as pd
import pytest
import torch
from analysis import loss_trace, schedule, steering

BINDER_LENGTH = 40


def distance_bin_midpoints() -> torch.Tensor:
    """The cookbook's 128 distance bin midpoints.

    Duplicated from ``analysis.design.losses.get_mid_points`` because that
    module needs ``esm``. :func:`test_bin_midpoints_match_the_cookbook` checks
    the duplication wherever ``esm`` is installed.
    """
    boundaries = torch.linspace(2, 52.0, 127)
    exp_boundaries = torch.cat((torch.tensor([1.0]), boundaries, torch.tensor([57.0])))
    return (exp_boundaries[:-1] + exp_boundaries[1:]) / 2


def helix_coords(n: int) -> np.ndarray:
    """Ideal alpha helix: 2.3 A radius, 1.5 A rise, 100 degrees per residue."""
    index = np.arange(n, dtype=np.float64)
    angle = np.deg2rad(100.0) * index
    return np.stack([2.3 * np.cos(angle), 2.3 * np.sin(angle), 1.5 * index], axis=1)


def strand_coords(n: int, y: float = 0.0, descending: bool = False) -> np.ndarray:
    """Ideal extended strand: 3.3 A rise per residue with a 1.9 A pleat."""
    index = np.arange(n, dtype=np.float64)
    pleat = 0.95 * np.where(index % 2 == 0, 1.0, -1.0)
    rise = 3.3 * (index[::-1] if descending else index)
    return np.stack([pleat, np.full(n, y), rise], axis=1)


def hairpin_coords(per_strand: int, gap: float = 4.8) -> np.ndarray:
    """Two antiparallel strands ``gap`` apart, in register end to end."""
    return np.concatenate(
        [strand_coords(per_strand), strand_coords(per_strand, y=gap, descending=True)]
    )


def distogram_from_coords(coords: np.ndarray, sharpness: float = 8.0) -> torch.Tensor:
    """Build peaked distogram logits centred on pairwise coordinate distances."""
    bins = distance_bin_midpoints().numpy()
    distances = np.linalg.norm(coords[:, None, :] - coords[None, :, :], axis=-1)
    logits = -sharpness * (bins[None, None, :] - distances[..., None]) ** 2
    return torch.tensor(logits, dtype=torch.float64).unsqueeze(0)


@pytest.fixture
def bins() -> torch.Tensor:
    return distance_bin_midpoints().to(dtype=torch.float64)


@pytest.fixture
def helix() -> torch.Tensor:
    return distogram_from_coords(helix_coords(BINDER_LENGTH))


@pytest.fixture
def strand() -> torch.Tensor:
    return distogram_from_coords(strand_coords(BINDER_LENGTH))


@pytest.fixture
def hairpin() -> torch.Tensor:
    return distogram_from_coords(hairpin_coords(BINDER_LENGTH // 2))


@pytest.mark.skipif(
    importlib.util.find_spec("esm") is None,
    reason="needs esm, which analysis.design.losses imports; the duplicated "
    "midpoints are used unchecked without it",
)
def test_bin_midpoints_match_the_cookbook() -> None:
    from analysis.design.losses import get_mid_points

    assert torch.equal(distance_bin_midpoints(), get_mid_points())


def test_ideal_geometry_has_the_distances_the_terms_assume() -> None:
    helix = helix_coords(12)
    strand = strand_coords(12)

    def offset_distance(coords: np.ndarray, offset: int) -> float:
        deltas = coords[offset:] - coords[:-offset]
        return float(np.linalg.norm(deltas, axis=-1).mean())

    assert offset_distance(helix, 4) == pytest.approx(6.2, abs=0.1)
    assert offset_distance(helix, 2) == pytest.approx(5.4, abs=0.1)
    assert offset_distance(strand, 1) == pytest.approx(3.8, abs=0.1)
    assert offset_distance(strand, 2) == pytest.approx(6.6, abs=0.1)


def test_helix_mean_is_high_on_a_helix_and_low_on_a_strand(
    helix: torch.Tensor, strand: torch.Tensor, bins: torch.Tensor
) -> None:
    on_helix = float(steering.helix_mean_loss(helix, BINDER_LENGTH, bins)[0])
    on_strand = float(steering.helix_mean_loss(strand, BINDER_LENGTH, bins)[0])
    assert on_helix > 0.99
    assert on_strand < 0.01


def test_terms_read_only_the_binder_block(bins: torch.Tensor) -> None:
    binder = strand_coords(BINDER_LENGTH)
    target = helix_coords(30) + np.array([60.0, 60.0, 0.0])
    alone = distogram_from_coords(binder)
    with_target = distogram_from_coords(np.concatenate([target, binder]))
    for key, spec in steering.TERM_SPECS.items():
        assert torch.allclose(
            spec.compute(alone, BINDER_LENGTH, bins),
            spec.compute(with_target, BINDER_LENGTH, bins),
            atol=1e-9,
        ), key


def test_vanilla_measures_every_term_and_steers_none(
    helix: torch.Tensor, bins: torch.Tensor
) -> None:
    logits = helix.requires_grad_(True)
    result = steering.compute_steering_losses(logits, BINDER_LENGTH, bins, {})
    assert set(result.terms) == set(steering.TERM_SPECS)
    assert all(not value.requires_grad for value in result.terms.values())
    assert float(result.total.detach()[0]) == 0.0
    assert not result.total.requires_grad


def test_only_weighted_terms_carry_gradient(helix: torch.Tensor, bins: torch.Tensor) -> None:
    logits = helix.requires_grad_(True)
    weighted = steering.compute_steering_losses(logits, BINDER_LENGTH, bins, {"helix_mean": 0.2})
    assert weighted.terms["helix_mean"].requires_grad
    measured = steering.compute_steering_losses(logits, BINDER_LENGTH, bins, {})
    assert not measured.terms["helix_mean"].requires_grad


def test_total_is_the_weighted_sum_of_the_active_terms(
    hairpin: torch.Tensor, bins: torch.Tensor
) -> None:
    weights = {"helix_mean": 0.5}
    result = steering.compute_steering_losses(hairpin, BINDER_LENGTH, bins, weights)
    expected = sum(weight * float(result.terms[name][0]) for name, weight in weights.items())
    assert float(result.total[0]) == pytest.approx(expected)


def test_batch_elements_are_scored_independently(bins: torch.Tensor) -> None:
    helix = distogram_from_coords(helix_coords(BINDER_LENGTH))
    strand = distogram_from_coords(strand_coords(BINDER_LENGTH))
    batched = torch.cat([helix, strand], dim=0)
    values = steering.helix_mean_loss(batched, BINDER_LENGTH, bins)
    assert values.shape == (2,)
    assert float(values[0]) > 0.99
    assert float(values[1]) < 0.01


def test_unknown_weight_name_is_rejected(helix: torch.Tensor, bins: torch.Tensor) -> None:
    with pytest.raises(KeyError, match="unknown steering term"):
        steering.compute_steering_losses(helix, BINDER_LENGTH, bins, {"helix": 1.0})


def test_short_binders_receive_the_mean_helix_score(bins: torch.Tensor) -> None:
    logits = distogram_from_coords(helix_coords(12))
    losses = steering.compute_steering_losses(logits, 12, bins, {"helix_mean": 1.0})
    assert float(losses.terms["helix_mean"]) > 0.99
    torch.testing.assert_close(losses.total, losses.terms["helix_mean"])


def test_binder_longer_than_the_distogram_is_rejected(
    helix: torch.Tensor, bins: torch.Tensor
) -> None:
    with pytest.raises(ValueError, match="fewer than"):
        steering.compute_steering_losses(helix, BINDER_LENGTH + 1, bins, {})


def test_cutoff_outside_the_bin_range_is_rejected(helix: torch.Tensor, bins: torch.Tensor) -> None:
    with pytest.raises(ValueError, match="excludes every distogram bin"):
        steering.contact_probability_matrix(helix, bins, 0.5)


def test_every_term_is_catalogued_with_a_unit() -> None:
    assert steering.TERM_SPECS
    for key, spec in steering.TERM_SPECS.items():
        assert spec.key == key
        assert spec.unit != ""


def test_constant_schedule_holds_for_the_whole_run() -> None:
    arm = schedule.constant("w02", "helix_mean", 0.2)
    assert arm.weights_at(0) == {"helix_mean": 0.2}
    assert arm.weights_at(149) == {"helix_mean": 0.2}


def test_vanilla_sets_no_weights() -> None:
    assert schedule.VANILLA.weights_at(0) == {}
    assert schedule.VANILLA.summary() == "no steering term"


@pytest.mark.parametrize(
    ("waypoints", "message"),
    [
        (
            (
                schedule.Waypoint(0, {"helix_mean": 0.2}),
                schedule.Waypoint(5, {"helix_mean": 0.1}),
                schedule.Waypoint(3, {"helix_mean": 0.3}),
            ),
            "sorted by step",
        ),
        (
            (
                schedule.Waypoint(0, {"helix_mean": 0.2}),
                schedule.Waypoint(0, {"helix_mean": 0.1}),
            ),
            "distinct steps",
        ),
        ((schedule.Waypoint(5, {"helix_mean": 0.2}),), "must be at step 0"),
    ],
)
def test_malformed_schedules_are_rejected(
    waypoints: tuple[schedule.Waypoint, ...], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        schedule.LossSchedule(key="bad", waypoints=waypoints)


def test_schedule_summary_and_record_describe_the_arm() -> None:
    arm = schedule.released("release20", "helix_mean", 0.5, 20)
    assert arm.summary() == ("helix_mean 0.5 for steps 0-19; helix_mean 0 from step 20 on")
    record = arm.to_dict()
    assert record["key"] == "release20"
    assert record["waypoints"] == [
        {"step": 0, "weights": {"helix_mean": 0.5}},
        {"step": 20, "weights": {"helix_mean": 0.0}},
    ]
    assert json.loads(json.dumps(record)) == record


def a_record(step: int = 0, seed: int = 0) -> loss_trace.StepRecord:
    return loss_trace.StepRecord(
        target="pdl1_igv",
        arm="helix_mean_w20",
        seed=seed,
        step=step,
        steps=150,
        temperature=0.5,
        confidence_step=False,
        inversion_model="ESMFold2-Experimental-Fast",
        binder_sequence="M" * 80,
        step_seconds=1.25,
        sequence_entropy_bits=1.97,
        distogram_iptm_proxy=0.42,
        iptm=None,
        losses={"total_loss": 3.5, "helix_mean": 1.2},
        weights={"helix_mean": 0.2},
    )


def test_every_logged_column_states_its_unit() -> None:
    row = a_record().row()
    quantities = [
        key
        for key in row
        if key not in ("target", "arm", "seed", "inversion model", "binder sequence")
    ]
    assert quantities
    for key in quantities:
        assert key.endswith(")"), key
        assert "(" in key, key


def test_logged_losses_use_their_own_units() -> None:
    row = a_record().row()
    assert "helix_mean (0-1)" in row
    assert "total_loss (dimensionless)" in row


def test_an_unregistered_loss_cannot_enter_the_log() -> None:
    with pytest.raises(KeyError, match="mystery_loss"):
        loss_trace.loss_column("mystery_loss")


def test_a_term_the_schedule_omits_is_null_not_zero() -> None:
    column = loss_trace.weight_column("helix_mean")
    assert a_record().row()[column] == 0.2
    unsteered = replace(a_record(), weights={})
    assert unsteered.row()[column] is None


def test_the_log_round_trips_through_json_lines(tmp_path) -> None:
    path = tmp_path / loss_trace.STEP_TRACE_NAME
    records = (a_record(step=1, seed=3), a_record(step=0, seed=3), a_record(step=0, seed=1))
    for record in records:
        loss_trace.append(record, path)
    frame = loss_trace.load(path)
    assert len(frame) == 3
    assert list(frame["seed"]) == [1, 3, 3]
    assert list(frame["step (index)"]) == [0, 0, 1]
    assert frame["iptm (0-1)"].isna().all()
    assert frame["binder length (residues)"].eq(80).all()


def test_loading_a_missing_log_says_how_to_make_one(tmp_path) -> None:
    with pytest.raises(FileNotFoundError, match="analysis.campaign collect"):
        loss_trace.load(tmp_path / "absent.jsonl")


def test_rows_flattens_a_whole_trace() -> None:
    flat = loss_trace.rows((a_record(step=0), a_record(step=1)))
    assert [row["step (index)"] for row in flat] == [0, 1]
    assert isinstance(loss_trace.frame(flat), pd.DataFrame)


def test_sequence_entropy_is_measured_in_bits() -> None:
    collapsed = torch.zeros(1, 30, 20)
    collapsed[..., 0] = 1.0
    uniform = torch.full((1, 30, 20), 1.0 / 20.0)
    assert loss_trace.sequence_entropy_bits(collapsed) == pytest.approx(0.0, abs=1e-4)
    assert loss_trace.sequence_entropy_bits(uniform) == pytest.approx(math.log2(20), abs=1e-4)
