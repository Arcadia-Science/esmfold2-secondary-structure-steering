"""The new timing arms must gate gradients without altering existing arms."""

import pytest
import torch
from analysis import schedule, targets


@pytest.mark.parametrize("hold", [False, True])
def test_schedule_boundary_and_base_gradient(hold):
    configuration = schedule.LossSchedule(
        "test",
        (schedule.Waypoint(0, {"helix_mean": 2.0}),)
        + (() if hold else (schedule.Waypoint(10, {"helix_mean": 0.0}),)),
        base_structure_onset=10,
    )
    for step in (0, 9, 10, 149):
        x = torch.tensor([2.0, 3.0], requires_grad=True)
        base = x[0] ** 2
        helix_weight = configuration.weights_at(step)["helix_mean"]
        steering = helix_weight * x[1] ** 2
        objective = configuration.structural_objective(base, steering, step)
        grad = torch.autograd.grad(objective, x)[0]
        expected_base = 0.0 if step < 10 else 4.0
        expected_helix = 12.0 if step < 10 or hold else 0.0
        torch.testing.assert_close(grad, torch.tensor([expected_base, expected_helix]))
        assert configuration.base_structure_weight_at(step) == float(step >= 10)
    assert configuration.to_dict()["base_structure_onset"] == 10


def test_existing_arms_keep_the_original_objective():
    for arm in targets.ARM_SPECS.values():
        for step in (0, 9, 10, 149):
            base = torch.tensor([0.3, 2.5])
            steering = torch.tensor([0.4, 0.0])
            assert torch.equal(
                arm.schedule.structural_objective(base, steering, step), base + steering
            )
            assert arm.schedule.base_structure_weight_at(step) == 1.0
        assert "base_structure_onset" not in arm.schedule.to_dict()


def test_negative_base_onset_is_rejected():
    with pytest.raises(ValueError, match="base_structure_onset"):
        schedule.LossSchedule("bad", (schedule.Waypoint(0, {}),), base_structure_onset=-1)
