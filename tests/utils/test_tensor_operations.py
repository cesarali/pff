import torch
import pytest
from pff.utils.tensors_operations import gather_distinct_times_per_substance
from pff.data.datasets.aicme_batch import AICMECompartmentsDataBatch

def make_fake_batch():
    B, c_ind, t_ind, Tmax = 2, 2, 2, 10

    # Context + target obs times
    # Substance 1: irregular, 3–4 obs per individual
    ctx_times_b0 = torch.tensor([
        [1., 3., 5., 0., 0., 0., 0., 0., 0., 0.],   # 3 obs
        [2., 4., 6., 7., 0., 0., 0., 0., 0., 0.]    # 4 obs
    ])
    tgt_times_b0 = torch.tensor([
        [1., 2., 3., 0., 0., 0., 0., 0., 0., 0.],   # 3 obs
        [2., 5., 0., 0., 0., 0., 0., 0., 0., 0.]    # 2 obs
    ])

    # Substance 2: denser, 5–10 obs per individual
    ctx_times_b1 = torch.tensor([
        [1., 2., 3., 4., 5., 0., 0., 0., 0., 0.],   # 5 obs
        [1., 3., 5., 7., 9., 10., 0., 0., 0., 0.]   # 6 obs
    ])
    tgt_times_b1 = torch.tensor([
        [2., 4., 6., 8., 10., 0., 0., 0., 0., 0.],  # 5 obs
        [1., 2., 3., 4., 5., 6., 7., 8., 9., 10.]   # 10 obs
    ])

    # Stack into [B, I, T, 1]
    context_obs_time = torch.stack([
        ctx_times_b0, ctx_times_b1
    ]).unsqueeze(-1)  # [B=2, c_ind=2, T=10, 1]

    target_obs_time = torch.stack([
        tgt_times_b0, tgt_times_b1
    ]).unsqueeze(-1)  # [B=2, t_ind=2, T=10, 1]

    # Masks: True where >0
    context_mask = context_obs_time.squeeze(-1) > 0
    target_mask = target_obs_time.squeeze(-1) > 0

    # Minimal fields for batch
    db = AICMECompartmentsDataBatch(
        target_obs=torch.zeros(2, 2, 10, 1),
        target_obs_time=target_obs_time,
        target_obs_mask=target_mask,
        target_rem_sim=torch.zeros(2, 2, 0, 1),
        target_rem_sim_time=torch.zeros(2, 2, 0, 1),
        target_rem_sim_mask=torch.zeros(2, 2, 0, dtype=torch.bool),
        context_obs=torch.zeros(2, 2, 10, 1),
        context_obs_time=context_obs_time,
        context_obs_mask=context_mask,
        context_rem_sim=torch.zeros(2, 2, 0, 1),
        context_rem_sim_time=torch.zeros(2, 2, 0, 1),
        context_rem_sim_mask=torch.zeros(2, 2, 0, dtype=torch.bool),
        target_dosing_amounts=torch.zeros(2, 2),
        target_dosing_route_types=torch.zeros(2, 2, dtype=torch.long),
        context_dosing_amounts=torch.zeros(2, 2),
        context_dosing_route_types=torch.zeros(2, 2, dtype=torch.long),
        mask_context_individuals=torch.ones(2, 2, dtype=torch.bool),
        mask_target_individuals=torch.ones(2, 2, dtype=torch.bool),
        study_name=["study0", "study1"],
        context_subject_name=[["c0", "c1"], ["c0", "c1"]],
        target_subject_name=[["t0", "t1"], ["t0", "t1"]],
        substance_name=["drug0", "drug1"],
        time_scales=torch.zeros(2, 2),
        is_empirical=False,
    )

    return db

def test_gather_distinct_times_per_substance():
    db = make_fake_batch()
    times, mask = gather_distinct_times_per_substance(db)

    # Expected unique counts: batch 0 (≈7), batch 1 (≈10)
    assert times.shape[0] == 2  # batch size
    assert times.shape[2] == 1  # last dim kept
    assert mask.shape == times.squeeze(-1).shape

    # Check that mask marks the right number of entries
    counts = mask.sum(dim=1).tolist()
    assert counts[0] == 7   # substance 0
    assert counts[1] == 10  # substance 1

    # Check that times are sorted ascending within each batch
    for b in range(times.shape[0]):
        valid_times = times[b, mask[b], 0]
        assert torch.all(valid_times[:-1] <= valid_times[1:])

    # Ensure padded entries are zero
    padded = times[~mask].view(-1)
    assert torch.allclose(padded, torch.zeros_like(padded))


def test_gather_distinct_times_per_substance_includes_remaining_and_respects_individual_masks():
    """Distinct times must include remainder streams and ignore masked individuals."""
    B, c_ind, t_ind, T_obs, T_rem = 1, 2, 2, 4, 3

    context_obs_time = torch.tensor(
        [[
            [1.0, 2.0, 0.0, 0.0],   # valid context individual
            [9.0, 11.0, 0.0, 0.0],  # should be ignored by mask_context_individuals
        ]]
    ).unsqueeze(-1)  # [1,2,4,1]
    target_obs_time = torch.tensor(
        [[
            [2.0, 3.0, 0.0, 0.0],   # valid target individual
            [13.0, 0.0, 0.0, 0.0],  # should be ignored by mask_target_individuals
        ]]
    ).unsqueeze(-1)  # [1,2,4,1]

    # Remainder carries times that must be included in the decode grid.
    context_rem_time = torch.tensor(
        [[
            [4.0, 0.0, 0.0],   # valid context remainder
            [15.0, 0.0, 0.0],  # masked individual, must be ignored
        ]]
    ).unsqueeze(-1)  # [1,2,3,1]
    target_rem_time = torch.tensor(
        [[
            [5.0, 6.0, 0.0],   # valid target remainder
            [17.0, 0.0, 0.0],  # masked individual, must be ignored
        ]]
    ).unsqueeze(-1)  # [1,2,3,1]

    context_obs_mask = context_obs_time.squeeze(-1) > 0
    target_obs_mask = target_obs_time.squeeze(-1) > 0
    context_rem_mask = context_rem_time.squeeze(-1) > 0
    target_rem_mask = target_rem_time.squeeze(-1) > 0

    db = AICMECompartmentsDataBatch(
        target_obs=torch.zeros(B, t_ind, T_obs, 1),
        target_obs_time=target_obs_time,
        target_obs_mask=target_obs_mask,
        target_rem_sim=torch.zeros(B, t_ind, T_rem, 1),
        target_rem_sim_time=target_rem_time,
        target_rem_sim_mask=target_rem_mask,
        context_obs=torch.zeros(B, c_ind, T_obs, 1),
        context_obs_time=context_obs_time,
        context_obs_mask=context_obs_mask,
        context_rem_sim=torch.zeros(B, c_ind, T_rem, 1),
        context_rem_sim_time=context_rem_time,
        context_rem_sim_mask=context_rem_mask,
        target_dosing_amounts=torch.zeros(B, t_ind),
        target_dosing_route_types=torch.zeros(B, t_ind, dtype=torch.long),
        context_dosing_amounts=torch.zeros(B, c_ind),
        context_dosing_route_types=torch.zeros(B, c_ind, dtype=torch.long),
        mask_context_individuals=torch.tensor([[True, False]]),
        mask_target_individuals=torch.tensor([[True, False]]),
        study_name=["study0"],
        context_subject_name=[["c0", "c1"]],
        target_subject_name=[["t0", "t1"]],
        substance_name=["drug0"],
        time_scales=torch.zeros(B, 2),
        is_empirical=False,
    )

    times, mask = gather_distinct_times_per_substance(db)
    valid_times = times[0, mask[0], 0]

    expected = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], dtype=valid_times.dtype)
    assert torch.equal(valid_times, expected)


def test_gather_distinct_times_per_substance_can_select_target_obs_only():
    """Source selection should restrict aggregation to the requested time tensors."""

    db = make_fake_batch()
    times, mask = gather_distinct_times_per_substance(db, time_sources=("target_obs_time",))

    expected_batch0 = torch.tensor([1.0, 2.0, 3.0, 5.0], dtype=times.dtype)
    expected_batch1 = torch.tensor(
        [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
        dtype=times.dtype,
    )

    assert torch.equal(times[0, mask[0], 0], expected_batch0)
    assert torch.equal(times[1, mask[1], 0], expected_batch1)


def test_gather_distinct_times_per_substance_rejects_unknown_sources():
    """Unknown time-source selectors should fail fast."""

    db = make_fake_batch()

    with pytest.raises(ValueError, match="unsupported entries"):
        gather_distinct_times_per_substance(db, time_sources=("unknown_time",))

if __name__=="__main__":
    test_gather_distinct_times_per_substance()
