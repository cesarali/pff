"""Tests for VPC StudyJSON conversion in :class:`NewGenerativeMixin`."""

from __future__ import annotations

from dataclasses import replace
from typing import Literal

import pytest
import torch

from pff.config_classes.data_config import MetaDosingConfig
from pff.data.datasets.aicme_batch import AICMECompartmentsDataBatch
from pff.models.amortized_inference.generative_pk import NewGenerativeMixin


class _DummyGenerativeVPC(NewGenerativeMixin):
    """Lightweight generative stub used to validate VPC wrapper behavior."""

    def __init__(
        self,
        *,
        layout: Literal["sbt1", "sb1t1", "sbit1"],
        meta_dosing: MetaDosingConfig | None,
        returned_time_offset: float = 0.0,
    ) -> None:
        self.layout = layout
        self.meta_dosing = meta_dosing
        self.returned_time_offset = float(returned_time_offset)
        self._call_index = 0

    def sample_new_individual(
        self,
        db: AICMECompartmentsDataBatch,
        sample_size: int = 10,
        decode_times=None,
        ignore_logvar: bool = True,
        num_steps: int = None,
    ):
        _ = ignore_logvar
        _ = num_steps

        if decode_times is None:
            raise ValueError("Dummy sampler requires decode_times for VPC tests.")

        times, mask = decode_times  # times: [B, T, 1], mask: [B, T]
        B, T, _ = times.shape
        c_ind = db.context_obs.shape[1]

        base = torch.zeros(sample_size, B, T, 1, dtype=times.dtype, device=times.device)
        for s in range(sample_size):
            for b in range(B):
                for t in range(T):
                    base[s, b, t, 0] = (
                        100 * self._call_index + 10 * s + b + 0.01 * t
                    )  # [S, B, T, 1]

        if self.layout == "sbt1":
            samples = base  # [S, B, T, 1]
        elif self.layout == "sb1t1":
            samples = base.unsqueeze(2)  # [S, B, 1, T, 1]
        elif self.layout == "sbit1":
            samples = torch.zeros(
                sample_size, B, c_ind, T, 1, dtype=base.dtype, device=base.device
            )  # [S, B, I, T, 1]
            for i in range(c_ind):
                samples[:, :, i, :, :] = base + 1000 * i
        else:  # pragma: no cover - guarded by Literal type
            raise ValueError(f"Unsupported dummy layout: {self.layout}")

        returned_times = times + self.returned_time_offset
        self._call_index += 1
        return samples, returned_times, mask


def _make_vpc_batch() -> AICMECompartmentsDataBatch:
    """Create a compact batch with heterogeneous context schedules."""

    B, c_ind, t_ind, T = 2, 3, 1, 5

    context_obs_time = torch.tensor(
        [
            [
                [3.0, 1.0, 3.0, 0.0, 0.0],  # i=0 -> raw valid [3, 1, 3]
                [2.0, 4.0, 0.0, 0.0, 0.0],  # i=1 -> raw valid [2, 4]
                [8.0, 8.0, 7.0, 0.0, 0.0],  # i=2 -> masked out by individual mask
            ],
            [
                [5.0, 2.0, 2.0, 1.0, 0.0],  # i=0 -> raw valid [5, 2, 2, 1]
                [6.0, 8.0, 8.0, 0.0, 0.0],  # i=1 -> raw valid [6, 8, 8]
                [9.0, 0.0, 0.0, 0.0, 0.0],  # i=2 -> masked out by individual mask
            ],
        ],
        dtype=torch.float32,
    ).unsqueeze(-1)  # [B, c_ind, T, 1]
    context_obs_mask = context_obs_time.squeeze(-1) > 0  # [B, c_ind, T]

    return AICMECompartmentsDataBatch(
        target_obs=torch.zeros(B, t_ind, 1, 1),
        target_obs_time=torch.zeros(B, t_ind, 1, 1),
        target_obs_mask=torch.zeros(B, t_ind, 1, dtype=torch.bool),
        target_rem_sim=torch.zeros(B, t_ind, 0, 1),
        target_rem_sim_time=torch.zeros(B, t_ind, 0, 1),
        target_rem_sim_mask=torch.zeros(B, t_ind, 0, dtype=torch.bool),
        context_obs=torch.zeros(B, c_ind, T, 1),
        context_obs_time=context_obs_time,
        context_obs_mask=context_obs_mask,
        context_rem_sim=torch.zeros(B, c_ind, 0, 1),
        context_rem_sim_time=torch.zeros(B, c_ind, 0, 1),
        context_rem_sim_mask=torch.zeros(B, c_ind, 0, dtype=torch.bool),
        target_dosing_amounts=torch.zeros(B, t_ind),
        target_dosing_route_types=torch.zeros(B, t_ind, dtype=torch.long),
        context_dosing_amounts=torch.tensor(
            [[10.0, 20.0, 30.0], [11.0, 0.0, 5.0]], dtype=torch.float32
        ),
        context_dosing_route_types=torch.tensor([[1, 0, 2], [0, 2, 1]], dtype=torch.long),
        mask_context_individuals=torch.tensor(
            [[True, True, False], [True, True, False]], dtype=torch.bool
        ),
        mask_target_individuals=torch.zeros(B, t_ind, dtype=torch.bool),
        study_name=["study_A", ""],
        context_subject_name=[
            ["ctx_0_0", "ctx_0_1", "ctx_0_2"],
            ["ctx_1_0", "ctx_1_1", "ctx_1_2"],
        ],
        target_subject_name=[["tgt_0"], ["tgt_1"]],
        substance_name=["drug_A", ""],
        time_scales=torch.zeros(B, 2),
        is_empirical=True,
    )


@pytest.mark.parametrize("layout", ["sbt1", "sb1t1", "sbit1"])
def test_sample_new_individuals_to_vpc_format(layout: str) -> None:
    """VPC wrapper should return ``[B][S]`` StudyJSONs preserving schedules."""

    batch = _make_vpc_batch()
    meta_dosing = replace(MetaDosingConfig(), route_options=["oral", "iv", "sc"], time=12.0)
    model = _DummyGenerativeVPC(layout=layout, meta_dosing=meta_dosing)

    sample_size = 3
    studies = model.sample_new_individuals_to_vpc_format(batch, sample_size=sample_size)

    # Nested output shape [B][S].
    assert len(studies) == 2
    assert all(len(studies_b) == sample_size for studies_b in studies)

    for b, studies_b in enumerate(studies):
        for s, study in enumerate(studies_b):
            assert set(study.keys()) == {"context", "target", "meta_data"}
            assert study["target"] == []
            assert len(study["context"]) == 2  # only unmasked context individuals

            if b == 0:
                assert study["meta_data"]["study_name"] == "study_A"
                assert study["meta_data"]["substance_name"] == "drug_A"
            else:
                assert study["meta_data"]["study_name"] == "study_1"
                assert study["meta_data"]["substance_name"] == "substance_1"

            # Context schedules preserve raw masked order from context_obs_time.
            expected_times_i0 = [3.0, 1.0, 3.0] if b == 0 else [5.0, 2.0, 2.0, 1.0]
            expected_times_i1 = [2.0, 4.0] if b == 0 else [6.0, 8.0, 8.0]
            assert study["context"][0]["observation_times"] == expected_times_i0
            assert study["context"][1]["observation_times"] == expected_times_i1

            # Metadata and dosing are preserved from the empirical context.
            assert study["context"][0]["name_id"] == ("ctx_0_0" if b == 0 else "ctx_1_0")
            assert study["context"][1]["name_id"] == ("ctx_0_1" if b == 0 else "ctx_1_1")
            assert study["context"][0]["dosing_times"] == [12.0]
            assert study["context"][1]["dosing_times"] == [12.0]
            assert study["context"][0]["dosing_type"] == (["iv"] if b == 0 else ["oral"])
            assert study["context"][1]["dosing_type"] == (["oral"] if b == 0 else ["sc"])
            assert study["context"][0]["dosing"] == ([10.0] if b == 0 else [11.0])
            assert study["context"][1]["dosing"] == ([20.0] if b == 0 else [0.0])

            # Sample axis must be distributed along inner list dimension [S].
            first_obs = study["context"][0]["observations"][0]
            assert isinstance(first_obs, float)
            if s > 0:
                prev_obs = studies_b[s - 1]["context"][0]["observations"][0]
                assert first_obs != prev_obs

            # Masked context index i=2 must be excluded.
            context_names = [ind.get("name_id", "") for ind in study["context"]]
            assert all(name not in {"ctx_0_2", "ctx_1_2"} for name in context_names)

    # In [S,B,I,T,1] layout, selecting index ``i`` should preserve per-index offsets.
    if layout == "sbit1":
        val_i0 = studies[0][0]["context"][0]["observations"][0]
        val_i1 = studies[0][0]["context"][1]["observations"][0]
        assert val_i1 > val_i0 + 500.0


def test_sample_new_individuals_to_vpc_format_requires_meta_dosing() -> None:
    """A clear error is raised when ``meta_dosing`` is unavailable."""

    batch = _make_vpc_batch()
    model = _DummyGenerativeVPC(layout="sbt1", meta_dosing=None)

    with pytest.raises(AttributeError, match="dosing"):
        _ = model.sample_new_individuals_to_vpc_format(batch, sample_size=2)


def test_sample_new_individuals_to_vpc_format_prefers_raw_decode_times() -> None:
    """VPC conversion should keep raw decode schedules even if model times drift."""

    batch = _make_vpc_batch()
    meta_dosing = replace(MetaDosingConfig(), route_options=["oral", "iv", "sc"], time=12.0)
    model = _DummyGenerativeVPC(
        layout="sbt1",
        meta_dosing=meta_dosing,
        returned_time_offset=1e-3,
    )

    studies = model.sample_new_individuals_to_vpc_format(batch, sample_size=1)
    study_b0 = studies[0][0]

    assert study_b0["context"][0]["observation_times"] == [3.0, 1.0, 3.0]
    assert study_b0["context"][1]["observation_times"] == [2.0, 4.0]


if __name__ == "__main__":
    test_sample_new_individuals_to_vpc_format_requires_meta_dosing()
