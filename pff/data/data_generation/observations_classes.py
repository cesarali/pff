from abc import ABC, abstractmethod
from typing import Callable, Optional, Tuple, List

import torch
from torch import Tensor
from torchtyping import TensorType
from pff.config_classes.data_config import ObservationsConfig, MetaStudyConfig
from pff.data.data_generation.observations_functions import fix_past_time_random_selection


def _sample_past_count_with_bias(
    low: int,
    high: int,
    *,
    generative_bias: bool,
    generator: torch.Generator,
    device: torch.device,
) -> int:
    """Sample the number of past observations under the configured bias mode."""

    if high <= 0:
        return 0

    if generative_bias:
        sample_zero = int(torch.randint(0, 2, (1,), generator=generator, device=device).item()) == 0
        if sample_zero:
            return 0

        rest_low = max(1, low)
        if rest_low > high:
            return 0
        if rest_low == high:
            return rest_low
        return int(
            torch.randint(
                rest_low,
                high + 1,
                (1,),
                generator=generator,
                device=device,
            ).item()
        )

    if low >= high:
        return int(high)

    return int(torch.randint(low, high + 1, (1,), generator=generator, device=device).item())


class ObservationStrategy(ABC):
    def __init__(self, observations_config: ObservationsConfig, meta_config: MetaStudyConfig):
        self.observations_config = observations_config
        self.meta_config = meta_config

    def _drop_non_positive_times_from_mask(self, times: Tensor, mask: Tensor) -> Tensor:
        """Optionally invalidate observations at non-positive timestamps.

        When ``drop_time_zero_observations=True`` in :class:`ObservationsConfig`,
        entries with ``time <= 0`` are excluded from downstream sampling.
        """
        if not getattr(self.observations_config, "drop_time_zero_observations", False):
            return mask
        return mask & (times > 0)

    def generate(
        self, full_simulation: Tensor, full_simulation_times: Tensor, **kwargs
    ) -> Tuple[Tensor, ...]:
        """Wrap raw generate: apply add_rem flag"""
        # call subclass raw generation
        obs, obs_time, obs_mask, rem_sim, rem_time, rem_mask = self._generate_raw(
            full_simulation, full_simulation_times, **kwargs
        )
        # drop remaining if not desired
        if not self.observations_config.add_rem:
            rem_sim = rem_time = rem_mask = None
        return obs, obs_time, obs_mask, rem_sim, rem_time, rem_mask, None

    @abstractmethod
    def _generate_raw(
        self, full_simulation: Tensor, full_simulation_times: Tensor, **kwargs
    ) -> Tuple[
        Tensor,
        TensorType["B", "T_obs"],
        TensorType["B", "T_obs"],
        TensorType["B", "T_rem"],
        TensorType["B", "T_rem"],
        TensorType["B", "T_rem"],
    ]:
        """Generate observations and remaining sims raw, regardless of add_rem"""
        pass

    def get_shapes(self) -> Tuple[int, int]:
        """Wrap raw shapes: apply add_rem flag"""
        max_obs, max_rem = self._get_shapes_raw()
        if not self.observations_config.add_rem:
            max_rem = 0
        return max_obs, max_rem

    @abstractmethod
    def _get_shapes_raw(self) -> Tuple[int, int]:
        """Return max observations and max remaining assuming add_rem=True"""
        pass


class PKPeakHalfLifeStrategy(ObservationStrategy):
    """Observation strategy tailored to pharmacokinetic (PK) curves.

    The strategy samples observations around the absorption peak and along the
    elimination phase of a PK simulation. It uses a canonical grid composed of
    four segments:

    1. Several points before the peak that are proportional to the configured
       peak time.
    2. The peak itself.
    3. Several points after the peak spaced by multiples of the provided
       half-life.
    4. Optional remainder points that are handed back to the caller when
       ``add_rem`` is enabled.

    For **synthetic simulations**, the strategy still uses this canonical grid
    and nearest-neighbour alignment.

    For **empirical data**, measurements are treated as already canonical:

    * No canonical time grid construction.
    * No time normalisation or template matching.
    * No interpolation or re-scaling to canonical coordinates.

    Empirical sequences are only padded / truncated to the internal capacity
    implied by :class:`ObservationsConfig` and :class:`MetaStudyConfig`, and
    then passed through the same past/future splitting logic.

    Past/future splitting
    ----------------------
    When ``split_past_future=True``, the canonical sequence for each row is
    split into:

    * a *past* observation block of fixed width (``max_obs``), and
    * an optional *remainder* block of width (``max_rem``).

    In the default mode (no fixed past selection), the number of past points
    is sampled according to ``generative_bias``:

    * ``False`` samples in ``[min_past, max_past]``.
    * ``True`` samples exactly ``0`` with probability 0.5 and, otherwise,
      samples uniformly in ``[max(1, min_past), max_past]``.

    Under ``generative_bias=False``, **short sequences** receive a special treatment: when
    the number of valid canonical points is less than or equal to the
    observation capacity, *all* valid points are placed in the observation
    block and none are shifted into the remainder.

    Fixed past selection
    --------------------
    Calling :meth:`fix_past_selection(k)` activates a strict mode in which
    the strategy tries to expose exactly ``k`` earliest valid timestamps as
    "past" for each series, subject to the following structural limits:

    1. The number of real data points available in the series.
    2. The observation capacity dictated by :meth:`_get_shapes_raw`.

    Concretely, for each row:

    * Let ``k`` be the fixed past count.
    * Let ``total_valid`` be the number of valid canonical points.
    * Let ``past_required = min(k, total_valid)``.

    The observation block receives

    * ``obs_count = min(past_required, max_obs)`` earliest valid points.

    If ``past_required > obs_count`` (for example because ``k`` exceeds the
    number of observation slots), the remaining required past events
    ``past_required - obs_count`` are the *first entries* in the remainder
    block (subject to the remainder capacity). This guarantees that, as long
    as data and shapes allow, the first ``k`` valid timestamps appear in
    ``obs`` + ``rem`` before any later timestamps.

    Calling :meth:`release_past_selection()` returns to the default stochastic
    behaviour governed by ``min_past``/``max_past``.
    """

    _PEAK_PHASE_MULTIPLIERS = (0.1, 0.2, 0.5, 0.8)
    _POST_PEAK_HALF_LIFE_MULTIPLIERS = (
        0.25,
        0.50,
        1.00,
        2.00,
        4.00,
        6.00,
        8.00,
        9.00,
        14.0,
        19.0,
        30.0,
    )
    _RAW_CANONICAL_POINTS = len(_PEAK_PHASE_MULTIPLIERS) + 1 + len(_POST_PEAK_HALF_LIFE_MULTIPLIERS)

    def __init__(
        self, observations_config: ObservationsConfig, meta_config: MetaStudyConfig
    ) -> None:
        super().__init__(observations_config, meta_config)
        self.max_num_obs = observations_config.max_num_obs
        self.split_past_future = observations_config.split_past_future
        self.min_past = observations_config.min_past
        self.max_past = observations_config.max_past
        self.generative_bias = observations_config.generative_bias
        # None → default random selection. When set, the strategy enforces a
        # strict fixed-past semantics as documented above.
        self._fixed_past_obs_count: Optional[int] = None

    def fix_past_selection(self, obs_count: int) -> None:
        """Activate strict ``k``-past behaviour.

        When this mode is active and ``split_past_future=True``, every call to
        :meth:`generate` or :meth:`generate_empirical` will:

        * expose up to ``obs_count`` earliest valid timestamps in the
          observation block, bounded by the available data and the observation
          capacity;
        * place any additional required past events (when ``obs_count`` is
          larger than the observation capacity) at the *front* of the remainder
          block (when a remainder is present).

        The strategy is allowed to allocate fewer than ``obs_count`` past
        events only when:

        * the series contains fewer real data points than ``obs_count``, or
        * the observation/remainder shapes leave insufficient slots.

        In all other cases the earliest valid timestamps are allocated in the
        order: observation block first, then remainder.
        """

        if not self.split_past_future:
            # No split → fixed past count is meaningless.
            return

        if obs_count < self.min_past or obs_count > self.max_past:
            raise ValueError(
                "Fixed past observation count must lie within the configured min/max bounds."
            )
        self._fixed_past_obs_count = int(obs_count)

    def release_past_selection(self) -> None:
        """Return to the default random past selection behaviour."""
        self._fixed_past_obs_count = None

    @classmethod
    def _build_canonical_grid(
        cls,
        *,
        t_peak: float,
        t_half: float,
        device: torch.device,
        dtype: torch.dtype,
    ) -> Tensor:
        """Construct the canonical grid for a single simulation.

        The grid covers the pre-peak, peak and post-peak regime of the curve by
        scaling two fundamental quantities supplied at runtime: the time of the
        peak concentration ``t_peak`` and the half-life ``t_half``. Both values
        are expected to be expressed in the same units as the simulation time
        axis.
        """
        before_peak = [mult * t_peak for mult in cls._PEAK_PHASE_MULTIPLIERS]
        after_peak = [t_peak + mult * t_half for mult in cls._POST_PEAK_HALF_LIFE_MULTIPLIERS]
        values = before_peak + [t_peak] + after_peak
        return torch.tensor(values, device=device, dtype=dtype)

    def _canonical_grid_capacity(self) -> int:
        """Return the number of canonical grid points available.

        The capacity is the minimum between the simulator resolution and the
        theoretical number of canonical points. This ensures that the
        observation tensors never attempt to gather indices outside the
        original simulation.
        """
        time_steps = getattr(self.meta_config, "time_num_steps", self.max_num_obs)
        return max(
            0,
            min(int(self.max_num_obs), int(time_steps), self._RAW_CANONICAL_POINTS),
        )

    def _get_shapes_raw(self) -> Tuple[int, int]:
        """Compute the maximum number of observation and remainder slots.

        Returns
        -------
        max_obs, max_rem : int, int
            * ``max_obs`` – maximum number of observation time-steps.
            * ``max_rem`` – maximum number of remainder time-steps when
              ``add_rem`` is enabled.

        Raises
        ------
        ValueError
            If a past/future split is requested but the canonical capacity
            cannot satisfy the configured ``min_past`` requirement.
        """
        canonical_cap = self._canonical_grid_capacity()
        if canonical_cap == 0:
            return 0, 0

        if self.split_past_future:
            if canonical_cap < self.min_past:
                raise ValueError("Canonical grid capacity is smaller than the configured min_past")
            max_obs = min(self.max_past, canonical_cap)
            max_rem = max(0, canonical_cap - self.min_past)
        else:
            max_obs = canonical_cap
            max_rem = canonical_cap

        return max_obs, max_rem

    @staticmethod
    def _deduplicate_sorted_indices(
        idx: Tensor, valid_mask: Optional[Tensor] = None
    ) -> Tuple[Tensor, Tensor]:
        """Collapse repeated gather indices while preserving alignment.

        ``idx`` is expected to be monotonically increasing. Consecutive
        duplicates are collapsed into a single entry at the front of the tensor
        and the corresponding ``valid_mask`` entries are shifted accordingly.
        """
        if valid_mask is None:
            valid_mask = torch.ones_like(idx, dtype=torch.bool)

        if idx.numel() <= 1:
            return idx, valid_mask

        duplicate_mask = torch.zeros_like(idx, dtype=torch.bool)
        duplicate_mask[1:] = idx[1:] == idx[:-1]

        if not duplicate_mask.any():
            return idx, valid_mask

        unique_mask = ~duplicate_mask
        kept_idx = idx[unique_mask]
        duplicate_idx = idx[duplicate_mask]

        padded_idx = torch.empty_like(idx)
        padded_idx[: kept_idx.numel()] = kept_idx
        padded_idx[kept_idx.numel() :] = duplicate_idx

        kept_valid = valid_mask[unique_mask]
        padded_mask = torch.zeros_like(valid_mask)
        padded_mask[: kept_valid.numel()] = kept_valid

        return padded_idx, padded_mask

    def _assemble_from_canonical(
        self,
        canonical_vals: Tensor,
        canonical_times: Tensor,
        canonical_mask: Tensor,
        *,
        generator: Optional[torch.Generator] = None,
    ) -> Tuple[Tensor, Tensor, Tensor, Optional[Tensor], Optional[Tensor], Optional[Tensor]]:
        """Convert canonical tensors into output observations.

        The canonical representation stores **all** admissible samples for a
        batch element. This helper slices the canonical tensors into the
        "past" observations that will be returned to the caller and (when
        requested) the "future" remainder.

        Allocation invariants
        ---------------------
        For each batch row:

        * Let ``valid_idx`` be the indices where ``canonical_mask`` is True,
          sorted in ascending order.
        * The observation block always receives the **earliest**
          ``obs_count`` indices from ``valid_idx``.
        * The remainder block (when present) receives later indices only; it
          never contains timestamps that precede those in the observation block.
        * Under ``generative_bias=False``, short sequences
          (``total_valid <= max_obs``) keep all valid points in the
          observation block and do not shift points to the remainder.

        When :meth:`fix_past_selection(k)` is active, we define::

            past_required = min(k, total_valid)

        and allocate:

        * ``obs_count = min(past_required, max_obs)`` to the observation
          block; and
        * any surplus past events ``past_required - obs_count`` at the **front**
          of the remainder block (subject to the remainder capacity), followed
          by any truly future points.

        Releasing the fixed selection returns to the stochastic behaviour
        controlled by ``generative_bias``.
        """
        max_obs, max_rem = self._get_shapes_raw()
        device = canonical_vals.device
        dtype = canonical_vals.dtype
        batch, _ = canonical_vals.shape

        obs_out = torch.zeros(batch, max_obs, dtype=dtype, device=device)
        obs_time = torch.zeros_like(obs_out)
        obs_mask = torch.zeros(batch, max_obs, dtype=torch.bool, device=device)

        rem_sim = rem_time = rem_mask = None
        if max_rem > 0:
            rem_sim = torch.zeros(batch, max_rem, dtype=dtype, device=device)
            rem_time = torch.zeros_like(rem_sim)
            rem_mask = torch.zeros(batch, max_rem, dtype=torch.bool, device=device)

        gen = generator if generator is not None else torch.default_generator

        for row in range(batch):
            valid_idx = canonical_mask[row].nonzero(as_tuple=True)[0]
            total_valid = int(valid_idx.numel())
            if total_valid == 0:
                continue

            fixed_k = self._fixed_past_obs_count if self.split_past_future else None

            # ------------------------------------------------------------------
            # 1) Decide obs_count
            # ------------------------------------------------------------------
            if self.split_past_future and fixed_k is not None:
                # Strict fixed-past semantics. Structural limits:
                #   - real data (total_valid)
                #   - observation capacity (max_obs)
                past_required = min(fixed_k, total_valid)
                obs_capacity = min(max_obs, total_valid)
                obs_count = min(past_required, obs_capacity)
            else:
                # Default stochastic behaviour; the short-series fix is kept
                # for the non-biased mode only.
                if self.split_past_future:
                    low = min(self.min_past, total_valid)
                    high = min(self.max_past, total_valid)

                    sampled = _sample_past_count_with_bias(
                        low=low,
                        high=high,
                        generative_bias=self.generative_bias,
                        generator=gen,
                        device=device,
                    )

                    if (not self.generative_bias) and total_valid <= max_obs:
                        # Short-series fix: never push valid points into the
                        # remainder just to satisfy a random split.
                        obs_count = total_valid
                    else:
                        obs_count = min(sampled, max_obs)
                else:
                    obs_count = min(total_valid, max_obs)

            # Safety clamp.
            obs_count = max(0, min(obs_count, min(max_obs, total_valid)))

            # ------------------------------------------------------------------
            # 2) Fill observation block (earliest obs_count indices)
            # ------------------------------------------------------------------
            if obs_count > 0:
                take = valid_idx[:obs_count]
                obs_out[row, :obs_count] = canonical_vals[row, take]
                obs_time[row, :obs_count] = canonical_times[row, take]
                obs_mask[row, :obs_count] = True

            # ------------------------------------------------------------------
            # 3) Fill remainder block (if enabled)
            # ------------------------------------------------------------------
            if rem_sim is not None:
                if self.split_past_future and fixed_k is not None:
                    # Remaining required past events plus genuine future.
                    past_required = min(fixed_k, total_valid)
                    # indices that are still part of the fixed past window
                    # but did not fit into the observation block
                    extra_past_idx = valid_idx[obs_count:past_required]
                    future_idx = valid_idx[past_required:]

                    candidates: List[Tensor] = []
                    if extra_past_idx.numel() > 0:
                        candidates.append(extra_past_idx)
                    if future_idx.numel() > 0:
                        candidates.append(future_idx)
                    if candidates:
                        remainder_candidates = torch.cat(candidates, dim=0)
                    else:
                        remainder_candidates = valid_idx.new_empty((0,), dtype=valid_idx.dtype)
                else:
                    # Default behaviour: everything after the obs window.
                    remainder_candidates = valid_idx[obs_count:]

                rem_count = min(int(remainder_candidates.numel()), max_rem)
                if rem_count > 0:
                    rem_idx = remainder_candidates[:rem_count]
                    rem_sim[row, :rem_count] = canonical_vals[row, rem_idx]
                    rem_time[row, :rem_count] = canonical_times[row, rem_idx]
                    rem_mask[row, :rem_count] = True

        return obs_out, obs_time, obs_mask, rem_sim, rem_time, rem_mask

    def _align_simulation_to_canonical(
        self,
        full_simulation: Tensor,
        full_simulation_times: Tensor,
        *,
        time_scales: Tensor,
        num_obs_sampler: Optional[Callable[[int], Tensor]] = None,
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        """Gather canonical samples from a simulated PK curve.

        Synthetic behaviour is unchanged compared to the original strategy:
        we build a canonical grid, snap it to the nearest simulation times and
        optionally subsample points via ``num_obs_sampler``.
        """
        device = full_simulation.device
        dtype = full_simulation.dtype
        batch, _ = full_simulation.shape
        time_steps = int(full_simulation_times.size(1))

        # DataLoader workers may receive empty row slices (B=0). In that case
        # there is no reference timeline to align against; return an empty
        # canonical block and let _assemble_from_canonical create [B, *] outputs.
        if batch == 0 or time_steps == 0:
            zero = torch.zeros(batch, 0, dtype=dtype, device=device)
            mask = torch.zeros(batch, 0, dtype=torch.bool, device=device)
            return zero, zero, mask, time_scales.clone()

        canonical_cap = self._canonical_grid_capacity()
        if canonical_cap == 0:
            zero = torch.zeros(batch, 0, dtype=dtype, device=device)
            mask = torch.zeros(batch, 0, dtype=torch.bool, device=device)
            return zero, zero, mask, time_scales.clone()

        grid = self._build_canonical_grid(
            t_peak=time_scales[0].item(),
            t_half=time_scales[1].item(),
            device=device,
            dtype=dtype,
        )[:canonical_cap]

        ref_times = full_simulation_times[0]
        min_time = ref_times.min()
        max_time = ref_times.max()
        grid_valid_mask = (grid >= min_time) & (grid <= max_time)

        idx = torch.cdist(grid[:, None], ref_times[:, None]).argmin(dim=1)
        idx, order = idx.sort()
        grid_valid_mask = grid_valid_mask[order]
        idx, grid_valid_mask = self._deduplicate_sorted_indices(idx, grid_valid_mask)

        gather_idx = idx[None, :].expand(batch, -1)
        batch_idx = torch.arange(batch, device=device)[:, None]

        canonical_vals = full_simulation[batch_idx, gather_idx]
        canonical_times = full_simulation_times[batch_idx, gather_idx]

        invalid_slots = ~grid_valid_mask
        if invalid_slots.any():
            canonical_vals[:, invalid_slots] = 0
            canonical_times[:, invalid_slots] = 0

        if num_obs_sampler is None:
            total_counts = torch.full((batch,), canonical_cap, dtype=torch.long, device=device)
        else:
            sampled = num_obs_sampler(batch).to(device=device).long()
            total_counts = sampled.clamp(min=0, max=canonical_cap)

        max_valid = int(grid_valid_mask.sum().item())
        if max_valid == 0:
            total_counts.zero_()
        else:
            total_counts.clamp_(max=max_valid)

        valid_order = grid_valid_mask.long().cumsum(dim=0) - 1
        valid_order = torch.where(
            grid_valid_mask,
            valid_order,
            torch.full_like(valid_order, -1, dtype=valid_order.dtype),
        )
        canonical_mask = grid_valid_mask[None, :] & (valid_order[None, :] < total_counts[:, None])
        canonical_mask = self._drop_non_positive_times_from_mask(canonical_times, canonical_mask)

        return canonical_vals, canonical_times, canonical_mask, time_scales.clone()

    def _align_empirical_to_canonical(
        self,
        empirical_obs: Tensor,
        empirical_times: Tensor,
        empirical_mask: Tensor,
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """(Legacy) Project empirical observations onto the canonical grid.

        This method is retained for backward compatibility but is **not** used
        by :meth:`generate_empirical`, which now treats empirical data as
        already canonical. New code should avoid calling this helper.
        """
        device = empirical_obs.device
        dtype = empirical_obs.dtype
        batch, _ = empirical_obs.shape
        canonical_cap = self._canonical_grid_capacity()

        canonical_vals = torch.zeros(batch, canonical_cap, dtype=dtype, device=device)
        canonical_times = torch.zeros_like(canonical_vals)
        canonical_mask = torch.zeros(batch, canonical_cap, dtype=torch.bool, device=device)

        if canonical_cap == 0:
            return canonical_vals, canonical_times, canonical_mask

        for row in range(batch):
            valid_idx = empirical_mask[row].nonzero(as_tuple=True)[0]
            if valid_idx.numel() == 0:
                continue

            obs_row = empirical_obs[row, valid_idx]
            time_row = empirical_times[row, valid_idx]
            max_time = torch.maximum(time_row.max(), torch.tensor(1.0, device=device))
            norm_time = time_row / max_time

            peak_idx = obs_row.argmax().item()
            t_peak = norm_time[peak_idx].item()
            post_times = norm_time[peak_idx:]
            post_obs = obs_row[peak_idx:]
            half_level = obs_row[peak_idx] / 2
            below_half = (post_obs <= half_level).nonzero(as_tuple=True)[0]
            if below_half.numel() == 0:
                half_time = post_times[-1].item()
            else:
                half_time = post_times[below_half[0]].item()
            t_half = max(half_time - t_peak, 1e-3)

            grid = self._build_canonical_grid(
                t_peak=t_peak if t_peak > 0 else 1e-3,
                t_half=t_half,
                device=device,
                dtype=dtype,
            )[:canonical_cap].clamp(max=1.0)

            actual_grid = grid * max_time
            distances = torch.cdist(actual_grid[:, None], time_row[:, None])
            nearest = distances.argmin(dim=1)

            usable = min(time_row.numel(), grid.numel())
            if usable == 0:
                continue

            canonical_vals[row, :usable] = obs_row[nearest[:usable]]
            canonical_times[row, :usable] = time_row[nearest[:usable]]
            canonical_mask[row, :usable] = True

        canonical_mask = self._drop_non_positive_times_from_mask(canonical_times, canonical_mask)

        return canonical_vals, canonical_times, canonical_mask

    def _prepare_empirical_as_canonical(
        self,
        empirical_obs: Tensor,
        empirical_times: Tensor,
        empirical_mask: Tensor,
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """Treat empirical observations as already canonical.

        This helper:

        * does **not** build any canonical grid;
        * does **not** normalise or re-scale time;
        * simply copies valid empirical points in their original order into
          fixed-size tensors, padding with zeros / False as needed.

        The resulting tensors have width equal to the canonical capacity so
        that they can be passed to :meth:`_assemble_from_canonical`.
        """
        device = empirical_obs.device
        dtype = empirical_obs.dtype
        batch, _ = empirical_obs.shape
        canonical_cap = self._canonical_grid_capacity()

        canonical_vals = torch.zeros(batch, canonical_cap, dtype=dtype, device=device)
        canonical_times = torch.zeros_like(canonical_vals)
        canonical_mask = torch.zeros(batch, canonical_cap, dtype=torch.bool, device=device)

        if canonical_cap == 0:
            return canonical_vals, canonical_times, canonical_mask

        for row in range(batch):
            valid_idx = empirical_mask[row].nonzero(as_tuple=True)[0]
            if valid_idx.numel() == 0:
                continue

            take_count = min(int(valid_idx.numel()), canonical_cap)
            take_idx = valid_idx[:take_count]

            canonical_vals[row, :take_count] = empirical_obs[row, take_idx]
            canonical_times[row, :take_count] = empirical_times[row, take_idx]
            canonical_mask[row, :take_count] = True

        canonical_mask = self._drop_non_positive_times_from_mask(canonical_times, canonical_mask)

        return canonical_vals, canonical_times, canonical_mask

    def _generate_raw(
        self, full_simulation: Tensor, full_simulation_times: Tensor, **kwargs
    ) -> Tuple[
        Tensor, Tensor, Tensor, Optional[Tensor], Optional[Tensor], Optional[Tensor], Tensor
    ]:
        """Deterministic canonical PK sampling for synthetic simulations."""
        time_scales: Optional[Tensor] = kwargs.get("time_scales")
        if time_scales is None:
            raise ValueError("time_scales must be provided for PKPeakHalfLifeStrategy")

        canonical_vals, canonical_times, canonical_mask, rescaled = (
            self._align_simulation_to_canonical(
                full_simulation,
                full_simulation_times,
                time_scales=time_scales,
                num_obs_sampler=kwargs.get("num_obs_sampler"),
            )
        )

        obs_out, obs_time, obs_mask, rem_sim, rem_time, rem_mask = self._assemble_from_canonical(
            canonical_vals,
            canonical_times,
            canonical_mask,
            generator=kwargs.get("generator"),
        )

        return obs_out, obs_time, obs_mask, rem_sim, rem_time, rem_mask, rescaled

    def _generate_random(
        self,
        full_simulation: Tensor,
        full_simulation_times: Tensor,
        *,
        time_scales: Tensor,
        generator: Optional[torch.Generator] = None,
    ) -> Tuple[
        Tensor, Tensor, Tensor, Optional[Tensor], Optional[Tensor], Optional[Tensor], Tensor
    ]:
        """Randomised variant of canonical observation generation.

        The pre- and post-peak segments are sampled from uniform distributions
        bounded by the canonical limits. This keeps the semantic meaning of the
        selected points while injecting stochasticity that can improve
        robustness during training.
        """
        device, dtype = full_simulation.device, full_simulation.dtype
        batch = full_simulation.size(0)
        time_steps = int(full_simulation_times.size(1))
        if batch == 0 or time_steps == 0:
            canonical_vals = torch.zeros(batch, 0, dtype=dtype, device=device)
            canonical_times = torch.zeros(batch, 0, dtype=dtype, device=device)
            canonical_mask = torch.zeros(batch, 0, dtype=torch.bool, device=device)
            obs_out, obs_time, obs_mask, rem_sim, rem_time, rem_mask = self._assemble_from_canonical(
                canonical_vals, canonical_times, canonical_mask, generator=generator
            )
            return obs_out, obs_time, obs_mask, rem_sim, rem_time, rem_mask, time_scales.clone()
        t_peak, t_half = time_scales[0].item(), time_scales[1].item()

        n_pre = len(self._PEAK_PHASE_MULTIPLIERS)
        n_post = len(self._POST_PEAK_HALF_LIFE_MULTIPLIERS)

        # Uniform samples before peak
        pre_times = torch.rand(n_pre, device=device, dtype=dtype) * t_peak
        # Always include the peak
        peak_time = torch.tensor([t_peak], device=device, dtype=dtype)
        # Uniform samples after peak
        post_times = []
        for mult in self._POST_PEAK_HALF_LIFE_MULTIPLIERS:
            t_end = t_peak + mult * t_half
            t_rand = torch.empty(1, device=device, dtype=dtype).uniform_(t_peak, t_end)
            post_times.append(t_rand)
        post_times = torch.cat(post_times, dim=0)

        # Truncate to canonical capacity
        grid = torch.cat([pre_times, peak_time, post_times], dim=0)
        canonical_cap = self._canonical_grid_capacity()
        grid = grid[:canonical_cap]

        # Map grid to nearest simulation points
        ref_times = full_simulation_times[0]
        idx = torch.cdist(grid[:, None], ref_times[:, None]).argmin(dim=1)
        idx, _ = idx.sort()
        valid_mask = torch.ones_like(idx, dtype=torch.bool)
        idx, valid_mask = self._deduplicate_sorted_indices(idx, valid_mask)
        gather_idx = idx[None, :].expand(batch, -1)
        batch_idx = torch.arange(batch, device=device)[:, None]

        canonical_vals = full_simulation[batch_idx, gather_idx]
        canonical_times = full_simulation_times[batch_idx, gather_idx]
        invalid_slots = ~valid_mask
        if invalid_slots.any():
            canonical_vals[:, invalid_slots] = 0
            canonical_times[:, invalid_slots] = 0

        canonical_mask = valid_mask[None, :].expand(batch, -1).clone()
        canonical_mask = self._drop_non_positive_times_from_mask(canonical_times, canonical_mask)

        obs_out, obs_time, obs_mask, rem_sim, rem_time, rem_mask = self._assemble_from_canonical(
            canonical_vals, canonical_times, canonical_mask, generator=generator
        )
        return obs_out, obs_time, obs_mask, rem_sim, rem_time, rem_mask, time_scales.clone()

    def generate(
        self,
        full_simulation: Tensor,
        full_simulation_times: Tensor,
        **kwargs,
    ) -> Tuple[
        Tensor, Tensor, Tensor, Optional[Tensor], Optional[Tensor], Optional[Tensor], Tensor
    ]:
        """Generate PK observations for synthetic simulations.

        With probability ``randomize_prob`` (default 0.5) the method delegates
        to :meth:`_generate_random`; otherwise the deterministic
        :meth:`_generate_raw` path is taken. Setting ``deterministic_only=True``
        forces the deterministic branch. Both paths require ``time_scales`` and
        honour the ``add_rem`` flag.
        """
        time_scales: Optional[Tensor] = kwargs.get("time_scales")
        if time_scales is None:
            raise ValueError("time_scales must be provided for PKPeakHalfLifeStrategy")

        deterministic_only = kwargs.pop("deterministic_only", False)

        use_random = False
        if not deterministic_only:
            use_random = torch.rand(()) < getattr(self, "randomize_prob", 0.5)

        if use_random:
            obs, obs_time, obs_mask, rem_sim, rem_time, rem_mask, rescaled = self._generate_random(
                full_simulation,
                full_simulation_times,
                time_scales=time_scales,
                generator=kwargs.get("generator"),
            )
        else:
            obs, obs_time, obs_mask, rem_sim, rem_time, rem_mask, rescaled = self._generate_raw(
                full_simulation,
                full_simulation_times,
                **kwargs,
            )

        if not self.observations_config.add_rem:
            rem_sim = rem_time = rem_mask = None

        return obs, obs_time, obs_mask, rem_sim, rem_time, rem_mask, rescaled

    def generate_empirical(
        self,
        empirical_obs: Tensor,
        empirical_times: Tensor,
        empirical_mask: Tensor,
        *,
        generator: Optional[torch.Generator] = None,
    ) -> Tuple[Tensor, Tensor, Tensor, Optional[Tensor], Optional[Tensor], Optional[Tensor]]:
        """Generate observations from empirical data.

        Empirical measurements are assumed to already live on their correct
        time grid. This routine:

        * does **not** perform canonical alignment or time normalisation;
        * only pads / truncates sequences to match the internal capacity;
        * applies past/future splitting via :meth:`_assemble_from_canonical`
          using the configuration in :class:`ObservationsConfig`.

        Synthetic simulations keep using the canonical alignment path.
        """
        canonical_vals, canonical_times, canonical_mask = self._prepare_empirical_as_canonical(
            empirical_obs,
            empirical_times,
            empirical_mask,
        )

        obs, obs_time, obs_mask, rem_sim, rem_time, rem_mask = self._assemble_from_canonical(
            canonical_vals,
            canonical_times,
            canonical_mask,
            generator=generator,
        )

        if not self.observations_config.add_rem:
            rem_sim = rem_time = rem_mask = None

        return obs, obs_time, obs_mask, rem_sim, rem_time, rem_mask


class PKPeakHalfLifeStrategyOld(ObservationStrategy):
    """Observation strategy tailored to pharmacokinetic (PK) curves.

    The strategy samples observations around the absorption peak and along the
    elimination phase of a PK simulation.  It uses a canonical grid composed of
    four segments:

    1. Several points before the peak that are proportional to the configured
       peak time.
    2. The peak itself.
    3. Several points after the peak spaced by multiples of the provided
       half-life.
    4. Optional remainder points that are handed back to the caller when
       ``add_rem`` is enabled.

    The resulting observation tensor can be optionally split into "past" and
    "future" observations according to :class:`ObservationsConfig`.

    Parameters
    ----------
    observations_config:
        Simulation-level configuration that defines sampling constraints such
        as ``max_num_obs`` or the minimum/maximum number of "past" points when
        a split is requested.
    meta_config:
        Meta-study configuration.  Only the ``time_num_steps`` attribute is
        used and allows clamping the canonical grid to the resolution of the
        simulator.
    """

    _PEAK_PHASE_MULTIPLIERS = (0.1, 0.2, 0.5, 0.8)
    _POST_PEAK_HALF_LIFE_MULTIPLIERS = (
        0.25,
        0.50,
        1.00,
        2.00,
        4.00,
        6.00,
        8.00,
        9.00,
        14.0,
        19.0,
        30.0,
    )
    _RAW_CANONICAL_POINTS = len(_PEAK_PHASE_MULTIPLIERS) + 1 + len(_POST_PEAK_HALF_LIFE_MULTIPLIERS)

    def __init__(
        self, observations_config: ObservationsConfig, meta_config: MetaStudyConfig
    ) -> None:
        super().__init__(observations_config, meta_config)
        self.max_num_obs = observations_config.max_num_obs
        self.split_past_future = observations_config.split_past_future
        self.min_past = observations_config.min_past
        self.max_past = observations_config.max_past
        self.generative_bias = observations_config.generative_bias
        # ``None`` indicates that the number of past observations should be
        # sampled according to the standard strategy.  When populated it forces
        # :meth:`_assemble_from_canonical` to always select the provided number
        # of past observations (within the valid range).
        self._fixed_past_obs_count: Optional[int] = None

    def fix_past_selection(self, obs_count: int) -> None:
        """Force the past observation count to ``obs_count`` when splitting.

        The override is only applied when ``split_past_future`` is enabled.  The
        provided ``obs_count`` must fall within ``[min_past, max_past]``.
        """

        if not self.split_past_future:
            return

        if obs_count < self.min_past or obs_count > self.max_past:
            raise ValueError(
                "Fixed past observation count must lie within the configured min/max bounds."
            )
        self._fixed_past_obs_count = int(obs_count)

    def release_past_selection(self) -> None:
        """Return to the default random past selection behaviour."""

        self._fixed_past_obs_count = None

    @classmethod
    def _build_canonical_grid(
        cls,
        *,
        t_peak: float,
        t_half: float,
        device: torch.device,
        dtype: torch.dtype,
    ) -> Tensor:
        """Construct the canonical grid for a single simulation.

        The grid covers the pre-peak, peak and post-peak regime of the curve by
        scaling two fundamental quantities supplied at runtime: the time of the
        peak concentration ``t_peak`` and the half-life ``t_half``.  Both values
        are expected to be expressed in the same units as the simulation time
        axis.

        Parameters
        ----------
        t_peak:
            Estimated time of the concentration peak.
        t_half:
            Estimated half-life used to position post-peak points.
        device, dtype:
            Torch device and dtype for the returned tensor so that it matches
            the simulation tensors that will be gathered later on.

        Returns
        -------
        torch.Tensor
            One-dimensional tensor containing monotonically increasing times
            representing the canonical sampling grid.
        """
        before_peak = [mult * t_peak for mult in cls._PEAK_PHASE_MULTIPLIERS]
        after_peak = [t_peak + mult * t_half for mult in cls._POST_PEAK_HALF_LIFE_MULTIPLIERS]
        values = before_peak + [t_peak] + after_peak
        return torch.tensor(values, device=device, dtype=dtype)

    def _canonical_grid_capacity(self) -> int:
        """Return the number of canonical grid points available.

        The capacity is the minimum between the simulator resolution and the
        theoretical number of canonical points.  This ensures that the
        observation tensors never attempt to gather indices outside the
        original simulation.

        Returns
        -------
        int
            Maximum number of grid points that can be sampled for each
            simulation in the batch.
        """
        time_steps = getattr(self.meta_config, "time_num_steps", self.max_num_obs)
        return max(
            0,
            min(int(self.max_num_obs), int(time_steps), self._RAW_CANONICAL_POINTS),
        )

    def _get_shapes_raw(self) -> Tuple[int, int]:
        """Compute the maximum number of observation and remainder slots.

        The method applies the canonical grid capacity alongside the
        ``split_past_future`` configuration to decide how many points can be
        surfaced directly as observations and how many should be exposed as
        "remaining" (future) points.

        Returns
        -------
        tuple[int, int]
            The first entry is the maximum number of observations.  The second
            entry is the maximum number of remaining observations when
            ``add_rem`` is enabled.

        Raises
        ------
        ValueError
            If a past/future split is requested but the canonical capacity
            cannot satisfy the configured ``min_past`` requirement.
        """
        canonical_cap = self._canonical_grid_capacity()
        if canonical_cap == 0:
            return 0, 0

        if self.split_past_future:
            if canonical_cap < self.min_past:
                raise ValueError("Canonical grid capacity is smaller than the configured min_past")
            max_obs = min(self.max_past, canonical_cap)
            max_rem = max(0, canonical_cap - self.min_past)
        else:
            max_obs = canonical_cap
            max_rem = canonical_cap

        return max_obs, max_rem

    @staticmethod
    def _deduplicate_sorted_indices(
        idx: Tensor, valid_mask: Optional[Tensor] = None
    ) -> Tuple[Tensor, Tensor]:
        """Collapse repeated gather indices while preserving alignment."""

        if valid_mask is None:
            valid_mask = torch.ones_like(idx, dtype=torch.bool)

        if idx.numel() <= 1:
            return idx, valid_mask

        duplicate_mask = torch.zeros_like(idx, dtype=torch.bool)
        duplicate_mask[1:] = idx[1:] == idx[:-1]

        if not duplicate_mask.any():
            return idx, valid_mask

        unique_mask = ~duplicate_mask
        kept_idx = idx[unique_mask]
        duplicate_idx = idx[duplicate_mask]

        padded_idx = torch.empty_like(idx)
        padded_idx[: kept_idx.numel()] = kept_idx
        padded_idx[kept_idx.numel() :] = duplicate_idx

        kept_valid = valid_mask[unique_mask]
        padded_mask = torch.zeros_like(valid_mask)
        padded_mask[: kept_valid.numel()] = kept_valid

        return padded_idx, padded_mask

    def _assemble_from_canonical(
        self,
        canonical_vals: Tensor,
        canonical_times: Tensor,
        canonical_mask: Tensor,
        *,
        generator: Optional[torch.Generator] = None,
    ) -> Tuple[Tensor, Tensor, Tensor, Optional[Tensor], Optional[Tensor], Optional[Tensor]]:
        """Convert canonical tensors into output observations.

        The canonical representation stores **all** admissible samples for a
        batch element.  This helper slices the canonical tensors into the
        "past" observations that will be returned to the caller and (when
        requested) the "future" remainder.  The selection proceeds row by row:

        1. ``canonical_mask`` is inspected to identify the indices that contain
           valid information.  These are the only points that may be surfaced.
        2. When ``split_past_future`` is ``False`` every valid point is treated
           as part of the observation history up to the configured capacity.
        3. Otherwise we randomly draw ``obs_count`` between ``min_past`` and
           ``max_past`` (capped by the number of valid canonical entries).  The
           first ``obs_count`` indices become past observations while the
           remaining valid points are placed in the remainder tensors.

        Parameters
        ----------
        canonical_vals, canonical_times:
            Tensors produced by aligning the simulation or empirical data to
            the canonical grid.
        canonical_mask:
            Boolean tensor marking valid entries for each batch element.
        generator:
            Optional random generator used when sampling ``obs_count`` in
            split-past/future mode.

        Returns
        -------
        tuple of tensors
            Observation and remaining tensors matching the shapes dictated by
            :meth:`_get_shapes_raw`.  All tensors share the same device and
            dtype as the inputs.  ``None`` is returned for remainder tensors
            when the capacity is zero.
        """
        max_obs, max_rem = self._get_shapes_raw()
        device = canonical_vals.device
        dtype = canonical_vals.dtype
        batch, _ = canonical_vals.shape

        obs_out = torch.zeros(batch, max_obs, dtype=dtype, device=device)
        obs_time = torch.zeros_like(obs_out)
        obs_mask = torch.zeros(batch, max_obs, dtype=torch.bool, device=device)

        rem_sim = rem_time = rem_mask = None
        if max_rem > 0:
            rem_sim = torch.zeros(batch, max_rem, dtype=dtype, device=device)
            rem_time = torch.zeros_like(rem_sim)
            rem_mask = torch.zeros(batch, max_rem, dtype=torch.bool, device=device)

        gen = generator if generator is not None else torch.default_generator

        for row in range(batch):
            valid_idx = canonical_mask[row].nonzero(as_tuple=True)[0]
            total_valid = valid_idx.numel()
            if total_valid == 0:
                continue

            if self.split_past_future:
                low = min(self.min_past, total_valid)
                high = min(self.max_past, total_valid)
                if self._fixed_past_obs_count is not None:
                    obs_count = min(self._fixed_past_obs_count, total_valid)
                else:
                    obs_count = _sample_past_count_with_bias(
                        low=low,
                        high=high,
                        generative_bias=self.generative_bias,
                        generator=gen,
                        device=device,
                    )
                obs_count = min(obs_count, max_obs)
            else:
                obs_count = min(total_valid, max_obs)

            if obs_count > 0:
                take = valid_idx[:obs_count]
                obs_out[row, :obs_count] = canonical_vals[row, take]
                obs_time[row, :obs_count] = canonical_times[row, take]
                obs_mask[row, :obs_count] = True

            if rem_sim is not None:
                rem_candidates = valid_idx[obs_count:]
                rem_count = min(rem_candidates.numel(), max_rem)
                if rem_count > 0:
                    rem_idx = rem_candidates[:rem_count]
                    rem_sim[row, :rem_count] = canonical_vals[row, rem_idx]
                    rem_time[row, :rem_count] = canonical_times[row, rem_idx]
                    rem_mask[row, :rem_count] = True

        return obs_out, obs_time, obs_mask, rem_sim, rem_time, rem_mask

    def _align_simulation_to_canonical(
        self,
        full_simulation: Tensor,
        full_simulation_times: Tensor,
        *,
        time_scales: Tensor,
        num_obs_sampler: Optional[Callable[[int], Tensor]] = None,
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        """Gather the canonical samples from a simulated PK curve.

        The routine creates the canonical grid described in the configuration
        (using the provided ``time_scales``) and then performs a nearest-neighbour
        lookup on the simulated trajectory.  Each grid location picks the
        closest time point from the reference simulation (the first batch row);
        the same indices are applied to every batch element so that values and
        times remain aligned across the batch.  ``num_obs_sampler`` can further
        prune the resulting grid by specifying how many of those canonical
        points should remain valid for each row.

        Parameters
        ----------
        full_simulation, full_simulation_times:
            Batched tensors representing the simulated concentration curve and
            its time axis.
        time_scales:
            Two-element tensor with ``t_peak`` and ``t_half`` scaling factors.
        num_obs_sampler:
            Optional callable that samples how many canonical points should be
            retained for each batch element.

        Returns
        -------
        tuple of torch.Tensor
            The canonical values, their corresponding times, a boolean mask of
            valid entries and the (cloned) ``time_scales`` tensor.  When the
            canonical capacity is zero, zero-sized tensors are returned for the
            first three entries.
        """
        device = full_simulation.device
        dtype = full_simulation.dtype
        batch, _ = full_simulation.shape
        time_steps = int(full_simulation_times.size(1))

        # Empty worker slices (B=0) and zero-step trajectories are valid edge
        # cases; return empty canonical tensors and keep shape assembly
        # delegated to _assemble_from_canonical.
        if batch == 0 or time_steps == 0:
            zero = torch.zeros(batch, 0, dtype=dtype, device=device)
            mask = torch.zeros(batch, 0, dtype=torch.bool, device=device)
            return zero, zero, mask, time_scales.clone()

        canonical_cap = self._canonical_grid_capacity()
        if canonical_cap == 0:
            zero = torch.zeros(batch, 0, dtype=dtype, device=device)
            mask = torch.zeros(batch, 0, dtype=torch.bool, device=device)
            return zero, zero, mask, time_scales.clone()

        grid = self._build_canonical_grid(
            t_peak=time_scales[0].item(),
            t_half=time_scales[1].item(),
            device=device,
            dtype=dtype,
        )[:canonical_cap]

        ref_times = full_simulation_times[0]
        min_time = ref_times.min()
        max_time = ref_times.max()
        grid_valid_mask = (grid >= min_time) & (grid <= max_time)

        idx = torch.cdist(grid[:, None], ref_times[:, None]).argmin(dim=1)
        idx, order = idx.sort()
        grid_valid_mask = grid_valid_mask[order]
        idx, grid_valid_mask = self._deduplicate_sorted_indices(idx, grid_valid_mask)

        gather_idx = idx[None, :].expand(batch, -1)
        batch_idx = torch.arange(batch, device=device)[:, None]

        canonical_vals = full_simulation[batch_idx, gather_idx]
        canonical_times = full_simulation_times[batch_idx, gather_idx]

        invalid_slots = ~grid_valid_mask
        if invalid_slots.any():
            canonical_vals[:, invalid_slots] = 0
            canonical_times[:, invalid_slots] = 0

        if num_obs_sampler is None:
            total_counts = torch.full((batch,), canonical_cap, dtype=torch.long, device=device)
        else:
            sampled = num_obs_sampler(batch).to(device=device).long()
            total_counts = sampled.clamp(min=0, max=canonical_cap)

        max_valid = int(grid_valid_mask.sum().item())
        if max_valid == 0:
            total_counts.zero_()
        else:
            total_counts.clamp_(max=max_valid)

        valid_order = grid_valid_mask.long().cumsum(dim=0) - 1
        valid_order = torch.where(
            grid_valid_mask,
            valid_order,
            torch.full_like(valid_order, -1, dtype=valid_order.dtype),
        )
        canonical_mask = grid_valid_mask[None, :] & (valid_order[None, :] < total_counts[:, None])
        canonical_mask = self._drop_non_positive_times_from_mask(canonical_times, canonical_mask)

        return canonical_vals, canonical_times, canonical_mask, time_scales.clone()

    def _align_empirical_to_canonical(
        self,
        empirical_obs: Tensor,
        empirical_times: Tensor,
        empirical_mask: Tensor,
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """Project empirical observations onto the canonical grid.

        The projection normalises the empirical time axis to estimate the peak
        and half-life from the data itself.  This allows harmonising real
        measurements with the canonical layout used during simulation-driven
        training.

        Parameters
        ----------
        empirical_obs, empirical_times, empirical_mask:
            Batched tensors storing empirical observations, the corresponding
            time stamps and a mask of valid entries.

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor, torch.Tensor]
            Canonical values, times and boolean masks aligned to the canonical
            sampling scheme.
        """
        device = empirical_obs.device
        dtype = empirical_obs.dtype
        batch, _ = empirical_obs.shape
        canonical_cap = self._canonical_grid_capacity()

        canonical_vals = torch.zeros(batch, canonical_cap, dtype=dtype, device=device)
        canonical_times = torch.zeros_like(canonical_vals)
        canonical_mask = torch.zeros(batch, canonical_cap, dtype=torch.bool, device=device)

        if canonical_cap == 0:
            return canonical_vals, canonical_times, canonical_mask

        for row in range(batch):
            valid_idx = empirical_mask[row].nonzero(as_tuple=True)[0]
            if valid_idx.numel() == 0:
                continue

            obs_row = empirical_obs[row, valid_idx]
            time_row = empirical_times[row, valid_idx]
            max_time = torch.maximum(time_row.max(), torch.tensor(1.0, device=device))
            norm_time = time_row / max_time

            peak_idx = obs_row.argmax().item()
            t_peak = norm_time[peak_idx].item()
            post_times = norm_time[peak_idx:]
            post_obs = obs_row[peak_idx:]
            half_level = obs_row[peak_idx] / 2
            below_half = (post_obs <= half_level).nonzero(as_tuple=True)[0]
            if below_half.numel() == 0:
                half_time = post_times[-1].item()
            else:
                half_time = post_times[below_half[0]].item()
            t_half = max(half_time - t_peak, 1e-3)

            grid = self._build_canonical_grid(
                t_peak=t_peak if t_peak > 0 else 1e-3,
                t_half=t_half,
                device=device,
                dtype=dtype,
            )[:canonical_cap].clamp(max=1.0)

            actual_grid = grid * max_time
            distances = torch.cdist(actual_grid[:, None], time_row[:, None])
            nearest = distances.argmin(dim=1)

            usable = min(time_row.numel(), grid.numel())
            if usable == 0:
                continue

            canonical_vals[row, :usable] = obs_row[nearest[:usable]]
            canonical_times[row, :usable] = time_row[nearest[:usable]]
            canonical_mask[row, :usable] = True

        canonical_mask = self._drop_non_positive_times_from_mask(canonical_times, canonical_mask)

        return canonical_vals, canonical_times, canonical_mask

    def _generate_raw(
        self, full_simulation: Tensor, full_simulation_times: Tensor, **kwargs
    ) -> Tuple[
        Tensor, Tensor, Tensor, Optional[Tensor], Optional[Tensor], Optional[Tensor], Tensor
    ]:
        time_scales: Optional[Tensor] = kwargs.get("time_scales")
        if time_scales is None:
            raise ValueError("time_scales must be provided for PKPeakHalfLifeStrategy")

        canonical_vals, canonical_times, canonical_mask, rescaled = (
            self._align_simulation_to_canonical(
                full_simulation,
                full_simulation_times,
                time_scales=time_scales,
                num_obs_sampler=kwargs.get("num_obs_sampler"),
            )
        )

        obs_out, obs_time, obs_mask, rem_sim, rem_time, rem_mask = self._assemble_from_canonical(
            canonical_vals,
            canonical_times,
            canonical_mask,
            generator=kwargs.get("generator"),
        )

        return obs_out, obs_time, obs_mask, rem_sim, rem_time, rem_mask, rescaled

    def _generate_random(
        self,
        full_simulation: Tensor,
        full_simulation_times: Tensor,
        *,
        time_scales: Tensor,
        generator: Optional[torch.Generator] = None,
    ) -> Tuple[
        Tensor, Tensor, Tensor, Optional[Tensor], Optional[Tensor], Optional[Tensor], Tensor
    ]:
        """Randomized variant of canonical observation generation.

        Instead of fixed multipliers, the pre- and post-peak segments are
        sampled from uniform distributions bounded by the canonical limits.
        This keeps the semantic meaning of the selected points while injecting
        stochasticity that improves robustness when training amortised
        inference models.
        """
        device, dtype = full_simulation.device, full_simulation.dtype
        batch = full_simulation.size(0)
        time_steps = int(full_simulation_times.size(1))
        if batch == 0 or time_steps == 0:
            canonical_vals = torch.zeros(batch, 0, dtype=dtype, device=device)
            canonical_times = torch.zeros(batch, 0, dtype=dtype, device=device)
            canonical_mask = torch.zeros(batch, 0, dtype=torch.bool, device=device)
            obs_out, obs_time, obs_mask, rem_sim, rem_time, rem_mask = self._assemble_from_canonical(
                canonical_vals, canonical_times, canonical_mask, generator=generator
            )
            return obs_out, obs_time, obs_mask, rem_sim, rem_time, rem_mask, time_scales.clone()
        t_peak, t_half = time_scales[0].item(), time_scales[1].item()

        n_pre = len(self._PEAK_PHASE_MULTIPLIERS)
        n_post = len(self._POST_PEAK_HALF_LIFE_MULTIPLIERS)

        # Uniform samples before peak
        pre_times = torch.rand(n_pre, device=device, dtype=dtype) * t_peak
        # Always include the peak
        peak_time = torch.tensor([t_peak], device=device, dtype=dtype)
        # Uniform samples after peak
        post_times = []
        for mult in self._POST_PEAK_HALF_LIFE_MULTIPLIERS:
            t_end = t_peak + mult * t_half
            t_rand = torch.empty(1, device=device, dtype=dtype).uniform_(t_peak, t_end)
            post_times.append(t_rand)
        post_times = torch.cat(post_times, dim=0)

        # Truncate to canonical capacity
        grid = torch.cat([pre_times, peak_time, post_times], dim=0)
        canonical_cap = self._canonical_grid_capacity()
        grid = grid[:canonical_cap]

        # Map grid to nearest simulation points
        ref_times = full_simulation_times[0]
        idx = torch.cdist(grid[:, None], ref_times[:, None]).argmin(dim=1)
        idx, _ = idx.sort()
        valid_mask = torch.ones_like(idx, dtype=torch.bool)
        idx, valid_mask = self._deduplicate_sorted_indices(idx, valid_mask)
        gather_idx = idx[None, :].expand(batch, -1)
        batch_idx = torch.arange(batch, device=device)[:, None]

        canonical_vals = full_simulation[batch_idx, gather_idx]
        canonical_times = full_simulation_times[batch_idx, gather_idx]
        invalid_slots = ~valid_mask
        if invalid_slots.any():
            canonical_vals[:, invalid_slots] = 0
            canonical_times[:, invalid_slots] = 0

        canonical_mask = valid_mask[None, :].expand(batch, -1).clone()
        canonical_mask = self._drop_non_positive_times_from_mask(canonical_times, canonical_mask)

        obs_out, obs_time, obs_mask, rem_sim, rem_time, rem_mask = self._assemble_from_canonical(
            canonical_vals, canonical_times, canonical_mask, generator=generator
        )
        return obs_out, obs_time, obs_mask, rem_sim, rem_time, rem_mask, time_scales.clone()

    def generate(
        self,
        full_simulation: Tensor,
        full_simulation_times: Tensor,
        **kwargs,
    ) -> Tuple[
        Tensor, Tensor, Tensor, Optional[Tensor], Optional[Tensor], Optional[Tensor], Tensor
    ]:
        """Generate PK observations using canonical or randomized schedules.

        With probability ``randomize_prob`` (default 0.5) the method delegates
        to :meth:`_generate_random`; otherwise the deterministic
        :meth:`_generate_raw` path is taken.  Setting the keyword argument
        ``deterministic_only=True`` forces the deterministic branch regardless
        of the random draw.  Both paths require the caller to provide
        ``time_scales`` specifying the peak and half-life.  The method honours
        the ``add_rem`` flag by optionally returning remainder tensors.
        """
        time_scales: Optional[Tensor] = kwargs.get("time_scales")
        if time_scales is None:
            raise ValueError("time_scales must be provided for PKPeakHalfLifeStrategy")

        deterministic_only = kwargs.pop("deterministic_only", False)

        use_random = False
        if not deterministic_only:
            use_random = torch.rand(()) < getattr(self, "randomize_prob", 0.5)

        if use_random:
            obs, obs_time, obs_mask, rem_sim, rem_time, rem_mask, rescaled = self._generate_random(
                full_simulation,
                full_simulation_times,
                time_scales=time_scales,
                generator=kwargs.get("generator"),
            )
        else:
            obs, obs_time, obs_mask, rem_sim, rem_time, rem_mask, rescaled = self._generate_raw(
                full_simulation,
                full_simulation_times,
                **kwargs,
            )

        if not self.observations_config.add_rem:
            rem_sim = rem_time = rem_mask = None

        return obs, obs_time, obs_mask, rem_sim, rem_time, rem_mask, rescaled

    def generate_empirical(
        self,
        empirical_obs: Tensor,
        empirical_times: Tensor,
        empirical_mask: Tensor,
        *,
        generator: Optional[torch.Generator] = None,
    ) -> Tuple[Tensor, Tensor, Tensor, Optional[Tensor], Optional[Tensor], Optional[Tensor]]:
        canonical_vals, canonical_times, canonical_mask = self._align_empirical_to_canonical(
            empirical_obs,
            empirical_times,
            empirical_mask,
        )

        obs, obs_time, obs_mask, rem_sim, rem_time, rem_mask = self._assemble_from_canonical(
            canonical_vals,
            canonical_times,
            canonical_mask,
            generator=generator,
        )

        if not self.observations_config.add_rem:
            rem_sim = rem_time = rem_mask = None

        return obs, obs_time, obs_mask, rem_sim, rem_time, rem_mask


class FixedRegularGridStrategy(ObservationStrategy):
    """Observation strategy using one deterministic subset of a regular solver grid.

    This strategy is intended for synthetic sample-experiment serialization
    where every generated time series must share the exact same observation
    schedule. The schedule is defined as an evenly spaced subset of the
    simulator grid:

    * ``k = min(max_num_obs, time_num_steps)``
    * if ``k == 0`` there are no observations
    * if ``k == 1`` the first simulator time point after the configured offset
      is selected
    * if ``k > 1`` the selected indices include both the first and last
      simulator indices in the truncated grid and are otherwise distributed as
      evenly as possible across the available solver grid

    The optional ``fixed_grid_start_index`` configuration trims the solver
    grid before the evenly spaced subset is built. This allows callers to skip
    the exact ``t=0`` point while still keeping a deterministic shared grid,
    e.g. by starting from solver step 2 or 3 instead of 0.

    The strategy never produces a remainder block. All selected points are
    returned in ``observations`` / ``observation_times`` only.
    """

    def __init__(self, observations_config: ObservationsConfig, meta_config: MetaStudyConfig) -> None:
        super().__init__(observations_config, meta_config)
        self.max_num_obs = int(observations_config.max_num_obs)
        self.fixed_grid_start_index = int(getattr(observations_config, "fixed_grid_start_index", 0))

    def _fixed_grid_capacity(self) -> int:
        """Return the fixed number of grid points exposed by the strategy."""

        time_steps = int(getattr(self.meta_config, "time_num_steps", self.max_num_obs))
        available_steps = max(0, time_steps - self.fixed_grid_start_index)
        return max(0, min(self.max_num_obs, available_steps))

    @staticmethod
    def _select_evenly_spaced_indices(
        time_steps: int,
        selected_points: int,
        *,
        start_index: int,
        device: torch.device,
    ) -> Tensor:
        """Return deterministic evenly spaced solver indices.

        The returned tensor has shape ``[k]`` and is guaranteed to include the
        first valid index after ``start_index`` and the last valid solver index
        when ``k > 1``.
        """

        if selected_points <= 0 or time_steps <= 0:
            return torch.zeros(0, dtype=torch.long, device=device)
        first_valid_index = max(0, int(start_index))
        if first_valid_index >= time_steps:
            return torch.zeros(0, dtype=torch.long, device=device)
        if selected_points == 1:
            return torch.full((1,), first_valid_index, dtype=torch.long, device=device)

        raw_positions = torch.linspace(
            first_valid_index,
            time_steps - 1,
            steps=selected_points,
            device=device,
            dtype=torch.float32,
        )
        indices = torch.round(raw_positions).to(torch.long)
        indices[0] = first_valid_index
        indices[-1] = time_steps - 1
        return indices

    def _get_shapes_raw(self) -> Tuple[int, int]:
        """Return fixed observation capacity and zero remainder capacity."""

        return self._fixed_grid_capacity(), 0

    def _generate_raw(
        self, full_simulation: Tensor, full_simulation_times: Tensor, **kwargs
    ) -> Tuple[
        Tensor,
        TensorType["B", "T_obs"],
        TensorType["B", "T_obs"],
        TensorType["B", "T_rem"],
        TensorType["B", "T_rem"],
        TensorType["B", "T_rem"],
    ]:
        """Gather one deterministic fixed grid directly from the solver outputs."""

        del kwargs

        device = full_simulation.device
        dtype = full_simulation.dtype
        batch_size = int(full_simulation.shape[0])
        max_obs, _ = self._get_shapes_raw()
        time_steps = int(full_simulation_times.shape[1]) if full_simulation_times.ndim >= 2 else 0
        selected_points = max(0, min(max_obs, time_steps))

        obs = torch.zeros(batch_size, max_obs, dtype=dtype, device=device)
        obs_time = torch.zeros(batch_size, max_obs, dtype=full_simulation_times.dtype, device=device)
        obs_mask = torch.zeros(batch_size, max_obs, dtype=torch.bool, device=device)

        if max_obs == 0 or batch_size == 0 or selected_points == 0:
            return obs, obs_time, obs_mask, None, None, None

        gather_idx = self._select_evenly_spaced_indices(
            time_steps,
            selected_points,
            start_index=self.fixed_grid_start_index,
            device=device,
        )  # [T_obs]
        batch_idx = torch.arange(batch_size, device=device)[:, None]  # [B, 1]

        # obs / obs_time: [B, T_obs]
        obs[:, :selected_points] = full_simulation[batch_idx, gather_idx]
        obs_time[:, :selected_points] = full_simulation_times[batch_idx, gather_idx]
        obs_mask[:, :selected_points] = True
        obs_mask = self._drop_non_positive_times_from_mask(obs_time, obs_mask)
        return obs, obs_time, obs_mask, None, None, None

    def generate_empirical(
        self,
        empirical_obs: Tensor,
        empirical_times: Tensor,
        empirical_mask: Tensor,
        *,
        generator: Optional[torch.Generator] = None,
    ) -> Tuple[Tensor, Tensor, Tensor, Optional[Tensor], Optional[Tensor], Optional[Tensor]]:
        """Copy empirical observations into the fixed-capacity observation block."""

        del generator

        device = empirical_obs.device
        dtype = empirical_obs.dtype
        batch_size = int(empirical_obs.shape[0])
        max_obs, _ = self._get_shapes_raw()

        obs = torch.zeros(batch_size, max_obs, dtype=dtype, device=device)
        obs_time = torch.zeros(batch_size, max_obs, dtype=empirical_times.dtype, device=device)
        obs_mask = torch.zeros(batch_size, max_obs, dtype=torch.bool, device=device)

        for row in range(batch_size):
            valid_idx = empirical_mask[row].nonzero(as_tuple=True)[0]
            if valid_idx.numel() == 0:
                continue

            start_offset = min(self.fixed_grid_start_index, int(valid_idx.numel()))
            shifted_valid_idx = valid_idx[start_offset:]
            take_count = min(int(shifted_valid_idx.numel()), max_obs)
            take_idx = shifted_valid_idx[:take_count]
            obs[row, :take_count] = empirical_obs[row, take_idx]
            obs_time[row, :take_count] = empirical_times[row, take_idx]
            obs_mask[row, :take_count] = True

        obs_mask = self._drop_non_positive_times_from_mask(obs_time, obs_mask)
        return obs, obs_time, obs_mask, None, None, None


class FixPastTimeRandomSelectionStrategy(ObservationStrategy):
    """Randomly sample observations and split with fixed-capacity past/future slots.

    For ``split_past_future=True`` this strategy enforces the contract:
    ``obs_capacity=max_past`` and ``rem_capacity=max_num_obs-max_past``
    (subject to ``fixed_M_max=min(max_num_obs, time_num_steps)``).
    """

    def __init__(self, config: ObservationsConfig, meta_config: MetaStudyConfig):
        super().__init__(config, meta_config)
        time_steps = getattr(meta_config, "time_num_steps", config.max_num_obs)
        self.fixed_M_max = min(config.max_num_obs, time_steps)
        self.split_past_future = config.split_past_future
        self.max_past = config.max_past
        self.min_past = config.min_past
        self.generative_bias = config.generative_bias
        self.boundary_ratio = getattr(config, "past_time_ratio", 0.1)

    def _generate_raw(self, full_simulation: Tensor, full_simulation_times: Tensor, **kwargs):
        return fix_past_time_random_selection(
            full_simulation=full_simulation,
            full_simulation_times=full_simulation_times,
            boundary_ratio=self.boundary_ratio,
            fixed_M_max=self.fixed_M_max,
            num_obs_sampler=kwargs.get("num_obs_sampler", None),
            generator=kwargs.get("generator", None),
        )

    def _get_shapes_raw(self) -> Tuple[int, int]:
        """Return fixed-capacity shapes for random split outputs.

        With ``split_past_future=True``:
        - ``max_obs`` is bounded by ``max_past``
        - ``max_rem`` is bounded by ``max_num_obs - max_past``
        """
        if self.split_past_future:
            if self.min_past is None or self.max_past is None:
                raise ValueError(
                    "min_past and max_past must be specified when split_past_future=True"
                )
            if self.fixed_M_max < self.min_past:
                raise ValueError("fixed_M_max is smaller than the configured min_past")
            max_obs = min(self.max_past, self.fixed_M_max)
            max_rem = max(0, self.fixed_M_max - self.max_past)
        else:
            max_obs = self.fixed_M_max
            max_rem = self.fixed_M_max

        return max_obs, max_rem

    def _split_by_boundary(
        self,
        obs: TensorType["B", "M"],
        obs_time: TensorType["B", "M"],
        obs_mask: TensorType["B", "M"],
        *,
        generator: Optional[torch.Generator] = None,
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
        """Split sampled observations into strict past and future blocks.

        The split is boundary-based and strict:
        - Past block samples ``k`` points from ``time <= boundary`` candidates,
          where ``k`` follows ``min_past``/``max_past`` (and ``generative_bias``),
          capped by available candidates and ``K_max``.
        - When ``k > 0``, remainder receives up to ``R_cap`` points sampled
          from ``time > boundary`` only (strict future).
        - When ``k == 0``, boundary splitting is ignored for remainder and
          points are sampled from all valid candidates.

        Extra past/future candidates are ignored, and missing entries are
        padded by zeros with mask=False.
        """
        B, M = obs.shape
        # K_max: capacity of the past block [B, K_max]
        K_max = min(int(self.max_past), int(M))
        K_min = min(int(self.min_past), K_max)
        # R_cap: fixed capacity of the remainder block [B, R_cap]
        R_cap = max(0, int(M) - K_max)

        boundary = self.meta_config.time_stop * self.boundary_ratio
        gen = generator if generator is not None else torch.default_generator

        past_obs = torch.zeros(B, K_max, dtype=obs.dtype, device=obs.device)
        past_time = torch.zeros_like(past_obs)
        past_mask = torch.zeros(B, K_max, dtype=torch.bool, device=obs.device)

        rem_obs = torch.zeros(B, R_cap, dtype=obs.dtype, device=obs.device)
        rem_time = torch.zeros_like(rem_obs)
        rem_mask = torch.zeros(B, R_cap, dtype=torch.bool, device=obs.device)

        for b in range(B):
            valid_idx = obs_mask[b].nonzero(as_tuple=True)[0]
            past_candidates = valid_idx[obs_time[b, valid_idx] <= boundary]
            future_candidates = valid_idx[obs_time[b, valid_idx] > boundary]

            if past_candidates.numel() > 1:
                order = torch.argsort(obs_time[b, past_candidates])
                past_candidates = past_candidates[order]
            if future_candidates.numel() > 1:
                order = torch.argsort(obs_time[b, future_candidates])
                future_candidates = future_candidates[order]

            # Past is sampled uniformly without replacement from pre-boundary points.
            k_high = min(K_max, int(past_candidates.numel()))
            k_low = min(K_min, k_high)
            k = _sample_past_count_with_bias(
                low=int(k_low),
                high=int(k_high),
                generative_bias=self.generative_bias,
                generator=gen,
                device=obs.device,
            )
            if k > 0 and past_candidates.numel() > 0:
                chosen_offsets = torch.randperm(
                    past_candidates.numel(),
                    generator=gen,
                    device=obs.device,
                )[:k]
                chosen_past = past_candidates[chosen_offsets]
                chosen_order = torch.argsort(obs_time[b, chosen_past])
                chosen_past = chosen_past[chosen_order]
            else:
                chosen_past = past_candidates[:0]

            num_past = chosen_past.numel()
            if num_past > 0:
                past_obs[b, :num_past] = obs[b, chosen_past]
                past_time[b, :num_past] = obs_time[b, chosen_past]
                past_mask[b, :num_past] = True

            # If no past point is selected, allow remainder sampling across the
            # whole valid domain. Otherwise keep strict future-only remainder.
            rem_pool = valid_idx if num_past == 0 else future_candidates
            if rem_pool.numel() > 1:
                order = torch.argsort(obs_time[b, rem_pool])
                rem_pool = rem_pool[order]

            if R_cap <= 0 or rem_pool.numel() == 0:
                chosen_rem = rem_pool[:0]
            elif rem_pool.numel() <= R_cap:
                chosen_rem = rem_pool
            else:
                chosen_offsets = torch.randperm(
                    rem_pool.numel(),
                    generator=gen,
                    device=obs.device,
                )[:R_cap]
                chosen_rem = rem_pool[chosen_offsets]
                chosen_order = torch.argsort(obs_time[b, chosen_rem])
                chosen_rem = chosen_rem[chosen_order]

            r = chosen_rem.numel()
            if r > 0:
                rem_obs[b, :r] = obs[b, chosen_rem]
                rem_time[b, :r] = obs_time[b, chosen_rem]
                rem_mask[b, :r] = True

        return past_obs, past_time, past_mask, rem_obs, rem_time, rem_mask

    def generate(
        self, full_simulation: Tensor, full_simulation_times: Tensor, **kwargs
    ) -> Tuple[Tensor, ...]:
        obs, obs_time, obs_mask, _, _, _ = self._generate_raw(
            full_simulation, full_simulation_times, **kwargs
        )
        obs_mask = self._drop_non_positive_times_from_mask(obs_time, obs_mask)

        if self.split_past_future:
            out = self._split_by_boundary(
                obs,
                obs_time,
                obs_mask,
                generator=kwargs.get("generator", None),
            )
        else:
            past_obs, past_time, past_mask = obs, obs_time, obs_mask
            rem_obs = rem_time = rem_mask = None
            out = (past_obs, past_time, past_mask, rem_obs, rem_time, rem_mask)

        if not self.observations_config.add_rem:
            out = out[:3] + (None, None, None)

        return (*out, None)


class ObservationStrategyFactory:
    @staticmethod
    def from_config(
        obs_config: ObservationsConfig, meta_config: MetaStudyConfig
    ) -> ObservationStrategy:
        # Legacy compatibility:
        # - omitted ``type`` defaults via dataclass to ``pk_peak_half_life``
        # - explicit YAML ``type: null`` is loaded as ``None`` and also falls
        #   back to ``pk_peak_half_life``
        strategy_type = getattr(obs_config, "type", None)
        if strategy_type is None:
            normalized_type = "pk_peak_half_life"
        elif isinstance(strategy_type, str):
            stripped = strategy_type.strip()
            if stripped == "" or stripped.lower() in {"null", "none"}:
                normalized_type = "pk_peak_half_life"
            else:
                normalized_type = stripped.lower()
        else:
            normalized_type = str(strategy_type).strip().lower()

        if normalized_type in {
            "observations_pk_peak_halflife",
            "pk_peak_half_life",
        }:
            return PKPeakHalfLifeStrategy(obs_config, meta_config)
        if normalized_type in {
            "fixed_regular_grid",
        }:
            return FixedRegularGridStrategy(obs_config, meta_config)
        if normalized_type in {
            "fix_past_time_random_selection",
            "random",
        }:
            return FixPastTimeRandomSelectionStrategy(obs_config, meta_config)
        raise ValueError(f"Unknown observation type: {strategy_type}")
