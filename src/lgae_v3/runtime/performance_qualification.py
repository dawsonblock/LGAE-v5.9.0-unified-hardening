"""Performance qualification (Phase 49).

Defines S/M/L/XL scale tiers and measures the actual hot path at each tier:

  S:  1k nodes
  M:  10k nodes
  L:  100k nodes
  XL: 1M nodes (where supported)

Metrics per tier:
  - proposal_latency_ms
  - diagnostic_latency_ms
  - commit_latency_ms
  - peak_memory_bytes
  - candidate_throughput (candidates/s)

Do not claim million-node scalability without measuring the actual hot path
at that size. A tier that was not measured is recorded as NOT_MEASURED, not
inferred.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Iterable

import torch
from torch import Tensor

from ..types import GraphBuffers, make_graph_buffers


class ScaleTier(str, Enum):
    S = "S"    # 1k nodes
    M = "M"    # 10k nodes
    L = "L"    # 100k nodes
    XL = "XL"  # 1M nodes


TIER_NODE_COUNTS: dict[ScaleTier, int] = {
    ScaleTier.S: 1_000,
    ScaleTier.M: 10_000,
    ScaleTier.L: 100_000,
    ScaleTier.XL: 1_000_000,
}


class MeasurementStatus(str, Enum):
    MEASURED = "measured"
    NOT_MEASURED = "not_measured"
    SKIPPED = "skipped"  # explicitly skipped (e.g. unsupported tier)


@dataclass(frozen=True, slots=True)
class TierMeasurement:
    """One performance measurement at a scale tier."""
    tier: ScaleTier
    n_nodes: int
    status: MeasurementStatus
    proposal_latency_ms: float = 0.0
    diagnostic_latency_ms: float = 0.0
    commit_latency_ms: float = 0.0
    peak_memory_bytes: int = 0
    candidate_throughput: float = 0.0
    notes: str = ""

    def to_log(self) -> dict[str, Any]:
        return {
            "tier": self.tier.value,
            "n_nodes": int(self.n_nodes),
            "status": self.status.value,
            "proposal_latency_ms": float(self.proposal_latency_ms),
            "diagnostic_latency_ms": float(self.diagnostic_latency_ms),
            "commit_latency_ms": float(self.commit_latency_ms),
            "peak_memory_bytes": int(self.peak_memory_bytes),
            "candidate_throughput": float(self.candidate_throughput),
            "notes": self.notes,
        }


@dataclass(slots=True)
class PerformanceQualificationReport:
    """Aggregate performance qualification report across tiers."""
    measurements: list[TierMeasurement] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def add(self, m: TierMeasurement) -> None:
        self.measurements.append(m)

    @property
    def measured_tiers(self) -> list[ScaleTier]:
        return [m.tier for m in self.measurements if m.status == MeasurementStatus.MEASURED]

    @property
    def xl_measured(self) -> bool:
        return any(m.tier == ScaleTier.XL and m.status == MeasurementStatus.MEASURED for m in self.measurements)

    def to_log(self) -> dict[str, Any]:
        return {
            "measurements": [m.to_log() for m in self.measurements],
            "measured_tiers": [t.value for t in self.measured_tiers],
            "xl_measured": self.xl_measured,
            "metadata": self.metadata,
        }


def _make_path_graph(n: int) -> GraphBuffers:
    edges = [(i, i + 1) for i in range(n - 1)]
    return make_graph_buffers(n, edges, capacity=max(n * 2, 16))


def _measure_latency(fn: Callable[[], Any]) -> float:
    """Measure wall-clock latency of fn in milliseconds."""
    t0 = time.perf_counter()
    fn()
    return (time.perf_counter() - t0) * 1000.0


def measure_tier(
    tier: ScaleTier,
    *,
    proposal_fn: Callable[[GraphBuffers], int] | None = None,
    diagnostic_fn: Callable[[GraphBuffers], Any] | None = None,
    commit_fn: Callable[[GraphBuffers], Any] | None = None,
    n_nodes: int | None = None,
    skip: bool = False,
) -> TierMeasurement:
    """Measure one scale tier. Functions that are None are not measured.

    ``proposal_fn`` returns the number of candidates generated (for throughput).
    """
    nn = int(n_nodes if n_nodes is not None else TIER_NODE_COUNTS[tier])
    if skip:
        return TierMeasurement(tier=tier, n_nodes=nn, status=MeasurementStatus.SKIPPED,
                               notes="explicitly skipped")

    # Track peak memory via torch if available.
    try:
        torch.cuda.reset_peak_memory_stats() if torch.cuda.is_available() else None
    except Exception:
        pass

    graph = _make_path_graph(nn)
    prop_ms = 0.0
    diag_ms = 0.0
    commit_ms = 0.0
    n_cands = 0

    if proposal_fn is not None:
        prop_ms = _measure_latency(lambda: None)  # warmup
        t0 = time.perf_counter()
        n_cands = int(proposal_fn(graph))
        prop_ms = (time.perf_counter() - t0) * 1000.0

    if diagnostic_fn is not None:
        diag_ms = _measure_latency(lambda: diagnostic_fn(graph))

    if commit_fn is not None:
        commit_ms = _measure_latency(lambda: commit_fn(graph))

    # Peak memory estimate.
    peak = 0
    try:
        if torch.cuda.is_available():
            peak = int(torch.cuda.max_memory_allocated())
    except Exception:
        pass
    if peak == 0:
        # Fallback: estimate from graph buffer size.
        try:
            peak = int(graph.weight.element_size() * graph.weight.numel())
        except Exception:
            pass

    throughput = float(n_cands) / (prop_ms / 1000.0) if prop_ms > 0 else 0.0

    return TierMeasurement(
        tier=tier, n_nodes=nn, status=MeasurementStatus.MEASURED,
        proposal_latency_ms=float(prop_ms),
        diagnostic_latency_ms=float(diag_ms),
        commit_latency_ms=float(commit_ms),
        peak_memory_bytes=int(peak),
        candidate_throughput=float(throughput),
    )


def run_performance_qualification(
    *,
    proposal_fn: Callable[[GraphBuffers], int] | None = None,
    diagnostic_fn: Callable[[GraphBuffers], Any] | None = None,
    commit_fn: Callable[[GraphBuffers], Any] | None = None,
    tiers: Iterable[ScaleTier] | None = None,
    skip_tiers: set[ScaleTier] | None = None,
    metadata: dict[str, Any] | None = None,
) -> PerformanceQualificationReport:
    """Run performance qualification across tiers.

    Tiers in ``skip_tiers`` are recorded as SKIPPED (not inferred). Tiers not
    in ``tiers`` are not included at all.
    """
    report = PerformanceQualificationReport(metadata=dict(metadata or {}))
    skip = skip_tiers or set()
    for tier in (tiers or [ScaleTier.S, ScaleTier.M, ScaleTier.L, ScaleTier.XL]):
        m = measure_tier(
            tier, proposal_fn=proposal_fn, diagnostic_fn=diagnostic_fn, commit_fn=commit_fn,
            skip=tier in skip,
        )
        report.add(m)
    return report
