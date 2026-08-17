"""Typed configuration for the canonical v5.10 runtime.

This is a thin orchestration-level config. It does not duplicate the
subsystem-level ``LGAEConfig``; it composes around it. Subsystem behavior
remains governed by ``LGAEConfig`` and its ``ProductionConfig`` /
``ResearchConfig`` presets.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Callable

import torch
from torch import Tensor

from ..types import GraphBuffers


class RuntimeMode(str, Enum):
    """Research vs production execution mode (Phase 44 distinction)."""
    RESEARCH = "research"
    PRODUCTION = "production"


@dataclass(slots=True)
class RuntimeConfig:
    """Canonical runtime configuration.

    Production mode fails closed: signed receipts, strict authority, and
    deterministic ordering are mandatory and cannot be silently relaxed.
    """
    mode: RuntimeMode = RuntimeMode.RESEARCH
    # Evidence / receipt persistence. When None, in-memory ledgers are used.
    evidence_path: str | Path | None = None
    receipt_path: str | Path | None = None
    signing_key: str | None = None
    require_signed_receipts: bool = False
    # Optional structural MPC planning (Phase 14). When horizon > 1 the
    # runtime plans before committing, but only ever executes the first
    # action of the chosen plan (receding horizon).
    mpc_horizon: int = 1
    mpc_max_branching: int = 8
    mpc_max_sequences: int = 64
    # Structural-learning loop knobs (delegated to StructuralLearningLoop).
    ensemble_size: int = 5
    max_candidates: int = 5
    # Optional external utility function used for MPC planning and credit.
    utility_fn: Callable[[GraphBuffers, Tensor], float] | None = None
    # Deterministic ordering guard: never rely on set/dict iteration order.
    deterministic_ordering: bool = True
    # Maximum stale-read retries before raising (Phase 3 seqlock enforcement).
    max_stale_read_retries: int = 4

    def __post_init__(self) -> None:
        if self.mode == RuntimeMode.PRODUCTION:
            # Production fails closed: receipts must be signed and persisted.
            if self.require_signed_receipts and self.signing_key is None:
                raise ValueError("production mode with require_signed_receipts needs a signing_key")
        if int(self.mpc_horizon) < 1:
            raise ValueError("mpc_horizon must be >= 1")
        if int(self.max_stale_read_retries) < 0:
            raise ValueError("max_stale_read_retries must be >= 0")

    @property
    def is_production(self) -> bool:
        return self.mode == RuntimeMode.PRODUCTION

    def to_summary(self) -> dict[str, Any]:
        return {
            "mode": self.mode.value,
            "mpc_horizon": int(self.mpc_horizon),
            "ensemble_size": int(self.ensemble_size),
            "max_candidates": int(self.max_candidates),
            "require_signed_receipts": bool(self.require_signed_receipts),
            "deterministic_ordering": bool(self.deterministic_ordering),
            "evidence_path": None if self.evidence_path is None else str(self.evidence_path),
            "receipt_path": None if self.receipt_path is None else str(self.receipt_path),
        }
