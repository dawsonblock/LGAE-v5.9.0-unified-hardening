"""v5.10 canonical runtime package.

One authoritative end-to-end governed cycle. The runtime orchestrates
existing engines (LGAEEngine, StructuralLearningLoop, StructuralReasoningLoop,
StructuralMPC, EvidenceLedger, receipts) and does not re-implement any
algorithm. Learned models propose; deterministic governance authorizes;
evidence proves.
"""
from __future__ import annotations

from .runtime_config import RuntimeConfig, RuntimeMode
from .runtime_state import RuntimeSnapshot
from .runtime_events import RuntimePhase, RuntimeEvent
from .runtime_result import RuntimeStepResult
from .authority import (
    AuthorityRole, AuthorityBoundary, AuthoritativeStateGuard,
    CommitChannel, DEFAULT_BOUNDARIES,
)
from .cache_coherence import MutationImpact, CacheRegistry, depends_on, declared_dependencies
from .canonical_runtime import LGAERuntime, UnauthorizedMutationError

__all__ = [
    "RuntimeConfig",
    "RuntimeMode",
    "RuntimeSnapshot",
    "RuntimePhase",
    "RuntimeEvent",
    "RuntimeStepResult",
    "LGAERuntime",
    "UnauthorizedMutationError",
    "AuthorityRole",
    "AuthorityBoundary",
    "AuthoritativeStateGuard",
    "CommitChannel",
    "DEFAULT_BOUNDARIES",
    "MutationImpact",
    "CacheRegistry",
    "depends_on",
    "declared_dependencies",
]
