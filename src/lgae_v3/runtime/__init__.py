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
from .adaptive_diagnostics import (
    DiagnosticLevel, DiagnosticEscalationPolicy, DiagnosticResult,
    DiagnosticCascade,
)
from .certification import (
    CertificationLevel, CertificationResult, CertificationError,
    minimum_level_for, meets_requirement,
)
from .candidates import (
    Candidate, CandidateUnion, candidate_id, build_candidate_union,
)
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
    "DiagnosticLevel",
    "DiagnosticEscalationPolicy",
    "DiagnosticResult",
    "DiagnosticCascade",
    "CertificationLevel",
    "CertificationResult",
    "CertificationError",
    "minimum_level_for",
    "meets_requirement",
    "Candidate",
    "CandidateUnion",
    "candidate_id",
    "build_candidate_union",
]
