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
from .candidate_retrieval import (
    RetrievalMetrics, RetrievalBenchmark, evaluate_retrieval, brute_force_top_k,
)
from .baseline_competition import (
    BaselineCompetition, CompetitionReport, PolicyResult,
    select_by_scores, learned_policy_from_scores,
)
from .observability import (
    MetricsSink, Counter, Gauge, Histogram, read_jsonl,
)
from .qualification import (
    SafetyCheckStatus, SafetyCheckResult, SafetyQualificationReport,
    SafetyGateError, run_safety_qualification, assert_safety_gate,
)
from .scientific_qualification import (
    ScientificMetric, ScientificQualificationReport,
    ScientificGateError, assert_scientific_gate,
)
from .performance_qualification import (
    ScaleTier, TIER_NODE_COUNTS, MeasurementStatus, TierMeasurement,
    PerformanceQualificationReport, measure_tier, run_performance_qualification,
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
    "RetrievalMetrics",
    "RetrievalBenchmark",
    "evaluate_retrieval",
    "brute_force_top_k",
    "BaselineCompetition",
    "CompetitionReport",
    "PolicyResult",
    "select_by_scores",
    "learned_policy_from_scores",
    "MetricsSink",
    "Counter",
    "Gauge",
    "Histogram",
    "read_jsonl",
    "SafetyCheckStatus",
    "SafetyCheckResult",
    "SafetyQualificationReport",
    "SafetyGateError",
    "run_safety_qualification",
    "assert_safety_gate",
    "ScientificMetric",
    "ScientificQualificationReport",
    "ScientificGateError",
    "assert_scientific_gate",
    "ScaleTier",
    "TIER_NODE_COUNTS",
    "MeasurementStatus",
    "TierMeasurement",
    "PerformanceQualificationReport",
    "measure_tier",
    "run_performance_qualification",
]
