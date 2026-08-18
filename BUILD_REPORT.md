# LGAE v5.11.0 Build Report

Release: **Canonical Runtime Convergence**

## Summary

LGAE v5.11.0 is a runtime-convergence release, not a feature release. The
objective is transactional correctness, crash-safety, determinism,
self-verification, and scientific honesty.

**Test suite: 1465 passed, 0 failed**

## Defects repaired (19 total)

### Transactional foundation (Phases 1-8)

- **D11-001**: Direct engine mutation bypasses CommitChannel → Fixed: engine is private (`self._engine`), `rt.engine` returns read-only `EngineFacade`
- **D11-002**: CommitChannel only logically exclusive → Fixed: capability-gated mutation primitives (`_AuthorityCapability` token)
- **D11-003**: Graph/fiber/gauge apply is non-atomic → Fixed: exception-atomic commit with rollback to pre-state
- **D11-004**: WAL records graph but not complete transaction state → Fixed: WAL serializes graph + fiber + gauge deltas
- **D11-005**: WAL COMMIT occurs after live mutation → Fixed: COMMIT written BEFORE APPLY (COMMIT-before-APPLY ordering), ABORT invalidates on rollback
- **D11-006**: WAL counters reset on reopen → Fixed: `_restore_counters()` scans existing records
- **D11-007**: Crash tests do not kill inside transaction stages → Fixed: subprocess SIGKILL crash matrix at 4 stages (before BEGIN, after BEGIN, after WRITE, after COMMIT)
- **D11-008**: FiberDelta fallback uses Python `hash()` → Fixed: raises `DeterminismError`, `FiberStateSnapshot.state_hash()` added
- **D11-009**: Authorization binding is optional/incomplete → Fixed: mandatory, non-nullable, `transaction_hash` binding
- **D11-010**: Fiber/gauge evaluation still mutate-and-restore → Fixed: shadow-only evaluation, restore before evaluation

### Learning integrity (Sprint 3)

- **D11-011**: `learn()` uses predicted delta as realized reward → Fixed: `realized_delta = U_after - U_before`
- **D11-012**: Calibration compares delta prediction against absolute utility → Fixed: `calibrator.update(predicted, realized_delta)`
- **D11-013**: Hierarchical credit not connected → Fixed: 6-field credit assignment (diagnostic/candidate/planner/action/governance/outcome)

### Qualification (Sprint 4)

- **D11-014**: Performance `MEASURED` can mean nothing executed → Fixed: NOT_RUN/INVALID/MEASURED/PASS/FAIL with thresholds
- **D11-018**: Final qualification asserts symbols instead of invariants → Fixed: behavioral invariant tests
- **D11-019**: Four "real graph" benchmarks remain synthetic surrogates → Verified: `is_real_data` flag, synthetic descriptions

### Release integrity (Sprint 5)

- **D11-015**: Hypothesis missing from dev dependencies → Fixed: added to `pyproject.toml`
- **D11-016**: Release manifest still stale → Updated
- **D11-017**: BUILD_REPORT remains v5.9 / 719 tests → Updated to v5.11.0 / 1458 tests

## Architecture

```
LGAERuntime
    │
    ├── immutable public APIs (rt.engine → EngineFacade)
    │
    └── _engine (private)
           │
           ├── _authority_capability (mutation token)
           │
           ├── graph
           ├── fibers
           ├── gauges
           ├── calibration
           ├── model
           ├── state_version
           └── state_hash
```

## Runtime invariant

```
S_{t+1} = Commit(S_t, T_t, A_t)
```

- `S_t`: immutable authoritative state
- `T_t`: deterministic structural transaction
- `A_t`: authorization bound to that exact transaction

## Crash invariant

```
S_restart ∈ {S_t, S_{t+1}}
```

Never: `S_restart = S_t + partial(T_t)`

## Deterministic replay invariant

```
F(S_t, O_t, C, M, R) = S_{t+1}
```

for identical state, observation, configuration, models, and deterministic randomness.

## Governing principle

> Learned models propose. Deterministic governance authorizes. Evidence proves.

## Test breakdown

- Unit tests: ~1300
- Integration tests: ~165
- Total: 1465 passed, 0 failed

## Dependencies

- Python >= 3.10
- PyTorch >= 2.0
- Dev: pytest, pytest-cov, hypothesis
