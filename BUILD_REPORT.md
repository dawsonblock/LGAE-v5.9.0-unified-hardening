# LGAE v5.9.0 Build Report

Release: **Unified Hardening Merge**

## Base

LGAE v5.8.4 remains the canonical trunk. Its adaptive geometry, cache coherence, concurrent snapshot protection, evidence/memory/reasoning layers, structural-intelligence stack, scalable candidate retrieval, and v5.8.4 joint topology/gauge runtime are retained.

## Restored from v5.3.3

- Domain-separated deterministic RNG streams (`DeterministicRNGContext`, `derive_seed`, `deterministic_mode`).
- Reproducibility metadata and deterministic qualification IDs.
- Canonical benchmark action ordering independent of `PYTHONHASHSEED`.
- `GraphFeatureBaseline` and graph feature extraction, exposed as an explicit credit-baseline option while retaining the v5.8 hash baseline as the compatibility default.
- Hierarchical ADD_EDGE candidate retrieval (top-K + latent KNN) with bounded candidate count.
- Dynamic-gauge generator-norm clamping and optional spectral normalization.
- Residual-aware latent equilibrium barrier.
- Bayesian Normal-Inverse-Gamma curvature hysteresis with predictive intervals/effective sample size.
- Named `ProductionConfig` / `ResearchConfig` profiles.
- Governor audit `CertificationLevel` metadata.
- `MutationAuthorityLevel` classification.
- Tensor-native topology signatures and Tarjan bridge detection, while preserving the newer bounded exact edge-connectivity prune floor.
- Checkpoint per-file SHA-256 commitments and Merkle root.
- Optional Ed25519-signed mutation receipts and signature verification.
- Bounded structural MPC and the permutation-equivariant executive reference implementation.
- Counterfactual Q-learning benchmark tooling and information-gain Task G.

## Compatibility decisions

- Information-gain Task G is available in `ALL_TASKS`, but is excluded from the legacy structural-diagnosis policy qualification gate so v5.9.0 does not silently redefine the historical release metric.
- `GraphHashBaseline` remains the default `MutationCreditTracker` estimator. `GraphFeatureBaseline` is selectable explicitly.
- The v5.8.4 gauge-override shadow rollout, cached Ollivier neighborhoods, and exact bounded edge-connectivity prune floor remain intact.

## Verification

- Full combined regression suite: **719 passed**.
- Reproducibility suite: **23/23 passed** under each `PYTHONHASHSEED` value: `0`, `1`, `2`, `42`, `123456`.
- Python source compile check: PASS.
- Source release is clean: generated `build/`, `dist/`, stale egg-info, and pytest caches are excluded.
- Release manifest is regenerated over the final source tree and independently reverified before packaging.

## Claim boundary

v5.9.0 is an engineering hardening/integration release. It does not change the existing scientific conclusion that learned structural policy superiority over strong reference heuristics on unseen topology families is not yet established. The restored mechanisms improve determinism, provenance, safety semantics, and research reliability; they do not by themselves prove improved OOD intelligence.
