"""v5.11 Phase 0: regression test documenting the manifest mismatch defect.

MANIFEST.sha256.json still references v5.9.0 schema and does not cover
new v5.10 files.

This test PASSES against v5.10, proving the defect exists.
After Phase 22, this test should be replaced with one that verifies
the manifest is complete and matches the current version.
"""
from __future__ import annotations

import json
import pathlib


def test_manifest_version_mismatch():
    """The manifest still references v5.9.0, not v5.10.0."""
    manifest_path = pathlib.Path("MANIFEST.sha256.json")
    if not manifest_path.exists():
        return  # no manifest to test
    manifest = json.loads(manifest_path.read_text())
    # DEFECT: manifest says 5.9.0 but code is 5.10.0.
    assert manifest.get("version") == "5.9.0", (
        f"Expected manifest version 5.9.0 (the defect), got {manifest.get('version')}. "
        "This test should FAIL after Phase 22 rebuilds the manifest."
    )


def test_manifest_missing_v510_files():
    """The manifest does not cover new v5.10 runtime files."""
    manifest_path = pathlib.Path("MANIFEST.sha256.json")
    if not manifest_path.exists():
        return
    manifest = json.loads(manifest_path.read_text())
    manifest_files = {f["path"] for f in manifest.get("files", [])}

    # New v5.10 files that should be in the manifest but aren't.
    new_files = [
        "src/lgae_v3/runtime/graph_ops.py",
        "src/lgae_v3/runtime/sparse_graph.py",
        "src/lgae_v3/runtime/gpu_path.py",
        "src/lgae_v3/runtime/batched_counterfactuals.py",
    ]
    missing = [f for f in new_files if f not in manifest_files]
    # DEFECT: new files are missing from the manifest.
    assert len(missing) > 0, (
        "Expected new v5.10 files to be missing from manifest (the defect). "
        "This test should FAIL after Phase 22 rebuilds the manifest."
    )
