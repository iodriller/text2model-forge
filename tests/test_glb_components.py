from __future__ import annotations

from pathlib import Path

import trimesh

from text2model_forge.glb_components import build_component_audit
from text2model_forge.hashing import sha256_file


def test_component_audit_builds_review_branches_without_mutating_source(tmp_path: Path) -> None:
    main = trimesh.creation.icosphere(subdivisions=2, radius=1)
    detail = trimesh.creation.icosphere(subdivisions=1, radius=0.1)
    detail.apply_translation((2, 0, 0))
    speck = trimesh.creation.box(extents=(0.01, 0.01, 0.01))
    speck.apply_translation((3, 0, 0))
    source = tmp_path / "source.glb"
    source.write_bytes(trimesh.exchange.gltf.export_glb(trimesh.Scene([main, detail, speck])))
    source_hash = sha256_file(source)

    result = build_component_audit(
        source,
        tmp_path / "audit",
        minimum_structural_faces=50,
        minimum_structural_fraction=0.01,
    )

    assert sha256_file(source) == source_hash
    assert result["audit"]["connected_components"] == 3
    assert result["audit"]["structural_component_indices"] == [0, 1]
    assert result["candidates"]["structural"]["components"] == 2
    assert result["candidates"]["primary"]["components"] == 1
    assert (tmp_path / "audit" / "report.md").is_file()
    assert (tmp_path / "audit" / "structural_candidate.glb").is_file()
