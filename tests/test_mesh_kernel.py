from __future__ import annotations

from pathlib import Path

from text2model_forge.mesh import TriangleMesh


FIXTURES = Path(__file__).resolve().parent / "fixtures" / "mesh"


def test_seeded_defect_fixture_exposes_expected_failures() -> None:
    mesh = TriangleMesh.from_obj((FIXTURES / "defective_tetrahedron.obj").read_text(encoding="utf-8"))
    health = mesh.health()

    assert health.duplicate_vertices == 1
    assert health.isolated_vertices == 0
    assert health.degenerate_faces == 1
    assert health.connected_components == 2
    assert health.boundary_edges == 3


def test_deterministic_repair_produces_closed_single_component_mesh() -> None:
    mesh = TriangleMesh.from_obj((FIXTURES / "defective_tetrahedron.obj").read_text(encoding="utf-8"))
    repaired = mesh.deterministic_repair()
    health = repaired.health()

    assert health.vertices == 4
    assert health.faces == 4
    assert health.duplicate_vertices == 0
    assert health.isolated_vertices == 0
    assert health.degenerate_faces == 0
    assert health.connected_components == 1
    assert health.boundary_edges == 0
    assert health.non_manifold_edges == 0
    assert health.inconsistent_winding_edges == 0
    assert TriangleMesh.from_obj(repaired.to_obj()).health() == health


def test_clean_fixture_is_stable_under_repair() -> None:
    source = TriangleMesh.from_obj((FIXTURES / "clean_tetrahedron.obj").read_text(encoding="utf-8"))
    repaired = source.deterministic_repair()
    assert repaired.to_obj() == source.to_obj()


def test_five_defect_classes_are_diagnosed_and_improved() -> None:
    mesh = TriangleMesh.from_obj((FIXTURES / "five_defects.obj").read_text(encoding="utf-8"))
    before = mesh.health()

    assert {
        "duplicate_vertices",
        "isolated_vertices",
        "degenerate_faces",
        "disconnected_components",
        "open_boundaries",
        "inconsistent_winding",
    }.issubset(before.diagnoses)

    repaired, decision = mesh.guarded_repair()
    after = repaired.health()

    assert decision.accepted
    assert after.diagnoses == []
    assert after.connected_components == 1
    assert after.boundary_edges == 0


def test_guard_rejects_a_destructive_repair_branch() -> None:
    source = TriangleMesh.from_obj((FIXTURES / "clean_tetrahedron.obj").read_text(encoding="utf-8"))
    selected, decision = source.guarded_repair(minimum_component_faces=5)

    assert not decision.accepted
    assert "repair removed all usable geometry" in decision.reasons
    assert selected.to_obj() == source.to_obj()


def test_bounded_hole_fill_closes_a_real_triangular_surface_hole() -> None:
    mesh = TriangleMesh.from_obj((FIXTURES / "open_tetrahedron.obj").read_text(encoding="utf-8"))
    assert mesh.health().boundary_edges == 3

    repaired, decision = mesh.guarded_repair()

    assert decision.accepted
    assert repaired.health().boundary_edges == 0
    assert repaired.health().faces == 4
