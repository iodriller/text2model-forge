"""Small deterministic OBJ diagnostics/repair kernel independent of Blender."""
from __future__ import annotations

import math
from dataclasses import dataclass

from pydantic import Field

from .schemas import StrictModel


Vec3 = tuple[float, float, float]
Face = tuple[int, int, int]


class MeshFormatError(ValueError):
    pass


class MeshHealth(StrictModel):
    vertices: int = Field(ge=0)
    faces: int = Field(ge=0)
    duplicate_vertices: int = Field(ge=0)
    isolated_vertices: int = Field(ge=0)
    degenerate_faces: int = Field(ge=0)
    connected_components: int = Field(ge=0)
    boundary_edges: int = Field(ge=0)
    non_manifold_edges: int = Field(ge=0)
    inconsistent_winding_edges: int = Field(ge=0)
    finite_coordinates: bool

    @property
    def hard_failures(self) -> list[str]:
        failures: list[str] = []
        if not self.finite_coordinates:
            failures.append("non_finite_coordinates")
        if self.degenerate_faces:
            failures.append("degenerate_faces")
        if self.non_manifold_edges:
            failures.append("non_manifold_edges")
        if self.inconsistent_winding_edges:
            failures.append("inconsistent_winding")
        return failures

    @property
    def diagnoses(self) -> list[str]:
        diagnoses = list(self.hard_failures)
        if self.duplicate_vertices:
            diagnoses.append("duplicate_vertices")
        if self.isolated_vertices:
            diagnoses.append("isolated_vertices")
        if self.connected_components > 1:
            diagnoses.append("disconnected_components")
        if self.boundary_edges:
            diagnoses.append("open_boundaries")
        return diagnoses


class MeshRepairDecision(StrictModel):
    accepted: bool
    reasons: list[str] = Field(default_factory=list)
    before: MeshHealth
    candidate: MeshHealth


@dataclass(frozen=True)
class TriangleMesh:
    vertices: list[Vec3]
    faces: list[Face]

    @classmethod
    def from_obj(cls, text: str) -> "TriangleMesh":
        vertices: list[Vec3] = []
        faces: list[Face] = []
        for line_number, raw in enumerate(text.splitlines(), start=1):
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if parts[0] == "v":
                if len(parts) < 4:
                    raise MeshFormatError(f"line {line_number}: vertex requires three coordinates")
                vertices.append((float(parts[1]), float(parts[2]), float(parts[3])))
            elif parts[0] == "f":
                if len(parts) != 4:
                    raise MeshFormatError(f"line {line_number}: only triangle faces are supported")
                indices: list[int] = []
                for token in parts[1:]:
                    index = int(token.split("/", 1)[0])
                    if index <= 0:
                        raise MeshFormatError(f"line {line_number}: only positive OBJ indices are supported")
                    zero_based = index - 1
                    if zero_based >= len(vertices):
                        raise MeshFormatError(f"line {line_number}: face index is out of range")
                    indices.append(zero_based)
                faces.append(tuple(indices))  # type: ignore[arg-type]
        if not vertices:
            raise MeshFormatError("OBJ contains no vertices")
        return cls(vertices=vertices, faces=faces)

    def to_obj(self) -> str:
        lines = ["# Asset Forge Darkness deterministic OBJ"]
        lines.extend(f"v {x:.9g} {y:.9g} {z:.9g}" for x, y, z in self.vertices)
        lines.extend(f"f {a + 1} {b + 1} {c + 1}" for a, b, c in self.faces)
        return "\n".join(lines) + "\n"

    @staticmethod
    def _area2(a: Vec3, b: Vec3, c: Vec3) -> float:
        ab = (b[0] - a[0], b[1] - a[1], b[2] - a[2])
        ac = (c[0] - a[0], c[1] - a[1], c[2] - a[2])
        cross = (
            ab[1] * ac[2] - ab[2] * ac[1],
            ab[2] * ac[0] - ab[0] * ac[2],
            ab[0] * ac[1] - ab[1] * ac[0],
        )
        return sum(item * item for item in cross)

    def degenerate_face_indices(self, epsilon: float = 1e-18) -> list[int]:
        result = []
        for index, face in enumerate(self.faces):
            if len(set(face)) < 3 or self._area2(*(self.vertices[item] for item in face)) <= epsilon:
                result.append(index)
        return result

    def duplicate_vertex_map(self, tolerance: float = 1e-9) -> dict[int, int]:
        if tolerance <= 0:
            raise ValueError("tolerance must be positive")
        seen: dict[tuple[int, int, int], int] = {}
        duplicates: dict[int, int] = {}
        for index, vertex in enumerate(self.vertices):
            key = tuple(round(value / tolerance) for value in vertex)
            if key in seen:
                duplicates[index] = seen[key]
            else:
                seen[key] = index
        return duplicates

    def edge_counts(self) -> dict[tuple[int, int], int]:
        counts: dict[tuple[int, int], int] = {}
        degenerate = set(self.degenerate_face_indices())
        for face_index, (a, b, c) in enumerate(self.faces):
            if face_index in degenerate:
                continue
            for edge in ((a, b), (b, c), (c, a)):
                key = tuple(sorted(edge))
                counts[key] = counts.get(key, 0) + 1
        return counts

    def edge_occurrences(self) -> dict[tuple[int, int], list[tuple[int, int, int]]]:
        occurrences: dict[tuple[int, int], list[tuple[int, int, int]]] = {}
        degenerate = set(self.degenerate_face_indices())
        for face_index, (a, b, c) in enumerate(self.faces):
            if face_index in degenerate:
                continue
            for start, end in ((a, b), (b, c), (c, a)):
                occurrences.setdefault(tuple(sorted((start, end))), []).append((face_index, start, end))
        return occurrences

    def inconsistent_winding_edges(self) -> int:
        inconsistent = 0
        for occurrences in self.edge_occurrences().values():
            if len(occurrences) != 2:
                continue
            _, first_start, first_end = occurrences[0]
            _, second_start, second_end = occurrences[1]
            if first_start == second_start and first_end == second_end:
                inconsistent += 1
        return inconsistent

    def face_components(self) -> list[list[int]]:
        vertex_faces: dict[int, list[int]] = {}
        for face_index, face in enumerate(self.faces):
            for vertex in face:
                vertex_faces.setdefault(vertex, []).append(face_index)
        remaining = set(range(len(self.faces)))
        components: list[list[int]] = []
        while remaining:
            seed = min(remaining)
            stack = [seed]
            component: list[int] = []
            remaining.remove(seed)
            while stack:
                current = stack.pop()
                component.append(current)
                for vertex in self.faces[current]:
                    for neighbor in vertex_faces[vertex]:
                        if neighbor in remaining:
                            remaining.remove(neighbor)
                            stack.append(neighbor)
            components.append(sorted(component))
        return components

    def health(self, tolerance: float = 1e-9) -> MeshHealth:
        edges = self.edge_counts()
        finite = all(math.isfinite(value) for vertex in self.vertices for value in vertex)
        used_vertices = {vertex for face in self.faces for vertex in face}
        return MeshHealth(
            vertices=len(self.vertices),
            faces=len(self.faces),
            duplicate_vertices=len(self.duplicate_vertex_map(tolerance)),
            isolated_vertices=len(self.vertices) - len(used_vertices),
            degenerate_faces=len(self.degenerate_face_indices()),
            connected_components=len(self.face_components()),
            boundary_edges=sum(count == 1 for count in edges.values()),
            non_manifold_edges=sum(count > 2 for count in edges.values()),
            inconsistent_winding_edges=self.inconsistent_winding_edges(),
            finite_coordinates=finite,
        )

    def weld_duplicates(self, tolerance: float = 1e-9) -> "TriangleMesh":
        duplicates = self.duplicate_vertex_map(tolerance)
        canonical: dict[int, int] = {}
        new_vertices: list[Vec3] = []
        for old_index, vertex in enumerate(self.vertices):
            root = duplicates.get(old_index, old_index)
            if root == old_index:
                canonical[old_index] = len(new_vertices)
                new_vertices.append(vertex)
            else:
                canonical[old_index] = canonical[root]
        new_faces = [tuple(canonical[index] for index in face) for face in self.faces]
        return TriangleMesh(new_vertices, new_faces)  # type: ignore[arg-type]

    def remove_degenerate_faces(self) -> "TriangleMesh":
        rejected = set(self.degenerate_face_indices())
        return TriangleMesh(self.vertices, [face for index, face in enumerate(self.faces) if index not in rejected])

    def remove_small_components(self, minimum_faces: int = 2) -> "TriangleMesh":
        if minimum_faces < 1:
            raise ValueError("minimum_faces must be positive")
        keep = {face for component in self.face_components() if len(component) >= minimum_faces for face in component}
        faces = [face for index, face in enumerate(self.faces) if index in keep]
        used = sorted({vertex for face in faces for vertex in face})
        remap = {old: new for new, old in enumerate(used)}
        return TriangleMesh(
            [self.vertices[index] for index in used],
            [tuple(remap[index] for index in face) for face in faces],  # type: ignore[list-item]
        )

    def fill_triangular_holes(self, minimum_component_faces: int = 3) -> "TriangleMesh":
        """Close only unambiguous three-edge holes on established surface components."""
        if minimum_component_faces < 1:
            raise ValueError("minimum_component_faces must be positive")
        face_component_sizes: dict[int, int] = {}
        for component in self.face_components():
            for face_index in component:
                face_component_sizes[face_index] = len(component)

        boundary_occurrences = {
            edge: occurrences[0]
            for edge, occurrences in self.edge_occurrences().items()
            if len(occurrences) == 1
        }
        adjacency: dict[int, set[int]] = {}
        for a, b in boundary_occurrences:
            adjacency.setdefault(a, set()).add(b)
            adjacency.setdefault(b, set()).add(a)

        unseen = set(boundary_occurrences)
        holes: list[tuple[int, int, int]] = []
        while unseen:
            seed = min(unseen)
            stack = [seed[0], seed[1]]
            vertices: set[int] = set()
            edges: set[tuple[int, int]] = set()
            while stack:
                current = stack.pop()
                if current in vertices:
                    continue
                vertices.add(current)
                for neighbor in adjacency.get(current, set()):
                    edge = tuple(sorted((current, neighbor)))
                    if edge in unseen:
                        unseen.remove(edge)
                        edges.add(edge)
                        stack.append(neighbor)
            if len(vertices) != 3 or len(edges) != 3:
                continue
            adjacent_faces = {boundary_occurrences[edge][0] for edge in edges}
            if not adjacent_faces or min(face_component_sizes[index] for index in adjacent_faces) < minimum_component_faces:
                continue
            a, b, c = sorted(vertices)
            first = TriangleMesh(self.vertices, [*self.faces, (a, b, c)])
            second = TriangleMesh(self.vertices, [*self.faces, (a, c, b)])
            chosen = min(
                (first, second),
                key=lambda item: (item.inconsistent_winding_edges(), item.to_obj()),
            )
            holes.append(chosen.faces[-1])
        return TriangleMesh(self.vertices, [*self.faces, *holes])

    def remove_isolated_vertices(self) -> "TriangleMesh":
        used = sorted({vertex for face in self.faces for vertex in face})
        remap = {old: new for new, old in enumerate(used)}
        return TriangleMesh(
            [self.vertices[index] for index in used],
            [tuple(remap[index] for index in face) for face in self.faces],  # type: ignore[list-item]
        )

    def orient_faces_consistently(self) -> "TriangleMesh":
        """Make adjacent manifold triangles use opposite directions on their shared edge."""
        neighbors: dict[int, list[tuple[int, bool]]] = {index: [] for index in range(len(self.faces))}
        for occurrences in self.edge_occurrences().values():
            if len(occurrences) != 2:
                continue
            first_face, first_start, first_end = occurrences[0]
            second_face, second_start, second_end = occurrences[1]
            same_direction = first_start == second_start and first_end == second_end
            neighbors[first_face].append((second_face, same_direction))
            neighbors[second_face].append((first_face, same_direction))

        flips: dict[int, bool] = {}
        for seed in range(len(self.faces)):
            if seed in flips:
                continue
            flips[seed] = False
            stack = [seed]
            while stack:
                current = stack.pop()
                for neighbor, parity_change in neighbors[current]:
                    expected = flips[current] ^ parity_change
                    if neighbor not in flips:
                        flips[neighbor] = expected
                        stack.append(neighbor)

        faces: list[Face] = []
        for index, (a, b, c) in enumerate(self.faces):
            faces.append((a, c, b) if flips.get(index, False) else (a, b, c))
        return TriangleMesh(self.vertices, faces)

    def deterministic_repair(self, *, tolerance: float = 1e-9, minimum_component_faces: int = 2) -> "TriangleMesh":
        return (
            self.weld_duplicates(tolerance)
            .remove_degenerate_faces()
            .fill_triangular_holes()
            .remove_small_components(minimum_component_faces)
            .remove_isolated_vertices()
            .orient_faces_consistently()
        )

    def guarded_repair(
        self,
        *,
        tolerance: float = 1e-9,
        minimum_component_faces: int = 2,
    ) -> tuple["TriangleMesh", MeshRepairDecision]:
        before = self.health(tolerance)
        candidate_mesh = self.deterministic_repair(
            tolerance=tolerance,
            minimum_component_faces=minimum_component_faces,
        )
        candidate = candidate_mesh.health(tolerance)
        reasons: list[str] = []
        if candidate.faces == 0 or candidate.vertices == 0:
            reasons.append("repair removed all usable geometry")
        before_failures = set(before.hard_failures)
        new_failures = sorted(set(candidate.hard_failures) - before_failures)
        if new_failures:
            reasons.append("repair introduced hard failures: " + ", ".join(new_failures))
        if candidate.boundary_edges > before.boundary_edges:
            reasons.append("repair increased open boundary edges")
        if candidate.boundary_edges:
            reasons.append("repair leaves unresolved open boundary edges")
        if candidate.hard_failures:
            reasons.append("repair leaves hard failures: " + ", ".join(candidate.hard_failures))
        before_defects = _defect_score(before)
        candidate_defects = _defect_score(candidate)
        if candidate_defects >= before_defects:
            reasons.append("repair did not reduce the measured defect score")
        decision = MeshRepairDecision(
            accepted=not reasons,
            reasons=reasons,
            before=before,
            candidate=candidate,
        )
        return (candidate_mesh if decision.accepted else self), decision


def _defect_score(health: MeshHealth) -> int:
    return (
        (0 if health.finite_coordinates else 1000000)
        + health.non_manifold_edges * 10000
        + health.degenerate_faces * 1000
        + health.inconsistent_winding_edges * 100
        + health.boundary_edges * 10
        + max(0, health.connected_components - 1) * 10
        + health.duplicate_vertices
        + health.isolated_vertices
    )
