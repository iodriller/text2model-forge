"""Small deterministic OBJ polygon parser and topology analyzer for retopology evidence."""
from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
import math


class PolygonObjError(ValueError):
    pass


@dataclass(frozen=True)
class PolygonObj:
    vertices: tuple[tuple[float, float, float], ...]
    faces: tuple[tuple[int, ...], ...]

    @classmethod
    def parse(cls, text: str) -> "PolygonObj":
        vertices: list[tuple[float, float, float]] = []
        faces: list[tuple[int, ...]] = []
        for line_number, raw_line in enumerate(text.splitlines(), start=1):
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            fields = line.split()
            if fields[0] == "v":
                if len(fields) < 4:
                    raise PolygonObjError(f"line {line_number}: vertex requires three coordinates")
                try:
                    vertex = tuple(float(value) for value in fields[1:4])
                except ValueError as exc:
                    raise PolygonObjError(f"line {line_number}: invalid vertex coordinate") from exc
                vertices.append(vertex)
            elif fields[0] == "f":
                if len(fields) < 4:
                    raise PolygonObjError(f"line {line_number}: face requires at least three vertices")
                face: list[int] = []
                for token in fields[1:]:
                    raw_index = token.split("/", 1)[0]
                    try:
                        obj_index = int(raw_index)
                    except ValueError as exc:
                        raise PolygonObjError(f"line {line_number}: invalid face index") from exc
                    if obj_index == 0:
                        raise PolygonObjError(f"line {line_number}: OBJ indices are one-based")
                    index = obj_index - 1 if obj_index > 0 else len(vertices) + obj_index
                    if index < 0 or index >= len(vertices):
                        raise PolygonObjError(f"line {line_number}: face index is out of range")
                    face.append(index)
                faces.append(tuple(face))
        if not vertices:
            raise PolygonObjError("OBJ contains no vertices")
        if not faces:
            raise PolygonObjError("OBJ contains no faces")
        return cls(vertices=tuple(vertices), faces=tuple(faces))

    def analyze(self) -> dict[str, int | float | bool | list[int]]:
        edge_counts: Counter[tuple[int, int]] = Counter()
        edge_faces: dict[tuple[int, int], list[int]] = defaultdict(list)
        used_vertices: set[int] = set()
        degenerate_faces = 0
        for face_index, face in enumerate(self.faces):
            used_vertices.update(face)
            if len(set(face)) != len(face) or _face_area(self.vertices, face) <= 1e-12:
                degenerate_faces += 1
            for offset, first in enumerate(face):
                second = face[(offset + 1) % len(face)]
                edge = (first, second) if first < second else (second, first)
                edge_counts[edge] += 1
                edge_faces[edge].append(face_index)

        adjacency: list[set[int]] = [set() for _ in self.faces]
        for linked_faces in edge_faces.values():
            for first in linked_faces:
                adjacency[first].update(second for second in linked_faces if second != first)
        remaining = set(range(len(self.faces)))
        component_sizes: list[int] = []
        while remaining:
            seed = min(remaining)
            remaining.remove(seed)
            stack = [seed]
            size = 0
            while stack:
                current = stack.pop()
                size += 1
                neighbors = adjacency[current] & remaining
                remaining.difference_update(neighbors)
                stack.extend(sorted(neighbors, reverse=True))
            component_sizes.append(size)
        component_sizes.sort(reverse=True)

        quads = sum(len(face) == 4 for face in self.faces)
        triangles = sum(len(face) == 3 for face in self.faces)
        ngons = sum(len(face) > 4 for face in self.faces)
        finite = all(math.isfinite(value) for vertex in self.vertices for value in vertex)
        return {
            "vertices": len(self.vertices),
            "faces": len(self.faces),
            "quads": quads,
            "triangles": triangles,
            "ngons": ngons,
            "non_quad_faces": len(self.faces) - quads,
            "quad_fraction": quads / len(self.faces),
            "connected_components": len(component_sizes),
            "component_faces": component_sizes,
            "isolated_vertices": len(self.vertices) - len(used_vertices),
            "degenerate_faces": degenerate_faces,
            "boundary_edges": sum(count == 1 for count in edge_counts.values()),
            "non_manifold_edges": sum(count > 2 for count in edge_counts.values()),
            "finite_coordinates": finite,
        }


def _face_area(vertices: tuple[tuple[float, float, float], ...], face: tuple[int, ...]) -> float:
    normal_x = normal_y = normal_z = 0.0
    for offset, index in enumerate(face):
        current = vertices[index]
        following = vertices[face[(offset + 1) % len(face)]]
        normal_x += (current[1] - following[1]) * (current[2] + following[2])
        normal_y += (current[2] - following[2]) * (current[0] + following[0])
        normal_z += (current[0] - following[0]) * (current[1] + following[1])
    return 0.5 * math.sqrt(normal_x * normal_x + normal_y * normal_y + normal_z * normal_z)
