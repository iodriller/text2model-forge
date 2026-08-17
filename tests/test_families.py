import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from text2model_forge.sprites.families import inspect_canonical_master, inspect_family_contract


def _write(path, value):
    Path(path).write_text(json.dumps(value), encoding="utf-8")


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _contract():
    return {
        "schema_version": 1,
        "stage": "family_contract",
        "id": "test_family",
        "body_plan": "biped",
        "topology_graph": {
            "points": ["hips", "head"],
            "edges": [["hips", "head"]],
            "radii": {"hips": 0.3, "head": 0.25},
        },
        "skeleton": [{"name": "spine", "head": "hips", "tail": "head"}],
        "required_actions": ["idle"],
    }


class FamilyContractTests(unittest.TestCase):
    def test_valid_contract(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "family.json"
            _write(path, _contract())
            self.assertTrue(inspect_family_contract(path)["passed"])

    def test_unknown_graph_point_fails(self):
        with tempfile.TemporaryDirectory() as folder:
            value = _contract()
            value["topology_graph"]["edges"] = [["hips", "missing"]]
            path = Path(folder) / "family.json"
            _write(path, value)
            report = inspect_family_contract(path)
            self.assertFalse(report["passed"])

    def test_master_requires_weights_uv_motion_and_hash(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            contract_path = root / "family.json"
            master_path = root / "master.blend"
            report_path = root / "report.json"
            _write(contract_path, _contract())
            master_path.write_bytes(b"master")
            _write(report_path, {
                "stage": "canonical_master",
                "production_eligible": False,
                "family": "test_family",
                "contract_sha256": _sha(contract_path),
                "master_sha256": _sha(master_path),
                "mesh": {"vertices": 200, "connected_components": 1, "boundary_edges": 0, "non_manifold_edges": 0, "has_uv": True},
                "weights": {"unweighted_vertices": 0, "max_sum_error": 0.0, "max_influences": 4},
                "missing_bones": [],
                "missing_actions": [],
                "motion_audit": {"passed": True},
            })
            self.assertTrue(inspect_canonical_master(contract_path, master_path, report_path)["passed"])
            value = json.loads(report_path.read_text())
            value["weights"]["unweighted_vertices"] = 3
            _write(report_path, value)
            failed = inspect_canonical_master(contract_path, master_path, report_path)
            self.assertFalse(failed["passed"])
            self.assertFalse(failed["canonical_stage_eligible"])


if __name__ == "__main__":
    unittest.main()
