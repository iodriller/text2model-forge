import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from assetforge.core import ForgeError
from assetforge.genesis import inspect_anatomy_target, inspect_genesis_spec, validate_genesis_spec


def _spec():
    return {
        "schema_version": 1,
        "stage": "genesis_spec",
        "id": "test_creature",
        "body_plan": "biped",
        "family_template": "short_biped_v1",
        "landmarks": {"hips": [0, 0, 1], "head": [0, 0, 2]},
        "volumes": [
            {"id": "torso", "type": "capsule", "region": "torso", "start": "hips", "end": "head", "radius": 0.3},
            {"id": "skull", "type": "ellipsoid", "region": "head", "center": "head", "radii": [0.3, 0.3, 0.35]},
        ],
        "anatomy_target": {"voxel_size": 0.03},
    }


def _write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2), encoding="utf-8")


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class GenesisTests(unittest.TestCase):
    def test_valid_spec_passes_and_is_hash_bound(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "creature.genesis.json"
            _write_json(path, _spec())
            report = validate_genesis_spec(path)
            self.assertTrue(report["passed"])
            self.assertEqual(_sha(path), report["spec_sha256"])

    def test_missing_family_template_blocks_spec(self):
        with tempfile.TemporaryDirectory() as folder:
            payload = _spec()
            payload.pop("family_template")
            path = Path(folder) / "creature.genesis.json"
            _write_json(path, payload)
            report = inspect_genesis_spec(path)
            self.assertFalse(report["passed"])
            self.assertTrue(any("family_template" in failure for failure in report["failures"]))

    def test_unknown_landmark_blocks_volume(self):
        with tempfile.TemporaryDirectory() as folder:
            payload = _spec()
            payload["volumes"][0]["end"] = "missing"
            path = Path(folder) / "creature.genesis.json"
            _write_json(path, payload)
            with self.assertRaises(ForgeError):
                validate_genesis_spec(path)

    def test_target_must_be_watertight_and_not_production_eligible(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            spec_path = root / "creature.genesis.json"
            target_path = root / "target.blend"
            report_path = root / "target.report.json"
            _write_json(spec_path, _spec())
            target_path.write_bytes(b"fixture")
            _write_json(report_path, {
                "stage": "anatomy_target",
                "production_eligible": True,
                "next_required_stage": "canonical_topology_fit",
                "spec_sha256": _sha(spec_path),
                "target_sha256": _sha(target_path),
                "mesh": {"connected_components": 2, "boundary_edges": 4, "non_manifold_edges": 4, "vertices": 200},
            })
            report = inspect_anatomy_target(spec_path, target_path, report_path)
            self.assertFalse(report["passed"])
            self.assertFalse(report["production_eligible"])
            self.assertTrue(any("production_eligible" in failure for failure in report["failures"]))
            self.assertTrue(any("watertight" in failure for failure in report["failures"]))

    def test_target_report_and_hash_pass(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            spec_path = root / "creature.genesis.json"
            target_path = root / "target.blend"
            report_path = root / "target.report.json"
            _write_json(spec_path, _spec())
            target_path.write_bytes(b"fixture")
            _write_json(report_path, {
                "stage": "anatomy_target",
                "production_eligible": False,
                "next_required_stage": "canonical_topology_fit",
                "spec_sha256": _sha(spec_path),
                "target_sha256": _sha(target_path),
                "mesh": {"connected_components": 1, "boundary_edges": 0, "non_manifold_edges": 0, "vertices": 200},
            })
            report = inspect_anatomy_target(spec_path, target_path, report_path)
            self.assertTrue(report["passed"], report["failures"])

    def test_wing_hull_and_rigid_box_detail_are_valid_schema(self):
        with tempfile.TemporaryDirectory() as folder:
            payload = _spec()
            payload["landmarks"]["wing_tip"] = [1.0, 0.0, 1.5]
            payload["volumes"].append({
                "id": "wing", "type": "hull", "region": "wing",
                "points": ["hips", "head", "wing_tip"], "thickness": 0.08,
            })
            payload["details"] = [{
                "id": "armor", "type": "box", "center": "hips",
                "scale": [0.2, 0.05, 0.3], "rotation_degrees": [0, 20, 0],
                "bone": "spine", "material": "leather",
            }]
            path = Path(folder) / "creature.genesis.json"
            _write_json(path, payload)
            report = inspect_genesis_spec(path)
            self.assertTrue(report["passed"], report["failures"])
            self.assertEqual(1, report["detail_count"])


if __name__ == "__main__":
    unittest.main()
