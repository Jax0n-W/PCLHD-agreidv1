from __future__ import absolute_import

import csv
import os.path as osp
import sys
import unittest
from pathlib import Path


ROOT = osp.dirname(osp.dirname(osp.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from agreid_identity import (
    AGREID_TEST_IDENTITIES, AGREID_TRAIN_IDENTITIES,
    compose_agreid_pid, parse_agreid_name, validate_identity_count)
from tools.generate_evaluation_protocols import parse_pid
from view_domain.common import parse_image_name


class AGReIDIdentityTests(unittest.TestCase):
    def test_official_filename_uses_p_t_a(self):
        parsed = parse_agreid_name("P0000T02140A0C0F121.jpg")
        self.assertEqual(parsed.pid, int("0000" + "02140" + "0"))
        self.assertEqual(parsed.camid, 0)
        self.assertEqual(parsed.source, "official")

    def test_rewritten_filename_recovers_a_from_encoded_camera(self):
        parsed = parse_agreid_name("P0034_T02210_C10_F19471.jpg")
        self.assertEqual(parsed.pid, int("0034" + "02210" + "1"))
        self.assertEqual(parsed.camid, 10)
        self.assertEqual(parsed.source, "rewritten")

    def test_official_and_rewritten_names_match(self):
        official = parse_agreid_name("P0001T04041A0C0F121.jpg")
        rewritten = parse_agreid_name("P0001_T04041_C00_F121.jpg")
        self.assertEqual(official.pid, rewritten.pid)

    def test_same_p_can_represent_different_identities(self):
        first = parse_agreid_name("P0001T04041A0C0F121.jpg").pid
        second = parse_agreid_name("P0001T09999A0C0F121.jpg").pid
        self.assertNotEqual(first, second)

    def test_invalid_name_fails_instead_of_falling_back_to_p(self):
        with self.assertRaises(ValueError):
            parse_agreid_name("P0001_unknown.jpg")
        with self.assertRaises(ValueError):
            parse_agreid_name("P0001_T04041_C0_F121.jpg")

    def test_protocol_generator_uses_official_parser(self):
        root = Path("/dataset/query_all_c0")
        path = root / "P0002T04041A0C0F151.jpg"
        self.assertEqual(
            parse_pid(path, root, "agreid"),
            compose_agreid_pid("0002", "04041", "0"))

    def test_view_manifest_uses_official_identity(self):
        pid, camera = parse_image_name(
            Path("P0001_T04041_C00_F121.jpg"))
        self.assertEqual(pid, str(compose_agreid_pid("0001", "04041", "0")))
        self.assertEqual(camera, "c00")

    def test_identity_count_guard(self):
        samples = [("a.jpg", i, 0) for i in range(AGREID_TRAIN_IDENTITIES)]
        self.assertEqual(validate_identity_count(
            samples, AGREID_TRAIN_IDENTITIES, "synthetic"),
            AGREID_TRAIN_IDENTITIES)
        with self.assertRaises(RuntimeError):
            validate_identity_count(samples[:-1], AGREID_TRAIN_IDENTITIES,
                                    "synthetic")

    def test_released_score_paths_recover_199_training_ids(self):
        score_file = osp.join(
            ROOT, "view_scores", "agreid_aerial_view_scores_foreground.csv")
        p_only = set()
        official = set()
        rows = 0
        with open(score_file, "r", encoding="utf-8-sig", newline="") as f:
            for row in csv.DictReader(f):
                rows += 1
                parsed = parse_agreid_name(row["image_path"])
                self.assertEqual(row["pid"], str(parsed.pid))
                p_only.add(int(parsed.person))
                official.add(parsed.pid)
        self.assertEqual(rows, 8154)
        self.assertEqual(len(p_only), 157)
        self.assertEqual(len(official), AGREID_TRAIN_IDENTITIES)

    def test_expected_official_test_count_constant(self):
        self.assertEqual(AGREID_TEST_IDENTITIES, 189)


if __name__ == "__main__":
    unittest.main()
