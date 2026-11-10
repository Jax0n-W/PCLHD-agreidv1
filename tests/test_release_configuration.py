from __future__ import absolute_import

import os.path as osp
import sys
import unittest


ROOT = osp.dirname(osp.dirname(osp.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import experiment_runner as runner


def parse(*items):
    return runner.build_parser().parse_args(list(items) + ["--dry-run"])


class ReleaseConfigurationTests(unittest.TestCase):
    def test_matrix_is_exactly_two_by_two(self):
        self.assertEqual(set(runner.EXPERIMENTS), {
            "baseline", "dynamic", "three_domain", "full"})
        self.assertEqual(runner.EXPERIMENTS["baseline"], {
            "dynamic_agva": False, "three_domain": False})
        self.assertEqual(runner.EXPERIMENTS["dynamic"], {
            "dynamic_agva": True, "three_domain": False})
        self.assertEqual(runner.EXPERIMENTS["three_domain"], {
            "dynamic_agva": False, "three_domain": True})
        self.assertEqual(runner.EXPERIMENTS["full"], {
            "dynamic_agva": True, "three_domain": True})

    def test_channel_mode_is_fixed_off_in_all_commands(self):
        for name in runner.EXPERIMENTS:
            args = parse("--experiment", name)
            stage1 = runner.build_stage1_command(args)
            stage2 = runner.build_stage2_command(args, "/tmp/stage1.pth.tar")
            self.assertIn("--channel-adap-gray", stage1)
            self.assertEqual(stage1[stage1.index("--channel-adap-gray") + 1],
                             "off")
            self.assertIn("--channel-adap-gray", stage2)
            self.assertEqual(stage2[stage2.index("--channel-adap-gray") + 1],
                             "off")

    def test_dynamic_agva_is_stage1_only(self):
        for name, spec in runner.EXPERIMENTS.items():
            args = parse("--experiment", name)
            stage1 = runner.build_stage1_command(args)
            stage2 = runner.build_stage2_command(args, "/tmp/stage1.pth.tar")
            self.assertEqual("--use-dynamic-agva" in stage1,
                             spec["dynamic_agva"])
            self.assertNotIn("--use-dynamic-agva", stage2)

    def test_three_domain_is_stage2_only(self):
        for name, spec in runner.EXPERIMENTS.items():
            args = parse("--experiment", name)
            stage1 = runner.build_stage1_command(args)
            stage2 = runner.build_stage2_command(args, "/tmp/stage1.pth.tar")
            self.assertTrue(stage1[1].endswith("_train_agreid_engine.py"))
            self.assertEqual(
                stage2[1].endswith("_train_agreid_three_domain_engine.py"),
                spec["three_domain"])

    def test_original_stage2_is_cmhard_without_ema_loss(self):
        for name in ("baseline", "dynamic"):
            args = parse("--experiment", name)
            command = runner.build_stage2_command(
                args, "/tmp/stage1.pth.tar")
            self.assertIn("--stage2-memorybank", command)
            self.assertEqual(
                command[command.index("--stage2-memorybank") + 1], "CMhard")
            self.assertIn("--no-ema-loss", command)

    def test_final_checkpoint_is_default(self):
        args = parse("--experiment", "baseline")
        stage1_name, _ = runner.default_log_names(args)
        self.assertTrue(
            runner.checkpoint_path(args, stage1_name).endswith(
                "checkpoint.pth.tar"))

    def test_training_engines_have_no_channel_conversion_calls(self):
        for filename in (
                "_train_agreid_engine.py",
                "_train_agreid_three_domain_engine.py"):
            with open(osp.join(ROOT, filename), "r", encoding="utf-8") as f:
                source = f.read()
            self.assertNotIn("ChannelAdapGray(", source)
            self.assertNotIn("ChannelExchange(", source)
            self.assertNotIn("Gray(", source)
            self.assertIn("ChannelRandomErasing(", source)


if __name__ == "__main__":
    unittest.main()
