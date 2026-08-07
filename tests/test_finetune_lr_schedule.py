import os
import sys
from types import SimpleNamespace
import unittest


PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)

from uer import finetune_lr_scheduler  # noqa: E402


class FinetuneLearningRateScheduleTest(unittest.TestCase):
    def test_cosine_schedule_warms_up_and_decays_to_minimum(self):
        kwargs = dict(
            total_steps=100,
            peak_lr=1e-3,
            min_lr=1e-5,
            warmup_steps=10,
            scheduler_name="cosine",
        )

        self.assertEqual(
            finetune_lr_scheduler.get_learning_rate(current_step=0, **kwargs),
            0.0,
        )
        self.assertAlmostEqual(
            finetune_lr_scheduler.get_learning_rate(current_step=5, **kwargs),
            5e-4,
        )
        self.assertAlmostEqual(
            finetune_lr_scheduler.get_learning_rate(current_step=10, **kwargs),
            1e-3,
        )
        self.assertAlmostEqual(
            finetune_lr_scheduler.get_learning_rate(current_step=55, **kwargs),
            (1e-3 + 1e-5) / 2,
        )
        self.assertAlmostEqual(
            finetune_lr_scheduler.get_learning_rate(current_step=100, **kwargs),
            1e-5,
        )

        last_update_lr = finetune_lr_scheduler.get_learning_rate(
            current_step=99,
            **kwargs,
        )
        self.assertGreaterEqual(last_update_lr, 1e-5)
        self.assertLess(last_update_lr, 1.1e-5)

    def test_cosine_schedule_without_warmup_starts_at_peak(self):
        kwargs = dict(
            total_steps=20,
            peak_lr=2e-4,
            min_lr=1e-6,
            warmup_steps=0,
            scheduler_name="cosine",
        )

        self.assertAlmostEqual(
            finetune_lr_scheduler.get_learning_rate(current_step=0, **kwargs),
            2e-4,
        )
        self.assertAlmostEqual(
            finetune_lr_scheduler.get_learning_rate(current_step=20, **kwargs),
            1e-6,
        )

    def test_constant_schedule_keeps_peak_learning_rate(self):
        learning_rate = finetune_lr_scheduler.get_learning_rate(
            current_step=17,
            total_steps=20,
            peak_lr=2e-5,
            min_lr=1e-6,
            warmup_steps=2,
            scheduler_name="constant",
        )

        self.assertEqual(learning_rate, 2e-5)

    def test_constant_config_ignores_unused_minimum_and_warmup(self):
        args = SimpleNamespace(
            epochs=3,
            max_steps=-1,
            lr=1e-7,
            lr_scheduler="constant",
            warmup_ratio=0.1,
            min_lr=1e-6,
        )

        config = finetune_lr_scheduler.build_lr_schedule_config(
            args=args,
            steps_per_epoch=12,
        )

        self.assertEqual(config["warmup_steps"], 0)
        self.assertEqual(config["peak_lr"], 1e-7)

    def test_schedule_config_respects_max_steps(self):
        args = SimpleNamespace(
            epochs=20,
            max_steps=150,
            lr=1e-4,
            lr_scheduler="cosine",
            warmup_ratio=0.1,
            min_lr=1e-6,
        )

        config = finetune_lr_scheduler.build_lr_schedule_config(
            args=args,
            steps_per_epoch=10,
        )

        self.assertEqual(config["total_steps"], 150)
        self.assertEqual(config["warmup_steps"], 15)

    def test_schedule_config_uses_epoch_length_without_max_steps(self):
        args = SimpleNamespace(
            epochs=3,
            max_steps=-1,
            lr=1e-4,
            lr_scheduler="cosine",
            warmup_ratio=0.1,
            min_lr=1e-6,
        )

        config = finetune_lr_scheduler.build_lr_schedule_config(
            args=args,
            steps_per_epoch=12,
        )

        self.assertEqual(config["total_steps"], 36)
        self.assertEqual(config["warmup_steps"], 3)

    def test_invalid_schedule_options_are_rejected(self):
        base_args = dict(
            epochs=3,
            max_steps=-1,
            lr=1e-4,
            lr_scheduler="cosine",
            warmup_ratio=0.1,
            min_lr=1e-6,
        )

        for changes in (
            {"lr_scheduler": "unknown"},
            {"warmup_ratio": -0.1},
            {"warmup_ratio": 1.0},
            {"min_lr": -1e-6},
            {"min_lr": 2e-4},
        ):
            with self.subTest(changes=changes):
                values = {**base_args, **changes}
                with self.assertRaises(ValueError):
                    finetune_lr_scheduler.build_lr_schedule_config(
                        args=SimpleNamespace(**values),
                        steps_per_epoch=12,
                    )

    def test_parameter_group_lr_scale_is_preserved(self):
        optimizer = SimpleNamespace(
            param_groups=[{}, {"lr_scale": 0.25}],
        )

        finetune_lr_scheduler.set_optimizer_learning_rate(optimizer, 1e-4)

        self.assertEqual(optimizer.param_groups[0]["lr"], 1e-4)
        self.assertEqual(optimizer.param_groups[1]["lr"], 2.5e-5)


if __name__ == "__main__":
    unittest.main()
