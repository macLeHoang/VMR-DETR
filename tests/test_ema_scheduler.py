import unittest

import torch

from vmr_detr.cli.train_utils import EMAScheduler, ModelEMA


class EMASchedulerTest(unittest.TestCase):
    def _model_and_ema(self, decay=0.9):
        model = torch.nn.Linear(1, 1, bias=False)
        with torch.no_grad():
            model.weight.fill_(1.0)
        return model, ModelEMA(model, decay)

    def test_cosine_schedule_starts_and_reaches_target(self):
        model, ema = self._model_and_ema()
        scheduler = EMAScheduler(
            ema,
            start_decay=0.9,
            target_decay=0.99,
            warmup_updates=5,
            schedule="cosine",
        )

        decays = []
        for _ in range(5):
            decays.append(scheduler.step(model))

        self.assertAlmostEqual(decays[0], 0.9)
        self.assertAlmostEqual(decays[-1], 0.99)
        self.assertTrue(all(before <= after for before, after in zip(decays, decays[1:])))

    def test_linear_schedule_is_monotonic(self):
        model, ema = self._model_and_ema()
        scheduler = EMAScheduler(
            ema,
            start_decay=0.9,
            target_decay=0.99,
            warmup_updates=3,
            schedule="linear",
        )

        decays = [scheduler.step(model) for _ in range(3)]

        for decay, expected in zip(decays, [0.9, 0.945, 0.99]):
            self.assertAlmostEqual(decay, expected)

    def test_update_every_skips_and_compensates_decay(self):
        model, ema = self._model_and_ema()
        scheduler = EMAScheduler(
            ema,
            start_decay=0.8,
            target_decay=0.8,
            warmup_updates=1,
            update_every=2,
            schedule="constant",
        )

        self.assertIsNone(scheduler.step(model))
        self.assertAlmostEqual(scheduler.step(model), 0.64)
        self.assertEqual(scheduler.num_updates, 2)
        self.assertEqual(scheduler.num_ema_updates, 1)

    def test_state_dict_restores_scheduler_counters_and_config(self):
        model, ema = self._model_and_ema()
        scheduler = EMAScheduler(
            ema,
            start_decay=0.9,
            target_decay=0.99,
            warmup_updates=5,
            update_every=2,
            schedule="linear",
        )
        scheduler.step(model)
        scheduler.step(model)

        _, new_ema = self._model_and_ema()
        restored = EMAScheduler(new_ema, start_decay=0.1, target_decay=0.2)
        restored.load_state_dict(scheduler.state_dict())

        self.assertEqual(restored.start_decay, scheduler.start_decay)
        self.assertEqual(restored.target_decay, scheduler.target_decay)
        self.assertEqual(restored.warmup_updates, scheduler.warmup_updates)
        self.assertEqual(restored.update_every, scheduler.update_every)
        self.assertEqual(restored.schedule, scheduler.schedule)
        self.assertEqual(restored.num_updates, scheduler.num_updates)
        self.assertEqual(restored.num_ema_updates, scheduler.num_ema_updates)

    def test_model_ema_reset_replaces_shadow_with_current_model(self):
        model, ema = self._model_and_ema()
        with torch.no_grad():
            model.weight.fill_(3.0)

        ema.reset(model)

        self.assertTrue(torch.equal(ema.shadow_params["weight"], model.weight.detach()))


if __name__ == "__main__":
    unittest.main()
