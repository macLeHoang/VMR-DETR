import unittest

from vmr_detr.evaluation.postprocessing_vmr_detr import PostProcessorDETR


class TestPostProcessorDurationClamp(unittest.TestCase):
    def _process(self, line):
        processor = PostProcessorDETR(process_func_names=("clip_ts",))
        return processor([line])[0]

    def test_tacos_like_duration_allows_late_window(self):
        out = self._process({
            "qid": 1,
            "_clamp_max_ts": 779,
            "pred_relevant_windows": [[600, 700, 0.9]],
        })

        self.assertEqual(out["pred_relevant_windows"], [[600.0, 700.0, 0.9]])

    def test_charades_like_duration_clamps_to_video_end(self):
        out = self._process({
            "qid": 1,
            "_clamp_max_ts": 30,
            "pred_relevant_windows": [[10, 45, 0.9]],
        })

        self.assertEqual(out["pred_relevant_windows"], [[10.0, 30.0, 0.9]])

    def test_missing_duration_uses_default_fallback(self):
        out = self._process({
            "qid": 1,
            "pred_relevant_windows": [[120, 170, 0.9]],
        })

        self.assertEqual(out["pred_relevant_windows"], [[120.0, 150.0, 0.9]])

    def test_invalid_duration_uses_default_fallback(self):
        out = self._process({
            "qid": 1,
            "_clamp_max_ts": "bad",
            "pred_relevant_windows": [[120, 170, 0.9]],
        })

        self.assertEqual(out["pred_relevant_windows"], [[120.0, 150.0, 0.9]])

    def test_temp_key_is_not_preserved(self):
        out = self._process({
            "qid": 1,
            "_clamp_max_ts": 30,
            "pred_relevant_windows": [[10, 45, 0.9]],
        })

        self.assertNotIn("_clamp_max_ts", out)

    def test_duration_is_per_line(self):
        processor = PostProcessorDETR(process_func_names=("clip_ts",))
        out = processor([
            {"qid": 1, "_clamp_max_ts": 30, "pred_relevant_windows": [[10, 45, 0.9]]},
            {"qid": 2, "_clamp_max_ts": 779, "pred_relevant_windows": [[600, 700, 0.8]]},
        ])

        self.assertEqual(out[0]["pred_relevant_windows"], [[10.0, 30.0, 0.9]])
        self.assertEqual(out[1]["pred_relevant_windows"], [[600.0, 700.0, 0.8]])


if __name__ == "__main__":
    unittest.main()
