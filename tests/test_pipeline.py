import unittest

from src.etf_quant.pipeline import run_pipeline


class PipelineTest(unittest.TestCase):
    def test_pipeline_runs(self):
        metrics = run_pipeline(bars=2200, freq="5min", horizon=6, embargo=2)
        self.assertGreater(metrics["n_predictions"], 0)
        self.assertGreater(metrics["final_equity"], 0)
        self.assertGreaterEqual(metrics["accuracy"], 0.0)
        self.assertLessEqual(metrics["accuracy"], 1.0)
        self.assertGreaterEqual(metrics["brier"], 0.0)
        self.assertIn("calmar", metrics)
        self.assertIn("win_rate", metrics)
        self.assertIn("avg_cost", metrics)
        self.assertGreaterEqual(metrics["avg_cost"], 0.0)

    def test_pipeline_supports_other_frequency(self):
        metrics = run_pipeline(bars=2200, freq="15min", horizon=3, embargo=1)
        self.assertGreater(metrics["n_predictions"], 0)
        self.assertGreater(metrics["final_equity"], 0)


if __name__ == "__main__":
    unittest.main()
