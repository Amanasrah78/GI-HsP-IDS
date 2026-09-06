import unittest

from models.proposed.repeated_evaluation import aggregate_metric_runs


class RepeatedEvaluationTests(unittest.TestCase):
    def test_aggregate_metric_runs_computes_mean_and_sample_std(self):
        runs = [
            {
                "loss": 0.7,
                "accuracy": 0.5,
                "precision": 0.5,
                "recall": 1.0,
                "f1": 2 / 3,
                "specificity": 0.5,
                "balanced_accuracy": 0.75,
                "mcc": 0.25,
            },
            {
                "loss": 0.5,
                "accuracy": 0.75,
                "precision": 0.8,
                "recall": 0.5,
                "f1": 0.6153846154,
                "specificity": 0.75,
                "balanced_accuracy": 0.625,
                "mcc": 0.3,
            },
        ]

        summary = aggregate_metric_runs(runs)

        self.assertEqual(summary["run_count"], 2)
        self.assertAlmostEqual(summary["loss"]["mean"], 0.6)
        self.assertAlmostEqual(summary["accuracy"]["mean"], 0.625)
        self.assertAlmostEqual(summary["accuracy"]["std"], 0.1767766953)


    def test_aggregate_metric_runs_single_run_uses_zero_std(self):
        runs = [
            {
                "loss": 0.4,
                "accuracy": 0.8,
                "precision": 0.75,
                "recall": 1.0,
                "f1": 0.8571428571,
                "specificity": 0.8,
                "balanced_accuracy": 0.9,
                "mcc": 0.7,
            },
        ]

        summary = aggregate_metric_runs(runs)

        self.assertEqual(summary["run_count"], 1)
        self.assertEqual(summary["loss"]["std"], 0.0)
        self.assertEqual(summary["f1"]["std"], 0.0)



    def test_aggregate_metric_runs_rejects_empty_input(self):
        with self.assertRaisesRegex(
            ValueError,
            "At least one metric run is required",
        ):
            aggregate_metric_runs([])



if __name__ == "__main__":
    unittest.main()
