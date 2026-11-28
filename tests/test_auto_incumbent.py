import unittest
from unittest.mock import patch

import pandas as pd

import training


class AutoTrainIncumbentTests(unittest.TestCase):
    def test_finished_solo_competes_with_previous_blend(self):
        X = pd.DataFrame({"x": range(12)})
        for metric, task, scores in [
            ("pr_auc", "binary", [0.22, 0.24, 0.32]),
            ("mae", "regression", [0.32, 0.30, 0.22]),
        ]:
            for final_mix in ("failure", "no_time", "better_blend"):
                with self.subTest(metric=metric, final_mix=final_mix):
                    greater = task == "binary"
                    high_tail = [1] * 11 + [20]
                    y = pd.Series(([0, 1] * 6) if greater else high_tail)
                    models = []

                    def fit(X, y, test, task_type, metric_name, config, time_budget):
                        index = len(models)
                        model = {
                            "config": dict(config, n_folds_done=3, n_folds=3, task_type=task_type,
                                           metric=metric_name),
                            "score": scores[index], "score_std": 0.01, "fit_time": 1.0,
                            "greater_is_better": greater, "predictions": [index],
                            "feature_candidates": {"sqrt_target": not greater},
                            "postprocess_time": 100.0 if index == 2 and final_mix == "no_time" else 0.,
                        }
                        models.append(model)
                        return model

                    def blend(members, y, metric_name):
                        if len(members) == 3 and final_mix != "better_blend":
                            raise ValueError("unavailable")
                        improvement = 0.01 if greater else -0.01
                        return members[-1] | {"score": members[-1]["score"] + improvement,
                                              "predictions": ["blend"]}

                    with patch.object(training, "train_and_predict", side_effect=fit), \
                            patch.object(training, "optimize_blend_weights", side_effect=blend) as blends:
                        result = training.auto_train(X, y, X.iloc[:1], task, metric,
                                                     120, config={"split": None}, max_seeds=1)
                    self.assertEqual(len(models), 3)
                    self.assertEqual(blends.call_count, 1 if final_mix == "no_time" else 2)
                    if final_mix == "better_blend":
                        self.assertEqual(result["predictions"], ["blend"])
                        self.assertAlmostEqual(result["score"], scores[-1] + (0.01 if greater else -0.01))
                    else:
                        self.assertEqual(result["predictions"], [2])
                        self.assertEqual(result["score"], scores[-1])
                    self.assertNotIn("member_results", models[-1])
