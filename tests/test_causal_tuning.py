import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from training import tune_lightgbm


class EarlyTuningTests(unittest.TestCase):
    def test_search_cannot_read_outer_validation_labels_or_live_test(self):
        dates = pd.date_range("2020-01-01", periods=120, freq="30min", tz="UTC")
        order = np.random.default_rng(13).permutation(len(dates))
        train = pd.DataFrame({"timestamp": dates[order], "x": order})
        durations = pd.Series(np.where(order < 19, 1., 360.))
        seen = []

        def fit(x, y, prediction_rows, task_type, metric, config, time_budget):
            self.assertTrue((x["timestamp"] < dates[20]).all())
            self.assertTrue(((x["timestamp"] + pd.to_timedelta(y, unit="min")) < dates[20]).all())
            self.assertEqual(prediction_rows["timestamp"].max(), dates[20])
            self.assertEqual(config["split"]["type"], "time")
            self.assertTrue(config["params"]["bagging_freq"])
            seen.append(len(x))
            return {"score": 2, "config": {"n_folds_done": 3, "n_folds": 3}}

        with patch("training.train_and_predict", side_effect=fit):
            best = tune_lightgbm(train, durations, train.tail(1), "regression", "mae",
                                {"split": {"type": "time", "column": "timestamp"},
                                 "label_delay_unit": "minutes", "n_folds": 5}, 30.)
        self.assertEqual(seen, [19] * len(seen))
        self.assertGreater(len(seen), 0)
        self.assertEqual(best["bagging_freq"], 1)

    def test_incomplete_trial_cannot_be_selected(self):
        x = pd.DataFrame({"timestamp": pd.date_range("2020", periods=120), "x": np.arange(120)})
        def partial(*args, **kwargs):
            return {"score": 1, "config": {"n_folds_done": 1, "n_folds": 3}}
        with patch("training.train_and_predict", side_effect=partial):
            params = tune_lightgbm(x, x["x"], x.tail(1), "regression", "mae",
                                   {"split": {"type": "time", "column": "timestamp"}}, 15)
        self.assertEqual(params, {})
