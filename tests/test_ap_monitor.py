import builtins
import runpy
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

import training


class AveragePrecisionMonitorTest(unittest.TestCase):
    def test_only_single_class_monitor_falls_back_to_logloss(self):
        import lightgbm as lgb

        frame = pd.DataFrame({"feature": np.arange(80, dtype=float)})
        labels = np.tile([0, 1], 40)
        cfg = training._resolve_config({"n_jobs": 1, "num_boost_round": 4,
                                        "early_stopping_rounds": 2,
                                        "params": {"metric": "average_precision"}})
        for monitor, expected in [(np.zeros(8), "binary_logloss"), (np.ones(8), "binary_logloss"),
                                  (labels[:8], "average_precision")]:
            with self.subTest(expected=expected, positives=int(sum(monitor))):
                with patch.object(lgb, "train", wraps=lgb.train) as train:
                    pv, pt, *_ = training._fit_lightgbm(
                        frame, labels, None, frame.iloc[:8], monitor, frame.iloc[:2], "binary",
                        training.get_metric("pr_auc", "binary"), 2, cfg, float("inf"),
                    )
                self.assertEqual(train.call_args.args[0]["metric"], expected)
                self.assertTrue(np.isfinite(pv).all() and np.isfinite(pt).all())
                self.assertEqual(cfg["params"]["metric"], "average_precision")

    def test_broken_lightgbm_does_not_disable_real_xgboost(self):
        real_import = builtins.__import__

        def missing_library(name, *args, **kwargs):
            if name == "lightgbm":
                raise OSError("libgomp.so.1 unavailable")
            return real_import(name, *args, **kwargs)

        with patch("builtins.__import__", missing_library):
            fresh = runpy.run_path(training.__file__)
            frame = pd.DataFrame({"x": np.arange(120, dtype=float)})
            target = pd.Series(np.sin(np.arange(120) / 15))
            cfg = {"model": "xgboost", "n_folds": 2, "n_jobs": 1,
                   "num_boost_round": 5, "early_stopping_rounds": 2}
            result = fresh["train_and_predict"](frame, target, frame.iloc[:3], "regression", "mae", cfg)
            self.assertEqual(result["config"]["model"], "xgboost")
            self.assertEqual(result["config"]["n_folds_done"], 2)
            self.assertTrue(np.isfinite(result["predictions"]).all())
            with self.assertRaisesRegex(OSError, "libgomp"):
                fresh["_fit_lightgbm"](None, None, None, None, None, None, "regression",
                                       fresh["get_metric"]("mae", "regression"), 0, {}, float("inf"))
