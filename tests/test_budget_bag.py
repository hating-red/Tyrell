"""Regression coverage for budgeted seed averaging and common validation windows."""
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

import training


class Clock:
    def __init__(self):
        self.now = 0.

    def monotonic(self):
        return self.now


def exercise(budget=18., *, partial_after=None, corrupt=None, task_type="regression"):
    clock = Clock()
    calls = []
    opts = {"model": "lightgbm", "seed": 123, "split": {"type": "time", "column": "t"}}
    metric = "mae" if task_type == "regression" else "pr_auc"
    y = pd.Series([0, 1, 0, 1, 0, 1])

    def fit(X, y, X_test, kind, metric, config, time_budget):
        calls.append((config, time_budget))
        n = len(calls)
        clock.now += 5.
        pred = .1 * n + .2 * np.arange(6) / 6
        pred_test = np.array([.1 * n, .2 * n])
        cfg = config | {"task_type": kind, "metric": metric, "n_folds": 2, "n_folds_done": 2,
                        "fold_windows": [{"train_rows": 2}, {"train_rows": 4}],
                        "early_stopping_windows": [{"fit_rows": 2}, {"fit_rows": 3}]}
        if partial_after is not None and n > partial_after:
            cfg["n_folds_done"] = 1
        pred[:2] = np.nan
        order, probs = None, None
        if kind != "regression":
            pred = np.column_stack([1.-pred, pred])
            probs = np.column_stack([1.-pred_test, pred_test])
            pred_test = (probs[:, 1] > .5).astype(int)
            order = [0, 1]
        member = {"score": float(n), "config": cfg, "fit_time": 4., "fold_ids": np.array([-1,-1,0,0,1,1]),
                  "class_order": order, "oof": pred, "predictions": pred_test, "probabilities": probs,
                  "meta_known_from": np.arange(6), "fold_scores": [float(n)] * 2,
                  "model_inputs": {"t": {"input_dtype":"int64"}}}
        if n == 2 and corrupt:
            corrupt(member)
        return member

    with patch.object(training, "time", clock), patch.object(training, "train_and_predict", side_effect=fit):
        bag = training.train_budget_bag(pd.DataFrame({"t": np.arange(6)}), y, pd.DataFrame({"t": [6,7]}),
                                        task_type, metric, opts, time_budget=budget)
    return bag, calls


def test_complete_members_averaged_once_without_score_selection_or_fold_reseeding():
    bag, calls = exercise()
    assert len(calls) == bag["members"]["count"] == 3
    assert len({cfg["seed"] for cfg, _ in calls}) == 3
    assert all(cfg["fold_seed"] == 42 for cfg, _ in calls)
    assert calls == exercise()[1]
    assert [seconds for _, seconds in calls] == pytest.approx([17.1, 12.1, 7.1])
    np.testing.assert_allclose(bag["predictions"], [.2, .4])
    np.testing.assert_allclose(bag["oof"][2:], .2 + .2 * np.arange(2,6)/6)
    assert np.isnan(bag["oof"][:2]).all()
    assert bag["fit_time"] == 15.
    assert bag["config"]["fold_windows"] == [{"train_rows": 2}, {"train_rows": 4}]
    assert bag["config"]["seed"] == 123 and "params" not in bag["config"]


def test_binary_averages_probabilities_and_recomputes_labels():
    bag, _ = exercise(task_type="binary")
    np.testing.assert_allclose(bag["probabilities"], [[.8,.2],[.6,.4]])
    np.testing.assert_array_equal(bag["predictions"], [0,0])
    assert bag["class_order"] == [0,1]


def test_partial_fit_cannot_enter_mean_and_first_complete_fit_is_preserved():
    result, calls = exercise(partial_after=1)
    assert len(calls) == 2 and result["members"]["count"] == 1
    np.testing.assert_allclose(result["predictions"], [.1,.2])
    assert result["fit_time"] == 10.
    with pytest.raises(TimeoutError, match="No complete"):
        exercise(partial_after=0)


@pytest.mark.parametrize("damage", [
    lambda r: r["config"]["fold_windows"][0].update(train_rows=1),
    lambda r: r["config"]["early_stopping_windows"][0].update(fit_rows=1),
    lambda r: r["fold_ids"].__setitem__(2, 1),
    lambda r: r["meta_known_from"].__setitem__(2, 0),
    lambda r: r.update(class_order=[1,0]),
    lambda r: r.update(probabilities=np.array([[.5,.5]])),
])
def test_incompatible_members_rejected_before_averaging(damage):
    with pytest.raises(ValueError):
        exercise(corrupt=damage, task_type="binary")


@pytest.mark.parametrize("backend, split", [
    ("lightgbm", {"type": "time", "column": "t"}),
    ("xgboost", {"type": "group", "column": "site"}),
    ("catboost", None),
])
def test_actual_backends_keep_identical_validation_rows_with_changing_seeds(backend, split):
    t = np.arange(126)
    data = pd.DataFrame({"t": t, "site": t % 7, "x": np.sin(t*.3)})
    y = pd.Series(3 + data["x"] + (t % 5) / 5.)
    train, test = data.iloc[:108], data.iloc[108:]
    clock = Clock()
    real_fit = training.train_and_predict
    members = []

    def actual_fit(*args, **kwargs):
        result = real_fit(*args, **kwargs)
        members.append(result)
        clock.now += 5.
        return result

    cfg = {"model": backend, "seed": 321, "fold_seed": 19, "split": split, "n_folds": 2,
           "n_jobs": 1, "num_boost_round": 4, "early_stopping_rounds": 2}
    with patch.object(training, "time", clock), patch.object(training, "train_and_predict", side_effect=actual_fit):
        bag = training.train_budget_bag(train, y.iloc[:len(train)], test, "regression", "mae", cfg, 12.)
    assert len(members) == bag["members"]["count"] == 2
    assert np.array_equal(bag["fold_ids"], members[0]["fold_ids"])
    assert members[0]["config"]["seed"] != members[1]["config"]["seed"]
    assert bag["config"]["fold_seed"] == 19
    assert np.isfinite(bag["predictions"]).all() and np.isfinite(bag["score"])
