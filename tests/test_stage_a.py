import numpy as np
import pandas as pd
import pytest

import pipeline
import solution
import time_budget
import training
from data_tools import detect_split, guess_task_spec
from features import add_auto_features, important_numeric


def test_fit_only_automated_features_ignore_changed_and_removed_queries():
    fit = pd.DataFrame({"key": ["a", "a", "b", None], "x": [2., 6., 8., 13.]})
    query = pd.DataFrame({"key": ["a", "b", "other"], "x": [200., 100., 200.]})
    expected, derived = add_auto_features(fit, query)
    changed = query.assign(key="b", x=1e9)
    for replay in (changed, query.iloc[:0]):
        actual, _ = add_auto_features(fit, replay)
        pd.testing.assert_frame_equal(actual, expected)
    assert derived["key__freq"].tolist() == [2., 1., 0.]
    np.testing.assert_allclose(derived["x__mean_by_key"], [4., 8., np.nan])


def test_pipeline_audits_fit_only_transforms_in_every_iid_fold(tmp_path, capsys, monkeypatch):
    rng = np.random.default_rng(3)
    frame = pd.DataFrame({"label": np.tile([0, 1], 60), "key": np.tile(["a", "b", "c"], 40),
                          "x": rng.normal(size=120)})
    frame.insert(0, "id", np.arange(len(frame)))
    test = pd.DataFrame({"id": [500, 501], "key": ["a", "unseen"], "x": [0., 1.]})
    frame.to_csv(tmp_path / "train.csv", index=False)
    test.to_csv(tmp_path / "test.csv", index=False)
    test[["id"]].iloc[::-1].assign(label=0).to_csv(tmp_path / "sample_submission.csv", index=False)
    spec, _ = guess_task_spec(tmp_path, "Average Precision")
    seen = []
    backend = training._BACKENDS["lightgbm"]

    def checked_fit(Xtr, ytr, wtr, Xva, yva, Xte, *args, **kwargs):
        if not args[3].get("final_fit"):
            assert "key__freq" in Xtr
            assert Xtr["key__freq"].max() == Xtr["key"].value_counts().max()
            seen.append(len(Xtr))
        return backend(Xtr, ytr, wtr, Xva, yva, Xte, *args, **kwargs)

    monkeypatch.setitem(training._BACKENDS, "lightgbm", checked_fit)
    result = pipeline.run_experiment(tmp_path, spec | {"audit_required": True},
                                     {"auto_features": True, "num_boost_round": 5,
                                      "early_stopping_rounds": 2}, 20.)
    assert result["complete"] and len(seen) == 5 and max(seen) < len(frame) * .8
    log = capsys.readouterr().err
    assert log.count('"group": "auto_features_fit_only"') == 10
    assert "status=fail" not in log
    assert pd.read_csv(tmp_path / "submission.csv")["id"].tolist() == [501, 500]


@pytest.mark.parametrize("grouped", [False, True])
def test_optuna_only_uses_first_outer_fit(monkeypatch, grouped):
    n = 150
    frame = pd.DataFrame({"position": np.arange(n), "key": np.repeat(np.arange(10), 15)})
    test = frame.tail(10).assign(position=-1, key=88)
    y = pd.Series(np.linspace(0, 5, n) if grouped else np.tile([0, 1], n // 2))
    kind, metric = ("regression", "mae") if grouped else ("binary", "pr_auc")
    split = detect_split(frame, test, group_column="key") if grouped else None
    groups = frame["key"].to_numpy() if grouped else None
    outer = training._make_folds(y.to_numpy(), kind, 5, 42, groups)
    fit, valid = outer[0]
    if grouped:
        assert set(frame["key"].iloc[fit]).isdisjoint(frame["key"].iloc[valid])
    seen = []

    def validate(X, target, prediction, task_type, metric_name, config, time_budget):
        np.testing.assert_array_equal(X["position"], fit)
        np.testing.assert_array_equal(target, y.iloc[fit])
        assert prediction["position"].tolist() == [fit[-1]]
        assert config["n_folds"] == 3 and config["split"] == split
        seen.append(config["params"])
        return {"score": 1., "config": {"n_folds_done": 3, "n_folds": 3}}

    monkeypatch.setattr(training, "train_and_predict", validate)
    diagnostics = {}
    params = training.tune_backend("xgboost", frame, y, test, kind, metric,
                                   {"split": split, "fold_seed": 42}, 20., diagnostics=diagnostics)
    assert params and seen and diagnostics["outer_fit_rows"] == len(fit)


def test_rare_dictionary_and_real_xgboost_with_ten_percent_unseen():
    fit = pd.DataFrame({"key": ["frequent"] * 5 + ["rare"] * 4 + [None], "x": np.arange(10)})
    query = pd.DataFrame({"key": ["frequent"] * 6 + ["unknown", None, "rare", "rare"], "x": np.arange(10)})
    Xtr, Xte, columns = training._prepare_features(fit, query, None, rare_min_freq=5)
    assert columns == ["key"]
    assert Xtr["key"].cat.categories.tolist() == ["frequent", "__rare__"]
    assert Xte["key"].iloc[6] == Xte["key"].iloc[8] == "__rare__"
    assert pd.isna(Xte["key"].iloc[7]) and pd.isna(Xtr["key"].iloc[9])
    fit = pd.concat([fit] * 12, ignore_index=True)
    result = training.train_and_predict(
        fit, pd.Series(np.tile([0, 1], len(fit) // 2)), query, "binary", "pr_auc",
        {"model": "xgboost", "rare_min_freq": 5, "num_boost_round": 8,
         "early_stopping_rounds": 3, "n_jobs": 1, "audit_required": True}, 15.,
    )
    assert result["config"]["model"] == "xgboost"
    assert np.isfinite(result["probabilities"]).all()


def test_catboost_temporal_order_and_high_cardinality_categories(monkeypatch):
    rng = np.random.default_rng(31)
    size = 360
    permutation = rng.permutation(size)
    frame = pd.DataFrame({"clock": permutation, "key": [f"v{i % 42}" for i in permutation]})
    target = pd.Series(permutation % 2)
    test = pd.DataFrame({"clock": [size, size + 1], "key": ["v0", "unseen"]})
    backend = training._BACKENDS["catboost"]
    seen = []

    def checked_fit(Xtr, ytr, wtr, Xva, yva, Xte, *args, **kwargs):
        assert "clock" not in Xtr
        assert kwargs["fit_times"].is_monotonic_increasing and kwargs["cat_cols"] == ["key"]
        np.testing.assert_array_equal(ytr, kwargs["fit_times"].to_numpy() % 2)
        assert Xtr["key"].map(type).eq(str).all()
        seen.append(Xtr["key"].nunique())
        return backend(Xtr, ytr, wtr, Xva, yva, Xte, *args, **kwargs)

    monkeypatch.setitem(training._BACKENDS, "catboost", checked_fit)
    cfg = {"model": "catboost", "split": {"type": "time", "column": "clock"}, "audit_required": True,
           "catboost_ordered_ctr": True, "num_boost_round": 8, "early_stopping_rounds": 3, "n_jobs": 1}
    result = training.train_and_predict(frame, target, test, "binary", "pr_auc", cfg, 20.)
    assert min(seen) > 32 and result["config"]["n_folds_done"] == 5
    with pytest.raises(AssertionError, match="sorted"):
        backend(frame, target, None, frame, target, test, "binary",
                training.get_metric("pr_auc", "binary"), 2, training._resolve_config(cfg), float("inf"),
                cat_cols=["key"], fit_times=frame["clock"])


@pytest.mark.parametrize("es_metric,native", [("auto", "aucpr"), ("logloss", "logloss")])
def test_xgboost_ap_monitor_is_native(monkeypatch, es_metric, native):
    import xgboost

    calls = []
    real = xgboost.train

    def train(params, *args, **kwargs):
        assert params["eval_metric"] == native and kwargs.get("custom_metric") is None
        calls.append(True)
        return real(params, *args, **kwargs)

    monkeypatch.setattr(xgboost, "train", train)
    frame = pd.DataFrame({"x": np.arange(100, dtype=float)})
    result = training.train_and_predict(
        frame, pd.Series(np.tile([0, 1], 50)), frame.tail(5), "binary", "pr_auc",
        {"model": "xgboost", "es_metric": es_metric, "n_jobs": 1, "n_folds": 3,
         "num_boost_round": 8, "early_stopping_rounds": 3}, 15.,
    )
    assert len(calls) == 3 and len(result["fold_scores"]) == 3


def test_missing_lightgbm_keeps_numeric_ranking_and_model(monkeypatch):
    original = training.backend_available
    monkeypatch.setattr(training, "backend_available", lambda name: name != "lightgbm" and original(name))
    rng = np.random.default_rng(4)
    fit = pd.DataFrame(rng.normal(size=(90, 9)))
    fit.columns = [f"x{i}" for i in fit]
    y = pd.Series(2. * fit["x8"])
    assert "x8" in important_numeric(fit, y, "regression")
    result = training.train_and_predict(
        fit, y, fit.tail(3), "regression", "mae",
        {"num_boost_round": 8, "early_stopping_rounds": 3, "n_jobs": 1}, 15.,
    )
    assert result["config"]["model"] != "lightgbm"
    assert np.isfinite(result["predictions"]).all()


def test_cpu_limit_respects_affinity_before_backend_import(monkeypatch):
    monkeypatch.setattr(time_budget.os, "sched_getaffinity", lambda pid: {0, 3, 5}, raising=False)
    monkeypatch.setenv("OMP_NUM_THREADS", "16")
    time_budget.limit_openmp()
    assert time_budget.os.environ["OMP_NUM_THREADS"] == "3"
    assert training._resolve_config({"n_jobs": 12})["n_jobs"] == 3


def test_quality_gate_checks_paired_noise_and_baseline_wins():
    control = {"fold_signature": "paired", "greater_is_better": True, "fold_scores": [.6] * 5,
               "fold_baselines": [{"score": .59}] * 5}
    candidate = control | {"fold_scores": [.63, .62, .63, .61, .62]}
    assert pipeline._passes_quality_gate(control, candidate, training.compare_results)
    assert not pipeline._passes_quality_gate(control, candidate | {"fold_signature": "different"},
                                             training.compare_results)
    assert not pipeline._passes_quality_gate(control, candidate | {"fold_scores": [.63, .62, .63, .61, .58]},
                                             training.compare_results)
    assert not pipeline._passes_quality_gate(control, candidate | {"fold_scores": [.601, .601, .601, .594, .97]},
                                             training.compare_results)


@pytest.mark.parametrize("models,stop", [(["lightgbm"], "queue_exhausted"),
                                         (["lightgbm", "catboost", "xgboost"], "stagnation")])
def test_bounded_queue_reserve_and_checkpoint(monkeypatch, tmp_path, models, stop):
    now = [0.]
    deadline = 5000.
    incumbent = {"complete": True, "recipe": "single", "score": 1., "greater_is_better": False,
                 "fold_scores": [1.] * 5, "fold_signature": "same", "fit_time": 95.,
                 "config": {"model": models[0]}}
    monkeypatch.setattr(pipeline, "experiment_history", lambda _: [incumbent])
    monkeypatch.setattr(solution.time, "monotonic", lambda: now[0])
    fits, tunes, checkpoints = [], [], []

    def fit(folder, spec, fit_deadline, budget, config, recipe="single", *, estimate):
        assert fit_deadline == deadline - 400.
        if recipe == "tune":
            tunes.append(config["model"])
            return {"tuned_params": {"seed_for_test": 12}}
        fits.append(config)
        now[0] += 180.
        if len(models) == 1:
            return incumbent | {"fold_scores": [.9] * 5, "score": .9}
        return incumbent

    monkeypatch.setattr(solution, "run_trial", fit)
    result = solution.search_remaining_budget(tmp_path, {"available_backends": models}, deadline, incumbent,
                                               {}, set(), checkpoint=lambda: checkpoints.append(now[0]))
    assert result["reason"] == stop and tunes == models
    assert result["completed"] == len(fits)
    assert bool(checkpoints) and checkpoints[0] == 900.
    if stop == "stagnation":
        assert len(fits) == 8
    else:
        assert len({f["seed"] for f in fits if "seed" in f}) == 4
