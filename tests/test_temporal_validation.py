import numpy as np
import pandas as pd
import pytest

import pipeline
import training
from data_tools import detect_split, guess_task_spec


def test_time_windows_keep_ties_and_only_look_back():
    rng = np.random.default_rng(21)
    times = np.repeat(pd.date_range("2025-01-01", periods=12, tz="UTC"), 5)
    times = pd.Series(rng.permutation(times))
    labels = np.arange(len(times)) % 2
    folds = training._time_folds(times, labels, 4, "binary", "pr_auc")
    assert len(folds) == 4
    for train, valid in folds:
        assert times.iloc[train].max() < times.iloc[valid].min()
        assert set(times.index[times.isin(times.iloc[valid])]) == set(valid)
    assert set(folds[0][0]).isdisjoint(np.concatenate([valid for _, valid in folds]))


@pytest.mark.parametrize(("task_type","metric"), [("binary","pr_auc"), ("regression","mae")])
def test_temporal_training_accepts_read_only_pandas_views(task_type, metric):
    x = np.arange(120) % 11
    frame = pd.DataFrame({"event_time": pd.date_range("2025-01-01", periods=len(x), freq="h"), "x": x})
    frame.loc[2, "event_time"] = pd.NaT
    y = pd.Series((x > 4) if task_type == "binary" else x, dtype=float)
    y.iloc[3] = np.nan
    original = frame.copy(), y.copy()
    config = {"num_boost_round": 3, "n_folds": 3, "n_jobs": 1}
    with pd.option_context("mode.copy_on_write", True):
        assert not y.notna().to_numpy().flags.writeable
        result = training.train_and_predict(frame, y, frame.iloc[-10:], task_type, metric, config, 20)
    assert result["config"]["n_folds_done"] == 3 and np.isfinite(result["score"])
    assert result["fold_ids"][2] == result["fold_ids"][3] == -1
    pd.testing.assert_frame_equal(frame, original[0])
    pd.testing.assert_series_equal(y, original[1])


def test_event_split_precedes_group_and_preserves_nontemporal_strategies():
    train = pd.DataFrame({"district": ["A", "B"], "event_time": ["2025-01-01", "2025-01-02"]})
    test = pd.DataFrame({"district": ["C"], "event_time": ["2025-02-01"]})
    assert detect_split(train, test) == {"type": "time", "column": "event_time"}
    assert detect_split(train[["district"]], test[["district"]]) == {"type": "group", "column": "district"}
    assert detect_split(train[["district"]], train[["district"]]) is None
    with pytest.raises(ValueError):
        detect_split(train, test, time_column="missing")


def test_time_oof_and_ensemble_never_fit_meta_weights_on_future(monkeypatch):
    n = 120
    frame = pd.DataFrame({"ts": np.tile(np.arange(24), 5).astype(float),
                          "x": np.arange(n) % 13, "entity": np.arange(n) % 3})
    frame.loc[3, "ts"] = np.nan
    y = pd.Series(np.arange(n), dtype=float)
    y.iloc[9] = np.nan
    split = {"type": "time", "column": "ts"}
    config = {"split": split, "n_jobs": 2, "num_boost_round": 8,
              "n_folds": 3, "target_encoding": ["entity"]}
    a = training.train_and_predict(frame, y, frame.iloc[:6], "regression", "mae", config, 20)
    b = training.train_and_predict(frame, y, frame.iloc[:6], "regression", "mae", config | {"seed": 4}, 20)
    fid = a["fold_ids"]
    np.testing.assert_array_equal(fid, b["fold_ids"])
    assert fid[3] == fid[9] == -1
    assert np.isnan(a["oof"][fid < 0]).all() and np.isfinite(a["oof"][fid >= 0]).all()
    assert a["config"]["target_encoding"] == ["entity"]
    assert "entity__te0" in a["model_inputs"]
    assert "entity" not in a["config"]["features"]

    fitted_labels = []
    def weights(oofs, labels, objective):
        assert np.isfinite(oofs).all()
        fitted_labels.append(set(labels))
        return np.array([.5, .5])
    monkeypatch.setattr(training, "_greedy_blend_weights", weights)
    blended = training.optimize_blend_weights([a, b], y, "mae")
    assert fitted_labels == [set(y[fid == 0]), set(y[(fid >= 0) & (fid < 2)]), set(y[fid >= 0])]
    np.testing.assert_array_equal(blended["fold_ids"], fid)
    with pytest.raises(ValueError, match="different validation rows"):
        training.optimize_blend_weights([a, b | {"fold_ids": np.roll(fid, 1)}], y, "mae")


def test_time_classification_can_score_one_class_validation_windows():
    t = np.repeat(np.arange(8), 5)
    y = np.repeat([0, 1, 1, 0, 0, 0, 1, 1], 5)
    folds = training._time_folds(t, y, 5, "binary", "pr_auc")
    assert all(len(set(y[tr])) == 2 for tr, _ in folds)
    assert any(len(set(y[va])) == 1 for _, va in folds)
    with pytest.raises(ValueError):
        training._time_folds(np.zeros(20), np.ones(20), 3, "regression", "mae")


def test_experiments_share_folds_and_failure_preserves_sample(tmp_path, monkeypatch):
    monkeypatch.setenv("TYRELL_OUTPUT_DIR", str(tmp_path))
    train = pd.DataFrame({"key": np.arange(90), "event_time": pd.date_range("2025-01-01", periods=90),
                          "district": list("ABC") * 30, "response_time_minutes": np.arange(90) % 7})
    train.sample(frac=1, random_state=4).to_csv(tmp_path / "train.csv", index=False)
    test = train.iloc[:6].drop(columns="response_time_minutes").copy()
    test["key"] = [f"k{i:03}" for i in range(6)]
    test.to_csv(tmp_path / "test.csv", index=False)
    sample = pd.DataFrame({"key": list(reversed(test["key"])), "response_time_minutes": 0.})
    sample.to_csv(tmp_path / "sample_submission.csv", index=False)
    spec, _ = guess_task_spec(tmp_path, "Predict response_time_minutes. Metric MAE.")
    assert pipeline.write_fallback(tmp_path, spec) == []
    fallback = (tmp_path / "submission.csv").read_bytes()
    pipeline.check_result(tmp_path, spec)
    with monkeypatch.context() as context:
        def fail(*args, **kwargs):
            raise RuntimeError("intentional model failure")
        context.setitem(training._BACKENDS, "lightgbm", fail)
        with pytest.raises(RuntimeError, match="intentional model failure"):
            pipeline.run_experiment(tmp_path, spec)
    assert (tmp_path / "submission.csv").read_bytes() == fallback
    config = {"num_boost_round": 4, "target_encoding": True}
    a = pipeline.run_experiment(tmp_path, spec, config, time_budget=20)
    b = pipeline.run_experiment(tmp_path, spec, config | {"auto_features": True}, time_budget=20)
    assert a["fold_signature"] == b["fold_signature"] and a["complete"] and b["complete"]
    assert a["target_encoding"] and b["split"]["type"] == "time"
    assert a["te_keys"] == {"duplicate": ["event_time", "district"]}
    ensemble = pipeline.run_ensemble(tmp_path, spec)
    assert ensemble["fold_signature"] == a["fold_signature"]
    assert ensemble["fold_windows"] == a["fold_windows"] == b["fold_windows"]
    assert len(a["fold_windows"]) == a["n_folds_done"]
    for window in a["fold_windows"]:
        assert pd.Timestamp(window["train_max"]) < pd.Timestamp(window["validation_min"])
        assert window["train_rows"] and window["validation_rows"]
    pipeline.check_result(tmp_path)
    produced = pd.read_csv(tmp_path / "submission.csv")
    assert list(produced["key"]) == list(sample["key"]) and np.isfinite(produced.iloc[:, 1]).all()
    assert (produced.iloc[:, 1] >= 0).all()


def test_training_resource_overrides():
    cfg = training._resolve_config({"n_jobs": 66, "params": {"n_jobs": -1, "tree_method": "gpu_hist",
                                                               "device": "cuda", "num_thread": 64}})
    assert 0 < cfg["n_jobs"] <= 8
    assert cfg["params"] == {"tree_method": "hist"}


def test_feature_code_cannot_reorder_predictions():
    a = pd.DataFrame({"x": [2, 4, 1]})
    b = pd.DataFrame({"x": [7, 5]})
    with pytest.raises(ValueError, match="row index and order"):
        pipeline.apply_feature_code("def make_features(df): return df.sort_values('x').reset_index(drop=True)", a, b)
    train, test = pipeline.apply_feature_code("def make_features(df): return df.assign(z=df.x * 2)", a, b)
    assert list(train.z) == [4, 8, 2] and list(test.z) == [14, 10]


def test_auto_hypothesis_params_only_reach_the_tested_backend(monkeypatch):
    configs = {}
    real_train = training.train_and_predict
    def observe(X, y, X_test, task_type, metric, config, time_budget):
        configs[config["model"]] = config["params"]
        return real_train(X, y, X_test, task_type, metric, config, time_budget)
    monkeypatch.setattr(training, "train_and_predict", observe)
    X = pd.DataFrame({"x": np.arange(100) % 17})
    training.auto_train(X, X.x, X.iloc[:3], "regression", "mae", 15,
                        config={"num_boost_round": 2, "n_folds": 2, "n_jobs": 2,
                                "model": "lightgbm", "params": {"min_child_samples": 3}}, max_seeds=1)
    assert configs["lightgbm"] == {"min_child_samples": 3}
    assert configs["xgboost"] == {}


def test_saved_ensemble_retains_postprocessed_labels(tmp_path, monkeypatch):
    monkeypatch.setenv("TYRELL_OUTPUT_DIR", str(tmp_path))
    spec = {"task_type": "binary", "metric": "f1"}
    sample_path = tmp_path / "sample_submission.csv"
    pd.DataFrame({"id": ["a", "b"], "y": 0}).to_csv(sample_path, index=False)
    data = {"test_ids": pd.Series(["a", "b"], name="id"), "sample_path": sample_path,
            "y_train": pd.Series([0, 1, 0, 1])}
    result = {"predictions": np.array([1, 1]), "probabilities": np.array([[.6, .4], [.7, .3]]),
              "oof": np.tile([.6, .4], (4, 1)), "fold_ids": [0, 0, 1, 1], "score": .5,
              "score_std": 0., "class_order": [0, 1], "fold_scores": [.5, .5],
              "fit_time": .1, "greater_is_better": True, "config": {"n_folds": 2, "n_folds_done": 2}}
    history = []
    report = {"task_spec": spec, "recipe": "single", "config": {}}
    pipeline._record(tmp_path, spec, data, history, result, report)
    pipeline._record(tmp_path, spec, data, history, result, report)
    monkeypatch.setattr(pipeline, "load_data", lambda *args: data)
    def observe(results, y, metric, **kwargs):
        assert all(list(r["predictions"]) == [1, 1] for r in results)
        stats = {"pool": [0, 1], "selected": [0], "best_single": 0, "duplicates": []}
        return results[0], ([0], lambda p: p[0]), stats
    monkeypatch.setattr(training, "select_ensemble", observe)
    pipeline.run_ensemble(tmp_path, spec)



def test_direct_training_recognizes_event_time_before_feature_filtering():
    timestamps = pd.date_range("2025-01-01", periods=36).astype(str)
    frame = pd.DataFrame({"event_time": timestamps, "x": np.arange(len(timestamps))})
    cfg = {"features": ["x"], "n_folds": 3, "num_boost_round": 2, "n_jobs": 1}
    result = training.train_and_predict(frame, frame.x, frame.iloc[:2], "regression", "mae", cfg, 15)
    assert result["config"]["split"] == {"type": "time", "column": "event_time"}
    assert np.all(result["fold_ids"][:9] == -1)
    assert np.all(result["fold_ids"][9:18] == 0)


@pytest.mark.parametrize("observed", [0, 1])
def test_fallback_binary_only_one_observed_class(tmp_path, monkeypatch, observed):
    monkeypatch.setenv("TYRELL_OUTPUT_DIR", str(tmp_path))
    train = pd.DataFrame({"event_id": [10, 11], "x": [2, 4], "engaged": [observed, observed]})
    train.to_csv(tmp_path / "train.csv", index=False)
    pd.DataFrame({"event_id": [13, 12], "x": [5, 2]}).to_csv(tmp_path / "test.csv", index=False)
    pd.DataFrame({"event_id": [12, 13], "engaged": 0}).to_csv(tmp_path / "sample_submission.csv", index=False)
    spec = {"id_columns": ["event_id"], "target_column": "engaged", "metric": "pr_auc",
            "task_type": "binary", "probabilities": True}
    assert pipeline.write_fallback(tmp_path, spec) == []
    pipeline.check_result(tmp_path, spec)
    frame = pd.read_csv(tmp_path / "submission.csv")
    assert frame.engaged.tolist() == [float(observed)] * 2
