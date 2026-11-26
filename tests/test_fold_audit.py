import hashlib

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import average_precision_score, mean_absolute_error

import pipeline
import training


def _frames():
    frame = pd.DataFrame({"event_time": pd.date_range("2025-01-01", periods=84, freq="min", tz="UTC"),
                          "value": np.arange(84) % 9})
    test = frame.iloc[:4].copy()
    test["event_time"] += pd.Timedelta(minutes=84)
    return frame, test


def _check_audit(y, result, frame):
    signature = hashlib.sha256(np.asarray(result["fold_ids"], dtype="<i4").tobytes()).hexdigest()
    data = {"X_train": frame, "y_train": y, "train_ids": pd.Series(np.arange(len(y)) + 10)}
    audit = pipeline._cv_audit(data, result, signature)
    assert audit["membership_signature"] and audit["fold_signature"] == signature
    assert audit["baseline_score"] == pytest.approx(np.mean([row["baseline_score"] for row in audit["folds"]]))
    return audit


@pytest.mark.parametrize("binary", [True, False])
def test_actual_baselines_and_past_only_early_stopping(binary):
    frame, test = _frames()
    y = pd.Series(((np.arange(len(frame)) % 7) < 2).astype(int) if binary else np.arange(len(frame)) % 13)
    kind, metric = ("binary", "pr_auc") if binary else ("regression", "mae")
    cfg = {"n_folds": 3, "num_boost_round": 5, "n_jobs": 1}
    result = training.train_and_predict(frame, y, test, kind, metric, cfg, 15)
    audit = _check_audit(y, result, frame)
    if binary:
        assert audit["training_period"]["positives"] == int(y.sum())
        assert audit["training_period"]["prevalence"] == pytest.approx(y.mean())
    for window, monitor in zip(audit["folds"], result["config"]["early_stopping_windows"]):
        valid = result["fold_ids"] == window["fold"]
        train = frame.event_time < frame.event_time[valid].min()
        assert pd.Timestamp(monitor["fit_max"]) < pd.Timestamp(monitor["monitor_min"])
        assert pd.Timestamp(monitor["monitor_max"]) < pd.Timestamp(window["validation_min"])
        if binary:
            expected = average_precision_score(y[valid], np.full(valid.sum(), y[train].mean()))
            assert window["train_positives"] == int(y[train].sum())
            assert window["validation_positives"] == int(y[valid].sum())
            probabilities = result["oof"][valid, 1]
            assert window["model_score"] == pytest.approx(average_precision_score(y[valid], probabilities))
            assert window["brier"] == pytest.approx(np.mean((y[valid] - probabilities) ** 2))
        else:
            expected = mean_absolute_error(y[valid], np.full(valid.sum(), y[train].median()))
            assert window["model_score"] == pytest.approx(mean_absolute_error(y[valid], result["oof"][valid]))
        assert window["baseline_score"] == pytest.approx(expected)
    changed = y.copy()
    valid = result["fold_ids"] == 0
    changed[valid] = 1 - changed[valid] if binary else changed[valid] + 1000
    other = training.train_and_predict(frame, changed, test, kind, metric, cfg, 15)
    np.testing.assert_array_equal(result["oof"][valid], other["oof"][valid])


def test_duration_labels_available_at_monitor_validation_and_test():
    frame, test = _frames()
    y = pd.Series(np.arange(len(frame)) % 5 + 2, dtype=float)
    y.iloc[19] = 25
    available = frame.event_time + pd.to_timedelta(y, unit="m")
    cfg = {"n_folds": 3, "num_boost_round": 4, "n_jobs": 1, "label_delay_unit": "minutes", "refit_full": True}
    result = training.train_and_predict(frame, y, test, "regression", "mae", cfg, 20)
    audit = _check_audit(y, result, frame)
    assert audit["training_period"]["observed_before_prediction"] == int((available < test.event_time.min()).sum())
    assert (result["fold_ids"][available >= test.event_time.min()] == -1).all()
    for window, monitor in zip(audit["folds"], result["config"]["early_stopping_windows"]):
        cutoff = pd.Timestamp(window["validation_cutoff"])
        known = available < cutoff
        missing = (frame.event_time < cutoff) & ~known
        assert window["train_pending"] == int(missing.sum()) > 0
        assert window["train_rows"] == int(known.sum())
        assert window["train_median"] == pytest.approx(y[known].median())
        assert monitor["fit_rows"] == (available < pd.Timestamp(monitor["monitor_min"])).sum()
    blend = training._prepare_blend([result, result], y, "mae")
    for i, (fit, valid) in enumerate(blend.folds):
        assert (available.iloc[fit] < frame.event_time.iloc[valid].min()).all()
    ensemble = training.blend_results([result, result], y, "mae")
    assert _check_audit(y, ensemble, frame)["folds"] == audit["folds"]


def test_ablation_uses_same_folds_and_keeps_submission(tmp_path, monkeypatch):
    frame, test = _frames()
    frame["district"] = np.arange(len(frame)) % 3
    test["district"] = 1
    frame.assign(id=np.arange(len(frame)), response=np.arange(len(frame)) % 7).to_csv(tmp_path / "train.csv", index=False)
    test.assign(id=np.arange(len(test)) + 100).to_csv(tmp_path / "test.csv", index=False)
    pd.DataFrame({"id": np.arange(len(test)) + 100, "response": 0.}).to_csv(tmp_path / "sample_submission.csv", index=False)
    spec = {"target_column": "response", "id_column": "id", "task_type": "regression", "metric": "mae"}
    monkeypatch.setenv("TYRELL_OUTPUT_DIR", str(tmp_path))
    monkeypatch.setenv("TYRELL_DIAGNOSTIC_ABLATIONS", "1")
    cfg = {"num_boost_round": 2}
    original = pipeline.run_experiment(tmp_path, spec, cfg, 20)
    path = tmp_path / "submission.csv"
    before = path.read_bytes()
    removed = pipeline.run_experiment(tmp_path, spec, cfg | {
        "diagnostic_group": "district", "exclude_columns": ["district"]}, 20)
    assert removed["cv_audit"]["membership_signature"] == original["cv_audit"]["membership_signature"]
    assert original["feature_coverage"] and not removed["saved"] and not removed["preds_file"]
    assert removed["diagnostic_group"] == "district" and path.read_bytes() == before


@pytest.mark.parametrize("test_offset, expected", [(3, 1.0), (0, None)])
def test_fallback_uses_only_published_durations(tmp_path, monkeypatch, test_offset, expected):
    start = pd.Timestamp("2025-01-01", tz="UTC")
    time = pd.date_range(start, periods=4, freq="min")
    frame = pd.DataFrame({"id": list("abcd"), "event_time": time, "duration": [1, 100, 100, 100]})
    frame.to_csv(tmp_path / "train.csv", index=False)
    pd.DataFrame({"id": ["test"], "event_time": [time[test_offset]]}).to_csv(tmp_path / "test.csv", index=False)
    pd.DataFrame({"id": ["test"], "duration": 0.}).to_csv(tmp_path / "sample_submission.csv", index=False)
    monkeypatch.setenv("TYRELL_OUTPUT_DIR", str(tmp_path))
    spec = {"id_column": "id", "target_column": "duration", "metric": "mae", "task_type": "regression",
            "time_column": "event_time", "label_delay_unit": "minutes"}
    if expected is None:
        with pytest.raises(ValueError, match="No eligible training labels"):
            pipeline.write_fallback(tmp_path, spec)
        assert not (tmp_path / "submission.csv").exists()
        return
    assert pipeline.write_fallback(tmp_path, spec) == []
    assert pd.read_csv(tmp_path / "submission.csv").duration.tolist() == [expected]
