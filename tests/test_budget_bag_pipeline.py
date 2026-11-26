import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import pipeline
import training


def test_bag_uses_shared_features_and_folds_and_saves_only_aggregate(tmp_path, monkeypatch):
    dates = pd.date_range("2025-01-01", periods=124, freq="h")
    frame = pd.DataFrame({"id": np.arange(124), "event_time": dates, "x": np.arange(124) % 11})
    frame.iloc[:120].assign(target=np.arange(120) % 11).to_csv(tmp_path / "train.csv", index=False)
    test = frame.iloc[120:]
    test.to_csv(tmp_path / "test.csv", index=False)
    test[["id"]].assign(target=0.).to_csv(tmp_path / "sample_submission.csv", index=False)
    spec = {"target_column": "target", "id_column": "id", "task_type": "regression", "metric": "mae",
            "time_column": "event_time", "label_delay_unit": "minutes"}
    monkeypatch.setenv("TYRELL_OUTPUT_DIR", str(tmp_path))
    members = {"count": 2, "model": "lightgbm", "batch_seed": 71, "first_seed": 71, "last_seed": 72}

    def bag(x, y, xt, task_type, metric, config, time_budget):
        assert len(x) == 120 and len(xt) == 4 and "event_time__hour" in x
        assert config["fold_seed"] == 42 and config["seed"] == 71 and config["n_folds"] == 5
        assert config["label_delay_unit"] == "minutes" and 0 < time_budget <= 20
        result = training.train_and_predict(x, y, xt, task_type, metric, config | {"n_jobs": 1}, time_budget)
        return result | {"members": members}

    monkeypatch.setattr(training, "train_budget_bag", bag, raising=False)
    report = pipeline.run_experiment(tmp_path, spec, {"auto_features": True, "seed": 71, "fold_seed": 71,
                                     "num_boost_round": 2}, 20, recipe="bag")
    assert report["complete"] and report["saved"] and report["members"] == members
    assert report["cv_audit"]["membership_signature"] and len(report["fold_scores"]) == 5
    assert list((tmp_path / pipeline.PREDS_DIR).glob("*.npz")) == [Path(report["preds_file"])]
    assert (tmp_path / "submission.csv").is_file()


@pytest.mark.parametrize("problem", ["partial", "different_rows", "different_training_membership", "absent_membership"])
def test_ineligible_bag_preserves_csv_and_cannot_enter_an_ensemble(tmp_path, monkeypatch, problem):
    monkeypatch.setenv("TYRELL_OUTPUT_DIR", str(tmp_path))
    csv_path = tmp_path / "submission.csv"
    csv_path.write_bytes(b"early prediction")
    folds = np.array([0, 0, 1, 1], dtype="<i4")
    prev_signature = hashlib.sha256(folds.tobytes()).hexdigest()
    history = []
    if problem != "partial":
        history.append({"saved": True, "complete": problem != "different_rows", "score": 10.,
                        "fold_signature": "old rows" if problem == "different_rows" else prev_signature,
                        "fold_scores": [10., 10.], "cv_audit": {"membership_signature": "old members"}})
    membership = {"different_training_membership": "other members", "absent_membership": None}.get(problem, "old members")
    monkeypatch.setattr(pipeline, "_cv_audit", lambda *args: {"membership_signature": membership})
    result = {"score": 1., "score_std": 0., "class_order": None, "fold_scores": [1., 1.], "fit_time": 1.,
              "greater_is_better": False, "fold_ids": folds,
              "config": {"n_folds": 2, "n_folds_done": 1 if problem == "partial" else 2}}
    report = pipeline._record(tmp_path, {"metric": "mae"}, {}, history, result, {"recipe": "bag"})
    assert not report["saved"] and report["preds_file"] is None
    assert csv_path.read_bytes() == b"early prediction"
