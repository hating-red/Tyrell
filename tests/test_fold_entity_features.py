import json

import numpy as np
import pandas as pd
import pytest

from features import add_fold_entity_history, temporal_entity_candidates
from pipeline import run_experiment
from training import train_and_predict


def test_history_has_strict_cutoff_and_frozen_query_batch():
    clock = pd.Timestamp("2025-06-01", tz="UTC")
    fit = pd.DataFrame({"entity_id": [7, 7, 8, 7, 7, None],
                        "when": clock + pd.to_timedelta([3, 1, 1, 0, 1, 1], unit="h")},
                       index=[8, 5, 0, 11, 2, 2])
    query = pd.DataFrame({"entity_id": [7, 8, 7, 7, 99, 7, 7],
                          "when": clock + pd.to_timedelta([2, 2, 1, 0, 2, 3, np.nan], unit="h")})
    past, held = add_fold_entity_history(fit, query, ["entity_id"], "when")
    np.testing.assert_array_equal(past.entity_id__past_events, [3, 1, 0, 0, 1, 0])
    np.testing.assert_array_equal(held.entity_id__past_events, [3, 1, 1, 0, 0, 3, np.nan])
    np.testing.assert_array_equal(held.entity_id__since_previous_s, [3600, 3600, 3600, np.nan, np.nan, 7200, np.nan])
    changed = query.sample(frac=1, random_state=6)
    _, shuffled = add_fold_entity_history(fit, changed, ["entity_id"], "when")
    pd.testing.assert_frame_equal(shuffled.sort_index(), held)
    assert list(fit.columns) == ["entity_id", "when"] and past.index.equals(fit.index)


def frame():
    n = 720
    rng = np.random.default_rng(19)
    ids = rng.integers(0, 17, n)
    dates = pd.date_range("2025-01-01", periods=n, freq="min", tz="UTC")
    return pd.DataFrame({"site_id": ids, "group_code": np.where(ids % 2, "X", "Y"),
                         "counter_id": np.arange(n), "date": dates})


def test_candidates_ignore_unique_ids_and_continuous_codes():
    data = frame()
    data["rate_code"] = data["site_id"] + .125
    candidates = temporal_entity_candidates(data, "date")
    assert "site_id" in candidates["categorical_ids"] and "rate_code" not in candidates["categorical_ids"]
    assert candidates["entity_history"][:2] == ["site_id", "group_code"]
    assert "counter_id" not in candidates["entity_history"]
    assert temporal_entity_candidates(data.assign(date=np.arange(len(data))), "date")["entity_history"] == []


@pytest.mark.parametrize("flag,added", [("entity_history", "site_id__past_events"), ("categorical_ids", "site_id")])
def test_pipeline_journal_preserves_folds_and_applied_features(tmp_path, monkeypatch, flag, added):
    data = frame()
    y = (data.site_id % 3 == 0).astype(int)
    data["event_id"] = np.arange(len(data)).astype(str)
    data.iloc[:-60].assign(target=y.iloc[:-60]).to_csv(tmp_path / "train.csv", index=False)
    data.iloc[-60:].to_csv(tmp_path / "test.csv", index=False)
    data.iloc[-60:][["event_id"]].assign(target=0).to_csv(tmp_path / "sample_submission.csv", index=False)
    monkeypatch.setenv("TYRELL_DIAGNOSTIC_ABLATIONS", "1")
    monkeypatch.delenv("TYRELL_OUTPUT_DIR", raising=False)
    spec = {"target_column": "target", "id_columns": ["event_id"], "time_column": "date",
            "task_type": "binary", "probabilities": True, "metric": "pr_auc", "positive_class": 1}
    config = {"params": {"n_estimators": 6}, "profile_categories": True}
    control = run_experiment(tmp_path, spec, config, time_budget=25)
    candidate = run_experiment(tmp_path, spec, config | {flag: True, "refit_full": True}, time_budget=25)
    assert candidate["complete"] and candidate["config"][flag]
    assert candidate["fold_windows"] == control["fold_windows"]
    assert candidate["fold_signature"] == control["fold_signature"]
    assert "site_id" in control["feature_candidates"][flag]
    assert candidate["model_inputs"][added]["model_dtype"] == ("category" if flag == "categorical_ids" else "float64")
    assert all(1 <= stat["best_iteration"] <= stat["last_iteration"] for stat in candidate["fold_diagnostics"])
    probes = control["category_probes"]["columns"]["site_id"]
    assert len(probes) == 5 and all(p["ap"] > p["baseline"] for p in probes)
    for probe, window, monitor in zip(probes, control["fold_windows"], control["early_stopping_windows"]):
        fit = data.site_id[data.date <= pd.to_datetime(monitor["fit_max"], utc=True)]
        rows = data.loc[data.date >= pd.to_datetime(window["validation_min"], utc=True), "site_id"].iloc[:window["validation_rows"]]
        delta = fit.value_counts(normalize=True).sub(rows.value_counts(normalize=True), fill_value=0.)
        assert probe["frequency_total_variation"] == pytest.approx(delta.abs().sum() / 2)
        assert probe["unseen_nonmissing_share"] == pytest.approx((~rows.isin(fit)).mean())
    assert len(json.loads((tmp_path / ".tyrell_experiments.json").read_text(encoding="utf-8"))) == 2


def test_selection_ignores_later_repetitions_and_test():
    data = frame()
    data["site_id"] = np.r_[np.arange(240) + 10000, np.zeros(len(data)-240)]
    y = pd.Series(np.arange(len(data)) % 2)
    cfg = {"n_folds": 3, "params": {"n_estimators": 5}, "split": {"type": "time", "column": "date"},
           "categorical_ids": True, "n_jobs": 1}
    with pytest.raises(ValueError, match="no eligible repeated IDs"):
        train_and_predict(data.iloc[:-60], y.iloc[:-60], data.iloc[-60:], "binary", "pr_auc", cfg)
