from contextlib import nullcontext
from unittest.mock import Mock

import numpy as np
import pandas as pd
import pytest

import pipeline
import solution
import training
from data_tools import _join_chunk, _prepare_lookup, _verify_lookup, enrich_postgres, guess_task_spec
from features import event_calendar, preprocess_fold, since_positive_drift, causal_target_history
from leak_audit import LeakAuditError
from postgres_loader import Table


@pytest.mark.parametrize("shift", [0, 15])
def test_hourly_lags_use_strict_ends_and_independent_audit(shift):
    stamps = pd.date_range("2024-01-01", periods=200, freq="h") + pd.Timedelta(minutes=shift)
    source = pd.DataFrame({"key": "site", "stamp": stamps, "value": np.arange(len(stamps), dtype=float)})
    end = stamps[-1] + pd.Timedelta(hours=1)
    events = pd.DataFrame({"key": ["site"] * 3, "event": [end, end + pd.Timedelta(minutes=30), pd.NaT]})
    lookup = _prepare_lookup(source, "key", "stamp", "inferred", ["value"], "src__",
                             cadence_before=end)
    joined = _join_chunk(events, lookup, "key", "event")
    features = joined.drop(columns=list(events))
    expected = {"lag1h": 199., "lag24h": 176., "lag168h": 32., "dev24h": 23., "dev168h": 167.,
                "delta1_2h": 1., "min6h": 194., "max6h": 199., "std6h": np.arange(6).std()}
    for name, value in expected.items():
        assert features.loc[1, "src__value__" + name] == pytest.approx(value)
    assert features.loc[0, "src__value__lag1h"] == 198.
    assert features.iloc[2].isna().all()
    assert "independent_interval_lags_and_rollups" in _verify_lookup(events, features, lookup, "key", "event", lambda: None)["checks"]
    damaged = features.copy()
    damaged.loc[1, "src__value__lag168h"] += 1
    with pytest.raises(LeakAuditError, match="lag168h"):
        _verify_lookup(events, damaged, lookup, "key", "event", lambda: None)


def test_nonhourly_lags_and_means_respect_missing_and_unpublished_slots():
    step = pd.Timedelta(days=1)
    stamps = pd.date_range("2024-01-01 08:00", periods=12, freq="D")
    source = pd.DataFrame({"key": "site", "stamp": stamps, "value": np.arange(12.)})
    source["release"] = stamps + step
    event_time = stamps[-1] + step + pd.Timedelta(minutes=30)
    source.loc[11, "release"] = event_time
    source = source.drop(index=9)
    events = pd.DataFrame({"key": ["site", "site"], "event": [event_time, event_time + pd.Timedelta(minutes=1)]})
    lookup = _prepare_lookup(source, "key", "stamp", "inferred", ["value"], "src__",
                             publication="release", cadence_before=event_time)
    joined = _join_chunk(events, lookup, "key", "event").drop(columns=list(events))
    assert not any(col.endswith("h") for col in joined)
    np.testing.assert_allclose(joined["src__value__lag1_step"], [np.nan, 11.], equal_nan=True)
    np.testing.assert_allclose(joined["src__value__lag2_step"], [10., 10.])
    np.testing.assert_allclose(joined["src__value__mean3_step"], [10., 10.5])
    np.testing.assert_allclose(joined["src__value__mean7_step"], [np.mean([5, 6, 7, 8, 10]), np.mean([5, 6, 7, 8, 10, 11])])
    _verify_lookup(events, joined, lookup, "key", "event", lambda: None)


def test_calendar_uses_row_clock_and_wraps_at_week_boundary():
    times = pd.Series(pd.to_datetime(["2024-02-04 23:30", "2024-02-05 00:00", None]))
    out = event_calendar(times)
    assert out.loc[0, "__event_weekend"] == 1 and out.loc[1, "__event_weekend"] == 0
    assert out.loc[0, "__event_minute"] == 30 and out.loc[1, "__event_weekday"] == 0
    assert out.loc[0, "__event_hour_sin"] == pytest.approx(np.sin(2 * np.pi * 23.5 / 24))
    assert out.loc[1, "__event_weekday_cos"] == 1 and out.iloc[2].isna().all()


def test_preprocessing_uses_only_fit_and_dates_strictly_before_event(capsys):
    size = 20
    clock = pd.date_range("2024-01-01", periods=size, freq="h")
    date = clock - pd.to_timedelta(np.arange(size) + 1, unit="h")
    fit = pd.DataFrame({"event": clock, "source_date": date.astype(str), "const": 3.,
                        "num": [str(i) for i in range(size - 1)] + ["bad"],
                        "dup": [float(i) for i in range(size - 1)] + [np.nan],
                        "bool": ["yes", "false"] * 10, "inf": [np.inf, -np.inf] + list(np.arange(18.))})
    query = fit.iloc[:3].copy()
    query["num"] = ["99", "bad", "98"]
    query["dup"] = [1e8, -1e8, 7.]
    query["source_date"] = [clock[0], clock[1] - pd.Timedelta(seconds=1), clock[2] + pd.Timedelta(seconds=1)]
    X, q = preprocess_fold(fit, query, "event", audit=True)
    replay, _ = preprocess_fold(fit, query.iloc[:0], "event")
    pd.testing.assert_frame_equal(X, replay)
    assert not {"event", "source_date", "const", "dup"}.intersection(X)
    assert X["bool"].iloc[:2].tolist() == [1., 0.]
    assert q["num"].iloc[[0, 2]].tolist() == [99., 98.]
    assert q["num__isna"].tolist() == [0., 1., 0.]
    assert X["inf"].iloc[:2].isna().all()
    np.testing.assert_allclose(q["source_date__age_s"], [np.nan, 1., np.nan])
    log = capsys.readouterr().err
    assert 'LEAK_AUDIT status=pass {"group": "preprocess_date_age"' in log and "PREPROC step=fit" in log


def test_annotated_lookup_dates_reach_only_the_fold_age_transform(tmp_path, monkeypatch):
    train_table = Table("s", "train", ("id", "region_key", "event_time", "response"))
    test_table = Table("s", "test", ("id", "region_key", "event_time"))
    reference = Table("s", "registry", ("region_key", "opened_date", "closed_date", "zone"),
                      column_types={"opened_date": "date", "closed_date": "date"},
                      comments={"opened_date": "Known before the event.", "closed_date": "Known after the event."})
    stamps = pd.date_range("2025-02-01", periods=7, freq="D")
    events = pd.DataFrame({"id": range(7), "region_key": ["A"] * 4 + ["B"] * 3, "event_time": stamps})
    events.iloc[:4].assign(response=range(4)).to_csv(tmp_path / "train.csv", index=False)
    events.iloc[4:].to_csv(tmp_path / "test.csv", index=False)
    events.iloc[4:][["id"]].assign(response=0).to_csv(tmp_path / "sample_submission.csv", index=False)
    names = []

    def read_lookup(conn, table, columns, deadline, **kwargs):
        names.extend(columns)
        return pd.DataFrame({"region_key": ["A", "B"], "opened_date": ["2025-01-31", "2025-02-06"],
                             "zone": [1, 2]})

    conn = Mock()
    conn.transaction.side_effect = nullcontext
    monkeypatch.setattr("data_tools._read_lookup", read_lookup)
    enrich_postgres(conn, {("s", "registry"): reference}, train_table, test_table, "id", "response", "", tmp_path)
    assert "opened_date" in names and "closed_date" not in names
    spec = {"id_columns": ["id"], "target_column": "response", "metric": "mae", "task_type": "regression",
            "audit_required": True, "time_column": "event_time"}
    base, _, _ = pipeline.prepare_features(tmp_path, spec, {"auto_features": True})
    assert not any(name.startswith("registry__opened_date") for name in base)
    fit, held, _ = pipeline.prepare_features(tmp_path, spec, {"auto_features": True, "fold_preprocess": True})
    assert [name for name in fit if name.startswith("registry__opened_date")] == ["registry__opened_date__reference_date"]
    _, transformed = preprocess_fold(fit, held, "event_time", audit=True)
    assert [name for name in transformed if name.startswith("registry__opened_date")] == ["registry__opened_date__age_s"]
    np.testing.assert_allclose(transformed["registry__opened_date__age_s"], [np.nan, np.nan, 86400.])


def test_sqrt_predicts_and_scores_in_original_units(monkeypatch):
    fit = pd.DataFrame({"x": np.arange(90, dtype=float)})
    test = pd.DataFrame({"x": [7., 13.]})
    y = fit["x"] ** 2

    def mock_backend(Xtr, ytr, weight, Xva, yva, Xte, *args):
        np.testing.assert_array_equal(ytr, Xtr["x"])
        return Xva["x"].to_numpy(), Xte["x"].to_numpy(), 1, False, 1

    monkeypatch.setitem(training._BACKENDS, "lightgbm", mock_backend)
    result = training.train_and_predict(fit, y, test, "regression", "mae", {"sqrt_target": True}, 10.)
    np.testing.assert_array_equal(result["predictions"], [49., 169.])
    assert result["score"] == 0.


def test_variant_losses_share_full_feature_config_and_history_candidates():
    choices = {"available_backends": ["lightgbm"], "audit_required": True}
    available = {"feature_candidates": dict.fromkeys(("hourly_extended", "event_activity", "causal_history",
                                                      "interval_features", "event_calendar", "sqrt_target"), True)}
    bundle = dict.fromkeys(("auto_features", "hourly_extended", "event_activity", "causal_history"), True)
    base = bundle | {"model": "lightgbm"}
    losses = solution.choose_variant(choices | {"metric": "mae"}, available)
    for config in ({}, {"mae_objective": "huber"}, {"log_target": True}, {"sqrt_target": True}):
        assert base | config in losses
    ap = solution.choose_variant(choices | {"metric": "pr_auc"}, available)
    assert base | {"regularized_ap": True, "es_metric": "logloss"} in ap
    assert base | {"history_since_positive": False} in ap
    assert all(base | {"history_smoothing": m} in ap for m in (5., 30.))
    assert solution._trial_key(base | {"history_since_positive": False}) != solution._trial_key(base)
    admitted = {"interval_features", "event_calendar"}
    enriched = base | dict.fromkeys(admitted, True)
    losses = solution._loss_variants(choices | {"metric": "mae"}, available, admitted)
    assert enriched in losses
    assert all(enriched | config in losses for config in ({"mae_objective": "huber"},
                                                          {"log_target": True}, {"sqrt_target": True}))


def test_frozen_history_drift_and_since_positive_ablation():
    source = pd.DataFrame({"key": "k", "event": pd.date_range("2024", periods=6, freq="D")})
    query = pd.DataFrame({"key": "k", "event": pd.date_range("2024-01-07", periods=8, freq="D")})
    options = {"window": 4, "smoothing": 5}
    labels = np.array([0, 0, 1, 0, 0, 0])
    full = causal_target_history(source, query, "event", source["event"], labels, ["key"], **options)
    removed = causal_target_history(source, query, "event", source["event"], labels, ["key"],
                                    since_positive=False, **options)
    pd.testing.assert_frame_equal(full.drop(columns="__history_key_since_positive"), removed)
    drift = since_positive_drift(query, full, "event", ["key"])["key"]
    assert drift["nondecreasing_share"] == 1
    assert drift["late_median_s"] - drift["early_median_s"] == 4 * 86400


def test_quality_rejection_keeps_oof_without_ensemble_eligibility(tmp_path, monkeypatch):
    size = 90
    train = pd.DataFrame({"id": np.arange(size), "x": np.arange(size, dtype=float),
                          "y": np.sin(np.arange(size) / 10)})
    test = pd.DataFrame({"id": [size, size + 1], "x": [0., 3.]})
    train.to_csv(tmp_path / "train.csv", index=False)
    test.to_csv(tmp_path / "test.csv", index=False)
    test[["id"]].assign(y=0).to_csv(tmp_path / "sample_submission.csv", index=False)
    spec, _ = guess_task_spec(tmp_path, "MAE")
    spec["audit_required"] = True
    archive = tmp_path / "archive"
    archive.mkdir()
    monkeypatch.setenv("TYRELL_OOF_RUN_DIR", str(archive))
    cfg = {"num_boost_round": 3, "early_stopping_rounds": 2}
    control = pipeline.run_experiment(tmp_path, spec, cfg, 15.)
    candidate = pipeline.run_experiment(tmp_path, spec, cfg | {"fold_preprocess": True}, 15.)
    assert control["saved"] and candidate["quality_flags_pass"] is False
    assert len(list((archive / pipeline.PREDS_DIR).glob("*.npz"))) == 2
    assert (archive / "experiments.json").is_file()
    with pytest.raises(ValueError):
        pipeline.run_ensemble(tmp_path, spec, time_budget=10.)
