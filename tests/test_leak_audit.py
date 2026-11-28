import numpy as np
import pandas as pd
import pytest

from data_tools import _audit_lookup, _join_chunk, _plan_relation, _prepare_lookup, target_constraints
from features import audit_event_kernel, event_activity
from leak_audit import AuditBudget, LeakAuditError
from postgres_loader import ForeignKey, Table
from training import backend_available, train_and_predict


@pytest.mark.parametrize("mode", ["start", "end"])
def test_hourly_replay_at_strict_boundary_and_with_delayed_publication(mode, capsys):
    end = pd.to_datetime(["2025-01-01 09:00", "2025-01-01 10:00", "2025-01-01 11:00"])
    stamps = end - pd.Timedelta(hours=1) if mode == "start" else end
    rows = pd.DataFrame({"zone_code": "a", "ts": stamps, "density": [8., 9., 10.]})
    events = pd.DataFrame({"zone_code": "a", "called_at": [
        "2025-01-01 10:30", "2025-01-01 11:00", "2025-01-01 11:00:00.000001"]})
    lookup = _prepare_lookup(rows, "zone_code", "ts", mode, ["density"], "grid__")
    joined = _join_chunk(events, lookup, "zone_code", "called_at")
    assert joined["grid__density"].tolist() == [9., 9., 10.]
    assert joined["grid__density__lag1h"].tolist() == [9., 9., 10.]
    assert joined["grid__density__mean24h"].tolist() == [8.5, 8.5, 9.]
    _audit_lookup([(events, joined.drop(columns=events.columns))], lookup, "zone_code",
                  "called_at", AuditBudget(), "grid")
    assert "LEAK_AUDIT status=pass" in capsys.readouterr().err

    rows["release"] = end + pd.Timedelta(minutes=5)
    delayed = _prepare_lookup(rows, "zone_code", "ts", mode, ["density"], "grid__", "release")
    delayed_out = _join_chunk(events, delayed, "zone_code", "called_at")
    assert delayed_out["grid__density"].tolist() == [9., 9., 9.]
    assert np.isnan(delayed_out["grid__density__lag1h"].iloc[-1])
    _audit_lookup([(events, delayed_out.drop(columns=events.columns))], delayed, "zone_code",
                  "called_at", AuditBudget(), "grid_delayed")


def test_audit_rejects_future_counts():
    source = pd.DataFrame({"ts": pd.date_range("2025", periods=3, freq="h"), "client_key": ["a"] * 3})

    def leaking(data, ask, *_):
        return event_activity(data, ask, "ts", ["client_key"]).assign(future_count=len(data))

    with pytest.raises(LeakAuditError, match="future_count"):
        audit_event_kernel(leaking, source, source["ts"], source, "ts", "future_count")


def test_replay_checks_a_leak_at_a_middle_boundary():
    times = pd.date_range("2025", periods=7, freq="h")
    source = pd.DataFrame({"ts": times, "value": ["same"] * len(times)})

    def leaking(frame, query, *_):
        out = pd.Series("same", index=query.index)
        if times[2] in query["ts"].array:
            value = frame.loc[frame["ts"].eq(times[2]), "value"]
            if not value.empty:
                out.loc[query["ts"].eq(times[2])] = value.iloc[0]
        return out.to_frame("value")

    with pytest.raises(LeakAuditError, match="feature value"):
        audit_event_kernel(leaking, source, source["ts"], source, "ts", "middle_boundary")


@pytest.mark.parametrize("metric", ["roc_auc", "rmse"])
def test_private_renamed_schema_and_other_metrics(metric, capsys):
    kind = "binary" if metric == "roc_auc" else "regression"
    backend = next((name for name in ("lightgbm", "catboost", "xgboost") if backend_available(name)), None)
    if backend is None:
        pytest.skip("no boosting backend installed")
    core = ("event_key", "client_key", "zone_code", "ts")
    metadata = {
        "column_types": {"ts": "timestamp without time zone"},
        "foreign_keys": (ForeignKey(("client_key",), ("public", "clients"), ("client_key",)),),
    }
    train, test = Table("public", "archive", (*core, "delay_min"), **metadata), Table("public", "new_jobs", core, **metadata)
    clients = Table("public", "clients", ("client_key", "segment", "total_orders"), primary_key=("client_key",))
    readings = Table("public", "readings", ("zone_code", "ts", "load"), primary_key=("zone_code", "ts"),
                     column_types={"ts": "timestamp without time zone"}, comments={"ts": "start of hour"})
    assert _plan_relation(clients, train, test, "event_key", {"delay_min"}, "") == (
        "client_key", "client_key", None, None, None, ["segment"])
    assert _plan_relation(readings, train, test, "event_key", {"delay_min"}, "") == (
        "zone_code", "zone_code", "ts", "ts", "start", ["load"])
    n = 520
    rows = np.arange(n)
    data = pd.DataFrame({"ts": pd.date_range("2025", periods=n, freq="35min"),
                         "client_key": rows % 15, "zone_code": rows % 5, "signal": rows % 29 / 29.})
    gauges = pd.DataFrame({"ts": pd.date_range("2024-12-31", periods=460, freq="h").repeat(5),
                           "zone_code": np.tile(np.arange(5), 460), "load": np.arange(2300) % 37})
    lookup = _prepare_lookup(gauges, "zone_code", "ts", "start", ["load"], "readings__")
    joined = _join_chunk(data, lookup, "zone_code", "ts")
    _audit_lookup([(data.iloc[:15], joined.iloc[:15].drop(columns=data.columns)),
                   (data.iloc[-15:], joined.iloc[-15:].drop(columns=data.columns))],
                  lookup, "zone_code", "ts", AuditBudget(), "readings")
    y = (data.signal > .5).astype(int) if kind == "binary" else 3 + 5 * data.signal
    result = train_and_predict(joined.iloc[:460], y.iloc[:460], joined.iloc[460:], kind, metric, config={
        "model": backend, "split": {"type": "time", "column": "ts"}, "n_folds": 3, "n_jobs": 2,
        "num_boost_round": 15, "event_activity": True, "causal_history": True, "audit_required": True,
        "label_delay_unit": "minutes" if kind == "regression" else None,
    }, time_budget=20.)
    assert np.isfinite(result["score"])
    assert len(result["fold_scores"]) == 3
    logs = capsys.readouterr().err
    for group in ("readings", "event_activity", "causal_history", "model_input"):
        assert f'LEAK_AUDIT status=pass {{"group": "{group}"' in logs


def test_delay_needs_numeric_plausible_declared_duration():
    rule = "delay_min is the duration from registration to response, in minutes."
    assert target_constraints(rule, "delay_min", pd.Series([5, 10]), "regression").get("label_delay_unit") == "minutes"
    for y in ([0, 0], [-1, 5], [55000, 95000]):
        assert "label_delay_unit" not in target_constraints(rule, "delay_min", pd.Series(y), "regression")
    assert "label_delay_unit" not in target_constraints("delay_min is a sensor reading.", "delay_min",
                                                       pd.Series([5, 10]), "regression")


def test_ordinary_labels_after_first_test_event_cannot_train_or_score(monkeypatch):
    import training

    hours = pd.date_range("2025", periods=48, freq="h")
    X = pd.DataFrame({"ts": hours, "registered_index": np.arange(48)})
    test = pd.DataFrame({"ts": [hours[25]], "registered_index": [100]})
    seen = []

    def fit(Xtr, ytr, weights, Xva, yva, query, *_):
        seen.append(int(Xtr["registered_index"].max()))
        return np.full(len(Xva), .5), np.full(len(query), .5), 1, False, 1

    monkeypatch.setitem(training._BACKENDS, "lightgbm", fit)
    result = train_and_predict(X, X.registered_index, test, "regression", "rmse", config={
        "split": {"type": "time", "column": "ts"}, "audit_required": True, "n_folds": 3,
    }, time_budget=20.)
    assert max(seen) == 24
    assert (result["fold_ids"][25:] == -1).all()


def test_created_at_attributes_and_undated_profile_counts_are_conservative():
    core = ("call_id", "client_key", "ts")
    train = Table("public", "archive", (*core, "label"))
    test = Table("public", "new_jobs", core)
    profile = Table("public", "clients", ("client_key", "created_at", "category", "total_orders", "credit_score"))
    assert _plan_relation(profile, train, test, "call_id", {"label"}, "")[-1] == ["category"]
    raw = pd.DataFrame({"client_key": [1, 1], "category": ["a", "b"]})
    static = _prepare_lookup(raw, "client_key", None, None, ["category"], "client__")
    assert list(static) == ["client__category"]
    assert static["client__category"].isna().all()


def test_explicit_source_restriction_and_llm_whitelist():
    from types import SimpleNamespace
    from solution import hypothesis

    core = ("call_id", "zone_code", "ts")
    train = Table("public", "archive", (*core, "label"))
    test = Table("public", "new_jobs", core)
    readings = Table("public", "readings", ("zone_code", "ts", "load"), comments={"ts": "start of hour"})
    with pytest.raises(ValueError, match="task.md"):
        _plan_relation(readings, train, test, "call_id", {"label"},
                       "readings available only after the investigation ends.")

    spec = {"metric": "pr_auc", "available_backends": ["catboost"]}
    for proposal in ({"model": "catboost", "feature_code": "arbitrary"},
                     {"model": "lightgbm", "event_activity": True},
                     {"model": "catboost", "mae_shift": True}):
        model = SimpleNamespace(ask=lambda *_: proposal)
        assert hypothesis(model, "", spec, [], [], None) is None


@pytest.mark.parametrize(("frequency", "column", "sql_type", "primary_key", "event", "stamps"), [
    ("h", "measured_at", "timestamp without time zone", (), "2025-01-01 10:30",
     pd.date_range("2025-01-01 06:00", periods=5, freq="h")),
    ("D", "day", "date", (), "2025-01-05 10:30", pd.date_range("2025-01-01", periods=5, freq="D")),
    ("D", "day", "date", ("zone", "day"), "2025-01-05 10:30", pd.date_range("2025-01-01", periods=5, freq="D")),
])
def test_inferred_step_renamed_table(frequency, column, sql_type, primary_key, event, stamps):
    fields = ("event_id", "zone", "call_time")
    train = Table("public", "archive", (*fields, "target"))
    test = Table("public", "new_jobs", fields)
    gauge = Table("public", "observations", ("zone", column, "reading"),
                  column_types={column: sql_type}, primary_key=primary_key)
    assert _plan_relation(gauge, train, test, "event_id", {"target"}, "") == (
        "zone", "zone", "call_time", column, "inferred", ["reading"])
    source = pd.DataFrame({"zone": "a", column: stamps, "reading": range(5)})
    query = pd.DataFrame({"zone": "a", "call_time": [event, stamps[-1] + pd.Timedelta(1, unit=frequency),
                                                   stamps[-1] + pd.Timedelta(1, unit=frequency) + pd.Timedelta(microseconds=1)]})
    lookup = _prepare_lookup(source, "zone", column, "inferred", ["reading"], "obs__",
                             cadence_before=pd.Timestamp(event))
    assert lookup.attrs["inferred_step"] == pd.Timedelta(1, unit=frequency)
    joined = _join_chunk(query, lookup, "zone", "call_time")
    assert joined["obs__reading"].tolist() == [3, 3, 4]
    assert ("obs__reading__lag1h" in joined) == (frequency == "h")
    _audit_lookup([(query, joined.drop(columns=query.columns))], lookup, "zone", "call_time",
                  AuditBudget(), "observations")


def test_irregular_and_undetermined_cadence_is_rejected():
    from data_tools import _infer_interval

    clock = pd.Series(pd.DatetimeIndex(["2025-01-01", "2025-01-02", "2025-01-04"]))
    keys = pd.Series(["a", "a", "a"])
    with pytest.raises(ValueError, match="irregular_step share="):
        _infer_interval(keys, clock, pd.Timestamp("2025-01-06"))
    with pytest.raises(ValueError, match="неопределимый шаг"):
        _infer_interval(keys, clock, pd.Timestamp("2025-01-03"))


def test_inferred_cadence_accepts_missing_hour_and_counts_duplicate_timestamp():
    from data_tools import _infer_interval

    times = pd.date_range("2025-01-01", periods=20, freq="h").delete(9)
    stamps = pd.Series(times.insert(4, times[4]))
    diagnostics = {}
    step = _infer_interval(pd.Series("a", index=stamps.index), stamps, pd.Timestamp("2025-01-02"), diagnostics=diagnostics)
    assert step == pd.Timedelta(hours=1)
    assert diagnostics["duplicate_timestamps"] == 1


def test_inferred_audit_rejects_artificially_early_release():
    times = pd.date_range("2025-01-01", periods=4, freq="D")
    source = pd.DataFrame({"zone": "a", "day": times, "reading": 1})
    query = pd.DataFrame({"zone": ["a"], "ts": ["2025-01-04 10:30"]})
    lookup = _prepare_lookup(source, "zone", "day", "inferred", ["reading"], "g__", cadence_before=times[-1])
    lookup["__available"] -= pd.Timedelta(hours=23)
    joined = _join_chunk(query, lookup, "zone", "ts")
    with pytest.raises(LeakAuditError, match="source timestamp"):
        _audit_lookup([(query, joined[["g__reading"]])], lookup, "zone", "ts", AuditBudget(), "tampered")


def test_inferred_join_logs_duplicate_timestamps_and_irregular_reason(tmp_path, monkeypatch, capsys):
    from contextlib import nullcontext
    from types import SimpleNamespace
    import data_tools

    fields = ("event_id", "zone_id", "ts")
    train = Table("public", "archive", (*fields, "target"))
    test = Table("public", "new_jobs", fields)
    regular = Table("public", "measurements", ("zone_id", "measured_at", "reading"),
                    column_types={"measured_at": "timestamp without time zone"})
    irregular = Table("public", "drift_measurements", regular.columns, column_types=regular.column_types)
    times = pd.date_range("2025-01-01", periods=20, freq="h").delete(9)
    source = pd.DataFrame({"zone_id": "a", "measured_at": times, "reading": range(len(times))})
    noisy = source.copy()
    noisy.loc[2, "measured_at"] += pd.Timedelta(minutes=11)
    monkeypatch.setattr(data_tools, "_dictionary_clocks", lambda *a: {})
    monkeypatch.setattr(data_tools, "_read_lookup", lambda conn, table, *a, **kw:
                        pd.concat([source, source.iloc[[0]]]) if table is regular else noisy)
    for split in ("train", "test"):
        pd.DataFrame({"event_id": [split], "zone_id": ["a"], "ts": ["2025-01-02"]}).to_csv(
            tmp_path / f"{split}.csv", index=False)
    conn = SimpleNamespace(transaction=nullcontext)
    tables = {(table.schema, table.name): table for table in (train, test, regular, irregular)}
    data_tools.enrich_postgres(conn, tables, train, test, "event_id", "target", "", tmp_path,
                               replay=AuditBudget(), event_counts={"train": 1, "test": 1})
    logs = capsys.readouterr().out
    assert "mode=inferred reason=regular_step step=0 days 01:00:00 duplicate_timestamps=1" in logs
    assert "public.drift_measurements skipped reason=irregular_step share=" in logs
    result = pd.read_csv(tmp_path / "test.csv")
    assert result["measurements__reading"].tolist() == [18]
    assert "drift_measurements__reading" not in result


def test_adaptive_audit_retries_each_group_with_its_own_budget(monkeypatch, capsys):
    import leak_audit

    now = [0.]
    monkeypatch.setattr(leak_audit.time, "monotonic", lambda: now[0])
    budget = AuditBudget()
    calls = []

    def verify(counts, check):
        calls.append(counts)
        now[0] += 34 if counts[0] == 2000 else 15 if counts[0] == 500 else 1
        check()
        return {"features": ["x"]}

    for group in ("first", "second"):
        budget.sampled(group, [2500, 2200], verify)
    assert calls == [(2000, 2000), (500, 500), (200, 200)] * 2
    stderr = capsys.readouterr().err
    assert stderr.count('"sample_sizes": [200, 200]') == 2
    assert "status=fail" not in stderr

def test_ambiguous_repeated_measurement_without_history_is_rejected():
    from data_tools import _reference_mode

    gauge = Table("public", "gauge", ("zone", "tick_at", "reading"),
                  column_types={"tick_at": "timestamp without time zone"})
    source = pd.DataFrame({"zone": ["a", "a"], "tick_at": ["2025-01-01", "2025-01-02"],
                           "reading": [1, 2]})
    with pytest.raises(ValueError, match="undetermined_step"):
        _reference_mode(gauge, source, "zone", "tick_at", "inferred", pd.Timestamp("2025-01-05"),
                        None, ["reading"])
