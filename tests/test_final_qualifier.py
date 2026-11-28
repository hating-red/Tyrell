import numpy as np
import pandas as pd
import pytest
import sys
from pathlib import Path
from threading import Event, Thread
from time import monotonic

import pipeline
import solution
from data_tools import _bridge_routes, _hour_clock, _join_chunk, _plan_relation, _prepare_lookup
from features import add_ad_history, ad_history_keys, target_encoding_keys
from postgres_loader import ForeignKey, Table
from publication import MODEL_SUCCESS, publication_run, published_result
from submission import make_submission
from solution import ResearchModel, parse_model_object, serve_research_model
from training import _past_encoding, train_and_predict


def test_duplicate_threshold_includes_split_column():
    train = pd.DataFrame({"x": [0., 2.], "period": [3, 4]})
    test = pd.DataFrame({"x": [-0.] * 1000, "period": [3] * 50 + [9] * 950})
    assert target_encoding_keys(train, test, exclude=["period"]) == {"duplicate": ["x", "period"]}
    test.loc[0, "period"] = 9
    assert "duplicate" not in target_encoding_keys(train, test, exclude=["period"])


@pytest.mark.parametrize("name", ["weather", "traffic", "air_quality", "power_grid", "public_transport"])
def test_public_hourly_clock_uses_completed_measurements(name):
    table = Table("public", name, ("district_id", "timestamp", "reading"))
    clock, mode = _hour_clock(table, f"{name}.timestamp is the start of the hour")
    assert (clock, mode) == ("timestamp", "start")
    assert _hour_clock(table, "Private task") == ("timestamp", "inferred")
    source = pd.DataFrame({"district_id": ["A", "A", "B"],
                           "timestamp": ["2025-01-01 09:00", "2025-01-01 10:00", "2025-01-01 09:00"],
                           "reading": [9, 10, 90]})
    lookup = _prepare_lookup(source, "district_id", clock, mode, ["reading"], "hour_")
    calls = pd.DataFrame({"district_id": ["B", "A", "A", "A", "A"],
                          "call": ["2025-01-01 10:30", "2025-01-01 10:30", "2025-01-01 09:30",
                                   "2025-01-01T16:00:00+05:00", "2025-01-01 11:00:00.000000001"]})
    out = _join_chunk(calls, lookup, "district_id", "call")
    np.testing.assert_allclose(out["hour_reading"], [90, 9, np.nan, 9, 10])
    np.testing.assert_allclose(out["hour_reading__lag1h"], [90, 9, np.nan, 9, 10])
    assert np.isnan(out.loc[3, "hour_reading__lag2h"])
    assert out.loc[3, "hour_reading__mean3h"] == 9
    assert out.loc[4, "hour_reading__mean3h"] == 9.5
    conflict = source.iloc[[0]].assign(timestamp="2025-01-01 09:00:00", reading=-1)
    rows = pd.concat([source, source.iloc[[2]], conflict], ignore_index=True)
    lookup = _prepare_lookup(rows, "district_id", clock, mode, ["reading"], "hour_", coalesce=True)
    calls.loc[5] = ["B", "2025-01-01 12:30"]
    out = _join_chunk(calls, lookup, "district_id", "call")
    np.testing.assert_allclose(out["hour_reading"], [90, np.nan, np.nan, np.nan, 10, 90])


def test_profile_versions_use_strict_past_and_preserve_order():
    profiles = pd.DataFrame({"client_key": ["x", "x", "y"], "created_at": [
        "2025-01-01 09:00:00Z", "2025-01-01 10:00:00", "2025-01-01 14:00:00+05:00"],
        "color": ["a", "b", "c"]})
    lookup = _prepare_lookup(profiles, "client_key", "created_at", "profile", ["color"], "profile_")
    events = pd.DataFrame({"client_key":["y", "x", "x"], "when":[
        "2025-01-01 09:30", "2025-01-01 10:00", "2025-01-01 10:01"]}, index=[9, 6, 2])
    out = _join_chunk(events, lookup, "client_key", "when")
    assert out.index.equals(events.index)
    assert out["profile_color"].tolist() == ["c", "a", "b"]
    assert out["profile_profile_count"].tolist() == [1, 1, 2]


def test_non_aligned_hour_needs_full_hour_to_finish():
    observations = pd.DataFrame({"zone": ["a", "a"], "timestamp": [
        "2026-01-01 08:00:00", "2026-01-01 09:15:00"], "value": [8., 9.]})
    visits = pd.DataFrame({"zone": ["a", "a", "a"], "visit": [
        "2026-01-01 10:01", "2026-01-01 10:15", "2026-01-01 10:16"]})
    lookup = _prepare_lookup(observations, "zone", "timestamp", "start", ["value"], "hour_")
    assert _join_chunk(visits, lookup, "zone", "visit")["hour_value"].tolist() == [8., 8., 9.]


def test_private_composite_clock_maps_between_postgres_timestamp_types():
    train = Table("public", "archived_jobs", ("job_id", "line_key", "opened", "minutes"),
                  column_types={"opened": "timestamp with time zone"})
    test = Table("public", "current_jobs", ("job_id", "line_key", "opened"),
                 column_types={"opened": "timestamp with time zone"})
    lookup = Table("public", "measurements", ("line_key", "bucket_clock", "density"),
                   column_types={"bucket_clock": "timestamp without time zone"},
                   primary_key=("line_key", "bucket_clock"))
    assert _plan_relation(lookup, train, test, "job_id", {"minutes"}, "") == (
        "line_key", "line_key", "opened", "bucket_clock", "inferred", ["density"],
    )


def test_hourly_source_uses_documented_district_bridge():
    train = Table("public", "train", ("event_id", "district_id", "timestamp", "label"))
    test = Table("public", "test", ("event_id", "district_id", "timestamp"))
    districts = Table("public", "districts", ("district_id", "power_zone_id"), primary_key=("district_id",))
    observations = Table("public", "power_grid", ("area_key", "timestamp", "load"),
                         primary_key=("area_key", "timestamp"),
                         foreign_keys=(ForeignKey(("area_key",), ("public", "districts"), ("power_zone_id",)),))
    bridges = _bridge_routes(observations, [(districts, ["power_zone_id"])])
    assert _plan_relation(observations, train, test, "event_id", {"label"},
                          "power_grid.timestamp is the start of the hour", bridges=bridges) == (
        "districts__power_zone_id", "area_key", "timestamp", "timestamp", "start", ["load"],
    )


def test_event_log_does_not_become_a_static_dimension():
    train = Table("public", "train", ("event_id", "district_id", "timestamp", "label"))
    test = Table("public", "test", ("event_id", "district_id", "timestamp"))
    calendar = Table("public", "city_events", ("district_id", "date", "actual_attendance"))
    with pytest.raises(ValueError, match="статический JOIN недопустим"):
        _plan_relation(calendar, train, test, "event_id", {"label"}, "")


def test_llm_retries_without_thinking_parameter(tmp_path, monkeypatch):
    import json
    import httpx
    from types import SimpleNamespace
    import openai

    calls = []

    def create(**parameters):
        calls.append(parameters)
        if "extra_body" in parameters:
            response = httpx.Response(400, request=httpx.Request("POST", "http://example.invalid"))
            raise openai.BadRequestError("unsupported chat_template_kwargs", response=response, body=None)
        message = SimpleNamespace(content=json.dumps({"model": "lightgbm"}))
        return SimpleNamespace(choices=[SimpleNamespace(message=message)])

    class Client:
        def __init__(self, **options):
            assert options["timeout"] <= 90
            self.chat = SimpleNamespace(completions=SimpleNamespace(create=create))

        def __enter__(self):
            return self

        def __exit__(self, *_):
            pass

    monkeypatch.setattr(openai, "OpenAI", Client)
    monkeypatch.setenv("TYRELL_LLM", "live")
    monkeypatch.setenv("TYRELL_LLM_CACHE", str(tmp_path / "cache"))
    monkeypatch.setenv("OPENAI_BASE_URL", "http://example.invalid")
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("OPENAI_MODEL", "test-model")
    stop = Event()
    broker = Thread(target=serve_research_model, args=(tmp_path, stop))
    broker.start()
    try:
        model = ResearchModel(tmp_path)
        assert model.ask("hypothesis", "suggest a model", {}, monotonic() + 10) == {"model": "lightgbm"}
    finally:
        stop.set()
        broker.join(timeout=3)
    assert len(calls) == 2 and all(call["max_tokens"] == 400 for call in calls)
    assert calls[0]["extra_body"] == {"chat_template_kwargs": {"enable_thinking": False}}
    assert "extra_body" not in calls[1]


def test_platform_deadline_and_worker_environment(tmp_path, monkeypatch):
    task = tmp_path / "task.md"
    task.write_text("Primary metric: MAE", encoding="utf-8")
    (tmp_path / "sample_submission.csv").write_text("id,y\n0002,0\n0001,0\n", encoding="utf-8")
    monkeypatch.setattr(sys, "argv", ["solution.py", str(task)])
    monkeypatch.setenv("TYRELL_TIME_LIMIT", "7200")
    preserved = {"PYTHONPATH": "retained-python-path", "LD_LIBRARY_PATH": "retained-library-path"}
    for name, value in preserved.items():
        monkeypatch.setenv(name, value)
    monkeypatch.setenv("OPENAI_API_KEY", "not-in-worker")
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
    started = solution.PROCESS_STARTED

    def check(command, deadline, *, env):
        assert float(command[command.index("--worker-deadline") + 1]) == started + 6470
        assert deadline == started + 6500
        assert all(env[name] == value for name, value in preserved.items())
        assert not any(name.startswith("OPENAI_") for name in env)
        return 124

    monkeypatch.setattr(solution, "supervised_exit", check)
    assert solution.main() == 0
    assert (tmp_path / "submission.csv").read_text(encoding="utf-8").splitlines() == [
        "id,y", "0002,1.0", "0001,1.0",
    ]


def test_temporal_encoding_waits_for_label_availability():
    observed = pd.Series([10, 10, 12])
    values = np.array([[0.], [2.], [8.]])
    encoded = _past_encoding([7, 7, 7], observed, values, [7, 7, 7, 8], pd.Series([10, 11, 13, 20]))
    np.testing.assert_allclose(encoded[:, 0], [np.nan, 1., 10./3, np.nan])
    np.testing.assert_allclose(encoded[:, 1], np.log1p([0, 2, 3, 0]))


def test_later_labels_cannot_change_first_window_predictions():
    frame = pd.DataFrame({"event_time": pd.date_range("2025-01-01", periods=90, freq="h"),
                          "value": np.arange(90) % 5, "__te_entity": np.arange(90) % 3})
    test = frame.iloc[-3:].assign(event_time=pd.date_range("2025-02-01", periods=3, freq="h"))
    cfg = {"n_jobs": 1, "n_folds": 3, "num_boost_round": 4, "target_encoding": ["__te_entity"]}
    y = pd.Series(np.arange(90) % 7, dtype=float)
    first = train_and_predict(frame, y, test, "regression", "mae", cfg)
    later = y.copy()
    later[first["fold_ids"] > 0] += 100
    second = train_and_predict(frame, later, test, "regression", "mae", cfg)
    mask = first["fold_ids"] == 0
    assert mask.any() and "__te_entity__te0" in first["model_inputs"]
    np.testing.assert_allclose(first["oof"][mask], second["oof"][mask])


def test_sample_provides_order_and_preserves_fractional_answers(tmp_path):
    sample = tmp_path / "sample_submission.csv"
    sample.write_text("engaged,event_id\n0,002\n0,001\n", encoding="utf-8")
    output = tmp_path / "submission.csv"
    ids = pd.Series(["001", "002"], name="event_id")
    assert make_submission([.25, .75], ids, sample, output, probabilities=True) == []
    assert output.read_text(encoding="utf-8").splitlines() == ["engaged,event_id", "0.75,002", "0.25,001"]


def test_failed_later_trial_preserves_published_model(tmp_path, monkeypatch):
    sample = tmp_path / "sample_submission.csv"
    sample.write_text("id,y\n002,-999\n001,-999\n", encoding="utf-8")
    output = tmp_path / "submission.csv"
    monkeypatch.setenv("TYRELL_OUTPUT_DIR", str(tmp_path))
    data = {"test_ids": pd.Series(["001", "002"], name="id"), "sample_path": sample}
    spec = {"task_type": "regression", "metric": "mae"}
    result = dict(score=2., score_std=0., greater_is_better=False, config={"n_folds": 2},
                  fold_ids=[0, 1], fold_scores=[2., 2.], oof=[1., 2.], fit_time=1.,
                  class_order=None, predictions=[1., 2.], probabilities=None)
    with publication_run():
        history = []
        pipeline._record(tmp_path, spec, data, history, result, {})
        old = output.read_bytes()
        better = result | {"score": 1., "fold_scores": [1., 1.], "predictions": [1.5, 2.5]}
        replace = Path.replace
        def fail_publication(path, target):
            if target == output:
                raise OSError("simulated write failure")
            return replace(path, target)
        with monkeypatch.context() as patch:
            patch.setattr(Path, "replace", fail_publication)
            with pytest.raises(OSError, match="simulated"):
                pipeline._record(tmp_path, spec, data, history, better, {})
        assert output.read_bytes() == old and published_result(output)["state"] == MODEL_SUCCESS
        assert [h["score"] for h in pipeline.experiment_history(tmp_path)] == [2.]
        def interrupt_after_publication(path, target):
            if path.suffix == ".pending":
                raise OSError("simulated interruption")
            return replace(path, target)
        with monkeypatch.context() as patch:
            patch.setattr(Path, "replace", interrupt_after_publication)
            with pytest.raises(OSError, match="simulated"):
                pipeline._record(tmp_path, spec, data, history, better, {})
        assert published_result(output)["score"] == 1.
        assert [h["score"] for h in pipeline.experiment_history(tmp_path)] == [2., 1.]


def test_ensemble_must_improve_majority_above_noise(tmp_path, monkeypatch):
    (tmp_path / "sample_submission.csv").write_text("id,y\na,0\nb,0\n", encoding="utf-8")
    monkeypatch.setenv("TYRELL_OUTPUT_DIR", str(tmp_path))
    data = {"test_ids": pd.Series(["a", "b"], name="id"), "sample_path": tmp_path / "sample_submission.csv"}
    spec = {"task_type": "regression", "metric": "mae"}
    history = []
    result = {
        "score": 1.96, "score_std": 0., "greater_is_better": False,
        "config": {"n_folds": 5, "fold_baselines": [{"score": 2.} for _ in range(5)]},
        "fold_ids": list(range(5)), "oof": [1., 2., 3., 4., 5.], "fit_time": 1.,
        "class_order": None, "predictions": [1., 2.], "probabilities": None,
        "fold_scores": [2.01, 2.01, 1.9, 1.9, 1.98],
    }
    pipeline._record(tmp_path, spec, data, history, result, {"recipe": "single"})
    almost = result | {"score": 1.932, "fold_scores": [1.96, 1.96, 1.91, 1.91, 1.92]}
    report = pipeline._record(tmp_path, spec, data, history, almost, {"recipe": "ensemble"})
    assert report["saved"] and report["vs_saved"]["n_better"] == 3
    weak = result | {"score": 1.62, "fold_scores": [2.01, 1.6, 1.5, 1.5, 1.5]}
    report = pipeline._record(tmp_path, spec, data, history, weak, {"recipe": "single"})
    assert not report["saved"]


def test_ad_history_uses_only_strict_fit_events_and_recent_labels():
    events = pd.DataFrame({
        "user_id": ["a", "a", "a", "a"], "ad_id": ["x", "y", "x", "x"],
        "ads__campaign_id": ["c", "c", "c", "d"], "ads__ad_type": ["video", "text", "video", "video"],
        "clock": pd.to_datetime(["2026-01-01 00:00", "2026-01-01 00:10", "2026-01-01 00:20", "2026-01-01 00:20"]),
    })
    query = events.iloc[[0, 0, 0, 0]].assign(
        clock=pd.to_datetime(["2026-01-01 00:10", "2026-01-01 00:20", "2026-01-01 00:30", None]))
    groups = ad_history_keys(events.columns)
    fit, actual = add_ad_history(events, query, [0, 1, 1, 0], "clock", groups, window=2, smoothing=20)
    np.testing.assert_allclose(actual["__adh_user_events"], [1, 2, 4, 0])
    np.testing.assert_allclose(actual["__adh_user_ad_type_events"], [1, 1, 3, 0])
    np.testing.assert_allclose(actual["__adh_user_campaign_events"], [1, 2, 3, 0])
    np.testing.assert_allclose(actual["__adh_user_last_2_rate"], [0, .5, .5, np.nan])
    np.testing.assert_allclose(actual["__adh_user_ad_past_impressions"], [1, 1, 2, 0])
    np.testing.assert_allclose(actual["__adh_user_seconds_since_engaged"], [np.nan, 600, 600, np.nan])
    np.testing.assert_allclose(fit["__adh_user_events"], [0, 1, 2, 2])
    assert np.isnan(actual["__adh_user_rate"].iloc[-1])


def test_model_answer_parser_accepts_fenced_json_after_thinking():
    fence = chr(96) * 3
    text = '<think>{"model":"ignored"}</think>\n' + fence + 'json\n{"model":"xgboost"}\n' + fence
    assert parse_model_object(text) == {"model": "xgboost"}


def test_trial_fit_includes_thirty_percent_safety_margin(monkeypatch):
    monkeypatch.setattr(solution.time, "monotonic", lambda: 1000.)
    assert not solution.trial_fits(100., 1130.)
    assert solution.trial_fits(100., 1140.)


@pytest.mark.parametrize("model", ["catboost", "xgboost"])
def test_optuna_tuning_sees_only_prefix(model, monkeypatch):
    import training

    clock = pd.date_range("2025", periods=120, freq="h")
    X = pd.DataFrame({"ts": clock, "feature": np.arange(len(clock))})
    y = pd.Series(np.arange(len(clock), dtype=float))
    seen = []

    def train(inner, labels, probe, task_type, metric, config, time_budget):
        seen.append(config["params"])
        assert inner["ts"].max() == clock[19]
        assert labels.max() == 19
        assert probe["ts"].iloc[0] == clock[20]
        assert config["model"] == model and config["n_folds"] == 3
        return {"config": {"n_folds_done": 3, "n_folds": 3}, "score": 1.}

    monkeypatch.setattr(training, "train_and_predict", train)
    params = training.tune_backend(model, X, y, X.iloc[[-1]], "regression", "mae",
                                   {"split": {"type": "time", "column": "ts"}}, 30.)
    assert seen and params in seen


def test_tune_recipe_and_outer_experiment_use_identical_features(tmp_path, monkeypatch, capsys):
    import training

    backend = next((name for name in ("catboost", "xgboost", "lightgbm") if training.backend_available(name)), None)
    if backend is None:
        pytest.skip("no available backend")
    clock = pd.date_range("2025", periods=420, freq="h")
    frame = pd.DataFrame({"ts": clock, "x": np.sin(np.arange(len(clock)) / 4),
                          "measurements__x__mean24h": np.arange(len(clock)) / 8})
    spec = {"task_type": "regression", "metric": "mae", "audit_required": True, "time_column": "ts"}
    config = {"model": backend, "auto_features": True, "hourly_extended": True, "n_folds": 99,
              "num_boost_round": 12, "early_stopping_rounds": 4}
    data = {"X_train": frame.iloc[:360], "X_test": frame.iloc[360:], "y_train": 3 + frame.x.iloc[:360]}
    monkeypatch.setattr(pipeline, "load_data", lambda *a: data)
    monkeypatch.setattr(pipeline, "experiment_history", lambda *a: [])
    monkeypatch.setattr(pipeline, "_record", lambda *a: a[-1])
    real_train = training.train_and_predict
    columns = []

    def collect(train, y, test, *args, **kwargs):
        columns.append(tuple(train.columns))
        cfg = args[2] if len(args) > 2 else kwargs["config"]
        assert not {"auto_features", "hourly_extended"}.intersection(cfg)
        return real_train(train, y, test, *args, **kwargs)

    monkeypatch.setattr(training, "train_and_predict", collect)
    pipeline.run_experiment(tmp_path, spec, config=config, time_budget=15.)
    tuned = pipeline.run_prefix_tuning(tmp_path, spec, config, 32.)
    assert tuned["tuned_params"] and tuned["tune_diagnostics"]["completed"]
    assert tuned["tune_diagnostics"]["prefix_max"] == str(clock[59])
    assert len(columns) > 1 and len(set(columns)) == 1
    assert "ts__hour" in columns[0] and "measurements__x__mean24h" in columns[0]
    assert capsys.readouterr().err.count('LEAK_AUDIT status=pass {"group": "calendar"') == 2


def test_ensemble_scores_the_whole_candidate_roster(tmp_path, monkeypatch):
    import training

    spec = {"task_type": "regression", "metric": "mae", "audit_required": True,
            "available_backends": ["lightgbm", "xgboost", "catboost", "hgb", "linear"]}
    history = []
    names = spec["available_backends"] * 2
    for i, model in enumerate(names):
        artifact = tmp_path / f"experiment{i}.npz"
        np.savez(artifact, test=[2.], oof=np.full(5, 2.), fold_ids=np.arange(5))
        history.append({"config": {"model": model, "seed": 42 + i}, "recipe": "single", "score": 2.,
                        "score_std": 0., "fold_scores": [2.] * 5, "fold_baselines": [{"score": 3.}] * 5,
                        "greater_is_better": False, "n_folds_done": 5, "fit_time": 1., "class_order": None,
                        "preds_file": str(artifact), "saved": i == 0})

    def select(members, y, metric, *, limit, **kwargs):
        assert len(members) == len(names) and limit == 16
        assert [m["config"]["model"] for m in members] == names
        stats = {"pool": list(range(len(members))), "selected": [0], "best_single": 0, "duplicates": []}
        return members[0] | {"config": {"blend_type": "best_single"}}, ([0], lambda p: p[0]), stats

    monkeypatch.setattr(pipeline, "experiment_history", lambda _: history)
    monkeypatch.setattr(pipeline, "load_data", lambda *_: {"y_train": pd.Series(np.zeros(5))})
    monkeypatch.setattr(pipeline, "_record", lambda *args: args[-1])
    monkeypatch.setattr(training, "select_ensemble", select)
    result = pipeline.run_ensemble(tmp_path, spec)
    assert result["config"]["blend_type"] == "best_single"
    assert len(result["members"]) == len(names)
