from contextlib import nullcontext
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import Mock, patch

import pandas as pd
import pytest

from data_tools import (_dictionary_clocks, _hour_clock, _plan_relation, _prepare_lookup,
                        enrich_postgres, feature_coverage, guess_task_spec)
from features import add_auto_features
from postgres_loader import Table


def test_join_audit_counts_key_matches_even_when_payload_is_null():
    train = Table("s", "train", ("eid", "district_id", "target"))
    test = Table("s", "test", ("eid", "district_id"))
    district = Table("s", "districts", ("district_id", "area"))
    unknown = Table("s", "weather", ("district_id", "timestamp", "wind"))
    repeated = Table("s", "untrusted", ("district_id", "other"), primary_key=("district_id",))
    tables = {(t.schema, t.name): t for t in (train, test, district, unknown, repeated)}
    rows = pd.DataFrame({"eid": [41, 17, 21, 14], "district_id": ["a", "b", "c", ""]})
    conn = Mock()
    conn.transaction.side_effect = nullcontext
    audit = []
    with TemporaryDirectory() as folder:
        path = Path(folder)
        rows.assign(target=0).to_csv(path / "train.csv", index=False)
        rows.iloc[[2, 1]].to_csv(path / "test.csv", index=False)
        sources = [pd.DataFrame({"district_id": ["a", "b"], "area": ["residential", None]}),
                   pd.DataFrame({"district_id": ["b", "b"], "other": [2, 3]})]
        with patch("data_tools._EVENT_CHUNK_ROWS", 2), patch("data_tools._read_lookup", side_effect=sources):
            enrich_postgres(conn, tables, train, test, "eid", "target", "", path, on_audit=audit.append)
        assert pd.read_csv(path / "test.csv")["eid"].tolist() == [21, 17]
    applied, skipped, dup = audit
    assert applied["output_columns"] == ["districts__area"]
    assert applied["status"] == "applied"
    assert applied["availability"] == "static_dimension"
    assert applied["train"] == {"rows": 4, "matched_rows": 2, "missing_key_rows": 1,
                                "feature_nonmissing": {"districts__area": 1}}
    assert applied["test"]["matched_rows"] == 1
    assert applied["test"]["feature_nonmissing"]["districts__area"] == 0
    assert applied["lookup_rows"] == applied["distinct_lookup_keys"] == 2
    assert skipped["status"] == "skipped" and skipped["output_columns"] == []
    assert dup["status"] == "applied" and dup["distinct_lookup_keys"] == 1
    assert dup["test"]["matched_rows"] == 1
    assert dup["test"]["feature_nonmissing"] == {"untrusted__other": 0}


def test_hourly_audit_checks_actual_availability_and_latency():
    train = Table("s", "calls", ("eid", "district_id", "call_time", "response"))
    test = Table("s", "test", ("eid", "district_id", "call_time"))
    hourly = Table("s", "weather", ("district_id", "hour_start", "temp"))
    rows = pd.DataFrame({"eid": [11, 22, 33], "district_id": ["a", "a", "a"],
                         "call_time": ["2025-06-01 11:00", None, "2025-06-01 11:10"]})
    source = pd.DataFrame({"district_id": ["a"], "hour_start": ["2025-06-01 10:00"], "temp": [3]})
    conn = Mock()
    conn.transaction.side_effect = nullcontext
    audit = []
    with TemporaryDirectory() as folder:
        path = Path(folder)
        rows.assign(response=0).to_csv(path / "train.csv", index=False)
        rows.iloc[[2]].to_csv(path / "test.csv", index=False)
        with patch("data_tools._EVENT_CHUNK_ROWS", 2), patch("data_tools._read_lookup", return_value=source):
            enrich_postgres(conn, {("s", "weather"): hourly}, train, test, "eid", "response", "",
                            path, on_audit=audit.append)
    stats = audit[0]["train"]
    assert stats["matched_rows"] == stats["invalid_event_time_rows"] == 1
    assert stats["min_lag_seconds"] == stats["max_lag_seconds"] == 600
    hourly.columns += ("published_at",)
    with pytest.raises(ValueError, match="публикации"):
        _plan_relation(hourly, train, test, "eid", {"response"}, "")


def test_feature_coverage_reports_unknown_categories_without_values():
    past = pd.DataFrame({"zone_id": ["private-a", "private-a", None], "level": [1, 2, 2],
                         "event_time": ["2025-01-01", "2025-01-02", "invalid"]})
    future = pd.DataFrame({"zone_id": ["private-b", "private-a", None], "level": [3, 1, 1],
                           "event_time": ["2025-01-04", "2025-01-05", "2025-01-06"]})
    report = feature_coverage(past, future, clock="event_time")
    zone = report["features"]["zone_id"]
    assert zone["train_unique"] == 1 and zone["test_unique"] == 2
    assert zone["test_unseen_rows"] == zone["test_known_rows"] == zone["test_missing"] == 1
    assert zone["train_most_frequent_rows"] == 2
    assert report["features"]["event_time"]["time_bounds"]["train"]["invalid_or_missing"] == 1
    assert report["features"]["level"]["test_unseen_rows"] == 1
    assert "private-" not in str(report)


def test_temporal_feature_values_do_not_depend_on_future_rows():
    past = pd.DataFrame({"call_time": ["2025-01-06 00:00", "2025-01-07 00:00"],
                         "category": ["bank", "hospital"]})
    future = pd.DataFrame({"call_time": ["2025-01-09 15:00"], "category": [None]})
    expanded, _ = add_auto_features(past, future, row_local=True)
    alone, _ = add_auto_features(past, past, row_local=True)
    pd.testing.assert_frame_equal(expanded, alone)
    assert alone["call_time__hour_of_week"].tolist() == [0, 24]


def test_time_without_calendar_date_cannot_timestamp_history():
    values = pd.DataFrame({"district": ["a"], "hour_start": ["11:00+00:00"], "value": [42]})
    with pytest.raises(ValueError, match="граница"):
        _prepare_lookup(values, "district", "hour_start", "start", ["value"], "weather__")


@pytest.mark.parametrize(("target", "condition", "expected"), [
    ("wait", "Label delay unit: minutes", True),
    ("response_time_minutes", "Предскажите число минут от регистрации вызова до прибытия службы.", True),
    ("response_time_minutes", "Измерение дано в минутах.", False),
    ("time_until_arrival_minutes", "The values describe minutes.", False),
    ("wait", "Label delay unit: minutes\nLabel delay unit: hours", False),
])
def test_label_delay_requires_explicit_event_relative_duration(tmp_path, target, condition, expected):
    pd.DataFrame({"eid": [1, 2], "timestamp": ["2025-02-01", "2025-02-02"], target: [7, 8]}).to_csv(
        tmp_path / "train.csv", index=False)
    pd.DataFrame({"eid": [3], "timestamp": ["2025-02-03"]}).to_csv(tmp_path / "test.csv", index=False)
    pd.DataFrame({"eid": [3], target: [0]}).to_csv(tmp_path / "sample_submission.csv", index=False)
    spec, _ = guess_task_spec(tmp_path, "MAE.\n" + condition)
    assert (spec.get("label_delay_unit") == "minutes") == expected

@pytest.mark.parametrize(("descriptions", "catalog", "expected"), [
    (["Start of the hour"], "", "start"),
    (["Начало часа"], "", "start"),
    (["End of the hour"], "End of the hour", "end"),
    (["Hourly measurement"], "", None),
    (["End of the hour"], "Start of the hour", None),
    (["Not the end of the hour"], "", None),
    (["Start of the hour or end of the hour."], "", None),
    (["Start of the hour", "Start of the hour"], "", None),
])
def test_dictionary_needs_one_scoped_unambiguous_clock(descriptions, catalog, expected):
    train = Table("s", "train", ("eid", "call_time", "district_id", "y"))
    test = Table("s", "test", ("eid", "call_time", "district_id"))
    weather = Table("s", "weather", ("timestamp", "district_id", "temp"), comments={"timestamp": catalog})
    dictionary = Table("s", "data_dictionary", ("table_name", "column_name", "description", "schema_name"))
    other = Table("private", "weather", weather.columns)
    tables = {(t.schema, t.name): t for t in (train, test, weather, other, dictionary)}
    conn = Mock()
    conn.transaction.side_effect = nullcontext
    rows = [("weather", "timestamp", text, "s") for text in descriptions]
    rows.append(("weather", "timestamp", "Start of the hour", None))  # ambiguous schema
    conn.execute.return_value.fetchall.return_value = rows
    result = _dictionary_clocks(conn, tables, train, test, None, lambda line: None)
    assert ("private", "weather") not in result
    assert _hour_clock(weather, "", result[("s", "weather")])[1] == expected
    assert conn.execute.call_args.args[0].lstrip().startswith("SELECT")
    assert "LIMIT 1001" in conn.execute.call_args.args[0]
    with pytest.raises(ValueError, match="метаданных"):
        _plan_relation(dictionary, train, test, "eid", {"y"}, "")


def test_dictionary_with_missing_columns_or_timeout_preserves_safe_catalog_clock():
    weather = Table("s", "weather", ("timestamp", "district_id"), comments={"timestamp": "End of the hour"})
    dictionary = Table("s", "data_dictionary", ("column_name", "description"))
    tables = {(t.schema, t.name): t for t in (weather, dictionary)}
    conn = Mock()
    conn.transaction.side_effect = nullcontext
    assert _dictionary_clocks(conn, tables, None, None, None, lambda line: None) == {}
    conn.execute.assert_not_called()
    dictionary.columns += ("table_name",)
    conn.execute.side_effect = TimeoutError
    result = _dictionary_clocks(conn, tables, None, None, None, lambda line: None)
    assert result == {}
    assert _hour_clock(weather, "", result.get(("s", "weather")))[1] == "end"
