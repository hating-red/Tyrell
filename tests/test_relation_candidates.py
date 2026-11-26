from contextlib import nullcontext
from unittest.mock import Mock, patch

import pandas as pd
import pytest

from data_tools import _dictionary_clocks, _pre_event_fields, _read_lookup, enrich_postgres
from postgres_loader import ForeignKey, Table


def fixture_tables():
    train = Table("s", "samples", ("event_id", "entity_id", "y"))
    test = Table("s", "future", ("event_id", "entity_id"))
    creative = Table("s", "creative", ("entity_id", "bucket_id", "kind"), primary_key=("entity_id",),
                     comments={"bucket_id": "Immutable; known before the event."},
                     foreign_keys=(ForeignKey(("bucket_id",), ("s", "buckets"), ("code",)),))
    bucket = Table("s", "buckets", ("code", "theme", "total_conversions", "description"))
    bucket.comments = {"theme": "Fixed at creation", "total_conversions": "Fixed at creation"}
    return train, test, creative, bucket


def join(tmp_path, tables, aux, text=""):
    train, test, *lookups = tables
    frame = pd.DataFrame({"event_id": ["04", "02", "07", "06", "08"], "entity_id": ["10", "20", "10", "30", "40"]})
    frame.assign(y=0).to_csv(tmp_path / "train.csv", index=False)
    frame.iloc[[3, 1, 0]].to_csv(tmp_path / "test.csv", index=False)
    conn = Mock()
    conn.transaction.side_effect = nullcontext
    conn.execute.return_value.fetchone.return_value = (4, 3, 2)
    def read(conn, table, columns, deadline):
        return aux[table.name][columns].copy()
    audit = []
    with patch("data_tools._read_lookup", side_effect=read):
        enrich_postgres(conn, {(t.schema, t.name): t for t in lookups}, train, test,
                        "event_id", "y", text, tmp_path, log=lambda msg: None, on_audit=audit.append)
    return ({x["table"]: x for x in audit},
            pd.read_csv(tmp_path / "test.csv", dtype=str, keep_default_na=False))


def test_bridge_uses_verified_point_in_time_features_independent_of_catalog_order(tmp_path):
    train, test, creative, bucket = fixture_tables()
    aux = {"creative": pd.DataFrame({"entity_id": ["10", "20", "30"], "bucket_id": ["A", "B", "B"], "kind": [1, 2, 3]}),
           "buckets": pd.DataFrame({"code": ["B", "A"], "theme": ["garden", "water"]})}
    audits, frame = join(tmp_path, (train, test, bucket, creative), aux)
    group = audits["s.buckets"]
    assert group["dependencies"] == ["s.creative"]
    assert group["bridge_availability"] == "catalog_pre_event_static_feature"
    assert group["left_key"] == "creative__bucket_id"
    assert group["right_key"] == "code"
    assert group["output_columns"] == ["buckets__theme"]
    assert group["train"]["matched_rows"] == 4
    assert frame["event_id"].tolist() == ["06", "02", "04"]
    assert frame["buckets__theme"].tolist() == ["garden", "garden", "water"]


def test_bridge_does_not_treat_fk_as_historical_availability(tmp_path):
    train, test, creative, bucket = fixture_tables()
    creative.comments.clear()
    aux = {"creative": pd.DataFrame({"entity_id": ["10"], "bucket_id": ["A"], "kind": [1]})}
    audit, frame = join(tmp_path, (train, test, creative, bucket), aux)
    assert audit["s.buckets"]["status"] == "skipped"
    assert audit["s.buckets"]["bridge_availability"] is None
    assert audit["s.buckets"]["key_profile"] == {"status": "complete", "rows": 4, "null_key_rows": 1,
                                                  "distinct_keys": 2, "unique_and_nonnull": False}
    assert "buckets__theme" not in frame


def test_ambiguous_indirect_paths_skip_child(tmp_path):
    train, test, creative, bucket = fixture_tables()
    second = Table("s", "other_parent", creative.columns, primary_key=creative.primary_key,
                   foreign_keys=creative.foreign_keys, comments=creative.comments)
    part = pd.DataFrame({"entity_id": ["10"], "bucket_id": ["A"], "kind": [1]})
    audit, frame = join(tmp_path, (train, test, creative, bucket, second), {"creative": part, "other_parent": part})
    assert "неоднозначный путь" in audit["s.buckets"]["reason"]
    assert "buckets__theme" not in frame


def test_ambiguous_static_values_masked_per_field_and_keys_remain_unique(tmp_path):
    train, test, _, _ = fixture_tables()
    lookup = Table("s", "profiles", ("entity_id", "x", "z", "recent_score"), primary_key=("entity_id",),
                   comments={"z": "Immutable, known before event"})
    source = pd.DataFrame({"entity_id": ["10", "10", "20", "20", "30", "30", None],
                           "x": ["a", "a", "b", "b", "c", None, "d"],
                           "z": ["identical", "identical", "left", "right", "a", "b", None],
                           "recent_score": range(7)})
    audit, frame = join(tmp_path, (train, test, lookup), {"profiles": source},
                        "Pre-event static feature: s.profiles.x")
    group = audit["s.profiles"]
    assert group["status"] == "applied"
    assert group["duplicate_resolution"] == {"missing_lookup_key_rows": 1, "duplicate_keys": 3,
        "identical_extra_rows": 1, "conflicts_per_column": {"x": 1, "z": 2}, "omitted_conflicting_keys": 1,
        "retained_keys": 2}
    assert group["test"]["matched_rows"] == 2
    assert frame["event_id"].tolist() == ["06", "02", "04"]
    assert frame["profiles__x"].tolist() == ["", "b", "a"]
    assert frame["profiles__z"].tolist() == ["", "", "identical"]
    assert "profiles__recent_score" not in frame


@pytest.mark.parametrize("text,comment", [
    ("Pre-event static feature: wrong.t.x", ""),
    ("Pre-event static feature: s.t.x", "Not known before the event"),
    ("", "The latest value after event registration"),
    ("", "Static"),
])
def test_feature_availability_is_scoped_and_unambiguous(text, comment):
    t = Table("s", "t", ("x",), comments={"x": comment})
    assert _pre_event_fields(t, ["x"], text) == {}


def test_qualified_clock_dictionary_rejects_unscoped_fields_and_schema_collisions():
    dictionary = Table("s", "data_dictionary", ("feature", "description"))
    source = Table("s", "weather", ("timestamp",))
    other = Table("other", "weather", ("timestamp",))
    conn = Mock()
    conn.transaction.side_effect = nullcontext
    conn.execute.return_value.fetchall.return_value = [(None, "timestamp", "Start of the hour", None),
        (None, "weather.timestamp", "Start of the hour", None),
        (None, "other.weather.timestamp", "End of the hour", None)]
    found = _dictionary_clocks(conn, {i:t for i,t in enumerate([dictionary, source, other])},
                                None, None, None, lambda _: None)
    assert found == {("other", "weather"): {"timestamp": "End of the hour"}}
    assert "weather.timestamp" in conn.execute.call_args.args[1][0]


def test_nullable_bigint_does_not_round_trip_through_float():
    from unittest.mock import MagicMock
    conn = MagicMock()
    conn.cursor.return_value.__enter__.return_value.fetchmany.side_effect = [[(2**60+1,), (None,)], []]
    table = Table("s", "lookup", ("key",))
    data = _read_lookup(conn, table, ["key"], None)
    assert data["key"].iloc[0] == 2**60+1
    assert data["key"].iloc[1] is None

def test_latest_completed_and_published_measurement_wins_even_if_older_hour_arrives_late(tmp_path):
    from data_tools import _join_chunk, _plan_relation, _prepare_lookup
    train = Table("s", "calls", ("eid", "call_time", "zone_id", "y"))
    test = Table("s", "queue", ("eid", "call_time", "zone_id"))
    source = Table("s", "weather", ("zone_id", "hour_start", "published_at", "wind"),
                   comments={"published_at": "First publication of the immutable observation."})
    assert _plan_relation(source, train, test, "eid", {"y"}, "") == (
        "zone_id", "zone_id", "call_time", "hour_start", "start", ["wind"])
    source_rows = pd.DataFrame({"zone_id": ["z", "z", "z", "z", "z"],
        "hour_start": ["2025-01-01 " + t for t in ("07:00", "08:00", "09:00", "10:00", "11:00")],
        "published_at": ["2025-01-01 " + t for t in ("12:05", "09:30", "10:05", "10:15", "15:00")],
        "wind": [7, 8, 9, 10, 11]})
    times = ("12:07", "09:30", "09:31", "10:05", "10:06", "11:00", "11:01", "15:00", "15:01")
    events = pd.DataFrame({"eid": ["r" + str(i) for i in range(len(times))], "zone_id": ["z"] * len(times),
                           "call_time": ["2025-01-01 " + t for t in times]})
    stats = dict(rows=0, matched_rows=0, missing_key_rows=0, feature_nonmissing={"weather__wind": 0})
    lookup = _prepare_lookup(source_rows, "zone_id", "hour_start", "start", ["wind"], "weather__", "published_at")
    actual = _join_chunk(events, lookup, "zone_id", "call_time", stats)
    assert actual.eid.tolist() == events.eid.tolist()
    assert actual.weather__wind.fillna(-1).tolist() == [10, -1, 8, 8, 9, 9, 10, 10, 11]
    assert stats["min_interval_age_seconds"] == 60
    # Verify publication control travels from relation planning to the real enrich entry point.
    (events.assign(y=0)).to_csv(tmp_path / "train.csv", index=False)
    events.iloc[[3, 0]].to_csv(tmp_path / "test.csv", index=False)
    conn = Mock()
    conn.transaction.side_effect = nullcontext
    audit = []
    with patch("data_tools._read_lookup", return_value=source_rows) as read:
        enrich_postgres(conn, {("s", "weather"): source}, train, test, "eid", "y", "",
                        tmp_path, log=lambda _: None, on_audit=audit.append)
    assert "published_at" in read.call_args.args[2]
    assert audit[0]["publication_clock"] == "published_at"
    assert pd.read_csv(tmp_path / "test.csv")["weather__wind"].tolist() == [8, 10]


def test_revisions_or_duplicate_interval_records_cannot_use_immutable_publication_mode():
    from data_tools import _plan_relation, _prepare_lookup
    train = Table("s", "calls", ("eid", "call_time", "zone_id", "y"))
    test = Table("s", "queue", ("eid", "call_time", "zone_id"))
    source = Table("s", "weather", ("zone_id", "hour_end", "published_at", "wind", "revision"),
                   comments={"published_at": "First publication of immutable observation"})
    with pytest.raises(ValueError, match="исправлений"):
        _plan_relation(source, train, test, "eid", {"y"}, "")
    rows = pd.DataFrame({"zone_id": ["z", "z"], "hour_end": ["2025-01-01 12:00"] * 2, "v": [2, 6],
                         "published_at": ["2025-01-01 12:01", "2025-01-01 13:02"]})
    with pytest.raises(ValueError, match="неуникальный"):
        _prepare_lookup(rows, "zone_id", "hour_end", "end", ["v"], "obs__", "published_at")
