"""Regression checks for safe reference data joins and schema discovery."""
import tempfile
import unittest
from contextlib import nullcontext
from pathlib import Path
from unittest.mock import Mock, patch

import pandas as pd

from data_tools import (_join_chunk, _plan_relation, _prepare_lookup, _reference_mode, detect_split,
                        enrich_postgres, guess_task_spec, load_data)
from features import add_auto_features
from postgres_loader import ForeignKey, Table


class SafeJoins(unittest.TestCase):
    def setUp(self):
        self.train = Table("s", "calls", ("eid", "call_time", "district_id", "delay"))
        self.test = Table("s", "future", ("eid", "call_time", "district_id"))
        self.events = pd.DataFrame({
            "eid": ["003", "001", "004", "002", "008", "NA"],
            "district_id": ["a", "a", "b", "a", "a", ""],
            "call_time": ["2025-06-01 14:25", "2025-06-01 14:00", "2025-06-01 14:25",
                          "2025-06-01 13:00", None, "2025-06-01 14:25"],
        }, index=[3, 0, 5, 77, 55, 2])

    def test_hour_start_end_and_strict_boundary(self):
        for mode, stamps in (("start", ["12:00", "13:00", "14:00", "13:00"]),
                             ("end", ["13:00", "14:00", "15:00", "14:00"])):
            with self.subTest(mode=mode):
                weather = pd.DataFrame({"key": ["a", "a", "a", "b"],
                                        "hour": ["2025-06-01 " + t for t in stamps],
                                        "temp": [12, 13, 99, 33]})
                lookup = _prepare_lookup(weather, "key", "hour", mode, ["temp"], "weather__")
                joined = _join_chunk(self.events, lookup, "district_id", "call_time")
                self.assertTrue(joined["eid"].equals(self.events["eid"]))
                self.assertEqual(joined["weather__temp"].iloc[:3].tolist(), [13, 12, 33])
                self.assertTrue(joined["weather__temp"].iloc[3:].isna().all())
                with self.assertRaisesRegex(ValueError, "неуникальный"):
                    _prepare_lookup(pd.concat([weather, weather.iloc[:1]]), "key", "hour", mode,
                                    ["temp"], "weather__")

    def test_static_lookup_does_not_expand_or_reorder(self):
        aux = pd.DataFrame({"key": ["a", "b"], "area": [5, 7]})
        lookup = _prepare_lookup(aux, "key", None, None, ["area"], "geo__")
        joined = _join_chunk(self.events, lookup, "district_id", None)
        self.assertTrue(joined["eid"].equals(self.events["eid"]))
        self.assertEqual(joined["geo__area"].iloc[:5].tolist(), [5, 5, 7, 5, 5])
        self.assertTrue(pd.isna(joined["geo__area"].iloc[5]))
        for bad in ["a", None, " "]:
            aux.loc[1, "key"] = bad
            lookup = _prepare_lookup(aux, "key", None, None, ["area"], "geo__")
            self.assertEqual(lookup.index.tolist(), ["a"])
            if bad == "a":
                self.assertTrue(pd.isna(lookup.loc["a", "geo__area"]))
            else:
                self.assertEqual(lookup.loc["a", "geo__area"], 5)

    def test_generic_timestamp_requires_semantics_and_group(self):
        weather = Table("s", "weather", ("district_id", "timestamp", "temp"))
        inferred = _plan_relation(weather, self.train, self.test, "eid", {"delay"}, "")
        self.assertEqual(inferred[:5], ("district_id", "district_id", "call_time", "timestamp", "inferred"))
        plan = _plan_relation(weather, self.train, self.test, "eid", {"delay"},
                              "weather.timestamp marks the start of the hour.")
        self.assertEqual(plan[:5], ("district_id", "district_id", "call_time", "timestamp", "start"))
        weather.comments["timestamp"] = "End of the hour"
        plan = _plan_relation(weather, self.train, self.test, "eid", {"delay"}, "")
        self.assertEqual(plan[4], "end")
        with self.assertRaises(ValueError):
            _plan_relation(weather, self.train, self.test, "eid", {"delay"},
                           "weather.timestamp marks the start of the hour.")

    def test_private_foreign_key_and_label_filter(self):
        lookup = Table("s", "spatial_reference", ("ref", "area", "delay", "mean_delay", "eid"))
        self.train.foreign_keys = (ForeignKey(("district_id",), ("s", lookup.name), ("ref",)),)
        plan = _plan_relation(lookup, self.train, self.test, "eid", {"delay"}, "")
        self.assertEqual(plan, ("district_id", "ref", None, None, None, ["area"]))
        self.train.foreign_keys = ()
        with self.assertRaisesRegex(ValueError, "связи"):
            _plan_relation(lookup, self.train, self.test, "eid", {"delay"}, "")

    def test_enrichment_skips_ambiguous_table_and_preserves_id_lexemes(self):
        weather = Table("s", "weather", ("district_id", "timestamp", "temp"))
        districts = Table("s", "districts", ("district_id", "area", "mean_delay"))
        tables = {(t.schema, t.name): t for t in (self.train, self.test, weather, districts)}
        aux = pd.DataFrame({"district_id": ["a", "b"], "area": [6, 7]})
        unknown_cadence = pd.DataFrame({"district_id": ["a"], "temp": [18], "timestamp": ["2025-06-01 11:00"]})
        conn = Mock()
        conn.transaction.side_effect = nullcontext
        messages = []
        def lookup_source(_conn, table, *_args, **_kwargs):
            return aux if table is districts else unknown_cadence

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)
            self.events.assign(delay=1).to_csv(path / "train.csv", index=False)
            self.events.to_csv(path / "test.csv", index=False)
            with patch("data_tools._read_lookup", side_effect=lookup_source) as fetch:
                enrich_postgres(conn, tables, self.train, self.test, "eid", "delay", "",
                                path, log=messages.append)
            self.assertEqual(fetch.call_count, 2)
            for name in ("train.csv", "test.csv"):
                df = pd.read_csv(path / name, dtype=str, keep_default_na=False)
                self.assertEqual(df.eid.tolist(), self.events.eid.tolist())
                self.assertIn("districts__area", df)
                self.assertNotIn("districts__mean_delay", df)
                self.assertNotIn("weather__temp", df)
            self.assertTrue(any("weather" in line and "undetermined_step" in line for line in messages))


    def test_late_chunks_and_missing_train_id(self):
        weather = Table("s", "weather", ("district_id", "hour_start", "temp"))
        self.train.columns = tuple(c for c in self.train.columns if c != "eid")
        aux = pd.DataFrame({"district_id": ["a", "b"], "hour_start": ["2025-06-01 13:00"] * 2,
                            "temp": [11, 22]})
        conn = Mock()
        conn.transaction.side_effect = nullcontext
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)
            self.events.drop(columns="eid").assign(delay=1).to_csv(path / "train.csv", index=False)
            self.events.to_csv(path / "test.csv", index=False)
            with patch("data_tools._read_lookup", return_value=aux), patch("data_tools._EVENT_CHUNK_ROWS", 2):
                enrich_postgres(conn, {("s", "weather"): weather}, self.train, self.test, "eid",
                                "delay", "", path, log=lambda line: None)
            for name in ("train.csv", "test.csv"):
                frame = pd.read_csv(path / name)
                values = frame["weather__temp"]
                self.assertEqual(values.iloc[[0, 2]].tolist(), [11, 22])
                self.assertTrue(values.iloc[[1, 3, 4, 5]].isna().all())

    def test_replace_error_rolls_back_both_snapshots(self):
        districts = Table("s", "districts", ("district_id", "area"))
        aux = pd.DataFrame({"district_id": ["a", "b"], "area": [6, 7]})
        conn = Mock()
        conn.transaction.side_effect = nullcontext
        replace = Path.replace
        def fail_on_test(source, target):
            if source.name == "test.csv":
                raise PermissionError("test")
            return replace(source, target)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)
            self.events.assign(delay=1).to_csv(path / "train.csv", index=False)
            self.events.to_csv(path / "test.csv", index=False)
            before = {p: p.read_bytes() for p in path.iterdir()}
            with patch("data_tools._read_lookup", return_value=aux), patch.object(Path, "replace", fail_on_test):
                enrich_postgres(conn, {("s", districts.name): districts}, self.train, self.test, "eid",
                                "delay", "", path, log=lambda line: None)
            self.assertEqual(before, {p: p.read_bytes() for p in path.iterdir()})

    def test_private_dimension_pk_and_unknown_interval_length(self):
        lookup = Table("s", "unfamiliar", ("district_id", "zone"), primary_key=("district_id",))
        plan = _plan_relation(lookup, self.train, self.test, "eid", {"delay"}, "")
        self.assertEqual(plan, ("district_id", "district_id", None, None, None, ["zone"]))
        interval = Table("s", "regional", ("district_id", "period_start", "power"))
        plan = _plan_relation(interval, self.train, self.test, "eid", {"delay"}, "")
        self.assertEqual(plan[4], "inferred")
        lookup_data = pd.DataFrame({"district_id": ["a"], "period_start": ["2025-06-01 11:00"], "power": [5]})
        with self.assertRaisesRegex(ValueError, "undetermined_step"):
            _reference_mode(interval, lookup_data, "district_id", "period_start", plan[4],
                            pd.Timestamp("2025-06-01 13:00"), None, ["power"])
        interval.comments["period_start"] = "Start of the hour"
        self.assertEqual(_plan_relation(interval, self.train, self.test, "eid", {"delay"}, "")[4], "start")

    def test_private_csv_without_dictionary(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)
            pd.DataFrame({"ref": range(12), "weight": range(12), "extra": range(12),
                          "price": range(12)}).to_csv(path / "train.csv", index=False)
            pd.DataFrame({"ref": ["001", "NA"], "weight": [2, 3]}).to_csv(path / "test.csv", index=False)
            pd.DataFrame({"ref": ["NA", "001"], "answer": [0, 0]}).to_csv(
                path / "sample_submission.csv", index=False)
            spec, confident = guess_task_spec(path, "Target column: price. MAE.")
            self.assertTrue(confident)
            self.assertEqual(spec["target_column"], "price")
            self.assertEqual(load_data(path, spec)["test_ids"].tolist(), ["001", "NA"])


    def test_single_label_with_zero_features(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)
            train = pd.DataFrame({"row_id": ["T"], "amount": [7]})
            train.to_csv(path / "train.csv", index=False)
            pd.DataFrame({"row_id": ["S"]}).to_csv(path / "test.csv", index=False)
            pd.DataFrame({"row_id": ["S"], "amount": [0]}).to_csv(path / "sample_submission.csv",
                                                                  index=False)
            spec, _ = guess_task_spec(path, "MAE")
            data = load_data(path, spec)
            self.assertEqual(data["X_train"].shape, (1, 0))
            self.assertEqual(data["y_train"].tolist(), [7])
            train.assign(amount=None).to_csv(path / "train.csv", index=False)
            with self.assertRaisesRegex(ValueError, "target"):
                load_data(path, spec)

    def test_future_numbered_periods(self):
        for name in ("Year_Factor", "period", "year_index"):
            with self.subTest(name=name):
                past = pd.DataFrame({name: [2, 1, 5, 2]})
                future = pd.DataFrame({name: [7, 6]})
                self.assertEqual(detect_split(past, future), {"type": "time", "column": name})
                self.assertIsNone(detect_split(past, pd.DataFrame({name: [3, 1]})))
        source = pd.DataFrame({"region": [1, 2, 3, 1]})
        self.assertEqual(detect_split(source, pd.DataFrame({"region": [7, 8]})),
                         {"type": "group", "column": "region"})

    def test_binary_ap_with_one_observed_class(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)
            pd.DataFrame({"eid": ["S"]}).to_csv(path / "test.csv", index=False)
            pd.DataFrame({"eid": ["S"], "engaged": [0]}).to_csv(path / "sample_submission.csv",
                                                                index=False)
            for observed in (0, 1):
                with self.subTest(label=observed):
                    pd.DataFrame({"eid": ["T"], "engaged": [observed]}).to_csv(path / "train.csv",
                                                                             index=False)
                    spec, _ = guess_task_spec(path, "Binary classification. PR-AUC.")
                    self.assertEqual((spec["task_type"], spec["metric"]), ("binary", "pr_auc"))
                    self.assertTrue(spec["probabilities"])
                    self.assertEqual(spec["positive_class"], 1)

    def test_unknown_clock_with_nonoverlapping_periods(self):
        past = pd.DataFrame({"batch_mark": ["2024-11-06", "2024-11-07"],
                             "birth_date": ["1980-01-01", "1981-01-01"]})
        future = pd.DataFrame({"batch_mark": ["2024-11-08"], "birth_date": ["1992-01-01"]})
        self.assertEqual(detect_split(past, future), {"type": "time", "column": "batch_mark"})
        self.assertIsNone(detect_split(past, past))

    def test_asof_normalizes_offsets_and_keeps_strict_boundary(self):
        aux = pd.DataFrame({"district_id": ["a"], "hour": ["2025-06-01 12:00+03:00"], "temp": [7]})
        lookup = _prepare_lookup(aux, "district_id", "hour", "end", ["temp"], "s__")
        clocks = ["2025-06-01 10:00Z", "2025-06-01 09:00", "2025-06-01 12:00+03:00",
                  "2025-06-01 10:01+01:00", None, "2025-06-01 10:00Z"]
        zoned = self.events.assign(call_time=clocks, district_id="a")
        joined = _join_chunk(zoned, lookup, "district_id", "call_time")
        self.assertEqual(joined["s__temp"].dropna().tolist(), [7, 7, 7])
        self.assertEqual(joined["s__temp"].notna().tolist(), [True, False, False, True, False, True])

    def test_row_local_features_and_clock_detection(self):
        X = self.events[["district_id", "call_time"]].reset_index(drop=True)
        X["n"] = range(len(X))
        split = detect_split(X, X)
        self.assertEqual(split, {"type": "time", "column": "call_time"})
        features, _ = add_auto_features(X, X, row_local=True)
        self.assertIn("call_time__hour", features)
        self.assertFalse(any("__freq" in c or "__mean_by" in c for c in features))
        self.assertIsNone(detect_split(X.rename(columns={"call_time": "birth_date"}),
                                      X.rename(columns={"call_time": "birth_date"})))
        with self.assertRaises(ValueError):
            detect_split(X, X, time_column="misspelled_time")
