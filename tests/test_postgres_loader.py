import csv
import sqlite3
import sys
import tempfile
import types
import unittest
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import patch

import data_tools
import postgres_loader as pg


def table(name, *cols, schema="main", **kwargs):
    return pg.Table(schema, name, cols, **kwargs)


class CatalogTest(unittest.TestCase):
    def test_private_unknown_schema(self):
        train = table("historical_cases", "case_key", "signal", "risk_17", "operator_note")
        test = table("unresolved_cases", "case_key", "signal")
        dictionary = table("lookup", "category", "description")
        tables = {(t.schema, t.name): t for t in (train, test, dictionary)}
        binding = pg._choose_pair(tables, None, ["case_key", "prediction"],
                                  "Target: risk_17. Metric: MAE.")
        self.assertEqual(binding, (train, test, "case_key", "risk_17"))

    def test_ambiguous_pair_is_rejected(self):
        train = table("train", "id", "val", "out")
        tests = [table(name, "id", "val") for name in ("orders1", "orders2")]
        with self.assertRaisesRegex(ValueError, "Несколько"):
            pg._choose_pair({t.name: t for t in [train, *tests]}, None, ["id", "out"], "")

    def test_key_is_not_an_order_contract(self):
        sample = table("sample_submission", "event_id", "y", primary_key=("event_id",))
        with self.assertRaisesRegex(ValueError, "Порядок"):
            pg._sample_order(sample, "Rows must match sample_submission.")
        key, desc, technical = pg._sample_order(sample, "sample_submission ORDER BY event_id DESC")
        self.assertEqual((key, desc, technical), ("event_id", True, []))

    def test_task_order_direction_and_expressions(self):
        sample = table("sample_submission", "event_id", "answer")
        for text in ("Шаблон отсортирован по event_id по убыванию.",
                     "Порядок строк шаблона по убыванию event_id.",
                     "sample_submission\nORDER BY event_id DESC"):
            self.assertEqual(pg._sample_order(sample, text), ("event_id", True, []))
        with self.assertRaisesRegex(ValueError, "Выражение ORDER BY"):
            pg._sample_order(sample, "sample_submission ORDER BY event_id % 7")

    def test_explicit_target_before_any_template_column(self):
        train = table("train", "case_key", "signal", "blue", "outcome", "score")
        test = table("test", "case_key", "signal", "blue", primary_key=("case_key",))
        pair = {"train": train, "test": test}
        for columns in (["case_key", "score"], ["case_key", "blue", "green"]):
            with self.subTest(columns=columns):
                self.assertEqual(pg._choose_pair(pair, None, columns, "Target column: outcome.")[-2:],
                                 ("case_key", "outcome"))

    def test_template_name_declared_by_task(self):
        sample = table("requested_predictions", "id", "target")
        self.assertIs(pg._sample_table({"sample": sample}, "Submission template: requested_predictions"), sample)

    def test_commented_ordinal(self):
        sample = table("sample_submission", "seq", "id", "y", column_types={"seq": "integer"},
                       comments={"seq": "original row position"})
        self.assertEqual(pg._sample_order(sample, ""), ("seq", False, ["seq"]))


class Connection:
    """Exercise the actual SELECT/export path without installing or contacting a server."""

    def __init__(self):
        self.db = sqlite3.connect(":memory:")
        self.statements = []
        self.db.executescript("""
            CREATE TABLE train(case_key TEXT, signal INT, cost INT);
            CREATE TABLE test(case_key TEXT, signal INT);
            CREATE TABLE sample_submission(case_key TEXT, cost INT, row_number INT);
            INSERT INTO train VALUES ('50', 3, 42), ('51', 4, 45);
            INSERT INTO test VALUES ('02', 2), ('01', 1);
            INSERT INTO sample_submission VALUES ('02', 0, 2), ('01', 0, 1);
        """)

    def execute(self, query, params=()):
        self.statements.append(query)
        if "set_config" in query or query.startswith("SET TRANSACTION"):
            return None
        return self.db.execute(query, params)

    @contextmanager
    def cursor(self, name):
        cursor = self.db.cursor()
        try:
            yield cursor
        finally:
            cursor.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass


class AdapterTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        folder = Path(self.temp.name)
        self.task_path = folder / "task.md"
        self.task_path.write_text("Target: cost. MAE.", encoding="utf-8")
        self.output = folder / "work"
        self.output.mkdir()
        self.conn = Connection()
        self.addCleanup(self.conn.db.close)
        train = table("train", "case_key", "signal", "cost")
        test = table("test", "case_key", "signal")
        self.sample = table("sample_submission", "case_key", "cost", "row_number",
                            column_types={"row_number": "integer"})
        self.catalog = {(t.schema, t.name): t for t in (train, test, self.sample)}

    def export(self, ready=None, on_stats=None):
        driver = types.SimpleNamespace(connect=lambda **opts: self.conn, Error=sqlite3.Error)
        with (patch.dict(sys.modules, {"psycopg": driver}),
              patch.dict("os.environ", dict.fromkeys(pg.PG_CONNECTION_ENV, "test") | {"PGPASSWORD": ""}),
              patch.object(pg, "_catalog", return_value=self.catalog) as read_catalog,
              patch.object(data_tools, "enrich_postgres", create=True) as enrich):
            pg.prepare_postgres(self.task_path, self.output, log=lambda msg: None,
                                on_core_ready=ready, on_stats=on_stats)
            read_catalog.assert_called_once()
            enrich.assert_called_once()
        self.assertIn("SET TRANSACTION ISOLATION LEVEL REPEATABLE READ, READ ONLY", self.conn.statements)

    def read_csv(self, name):
        with (self.output / name).open(encoding="utf-8-sig", newline="") as file:
            return list(csv.reader(file))

    def test_interrupted_database_template_is_never_visible(self):
        cursor = self.conn.cursor

        @contextmanager
        def interrupted(name):
            with cursor(name) as real:
                def fetch(_):
                    if (self.output / "sample_submission.csv").exists():
                        self.fail("Database template was visible before the cursor finished")
                    if real.fetchone() is None:
                        raise TimeoutError("second batch interrupted")
                    return [("0002", 0, 1)]

                yield types.SimpleNamespace(execute=real.execute, fetchmany=fetch)

        self.conn.db.execute("DELETE FROM sample_submission WHERE case_key = '01'")
        with patch.object(self.conn, "cursor", interrupted):
            with self.assertRaises(TimeoutError):
                self.export()
        self.assertFalse((self.output / "sample_submission.csv").exists())
        self.assertFalse(list(self.output.glob(".template-*")))

    def test_database_order_snapshot_and_callback(self):
        def ready():
            self.assertEqual(self.read_csv("test.csv")[1][0], "02")
            self.assertEqual(self.read_csv("sample_submission.csv"), [["case_key", "cost"], ["01", "0"], ["02", "0"]])
            self.assertEqual(self.read_csv("data_dictionary.csv")[-1][:2], ["cost", "target"])

        self.export(ready)

    def test_measures_rows_and_bytes_without_count_queries(self):
        stats = {}
        self.export(on_stats=stats.update)
        for part in ("test", "train"):
            self.assertEqual(stats[part + "_rows"], 2)
            self.assertTrue(stats["exports"][part]["complete"])
            self.assertEqual(stats["exports"][part]["bytes"], (self.output / f"{part}.csv").stat().st_size)
        self.assertGreaterEqual(stats["data_load_seconds"], stats["snapshot_seconds"])
        self.assertGreaterEqual(stats["join_seconds"], 0)
        self.assertFalse(any("COUNT(" in query.upper() for query in self.conn.statements))

    def test_dimension_fetch_limits_apply_to_adaptive_batches(self):
        source = self.catalog[("main", "test")]
        with patch.object(data_tools, "_AUX_ROW_LIMIT", 2):
            lookup = data_tools._read_lookup(self.conn, source, ["case_key"], deadline=None)
        self.assertEqual(lookup["case_key"].tolist(), ["01", "02"])
        with patch.object(data_tools, "_AUX_ROW_LIMIT", 1):
            with self.assertRaisesRegex(ValueError, "бюджет"):
                data_tools._read_lookup(self.conn, source, ["case_key"], deadline=None)

    def test_authentication_errors_hide_connection_details(self):
        error = RuntimeError("password authentication failed (test sentinel)")
        self.assertIn("password authentication failed", pg.pg_error_reason(error))
        self.assertNotIn("sentinel", pg.pg_error_reason(error))
        self.assertIn("password required", pg.pg_error_reason(RuntimeError("no password supplied")))

    def test_featureless_tables_fall_back_with_sample_order(self):
        from pipeline import check_result, write_fallback

        for name in ("train", "test"):
            self.conn.db.execute(f"ALTER TABLE {name} DROP COLUMN signal")
            self.catalog[("main", name)].columns = tuple(c for c in self.catalog[("main", name)].columns
                                                         if c != "signal")
        spec = {"id_columns": ["case_key"], "target_column": "cost", "task_type": "regression", "metric": "mae"}

        def ready():
            self.assertEqual(write_fallback(self.output, spec), [])
            check_result(self.output, spec)

        self.export(ready)
        self.assertEqual([row[0] for row in self.read_csv("submission.csv")], ["case_key", "01", "02"])
        unknown = [table("people", "case_key", "cost"), table("stores", "case_key")]
        with self.assertRaisesRegex(ValueError, "Не удалось определить"):
            pg._choose_pair({t.name: t for t in unknown}, None, ["case_key", "cost"], "")

    def test_nested_local_template_is_used_only_when_unique(self):
        first = self.task_path.parent / "data" / "sample_submission.csv"
        first.parent.mkdir()
        first.write_text("case_key,cost\n02,0\n01,0\n", encoding="utf-8")
        self.task_path.write_text("Use sample_submission.csv; target: cost", encoding="utf-8")
        self.export()
        self.assertEqual((self.output / first.name).read_bytes(), first.read_bytes())
        self.assertIsNone(pg._local_sample(self.task_path, "Use sample_submission.csv"))
        self.assertEqual(pg._local_sample(self.task_path, "Use data/sample_submission.csv"), first)

    def test_identifier_named_row_number_is_preserved(self):
        self.conn.db.executescript("""
            ALTER TABLE train RENAME COLUMN case_key TO row_number;
            ALTER TABLE test RENAME COLUMN case_key TO row_number;
            DROP TABLE sample_submission;
            CREATE TABLE sample_submission (row_number INTEGER, cost INT);
            INSERT INTO sample_submission VALUES (2,0), (1,0);
        """)
        self.catalog = {(t.schema, t.name): t for t in (
            table("train", "row_number", "signal", "cost"), table("test", "row_number", "signal"),
            table("sample_submission", "row_number", "cost", column_types={"row_number": "integer"}))}
        self.task_path.write_text("target: cost\nSample_submission ORDER BY row_number DESC.", encoding="utf-8")
        self.export()
        self.assertEqual(self.read_csv("sample_submission.csv"), [["row_number", "cost"], ["2", "0"], ["1", "0"]])

    def test_missing_template_uses_task_header_and_test_cursor_order(self):
        self.catalog.pop(("main", "sample_submission"))
        self.task_path.write_text("Target: cost. Output columns: prediction,case_key\n01,02", encoding="utf-8")
        messages = {}
        self.export(on_stats=messages.update)
        self.assertEqual(self.read_csv("sample_submission.csv"),
                         [["prediction", "case_key"], ["0", "02"], ["0", "01"]])
        self.assertEqual(messages["template_source"], "task_header_and_test_ids")
        self.assertFalse(messages["template_official"])
        self.assertIsNone(messages["template_order"])

    def test_missing_template_declared_order_uses_unexported_test_ordinal(self):
        self.catalog.pop(("main", "sample_submission"))
        test = self.catalog[("main", "test")]
        self.conn.db.execute("ALTER TABLE test ADD COLUMN seq INTEGER")
        self.conn.db.execute("UPDATE test SET seq=CAST(case_key AS INTEGER)")
        test.columns += ("seq",)
        self.task_path.write_text("Submission columns: case_key,answer\nsubmission ORDER BY seq ASC\nTarget: cost",
                                  encoding="utf-8")
        self.export()
        self.assertEqual(self.read_csv("sample_submission.csv"),
                         [["case_key", "answer"], ["01", "0"], ["02", "0"]])
        self.assertEqual(self.read_csv("test.csv")[0], ["case_key", "signal"])

    def test_missing_template_requires_unique_output_header(self):
        for text in ["Target: cost. ID: case_key.", "Train columns: case_key,signal",
                     "Output columns: id,out\nSubmission columns: id,other"]:
            with self.subTest(text=text), self.assertRaises(ValueError):
                pg._declared_submission_columns(text)
        for text in ["Submission columns: id,prediction\n123,0\n456,0",
                     "Колонки файла submission.csv: \x60id\x60 и \x60prediction\x60.",
                     "## Output format\n- \x60id\x60 — row identifier\n- \x60prediction\x60 — model output",
                     "## Формат ответа\n| id | prediction |\n|---|---|\n|01|0|",
                     "## Output format\n\x60\x60\x60csv\nid,prediction\n001,0\n\x60\x60\x60"]:
            with self.subTest(text=text):
                self.assertEqual(pg._declared_submission_columns(text), ["id", "prediction"])

    def test_missing_template_rejects_unsupported_order_even_with_ordinal(self):
        test = table("test", "id", "seq", "row_order", column_types={"row_order":"integer"})
        for text in ("sample_submission ORDER BY ID", "sample_submission sorted by ID, as in the input."):
            self.assertEqual(pg._sample_order(test, text), ("id", False, ["row_order"]))
        for clause in ("id, seq", "id then by seq", "ABS(id)", "unknown", "(id)", "id % 7"):
            with self.subTest(clause=clause), self.assertRaisesRegex(ValueError, "ORDER BY"):
                pg._sample_order(test, "submission ORDER BY " + clause, optional=True)

    def test_multiple_nested_templates_never_reconstructed(self):
        self.catalog.pop(("main", "sample_submission"))
        self.task_path.write_text("Submission columns: case_key,cost", encoding="utf-8")
        for name in ("one", "two"):
            folder = self.task_path.parent / name
            folder.mkdir()
            (folder / "sample_submission.csv").write_text("case_key,cost\n", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "Несколько локальных"):
            self.export()

    def test_memory_budget_checked_before_pandas(self):
        with self.assertRaisesRegex(ValueError, "бюджет загрузки"):
            pg._export(self.conn, self.sample, ["case_key", "cost"], self.output / "limited.csv",
                       None, memory_budget=10)

    def test_local_template_overrides_unordered_database(self):
        sample_path = self.task_path.parent / "sample_submission.csv"
        original = b"case_key,cost\n02,0\n01,0\n"
        sample_path.write_bytes(original)
        self.sample.column_types.clear()
        self.export()
        self.assertEqual((self.output / sample_path.name).read_bytes(), original)
        self.assertEqual(sample_path.read_bytes(), original)

    def test_duplicate_or_null_ordinals_rejected(self):
        for invalid in (1, None):
            with self.subTest(invalid=invalid):
                self.conn.db.execute("UPDATE sample_submission SET row_number=? WHERE case_key='02'", (invalid,))
                temp = self.output / "sample_export.csv"
                with self.assertRaisesRegex(ValueError, "NULL или повторы"):
                    pg._export(self.conn, self.sample, ["case_key", "cost"], temp, None,
                               order="row_number", check_order=True)
                temp.unlink()


if __name__ == "__main__":
    unittest.main()
