import sys
from pathlib import Path
from contextlib import contextmanager
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import average_precision_score

import pipeline
import postgres_loader as pg
import solution
import training
from data_tools import guess_task_spec, load_data
from publication import MODEL_SUCCESS, publication_run, published_result
from submission import make_submission


def fixture_files(directory, labels, *, test_rows=2, clock=None):
    train = pd.DataFrame({"id": np.arange(len(labels)), "x": np.arange(len(labels)), "y": labels})
    test = pd.DataFrame({"id": [f"{i:04}" for i in reversed(range(test_rows))], "x": 0})
    if clock is not None:
        train["event_time"], test["event_time"] = clock
    train.to_csv(directory / "train.csv", index=False)
    test.to_csv(directory / "test.csv", index=False)
    sample = directory / "sample_submission.csv"
    test[["id"]].iloc[::-1].assign(y=0).to_csv(sample, index=False)
    return sample


def test_interrupted_reconstructed_template_cannot_become_a_success(tmp_path, monkeypatch):
    task = tmp_path / "task.md"
    task.write_text("Evaluation metric: MAE.\nsubmission.csv columns: id, y.\n", encoding="utf-8")
    out = tmp_path / "submission.csv"
    out.write_text("id,y\n0001,7\n", encoding="utf-8")
    old = out.read_bytes()
    monkeypatch.setattr(sys, "argv", ["solution.py", str(task)])
    monkeypatch.setenv("TYRELL_LLM", "mock")

    def interrupted(command, deadline, **kwargs):
        work = Path(command[-2])
        with pytest.raises(TimeoutError):
            pg._export(export_connection(interrupted=True), pg.Table("s", "test", ("id",)), ["id"],
                       work / "test.csv", None, id_column="id", template_columns=("id", "y"))
        assert not (work / "sample_submission.csv").exists()
        assert not list(work.glob(".template-*"))
        return 124

    monkeypatch.setattr(solution, "supervised_exit", interrupted)
    assert solution.main() == 124 and out.read_bytes() == old


def export_connection(*, interrupted=False):
    batches = iter([[("0008",), ("0002",)], []])

    def fetch(_):
        batch = next(batches)
        if not batch and interrupted:
            raise TimeoutError("interrupted after the first batch")
        return batch

    @contextmanager
    def cursor(**kwargs):
        yield SimpleNamespace(execute=lambda _: None, fetchmany=fetch)

    return SimpleNamespace(cursor=cursor, execute=lambda *args: None)


def test_reconstructed_template_is_complete_and_preserves_cursor_order(tmp_path):
    pg._export(export_connection(), pg.Table("s", "test", ("id",)), ["id"], tmp_path / "test.csv", None,
               id_column="id", template_columns=("y", "id"))
    assert (tmp_path / "sample_submission.csv").read_text().splitlines() == ["y,id", "0,0008", "0,0002"]


def test_numeric_contract_keeps_prior_csv_and_proof(tmp_path):
    sample = fixture_files(tmp_path, [0, 0, 90])
    out = tmp_path / "submission.csv"
    ids = pd.Series(["0001", "0000"], name="id")
    with publication_run():
        assert not make_submission([1., 3.], ids, sample, out, answer_kind="regression", publication_state=MODEL_SUCCESS)
        before = out.read_bytes(), published_result(out)
        assert make_submission(["oops", "inf"], ids, sample, out, answer_kind="regression", publication_state=MODEL_SUCCESS)
        assert (out.read_bytes(), published_result(out)) == before
        assert not make_submission(["fraud", "ok"], ids, sample, out, {"fraud", "ok"}, answer_kind="labels")


@pytest.mark.parametrize(("negative", "positive"), [("ok", "fraud"), (1, 0)])
def test_declared_positive_class_matches_outer_ap_and_csv(tmp_path, monkeypatch, negative, positive):
    y = np.where(np.arange(60) % 3 == 0, positive, negative)
    sample = fixture_files(tmp_path, y)
    task = f"Target: y. Metric: PR-AUC. **Positive class:** `{positive}`."
    spec, _ = guess_task_spec(tmp_path, task)
    assert spec["positive_class"] == positive
    data = load_data(tmp_path, spec)
    x, xt, config = pipeline.prepare_features(tmp_path, spec, {}, data=data)
    xt["x"] = [0, 1]

    def backend(xtr, ytr, wtr, xva, yva, xte, *args, **kwargs):
        np.testing.assert_array_equal(ytr, (xtr["x"].to_numpy() % 3 == 0).astype(int))
        def probabilities(frame):
            score = np.where(frame["x"].to_numpy() % 3 == 0, .9, .1)
            return np.column_stack([1 - score, score])
        return probabilities(xva), probabilities(xte), 1, False, 1

    monkeypatch.setattr(training, "_check_backend", lambda name: name)
    monkeypatch.setitem(training._BACKENDS, "lightgbm", backend)
    result = training.train_and_predict(x, data["y_train"], xt, "binary", "pr_auc", config)
    assert result["class_order"] == [negative, positive]
    assert result["score"] == average_precision_score(y == positive, result["oof"][:, 1]) == 1.
    out = tmp_path / "submission.csv"
    assert not pipeline._write_submission(out, spec, data, result)
    assert pd.read_csv(out)["y"].tolist() == [.1, .9]
    assert not pipeline.write_fallback(tmp_path, spec)
    np.testing.assert_allclose(pd.read_csv(out)["y"], 1 / 3)


def test_multitarget_has_explicit_diagnostic_and_schema_fallback(tmp_path):
    sample = fixture_files(tmp_path, [0, 0, 90])
    pd.read_csv(sample, dtype={"id": str}).assign(y2=0).to_csv(sample, index=False)
    with pytest.raises(ValueError, match="мультитаргетная регрессия"):
        guess_task_spec(tmp_path, "MAE")
    out = tmp_path / "submission.csv"
    pipeline.schema_fallback(sample, "MAE", out)
    pipeline.verify_template_output(sample, "MAE", out)
    assert pd.read_csv(out)[["y", "y2"]].sum(axis=1).tolist() == [2., 2.]


@pytest.mark.parametrize(("metric", "filler"), [("mae", 0.), ("rmse", 30.)])
def test_regression_repair_and_early_fallback_use_metric(tmp_path, capsys, metric, filler):
    fixture_files(tmp_path, [0., 0., 90.])
    spec = {"task_type": "regression", "target_column": "y", "metric": metric, "nonnegative": True}
    assert not pipeline.write_fallback(tmp_path, spec)
    assert pd.read_csv(tmp_path / "submission.csv")["y"].tolist() == [filler, filler]
    np.testing.assert_array_equal(training._finalize_regression(np.array([np.nan, 7.]), np.array([0., 0., 90.]),
                                                               True, metric), [filler, 7.])
    assert "TRAIN_EDA available_labels=3 target_missing_fraction=0.000000" in capsys.readouterr().err


def test_prior_repairs_whole_rows_after_reordering_classes(tmp_path, capsys):
    ids = pd.Series(range(5), name="id")
    sample = tmp_path / "sample.csv"
    pd.DataFrame({"id": ids, "ok": 0, "fraud": 0}).to_csv(sample, index=False)
    output = tmp_path / "submission.csv"
    raw = [[np.nan, .8], [np.inf, .8], [-np.inf, .8], [-7., 1.], [.3, .7]]
    assert not make_submission(raw, ids, sample, output, class_order=["fraud", "ok"], filler=[.15, .85])
    assert pd.read_csv(output)[["ok", "fraud"]].to_numpy().tolist() == [[.85, .15]] * 4 + [[.7, .3]]
    assert "Warning: PREDICTION_REPAIR stage=submission invalid_rows=4/5" in capsys.readouterr().err


def test_ensemble_repairs_before_clipping_and_logs_rows_once(capsys):
    values = np.array([[-np.inf, -7.], [.4, .6]])
    result = training._constrain_ensemble(values, None, True, False, [.15, .85])
    np.testing.assert_allclose(result, [[.15, .85], [.4, .6]])
    assert "Warning: PREDICTION_REPAIR stage=ensemble invalid_rows=1/2" in capsys.readouterr().err
    with pytest.raises(ValueError, match="All predictions"):
        training._constrain_ensemble(values[:1], None, True, False, [.15, .85])


@pytest.mark.parametrize("n", [60, 100])
def test_warning_threshold_and_all_invalid_rejection(tmp_path, capsys, n):
    sample = fixture_files(tmp_path, [0, 0, 90], test_rows=n)
    ids = pd.Series([f"{i:04}" for i in range(n)], name="id")
    output = tmp_path / "submission.csv"
    with publication_run():
        assert not make_submission([np.nan] + [2.] * (n - 1), ids, sample, output, filler=0.,
                                   answer_kind="regression", publication_state=MODEL_SUCCESS)
        assert f"Warning: PREDICTION_REPAIR stage=submission invalid_rows=1/{n}" in capsys.readouterr().err
        before = output.read_bytes(), published_result(output)
        assert make_submission([np.nan] * n, ids, sample, output, filler=0., answer_kind="regression")
        assert (output.read_bytes(), published_result(output)) == before


def test_eda_ignores_late_labels_and_drift_cannot_block_publication(tmp_path, capsys):
    clock = pd.Timestamp("2025-01-01")
    fixture_files(tmp_path, [1., 3., 90., np.nan], clock=(
        [clock, clock, clock, clock], [clock + pd.Timedelta(minutes=5)] * 2))
    spec = {"task_type": "regression", "target_column": "y", "metric": "mae",
            "time_column": "event_time", "label_delay_unit": "minutes"}
    assert not pipeline.write_fallback(tmp_path, spec)
    baseline_log = capsys.readouterr().err
    assert "available_labels=2 target_missing_fraction=0.333333 fallback=2.0 reason=median" in baseline_log
    test_path = tmp_path / "test.csv"
    pd.read_csv(test_path, dtype={"id": str}).assign(x=1e9).to_csv(test_path, index=False)
    assert not pipeline.write_fallback(tmp_path, spec)
    assert capsys.readouterr().err == baseline_log
    result = {"predictions": [100., 100.], "class_order": None, "score": 1.}
    assert not pipeline._write_submission(tmp_path / "submission.csv", spec, load_data(tmp_path, spec), result)
    assert "Warning: PREDICTION_DRIFT" in capsys.readouterr().err
