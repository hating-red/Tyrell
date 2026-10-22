"""Тесты submission.py. Запуск: python -m pytest tests -q"""

import numpy as np
import pandas as pd
import pytest

from submission import make_submission, validate_submission

IDS = pd.Series([10, 11, 12, 13])


@pytest.fixture
def sample_reg(tmp_path):
    path = tmp_path / "sample_reg.csv"
    pd.DataFrame({"id": IDS, "site_eui": 0.0}).to_csv(path, index=False)
    return path


@pytest.fixture
def sample_clf(tmp_path):
    path = tmp_path / "sample_clf.csv"
    pd.DataFrame({"id": IDS, "attack_cat": "Normal"}).to_csv(path, index=False)
    return path


def test_regression_written(tmp_path, sample_reg):
    out = tmp_path / "submission.csv"
    assert make_submission([1.5, 2.0, 3.0, 4.0], IDS, sample_reg, out) == []
    written = pd.read_csv(out)
    assert list(written.columns) == ["id", "site_eui"]
    assert written["id"].tolist() == IDS.tolist()


def test_nan_and_inf_repaired(tmp_path, sample_reg):
    out = tmp_path / "submission.csv"
    assert make_submission([1.0, np.nan, np.inf, 3.0], IDS, sample_reg, out) == []
    assert np.isfinite(pd.read_csv(out)["site_eui"]).all()


def test_column_vector_flattened(tmp_path, sample_reg):
    out = tmp_path / "submission.csv"
    assert make_submission(np.array([[1.0], [2.0], [3.0], [4.0]]), IDS, sample_reg, out) == []


def test_wrong_length_keeps_old_file(tmp_path, sample_reg):
    out = tmp_path / "submission.csv"
    make_submission([1.0, 2.0, 3.0, 4.0], IDS, sample_reg, out)
    before = out.read_text()
    problems = make_submission([9.0, 9.0], IDS, sample_reg, out)
    assert problems and out.read_text() == before  # плохой ответ не затёр хороший


def test_unknown_class_rejected(tmp_path, sample_clf):
    out = tmp_path / "submission.csv"
    problems = make_submission(["Normal", "DoS", "Martians", "DoS"], IDS, sample_clf, out, {"Normal", "DoS"})
    assert any("вне допустимого" in p for p in problems)
    assert not out.exists()


def test_missing_class_filled_with_most_common(tmp_path, sample_clf):
    out = tmp_path / "submission.csv"
    assert make_submission(["DoS", None, "DoS", "Normal"], IDS, sample_clf, out, {"Normal", "DoS"}) == []
    assert pd.read_csv(out)["attack_cat"].tolist() == ["DoS", "DoS", "DoS", "Normal"]


@pytest.mark.parametrize("broken, expected", [
    (pd.DataFrame({"id": [11, 10, 12, 13], "y": 1.0}), "порядок"),
    (pd.DataFrame({"id": [10, 11, 12], "y": 1.0}), "строк"),
    (pd.DataFrame({"id": IDS, "y": 1.0, "extra": 1}), "столбцы"),
    (pd.DataFrame({"id": [10, 10, 12, 13], "y": 1.0}), "повторяющиеся"),
])
def test_validate_catches_format_errors(broken, expected):
    problems = validate_submission(broken, IDS, ["id", "y"])
    assert any(expected in p for p in problems)


# ---------- Форматы скрытых задач ----------

def write_sample(tmp_path, frame, name="sample.csv"):
    path = tmp_path / name
    frame.to_csv(path, index=False)
    return path


def test_probabilities_written_and_rounding_clipped(tmp_path):
    sample = write_sample(tmp_path, pd.DataFrame({"id": IDS, "default": 0.5}))
    out = tmp_path / "submission.csv"
    assert make_submission([0.1, 1 + 1e-9, -1e-9, 0.7], IDS, sample, out, probabilities=True) == []
    assert pd.read_csv(out)["default"].between(0, 1).all()


def test_labels_instead_of_probabilities_rejected(tmp_path):
    sample = write_sample(tmp_path, pd.DataFrame({"id": IDS, "default": 0.5}))
    out = tmp_path / "submission.csv"
    problems = make_submission([0, 1, 3, 1], IDS, sample, out, probabilities=True)
    assert any("вне [0, 1]" in p for p in problems) and not out.exists()


def test_class_probabilities_from_array_with_class_order(tmp_path):
    sample = write_sample(tmp_path, pd.DataFrame({"id": IDS, "cat": 0.0, "dog": 0.0, "fox": 0.0}))
    out = tmp_path / "submission.csv"
    proba = np.array([[0.7, 0.2, 0.1]] * 4)  # столбцы в порядке dog, cat, fox
    assert make_submission(proba, IDS, sample, out, class_order=["dog", "cat", "fox"]) == []
    written = pd.read_csv(out)
    assert list(written.columns) == ["id", "cat", "dog", "fox"]
    assert written.loc[0, "dog"] == 0.7 and written.loc[0, "cat"] == 0.2


def test_class_probabilities_from_dataframe_renormalized(tmp_path):
    sample = write_sample(tmp_path, pd.DataFrame({"id": IDS, "A": 0.0, "B": 0.0}))
    out = tmp_path / "submission.csv"
    proba = pd.DataFrame({"B": [0.5, 0.3, 0.6, 0.1], "A": [0.501, 0.7, 0.4, 0.9]})
    assert make_submission(proba, IDS, sample, out) == []
    sums = pd.read_csv(out)[["A", "B"]].sum(axis=1)
    assert np.allclose(sums, 1)


def test_class_probabilities_bad_sum_rejected(tmp_path):
    sample = write_sample(tmp_path, pd.DataFrame({"id": IDS, "A": 0.0, "B": 0.0}))
    problems = make_submission(np.full((4, 2), 0.9), IDS, sample, tmp_path / "submission.csv")
    assert any("сумма" in p for p in problems)


def test_integer_labels_from_floats_and_bools(tmp_path):
    sample = write_sample(tmp_path, pd.DataFrame({"id": IDS, "label": 0}))
    out = tmp_path / "submission.csv"
    assert make_submission(np.array([1.0, 0.0, 1.0, 1.0]), IDS, sample, out, allowed_labels={0, 1}) == []
    assert pd.read_csv(out)["label"].tolist() == [1, 0, 1, 1]
    assert make_submission(np.array([True, False, False, True]), IDS, sample, out, allowed_labels={0, 1}) == []
    assert pd.read_csv(out)["label"].tolist() == [1, 0, 0, 1]


def test_id_column_found_by_name_not_position(tmp_path):
    sample = write_sample(tmp_path, pd.DataFrame({"price": 0.0, "row_id": IDS}))
    out = tmp_path / "submission.csv"
    assert make_submission([1.0, 2.0, 3.0, 4.0], IDS.rename("row_id"), sample, out) == []
    written = pd.read_csv(out)
    assert list(written.columns) == ["price", "row_id"] and written["row_id"].tolist() == IDS.tolist()
