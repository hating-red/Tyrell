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
