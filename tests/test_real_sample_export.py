"""Validation of local COPY templates when source order is arbitrary."""

import csv
import sys
import time
from pathlib import Path

import pytest

LOCAL_TOOLS = Path(__file__).resolve().parents[1] / "runs" / "real-public"
sys.path.insert(0, str(LOCAL_TOOLS))
from sample_export import _validate


def check(tmp_path, ids, *, indexed=False, expected=None):
    source = tmp_path / "sample.csv"
    header = ["event_id", "target"]
    with source.open("w", newline="", encoding="utf-8") as out:
        writer = csv.writer(out)
        writer.writerow(header)
        writer.writerows((key, 0) for key in ids)
    report = {}
    _validate(source, header, tmp_path / "work.sqlite", len(ids) if expected is None else expected,
              indexed, report, time.perf_counter() + 10)
    return report


@pytest.mark.parametrize("indexed", [True, False])
def test_unordered_literal_and_quoted_ids(tmp_path, indexed):
    ids = ["z", "007", 'й",\n', "NA", "0"]
    assert check(tmp_path, ids, indexed=indexed)["validated_rows"] == len(ids)
    assert (tmp_path / "work.sqlite").exists() is not indexed


def test_nonadjacent_duplicate_crosses_batch(tmp_path):
    with pytest.raises(ValueError, match="Duplicate ID"):
        check(tmp_path, ["abc", *(str(i) for i in range(5000)), "abc"])


@pytest.mark.parametrize("ids", [["007", ""], [" "], [None]])
def test_invalid_key(tmp_path, ids):
    with pytest.raises(ValueError, match="Invalid diagnostic ID"):
        check(tmp_path, ids)


@pytest.mark.parametrize("ids, expected", [([], 0), (["one"], 2)])
def test_truncation_or_empty(tmp_path, ids, expected):
    with pytest.raises(ValueError, match="PostgreSQL COPY reported"):
        check(tmp_path, ids, expected=expected)
