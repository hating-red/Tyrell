"""Point-in-time checks for feature builders and input availability rules."""
from contextlib import contextmanager
import json
import re
import sys
import time

import numpy as np
import pandas as pd


class LeakAuditError(ValueError):
    pass


def audit_log(status, group, **details):
    print("LEAK_AUDIT status=" + status + " " + json.dumps({"group": group, **details}, ensure_ascii=False),
          file=sys.stderr, flush=True)


class AuditBudget:
    def __init__(self, seconds=60.):
        self.limit = seconds
        self.remaining = seconds

    @property
    def elapsed(self):
        return round(self.limit - self.remaining, 6)

    @contextmanager
    def check(self, group):
        start = time.monotonic()

        def check_time():
            if time.monotonic() - start >= self.remaining:
                raise LeakAuditError("audit time budget exhausted")

        try:
            check_time()
            yield check_time
            check_time()
        except LeakAuditError as error:
            audit_log("fail", group, reason=str(error))
            raise
        finally:
            self.remaining -= time.monotonic() - start


def sample_positions(size, *, limit=2000, seed=42):
    if size <= limit:
        return np.arange(size)
    return np.sort(np.random.default_rng(seed).choice(size, limit, replace=False))


def check_equal(expected, actual):
    if list(expected.columns) != list(actual.columns) or len(expected) != len(actual):
        raise LeakAuditError("feature columns or row count changed under future replay")
    for column in expected:
        a, b = expected[column].reset_index(drop=True), actual[column].reset_index(drop=True)
        if pd.api.types.is_numeric_dtype(a) and pd.api.types.is_numeric_dtype(b):
            equal = np.allclose(a.to_numpy(dtype=float, na_value=np.nan),
                                b.to_numpy(dtype=float, na_value=np.nan), atol=1e-6, rtol=0, equal_nan=True)
        else:
            equal = bool((a.astype("string").eq(b.astype("string")).fillna(False) |
                          (a.isna() & b.isna())).all())
        if not equal:
            raise LeakAuditError(f"feature {column} changed under future replay")


def exact_cutoffs(query_clock):
    """Group only rows at the exact same event time."""
    query = pd.Series(query_clock).reset_index(drop=True).dropna()
    for cutoff, positions in query.groupby(query, sort=False).groups.items():
        yield positions.to_numpy(), cutoff


def replay_payload(values, released, cutoff, *, corrupt_future):
    """Apply a query-specific mask to gathered data with an implicit noisy suffix."""
    released, cutoff = np.asarray(released), np.asarray(cutoff)
    visible = released < cutoff
    unavailable = (released >= cutoff) | pd.isna(released)
    if values.ndim > visible.ndim:
        visible = visible[..., None]
        unavailable = unavailable[..., None]
    payload = np.where(unavailable, 987654.321, values) if corrupt_future else values
    return np.where(visible, payload, np.nan)


def restricted_inputs(task, names):
    """Resolve scoped lists/prohibitions without treating ordinary task prose as a column rule."""
    restricted = {}
    for line in task.splitlines():
        forbid = re.search(r"do not use|must not use|forbidden|unavailable at|нельзя использ|"
                           r"не используй|запрещено|недоступн", line, re.I)
        scoped_rule = re.search(r"\b(?:availability|available only|use only before)\b|"
                                r"доступност[ьи]?\s*:|доступн\w* только", line, re.I)
        if not (forbid or scoped_rule):
            continue
        for name in names:
            if not re.search(rf"(?<![\w.]){re.escape(name)}(?!\w)", line, re.I):
                continue
            if forbid:
                restricted[name] = "task.md prohibition"
            elif not re.search(r"(?:before (?:the )?(?:event|call)|раньше (?:события|вызова))\s*[.]?\s*$", line, re.I):
                restricted[name] = "unrecognised task.md availability rule"
    return restricted
