"""Provenance of atomic CSV publications by this invocation and its workers."""

from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import stat
import tempfile


MODEL_SUCCESS = "MODEL_SUCCESS"
TRAINED_FALLBACK = "TRAINED_FALLBACK"
TEMPLATE_ONLY = "TEMPLATE_ONLY"
_ENV = "TYRELL_PUBLICATION_DIR"


@contextmanager
def publication_run():
    previous = os.environ.get(_ENV)
    with tempfile.TemporaryDirectory(prefix="tyrell-publications-") as directory:
        os.environ[_ENV] = directory
        try:
            yield
        finally:
            if previous is None:
                os.environ.pop(_ENV, None)
            else:
                os.environ[_ENV] = previous


def _identity(info):
    return f"{info.st_dev}-{info.st_ino}-{info.st_mtime_ns}-{info.st_size}"


def prepare_publication(staged, state, score=None):
    directory = os.environ.get(_ENV)
    if not directory or state is None:
        return
    if state not in (MODEL_SUCCESS, TRAINED_FALLBACK):
        raise ValueError("Unknown submission state")
    # Prepare proof before rename. It can only match the output once os.replace
    # publishes this exact file. Earlier successful publications keep their proof.
    with staged.open("rb") as stream:
        key = _identity(os.fstat(stream.fileno()))
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    record = {"state": state, "sha256": digest, "score": score}
    (Path(directory) / key).write_text(json.dumps(record), encoding="utf-8")


def published_result(path):
    directory = os.environ.get(_ENV)
    if not directory:
        return None
    try:
        info = path.lstat()
        if not stat.S_ISREG(info.st_mode):
            return None
        key = _identity(info)
        record = json.loads((Path(directory) / key).read_text(encoding="utf-8"))
        with path.open("rb") as stream:
            if _identity(os.fstat(stream.fileno())) != key:
                return None
            digest = hashlib.file_digest(stream, "sha256").hexdigest()
        if (record["state"] in (MODEL_SUCCESS, TRAINED_FALLBACK)
                and digest == record["sha256"] and _identity(path.lstat()) == key):
            return record
    except (OSError, ValueError, KeyError, TypeError):
        pass
    return None
