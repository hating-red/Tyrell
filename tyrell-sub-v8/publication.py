"""Provenance of atomic CSV publications by this invocation and its workers."""

from contextlib import contextmanager
from contextvars import ContextVar
import hashlib
import json
import os
from pathlib import Path
import stat
import tempfile


MODEL_SUCCESS = "MODEL_SUCCESS"
TRAINED_FALLBACK = "TRAINED_FALLBACK"
TEMPLATE_ONLY = "TEMPLATE_ONLY"
SCHEMA_FALLBACK = "SCHEMA_FALLBACK"
receipt_directory = ContextVar("receipt_directory", default=None)


@contextmanager
def publication_run():
    with tempfile.TemporaryDirectory(prefix="tyrell-publications-") as directory:
        token = receipt_directory.set(directory)
        try:
            yield directory
        finally:
            receipt_directory.reset(token)


def _identity(info):
    return f"{info.st_dev}-{info.st_ino}-{info.st_mtime_ns}-{info.st_size}"


def csv_digest(stream):
    digest = hashlib.sha256()
    for block in iter(lambda: stream.read(1024 * 1024), b""):
        digest.update(block)
    return digest.hexdigest()


def prepare_publication(staged, state, score=None):
    directory = receipt_directory.get()
    if not directory or state is None:
        return
    if state not in (MODEL_SUCCESS, TRAINED_FALLBACK, SCHEMA_FALLBACK):
        raise ValueError("Unknown submission state")
    # Prepare proof before rename. It can only match the output once os.replace
    # publishes this exact file. Earlier successful publications keep their proof.
    with staged.open("rb") as stream:
        key = _identity(os.fstat(stream.fileno()))
        digest = csv_digest(stream)
    record = {"state": state, "sha256": digest, "score": score}
    (Path(directory) / key).write_text(json.dumps(record), encoding="utf-8")


def published_result(path):
    directory = receipt_directory.get()
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
            digest = csv_digest(stream)
        if (record["state"] in (MODEL_SUCCESS, TRAINED_FALLBACK, SCHEMA_FALLBACK)
                and digest == record["sha256"] and _identity(path.lstat()) == key):
            return record
    except (OSError, ValueError, KeyError, TypeError):
        pass
    return None
