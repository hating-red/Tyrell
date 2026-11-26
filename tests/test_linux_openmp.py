import ctypes
from importlib import metadata
import os
from pathlib import Path
import subprocess
from unittest.mock import Mock

import pytest

import time_budget


def test_available_system_openmp_skips_wheel_lookup(monkeypatch):
    monkeypatch.setattr(ctypes, "CDLL", lambda name: object())
    monkeypatch.setattr(metadata, "distribution", Mock(side_effect=AssertionError))
    assert time_budget._bundled_openmp() is None


def test_missing_or_unusable_wheel_falls_back_to_next(monkeypatch, tmp_path):
    def distribution(name):
        directory = name.replace("-", "_") + ".libs"
        matching = metadata.PackagePath(f"{directory}/libgomp-hash.so.1.0.0")
        unrelated = metadata.PackagePath("other.libs/libgomp-hash.so.1.0.0")
        return Mock(files=[unrelated, matching], locate_file=lambda entry: tmp_path / entry)

    expected = tmp_path / "xgboost.libs" / "libgomp-hash.so.1.0.0"

    def load(path):
        if path != str(expected):
            raise OSError("library not usable")
        return object()

    monkeypatch.setattr(ctypes, "CDLL", load)
    monkeypatch.setattr(metadata, "distribution", distribution)
    assert time_budget._bundled_openmp() == expected
    monkeypatch.setattr(metadata, "distribution", Mock(side_effect=metadata.PackageNotFoundError))
    assert time_budget._bundled_openmp() is None


@pytest.mark.parametrize("outcome", ["completed", "expired"])
def test_private_worker_alias_survives_until_exit(monkeypatch, tmp_path, outcome):
    runtime = tmp_path / "vendored.so"
    runtime.touch()
    monkeypatch.setattr(time_budget.sys, "platform", "linux")
    monkeypatch.setattr(time_budget, "_bundled_openmp", lambda: runtime)
    if os.name == "nt":
        # Creating symlinks requires an extra privilege in some Windows test environments.
        monkeypatch.setattr(Path, "symlink_to", lambda alias, target: alias.write_text(str(target)))
    original = {"LD_LIBRARY_PATH": "/prior/path", "PRESERVE": "abc"}
    directories = []

    def run(command, deadline, **options):
        assert command == ["worker"] and deadline == 12345
        assert options["cwd"] == tmp_path
        env = options["env"]
        assert env["PRESERVE"] == "abc"
        first, tail = env["LD_LIBRARY_PATH"].split(os.pathsep)
        assert tail == "/prior/path"
        directory = Path(first)
        alias = directory / "libgomp.so.1"
        if os.name == "nt":
            assert alias.read_text() == str(runtime)
        else:
            assert alias.is_symlink() and alias.resolve() == runtime
        directories.append(directory)
        if outcome == "expired":
            raise subprocess.TimeoutExpired(command, 0)
        return subprocess.CompletedProcess(command, 3)

    monkeypatch.setattr(time_budget, "run_until", run)
    code = time_budget.supervised_exit(["worker"], 12345, cwd=tmp_path, env=original)
    assert code == (124 if outcome == "expired" else 3)
    assert original["LD_LIBRARY_PATH"] == "/prior/path"
    assert len(directories) == 1 and not directories[0].exists()


def test_non_linux_skips_runtime_discovery(monkeypatch):
    monkeypatch.setattr(time_budget.sys, "platform", "win32")
    monkeypatch.setattr(time_budget, "_bundled_openmp", Mock(side_effect=AssertionError))
    runner = Mock(return_value=subprocess.CompletedProcess(["worker"], 0))
    monkeypatch.setattr(time_budget, "run_until", runner)
    assert time_budget.supervised_exit(["worker"], 42) == 0
    runner.assert_called_once_with(["worker"], 42)
