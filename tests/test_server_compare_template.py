import hashlib
import importlib.util
import io
import json
from pathlib import Path
import stat
from zipfile import ZipFile

import pytest


PATH = Path(__file__).resolve().parents[1] / "runs" / "real-public" / "server_compare.py"
SPEC = importlib.util.spec_from_file_location("server_compare_local", PATH)
driver = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(driver)

SAMPLE = b"event_id,engaged\r\n010,0\r\nNA,1\r\n001,0\r\n"
EXPECTED = {"001", "NA", "010"}


@pytest.fixture
def inputs(tmp_path, monkeypatch):
    monkeypatch.setattr(driver, "ROOT", tmp_path)
    official = tmp_path / "official" / "task1"
    official.mkdir(parents=True)
    (official / "task.md").write_bytes(b"Official task.")
    (official / "sample_submission.csv").write_bytes(SAMPLE)
    (official / "baseline.py").write_bytes(b"raise AssertionError('Not an entry point')")
    return official


def test_order_and_numeric_comparison(inputs, tmp_path):
    work = tmp_path / "work"
    work.mkdir()
    template = driver.prepare_inputs(work, "task1", EXPECTED)
    identical = driver.snapshot(SAMPLE.replace(b",0\r", b",0.00\r"), "task1", EXPECTED, template)
    assert identical["csv_valid"] and identical["order_verified"]
    assert identical["predictions_changed_from_template"] == 0
    wrong = driver.snapshot(b"event_id,engaged\n001,0.4\nNA,1\n010,0\n", "task1", EXPECTED, template)
    assert wrong["test_ids_match"] and wrong["template_official"]
    assert not wrong["csv_valid"] and not wrong["official_order_verified"]
    assert wrong["predictions_changed_from_template"] == 1
    with pytest.raises(ValueError, match="header"):
        driver.snapshot(SAMPLE.replace(b"engaged", b"Engaged"), "task1", EXPECTED, template)


def test_pristine_official_inputs_and_final_csv(inputs, tmp_path):
    archive = "tyrell-submission-v5.zip"
    with ZipFile(tmp_path / archive, "w") as z:
        z.writestr("solution.py", """
from pathlib import Path
assert Path("task.md").read_bytes() == b"Official task."
assert Path("baseline.py").is_file()
template = Path("sample_submission.csv").read_bytes()
Path("submission.csv").write_bytes(template.replace(b",1\\r", b",0.8\\r"))
""")
        z.writestr("requirements.txt", "")
    dest = tmp_path / "output"
    report = driver.one(archive, "task1", dest, EXPECTED, {}, lambda: None)
    assert report["status"] == "ok" and report["order_verified"] and report["template_intact"]
    digest = hashlib.sha256(SAMPLE).hexdigest()
    assert report["template_sha256_before"] == report["template_sha256_after"] == digest
    assert report["copied_template_sha256_after"] == digest
    assert report["final_csv"]["predictions_changed_from_template"] == 1
    for source in inputs.iterdir():
        copy = dest / "run" / source.name
        assert not copy.stat().st_mode & stat.S_IWRITE
        assert source.read_bytes() == copy.read_bytes()


def test_tampered_copy_does_not_modify_official_template(inputs, tmp_path):
    work = tmp_path / "work"
    work.mkdir()
    template = driver.prepare_inputs(work, "task1", EXPECTED)
    copy = work / "sample_submission.csv"
    copy.chmod(0o644)
    copy.write_bytes(b"modified")
    integrity = driver.template_integrity(template)
    assert not integrity["template_intact"]
    assert integrity["template_sha256_after"] == integrity["template_sha256_before"]
    assert (inputs / copy.name).read_bytes() == SAMPLE


def test_without_bundle_preserves_diagnostic_behavior(tmp_path, monkeypatch):
    monkeypatch.setattr(driver, "ROOT", tmp_path)
    (tmp_path / "task1.md").write_text("Diagnostic task", encoding="utf-8")
    work = tmp_path / "work"
    work.mkdir()
    template = driver.prepare_inputs(work, "task1", EXPECTED)
    assert template is None and not (work / "sample_submission.csv").exists()
    report = driver.snapshot(SAMPLE, "task1", EXPECTED, template)
    assert report["csv_valid"] and not report["template_official"] and not report["order_verified"]
    assert report["predictions_changed_from_template"] is None


def test_incomplete_official_bundle_fails_closed(inputs, tmp_path):
    (inputs / "sample_submission.csv").unlink()
    work = tmp_path / "work"
    work.mkdir()
    with pytest.raises(FileNotFoundError, match="Missing official input"):
        driver.prepare_inputs(work, "task1", EXPECTED)


def test_qwen_environment_is_explicit_and_capped(tmp_path, monkeypatch):
    for key in driver.QWEN_ENV:
        monkeypatch.setenv(key, "fake-" + key)
    monkeypatch.setenv("OPENAI_LOG", "debug")
    monkeypatch.setenv("TYRELL_MAX_LLM_CALLS", "99")
    for enabled in (True, False):
        env = driver.child_env("task2", tmp_path, enabled)
        names = {k for k in env if k.startswith("OPENAI_")}
        assert names == (set(driver.QWEN_ENV) if enabled else set())
        assert env["TYRELL_MAX_LLM_CALLS"] == ("2" if enabled else "0")
    monkeypatch.delenv("OPENAI_API_KEY")
    with pytest.raises(ValueError, match="OPENAI_API_KEY"):
        driver.child_env("task2", tmp_path, True)


def test_qwen_log_allows_only_local_milestones():
    source = io.StringIO("Authorization: secret\n"
                         "untrusted model response with no recognizable secret\n"
                         "Страховочный submission.csv готов. task_spec: hidden\n"
                         "LLM: запросов 1, токенов на входе 777, на выходе 13\n"
                         "Готово: submission.csv проверен, mae=4.303.\n")
    dest = io.StringIO()
    driver.log_worker(source, dest, set(), qwen=True)
    assert dest.getvalue().splitlines() == [
        "Страховочный submission.csv готов.",
        "LLM: запросов 1, токенов на входе 777, на выходе 13",
        "Готово: submission.csv проверен, mae=4.303."]


def test_qwen_corroborates_trials_using_complete_fold_and_config(tmp_path):
    trial = {"tool": "run_experiment", "config": {"model": "lightgbm"}, "recipe": "single",
             "journal_index": 1, "complete": True, "score": 4.6}
    calls = {"status": "ok", "attempted_calls": 2, "responses": 1, "tool_calls": 1, "completed_experiments": 1,
             "prompt_tokens": 777, "completion_tokens": 13, "trials": [trial]}
    (tmp_path / "runtime.json").write_text(json.dumps({"qwen_tool_run": calls}), encoding="utf-8")
    entry = {"config": {"model": "lightgbm"}, "recipe": "single", "fold_signature": "time",
             "fold_scores": [4.5, 4.7], "complete": True, "score": 4.6, "metric": "mae", "saved": False}
    hist = [{"saved": True, "score": 4.3}, entry]
    result = driver.qwen_evidence(tmp_path, hist, "mae")
    assert result["verified"] and result["hypotheses"] == 1 and result["attempted_calls"] == 2
    assert result["actual_cost_rub"] is None
    assert result["verified_experiments"] == [{"journal_index": 1, "previous_saved_score": 4.3, "score": 4.6,
                                              "fold_scores": [4.5, 4.7], "accepted": False}]
    entry["config"]["model"] = "xgboost"
    result = driver.qwen_evidence(tmp_path, hist, "mae")
    assert not result["verified"] and not result["verified_experiments"]


def test_resume_cannot_reuse_results_from_another_qwen_mode(tmp_path, monkeypatch):
    monkeypatch.setattr(driver, "ROOT", tmp_path)
    (tmp_path / "status.json").write_text(json.dumps({"status": "complete", "results": []}), encoding="utf-8")
    with pytest.raises(ValueError, match="different Qwen mode"):
        driver.run([driver.QWEN_ARCHIVE], ["task1"], qwen=True)
    with pytest.raises(ValueError, match="v5 archive"):
        driver.run(["tyrell-submission-v2.zip"], ["task1"], qwen=True)
