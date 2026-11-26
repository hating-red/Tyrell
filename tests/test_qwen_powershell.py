"""Exercise Windows PowerShell 5.1 with a temporary .env and stub processes only."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import socket
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
PS = shutil.which("powershell.exe")
if not PS and os.name == "nt":
    candidate = Path(os.environ["SystemRoot"]) / "System32/WindowsPowerShell/v1.0/powershell.exe"
    if candidate.is_file():
        PS = str(candidate)
pytestmark = pytest.mark.skipif(not PS, reason="Windows PowerShell required")


@pytest.fixture
def runner(tmp_path):
    root = tmp_path / "Tyrell runner"
    diagnostics = root / "runs" / "real-public"
    diagnostics.mkdir(parents=True)
    shutil.copy2(ROOT / "run-real.ps1", root / "run-real.ps1")
    secret = 'local#sentinel$([IO.File]::WriteAllText("injected.txt","yes"))'
    (root / ".env").write_text(
        "# Only the three Qwen variables may reach the fake API.\r\n"
        f"export OPENAI_API_KEY = '{secret}' # literal\r\n"
        'OPENAI_BASE_URL="https://endpoint.invalid/v1" # comment\r\n'
        "OPENAI_MODEL=file-fixture\r\n"
        "TYRELL_UNRELATED=forbidden\r\n",
        encoding="utf-8-sig")
    (diagnostics / "qwen_check.py").write_text(
        "import hashlib, json, os, sys\nfrom pathlib import Path\n"
        "ok = (set(k for k in os.environ if k.startswith('OPENAI_')) == "
        "{'OPENAI_API_KEY','OPENAI_BASE_URL','OPENAI_MODEL'} "
        "and not any(k.startswith('PG') for k in os.environ) "
        "and os.environ.get('OPENAI_MODEL') == 'session-fixture' "
        "and os.environ.get('OPENAI_BASE_URL') == 'https://endpoint.invalid/v1' "
        "and 'TYRELL_UNRELATED' not in os.environ "
        f"and hashlib.sha256(os.environ['OPENAI_API_KEY'].encode()).hexdigest() == '{hashlib.sha256(secret.encode()).hexdigest()}')\n"
        "if not ok: sys.exit(8)\n"
        "with Path('api-fixture.log').open('a') as f: print(sys.argv[1],file=f)\n",
        encoding="utf-8")
    # Stand-ins for the DB preflight and the ML process. The listener only acknowledges TCP.
    (root / "postgres_loader.py").write_text("def pg_error_reason(exc): return type(exc).__name__\n")
    (root / "psycopg.py").write_text(
        "import os\nclass Conn:\n"
        " def __enter__(self): return self\n def __exit__(self,*args): pass\n"
        " def execute(self,q): return self\n"
        " def fetchone(self): return os.environ['PGDATABASE'],os.environ['PGUSER'],'on'\n"
        "def connect(**kw):\n"
        " if any(k.startswith('OPENAI_') for k in os.environ): raise RuntimeError('credential routing')\n"
        " return Conn()\n", encoding="utf-8")
    (diagnostics / "diagnose.py").write_text(
        "import json,os,sys\nfrom pathlib import Path\n"
        "args=sys.argv[1:]; out=Path(args[args.index('--run-dir')+1]); out.mkdir(parents=True)\n"
        "qwen='--qwen-tools' in args\n"
        "ok=(int(os.environ['TYRELL_MAX_LLM_CALLS'])==(2 if qwen else 0) and "
        "bool(os.environ.get('OPENAI_API_KEY'))==qwen and ('--pair-arm' not in args) and "
        "int(args[args.index('--seconds')+1])==600)\n"
        "out.joinpath('report.json').write_text(json.dumps({'status':'ok' if ok else 'bad_env', "
        "'task':out.parent.name,'research_mode':'qwen_tools' if qwen else 'automl'}))\n"
        "sys.exit(0 if ok else 9)\n", encoding="utf-8")
    env = {k: v for k, v in os.environ.items() if not k.upper().startswith(("OPENAI_", "PG", "TYRELL_", "PYTHONPATH"))}
    env.update(OPENAI_MODEL="session-fixture", OPENAI_LOG="debug", PGPASSWORD="unused_pg_fixture",
               PATH=str(Path(sys.executable).parent) + os.pathsep + env["PATH"])
    def run(*args):
        proc = subprocess.run(
            [PS, "-NoProfile", "-NonInteractive", "-ExecutionPolicy", "Bypass", "-File",
             str(root / "run-real.ps1"), "-NonInteractive", *args], env=env, cwd=tmp_path,
            stdin=subprocess.DEVNULL, capture_output=True, timeout=25)
        assert secret.encode() not in proc.stdout + proc.stderr
        assert b"unused_pg_fixture" not in proc.stdout + proc.stderr
        assert not (root / "injected.txt").exists() and not (tmp_path / "injected.txt").exists()
        return proc
    return root, run


@pytest.mark.parametrize("mode,expected", [("-QwenCheck", "--check"), ("-QwenSmoke", "--smoke")])
def test_api_only_modes_read_dotenv_without_pg(runner, mode, expected):
    root, run = runner
    assert run(mode).returncode == 0
    assert (root / "api-fixture.log").read_text().splitlines() == [expected]
    assert not list(root.glob("runs/real-public/run-*.json"))


def test_bad_dotenv_never_echoes_source_or_starts_client(runner):
    root, run = runner
    (root / ".env").write_text('OPENAI_API_KEY="malformed_sent1nel\n')
    result = run("-QwenSmoke")
    assert result.returncode != 0 and b"malformed_sent1nel" not in result.stdout + result.stderr
    assert not (root / "api-fixture.log").exists()


@pytest.mark.parametrize("qwen,expected_tasks", [(True, ["task2"]), (False, ["task1", "task2"])])
def test_task_filter_and_credentials(runner, qwen, expected_tasks):
    root, run = runner
    if not qwen:
        # The default runner should completely ignore file contents.
        (root / ".env").write_text('OPENAI_API_KEY="malformed\n')
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        listener.listen()
        args = ["-Port", str(listener.getsockname()[1])]
        if qwen:
            args += ["-QwenTask2", "-Seconds", "7200"]
        result = run(*args)
    assert result.returncode == 0
    summary_path, = root.glob("runs/real-public/run-*.json")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert [row["Task"] for row in summary["results"]] == expected_tasks
    if qwen:
        assert (root / "api-fixture.log").read_text().splitlines() == ["--check"]
    else:
        assert not (root / "api-fixture.log").exists()
