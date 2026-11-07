"""The runner's bookkeeping: what a result records about the system, and what it refuses."""

import re
from pathlib import Path

import pytest

from groundhog import run
from groundhog.model import Model

ROOT = Path(__file__).resolve().parent.parent

CACHE = """\
from http.server import ThreadingHTTPServer

from helpers import Handler


class Cache(ThreadingHTTPServer):
    def __init__(self, port, upstream):
        super().__init__(("127.0.0.1", port), Handler)
"""

HELPERS = "from http.server import BaseHTTPRequestHandler as Handler\n"
CHANGED = "from http.server import SimpleHTTPRequestHandler as Handler\n"


@pytest.fixture
def project(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "mycache.py").write_text(CACHE)
    (tmp_path / "helpers.py").write_text(HELPERS)
    return tmp_path


def test_a_python_cache_is_digested_with_its_own_imports(project):
    sources = run.system_sources("py:mycache:Cache")
    assert sorted(Path(p).name for p in sources) == ["helpers.py", "mycache.py"]
    before = run.system_digest("py:mycache:Cache")
    (project / "helpers.py").write_text(CHANGED)
    assert run.system_digest("py:mycache:Cache") != before


def test_a_script_is_digested_and_not_the_interpreter(project):
    sources = run.system_sources("cmd:python mycache.py SYSTEM_PORT MODEL_PORT")
    assert sorted(Path(p).name for p in sources) == ["helpers.py", "mycache.py"]


def test_installed_libraries_are_left_to_the_version_record(project):
    (project / "mycache.py").write_text("import json\nimport http.server\n")
    assert [Path(p).name for p in run.system_sources("py:mycache:Cache")] == ["mycache.py"]


def test_the_examples_are_digested(monkeypatch):
    monkeypatch.chdir(ROOT)
    sources = run.system_sources("py:examples.exact_cache:ExactCache")
    assert [Path(p).name for p in sources] == ["exact_cache.py", "proxy.py"]


def test_a_command_gets_settings_credentials_and_plumbing_only():
    environ = {
        "PATH": "/bin",
        "SYSTEMROOT": "C:\\Windows",
        "GROUNDHOG_THRESHOLD": "0.8",
        "OPENROUTER_API_KEY": "k",
        "UNRELATED": "x",
    }
    assert set(run.child_environment(environ)) == {
        "PATH",
        "SYSTEMROOT",
        "GROUNDHOG_THRESHOLD",
        "OPENROUTER_API_KEY",
    }


def test_settings_are_recorded_and_credentials_withheld():
    environ = {"GROUNDHOG_THRESHOLD": "0.8", "GROUNDHOG_API_KEY": "s3cr3t", "HOME": "/"}
    recorded = run.settings_of(environ)
    assert recorded["GROUNDHOG_THRESHOLD"] == "0.8"
    assert "s3cr3t" not in recorded["GROUNDHOG_API_KEY"]
    assert "HOME" not in recorded


def test_the_overall_line_pools_by_baseline_cost():
    rows = [
        {"control_tokens": 300, "saved": 0.5, "earned": 0.5, "gate": "pass", "wrong_writes": 0},
        {"control_tokens": 100, "saved": -0.2, "earned": 0.0, "gate": "FAIL", "wrong_writes": 4},
    ]
    line = run.summary(rows)
    assert "32.5% gross, 37.5% earned" in line
    assert "a wrong write on 1 of 2 pairs (4 wrong writes in all)" in line


def test_a_saved_baseline_of_another_length_is_refused():
    model = Model(0)
    try:
        saved = {"support/lookup": {"runs": [{"effects": []}], "problems": []}}
        with pytest.raises(SystemExit, match="same --limit"):
            run.evaluate(
                None, model, ["lookup"], 0, domains=("support",), limit=2, control_in=saved
            )
    finally:
        model.server_close()


def test_every_readme_command_names_a_real_option():
    options = {a for action in run.build()._actions for a in action.option_strings}
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    guide = (ROOT / "docs" / "writing-a-cache.md").read_text(encoding="utf-8")
    for line in re.findall(r"^groundhog .*$", readme + guide, re.M):
        for flag in re.findall(r"(?<!\S)(--[a-z-]+)", line):
            assert flag in options, line


def test_the_readme_shows_the_example_as_it_is():
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    shown = next(b for b in re.findall(r"```python\n(.*?)```", readme, re.S) if "ExactCache" in b)
    assert shown in (ROOT / "examples" / "exact_cache.py").read_text(encoding="utf-8")


def test_the_readme_dataset_example_runs(capsys):
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    shown = next(b for b in re.findall(r"```python\n(.*?)```", readme, re.S) if "records" in b)
    exec(shown, {})
    assert len(capsys.readouterr().out.splitlines()) == 3


def test_an_unknown_family_is_refused(monkeypatch):
    monkeypatch.setattr("sys.argv", ["groundhog", "--families", "lookup,lookp"])
    with pytest.raises(SystemExit, match="lookp"):
        run.main()
