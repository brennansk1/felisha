from __future__ import annotations

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from causalrag import __version__
from causalrag.cli.main import PROTOCOL_FILENAME, app
from causalrag.core.protocol import StudyProtocol

runner = CliRunner()


def test_version_flag() -> None:
    result = runner.invoke(app, ["--version"])
    assert result.exit_code == 0
    assert __version__ in result.stdout


def test_init_creates_skeleton(tmp_path: Path) -> None:
    result = runner.invoke(app, ["init", "demo", "--path", str(tmp_path / "demo")])
    assert result.exit_code == 0, result.stdout
    project = tmp_path / "demo"
    assert (project / PROTOCOL_FILENAME).exists()
    assert (project / "data").is_dir()
    assert (project / "reports").is_dir()
    assert (project / ".causalrag" / "cassettes").is_dir()
    assert (project / ".causalrag" / "history.jsonl").exists()
    assert (project / "README.md").exists()

    p = StudyProtocol.read_yaml(project / PROTOCOL_FILENAME)
    assert p.name == "demo"
    assert p.tier == "academic"


def test_init_refuses_nonempty_without_force(tmp_path: Path) -> None:
    target = tmp_path / "demo"
    target.mkdir()
    (target / "existing.txt").write_text("hi")
    result = runner.invoke(app, ["init", "demo", "--path", str(target)])
    assert result.exit_code == 1
    assert "not empty" in result.stdout


def test_init_respects_tier_option(tmp_path: Path) -> None:
    result = runner.invoke(
        app,
        ["init", "demo", "--path", str(tmp_path / "demo"), "--tier", "domain-expert"],
    )
    assert result.exit_code == 0
    p = StudyProtocol.read_yaml(tmp_path / "demo" / PROTOCOL_FILENAME)
    assert p.tier == "domain-expert"


def test_init_rejects_unknown_tier(tmp_path: Path) -> None:
    result = runner.invoke(
        app, ["init", "demo", "--path", str(tmp_path / "demo"), "--tier", "nonsense"]
    )
    assert result.exit_code == 2


def test_validate_on_scaffolded_project(tmp_path: Path) -> None:
    project = tmp_path / "demo"
    runner.invoke(app, ["init", "demo", "--path", str(project)])
    result = runner.invoke(app, ["validate", str(project / PROTOCOL_FILENAME)])
    assert result.exit_code == 0, result.stdout
    assert "OK" in result.stdout


def test_validate_reports_missing_file(tmp_path: Path) -> None:
    result = runner.invoke(app, ["validate", str(tmp_path / "missing.yaml")])
    assert result.exit_code == 2


def test_validate_reports_corrupt_file(tmp_path: Path) -> None:
    bad = tmp_path / "bad.yaml"
    bad.write_text("name: demo\ntier: not_a_real_tier\n")
    result = runner.invoke(app, ["validate", str(bad)])
    assert result.exit_code == 1


def test_doctor_json_output() -> None:
    result = runner.invoke(app, ["doctor", "--json"])
    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert "python_version" in payload
    assert "tier" in payload
    assert "tier_label" in payload
    assert isinstance(payload["warnings"], list)
    assert "recommended_models" in payload
    for slot in ("discovery", "hypothesize", "utility"):
        assert slot in payload["recommended_models"]


def test_doctor_save_into_project(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    project = tmp_path / "demo"
    runner.invoke(app, ["init", "demo", "--path", str(project)])
    monkeypatch.chdir(project)
    result = runner.invoke(app, ["doctor", "--save", "--json"])
    assert result.exit_code == 0
    saved = project / ".causalrag" / "hardware.json"
    assert saved.exists()
    payload = json.loads(saved.read_text())
    assert "python_version" in payload
    assert "recommended_models" in payload
