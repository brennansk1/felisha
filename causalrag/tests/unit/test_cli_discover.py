from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from typer.testing import CliRunner

from causalrag.cli.main import PROTOCOL_FILENAME, app
from causalrag.core.protocol import StudyProtocol

runner = CliRunner()


def _write_synth(path: Path, n: int = 300) -> None:
    rng = np.random.default_rng(0)
    df = pd.DataFrame(
        {
            "age": rng.integers(18, 55, size=n),
            "income": rng.gamma(2.0, 1000, size=n),
            "treat": rng.integers(0, 2, size=n),
            "outcome": rng.gamma(2.0, 1500, size=n),
        }
    )
    df.to_csv(path, index=False)


def test_discover_no_llm_updates_protocol(tmp_path: Path) -> None:
    project = tmp_path / "demo"
    runner.invoke(app, ["init", "demo", "--path", str(project)])
    csv = project / "data" / "tiny.csv"
    _write_synth(csv)

    result = runner.invoke(
        app,
        [
            "discover",
            str(csv),
            "--project",
            str(project),
            "--treatment",
            "treat",
            "--outcome",
            "outcome",
            "--no-llm",
        ],
    )
    assert result.exit_code == 0, result.stdout

    sidecar = project / ".causalrag" / "discovery.json"
    assert sidecar.exists()
    payload = json.loads(sidecar.read_text())
    assert "binary_treatment" in payload["flags"]
    assert "continuous_outcome" in payload["flags"]

    protocol = StudyProtocol.read_yaml(project / PROTOCOL_FILENAME)
    assert protocol.discovery is not None
    assert {c.name for c in protocol.discovery.columns} == {
        "age",
        "income",
        "treat",
        "outcome",
    }
    from causalrag.core.flags import DataFlag

    assert DataFlag.BINARY_TREATMENT in protocol.flags
    assert protocol.dataset is not None
    assert protocol.dataset.n_rows == 300


def test_discover_without_init_errors(tmp_path: Path) -> None:
    csv = tmp_path / "tiny.csv"
    _write_synth(csv)
    result = runner.invoke(
        app, ["discover", str(csv), "--project", str(tmp_path), "--no-llm"]
    )
    assert result.exit_code == 2
    assert "No StudyProtocol" in result.stdout
