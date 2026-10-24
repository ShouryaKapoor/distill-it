from typer.testing import CliRunner

import distill_it
from distill_it.cli import app


def test_package_has_version():
    assert distill_it.__version__


def test_cli_version():
    result = CliRunner().invoke(app, ["--version"])
    assert result.exit_code == 0
    assert distill_it.__version__ in result.output
