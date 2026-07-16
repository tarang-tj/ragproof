"""CLI smoke tests -- version, help, and the offline demo (no model downloads)."""
from ragproof.cli import main


def test_version_prints(capsys):
    assert main(["version"]) == 0
    assert "ragproof" in capsys.readouterr().out


def test_no_command_prints_help(capsys):
    assert main([]) == 0
    assert "usage" in capsys.readouterr().out.lower()


def test_demo_command_runs_offline(capsys):
    assert main(["demo"]) == 0
    assert "hybrid" in capsys.readouterr().out
