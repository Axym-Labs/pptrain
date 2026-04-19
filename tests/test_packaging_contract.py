from pathlib import Path


def test_default_install_includes_replication_dependencies() -> None:
    pyproject = Path(__file__).resolve().parents[1] / "pyproject.toml"
    contents = pyproject.read_text(encoding="utf-8")
    project_block = contents.split("[project.optional-dependencies]", maxsplit=1)[0]

    assert '"datasets>=2.18"' in project_block
    assert '"pandas>=2.2"' in project_block
