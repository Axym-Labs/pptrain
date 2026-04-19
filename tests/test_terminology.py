from __future__ import annotations

import importlib

import pytest

import pptrain
from pptrain import cli


def test_public_api_uses_task_terminology_only() -> None:
    legacy_create = "create_" + "mecha" + "nism"
    legacy_registered = "registered_" + "mecha" + "nisms"
    legacy_preset = "Mecha" + "nismPreset"
    assert hasattr(pptrain, "create_task")
    assert hasattr(pptrain, "registered_tasks")
    assert hasattr(pptrain, "TaskPreset")
    assert not hasattr(pptrain, legacy_create)
    assert not hasattr(pptrain, legacy_registered)
    assert not hasattr(pptrain, legacy_preset)


def test_cli_rejects_legacy_task_alias_command() -> None:
    legacy_command = "mecha" + "nisms"
    with pytest.raises(SystemExit):
        cli.build_parser().parse_args([legacy_command])


def test_legacy_package_alias_is_not_importable() -> None:
    legacy_package = "pptrain." + "mecha" + "nisms"
    with pytest.raises(ModuleNotFoundError):
        importlib.import_module(legacy_package)
