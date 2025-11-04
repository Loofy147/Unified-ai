
import pytest
import os
from pathlib import Path
import sys

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from core.config import Settings

def test_settings_load_defaults():
    """
    Tests that the Settings class loads the default values when no
    environment variables are set.
    """
    # Clear any potential env vars
    for key in os.environ:
        if key.startswith("MODEL_ZOO_") or key.startswith("RESOURCE_MANAGER_"):
            del os.environ[key]

    settings = Settings()

    # Assert default values
    assert settings.model_zoo.storage_path == "data/models"
    assert settings.resource_manager.max_cpu == 100.0
    assert settings.resource_manager.max_memory_mb == 4096.0

def test_settings_load_from_env_vars(monkeypatch):
    """
    Tests that the Settings class correctly overrides defaults with values
    from environment variables.
    """
    # Use monkeypatch to set environment variables for the duration of the test
    monkeypatch.setenv("MODEL_ZOO_STORAGE_PATH", "/tmp/test_models")
    monkeypatch.setenv("RESOURCE_MANAGER_MAX_CPU", "200.5")
    monkeypatch.setenv("RESOURCE_MANAGER_MAX_MEMORY_MB", "8192")

    # The settings object needs to be re-instantiated to load the new env vars
    settings = Settings()

    # Assert that the values were loaded from the environment
    assert settings.model_zoo.storage_path == "/tmp/test_models"
    assert settings.resource_manager.max_cpu == 200.5
    assert settings.resource_manager.max_memory_mb == 8192.0
