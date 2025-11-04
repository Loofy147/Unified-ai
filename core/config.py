
"""
Centralized Configuration Management for the Unified AI System.

This module uses Pydantic's BaseSettings to provide a type-safe,
environment-aware configuration system. It allows for easy management of
settings for different environments (dev, test, prod) and centralizes all
configurable parameters.
"""
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field

class ModelZooSettings(BaseSettings):
    """Configuration for the ModelZoo."""
    model_config = SettingsConfigDict(env_prefix="MODEL_ZOO_")

    storage_path: str = Field(
        "data/models",
        description="The file system path where models should be stored."
    )

class ResourceManagerSettings(BaseSettings):
    """Configuration for the ResourceManager."""
    model_config = SettingsConfigDict(env_prefix="RESOURCE_MANAGER_")

    max_cpu: float = Field(
        100.0,
        description="Maximum total CPU units available for allocation."
    )
    max_memory_mb: float = Field(
        4096.0,
        description="Maximum total memory in MB available for allocation."
    )

class Settings(BaseSettings):
    """
    Main settings class that aggregates all component settings.
    It loads configuration from a .env file and environment variables.
    """
    model_config = SettingsConfigDict(
        env_file='.env',
        env_nested_delimiter='__',
        extra='ignore'
    )

    # Component-specific settings are instantiated on-demand
    model_zoo: ModelZooSettings = Field(default_factory=ModelZooSettings)
    resource_manager: ResourceManagerSettings = Field(default_factory=ResourceManagerSettings)

# Singleton instance of the settings
settings = Settings()
