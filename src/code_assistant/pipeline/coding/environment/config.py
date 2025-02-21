"""
Configuration classes for execution environments.

This module defines the configuration structures for different execution
environments, providing a clean way to customize environment behavior.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class BaseEnvironmentConfig:
    """Base configuration for all execution environments."""
    timeout: int = 300  # Timeout in seconds
    env_vars: Dict[str, str] = field(default_factory=dict)

    # Git configuration
    git_branch: str = "main"
    git_user_name: str = "Code Assistant"
    git_user_email: str = "code-assistant@example.com"


@dataclass
class DockerConfig(BaseEnvironmentConfig):
    """Configuration for Docker environment."""
    base_image: str = "python:3.8"
    memory_limit: str = "2g"
    cpu_limit: int = 2
    working_dir: str = "/workspace"

    # Container network settings
    network_mode: str = "bridge"
    ports: Dict[str, str] = field(default_factory=dict)

    # Additional packages to install
    apt_packages: List[str] = field(
        default_factory=lambda: ["git", "curl", "build-essential"])

    # Python-specific settings
    install_requirements: bool = True
    pip_packages: List[str] = field(
        default_factory=lambda: ["pytest", "pytest-cov"])
