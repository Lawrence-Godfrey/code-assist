from dataclasses import dataclass
from typing import Optional

from code_assistant.pipeline.coding.environment.config import DockerConfig
from code_assistant.pipeline.coding.environment.docker import DockerEnvironment


@dataclass
class EnvironmentFactory:
    """Factory for creating execution environments."""

    @staticmethod
    def create_environment(env_type: str, config: Optional[dict] = None):
        """
        Create appropriate environment based on type.

        Args:
            env_type: Type of environment to create ("docker", "venv")
            config: Configuration dictionary

        Returns:
            ExecutionEnvironment implementation
        """
        if env_type == "docker":
            docker_config = DockerConfig(**(config or {}))
            return DockerEnvironment(docker_config)
        else:
            raise ValueError(f"Unknown environment type: {env_type}")