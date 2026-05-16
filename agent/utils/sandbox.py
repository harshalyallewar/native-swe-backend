import os

from agent.integrations.daytona import create_daytona_sandbox


def create_sandbox(sandbox_id: str | None = None):
    """Create or reconnect to a sandbox using Daytona.

    Args:
        sandbox_id: Optional existing sandbox ID to reconnect to.

    Returns:
        A sandbox backend implementing SandboxBackendProtocol.
    """
    return create_daytona_sandbox(sandbox_id)


def validate_sandbox_startup_config() -> None:
    """Validate the configured sandbox provider's env vars at server startup.

    Raises ValueError if the active provider's configuration is invalid.
    Called from the FastAPI lifespan hook so errors surface at boot rather
    than on the first sandbox creation.
    """
    if not os.getenv("DAYTONA_API_KEY"):
        raise ValueError("DAYTONA_API_KEY environment variable is required")
