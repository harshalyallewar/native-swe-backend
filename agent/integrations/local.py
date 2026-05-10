import os

from deepagents.backends import LocalShellBackend


class PersistentLocalShellBackend(LocalShellBackend):
    def __init__(self, sandbox_id: str | None = None, **kwargs):
        super().__init__(**kwargs)
        self._sandbox_id = sandbox_id or super().id

    @property
    def id(self) -> str:
        return self._sandbox_id


def create_local_sandbox(sandbox_id: str | None = None):
    """Create a local shell sandbox with no isolation.

    WARNING: This runs commands directly on the host machine with no sandboxing.
    Only use for local development with human-in-the-loop enabled.

    The root directory defaults to the current working directory and can be
    overridden via the LOCAL_SANDBOX_ROOT_DIR environment variable.

    Args:
        sandbox_id: Optional existing sandbox ID to preserve identity across reconnects.

    Returns:
        LocalShellBackend instance implementing SandboxBackendProtocol.
    """
    root_dir = os.getenv("LOCAL_SANDBOX_ROOT_DIR", os.getcwd())

    return PersistentLocalShellBackend(
        sandbox_id=sandbox_id,
        root_dir=root_dir,
        inherit_env=True,
    )
