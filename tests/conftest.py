"""Shared pytest fixtures."""
import urllib.request

import pytest

from needlestack_core.constants import OLLAMA_URL


@pytest.fixture(scope="session")
def ollama_running():
    """Skip the test if Ollama is not reachable."""
    try:
        with urllib.request.urlopen(f"{OLLAMA_URL}/api/tags", timeout=3):
            pass
    except Exception:
        pytest.skip("Ollama not reachable — start Ollama to run integration tests")
