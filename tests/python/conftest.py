import pytest
import os
from pathlib import Path

@pytest.fixture(scope="session")
def jazelle_data_path():
    """
    Returns the path to the private jazelle file from env var.
    Skips tests if not found.
    """
    path = os.environ.get("JAZELLE_TEST_FILE")
    if not path or not os.path.exists(path):
        pytest.skip("JAZELLE_TEST_FILE environment variable not set or file missing.")
    return path

@pytest.fixture(scope="session")
def sample_event(jazelle_data_path):
    """Opens the file and returns the first event for inspection."""
    import jazelle
    with jazelle.open(jazelle_data_path) as f:
        yield f[0] # Return first event