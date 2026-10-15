import pytest

from src.config import configure_torch, load_config


@pytest.fixture(scope="session")
def cfg():
    c = load_config()
    configure_torch(c)
    return c
