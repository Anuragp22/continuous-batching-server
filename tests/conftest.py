from collections.abc import Iterator

import pytest
from fastapi.testclient import TestClient

from cbserver.api.app import create_app


@pytest.fixture(autouse=True)
def _reset_sse_starlette_appstatus() -> None:
    from sse_starlette.sse import AppStatus

    AppStatus.should_exit_event = None


@pytest.fixture
def client() -> Iterator[TestClient]:
    with TestClient(create_app()) as test_client:
        yield test_client
