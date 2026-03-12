"""Tests for the /ping health check endpoint."""

import pytest
from fastapi.testclient import TestClient

from dashboard.server import app


@pytest.fixture
def client():
    return TestClient(app)


def test_ping_get(client):
    resp = client.get("/ping")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}


def test_ping_head(client):
    resp = client.head("/ping")
    assert resp.status_code == 200
    # HEAD responses have no body
    assert resp.content == b""
