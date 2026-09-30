"""The Chinese error boundary must itself be JSON-safe on malformed input."""
import pytest
from fastapi.testclient import TestClient

from backend.main import create_app
from tests.fakes import FakeMLBackend


@pytest.mark.parametrize("body", [
    '{"model":"kmeans.optimized","dataset":"seeds","test_size":NaN}',
    '{"model":"kmeans.optimized","dataset":"seeds","test_size":Infinity}',
    '"not an object"',
    '{"model":',
])
def test_malformed_input_returns_validation_error_not_server_error(body):
    with TestClient(create_app(FakeMLBackend()), raise_server_exceptions=False) as client:
        response = client.post('/api/experiments', content=body,
                               headers={'Content-Type': 'application/json'})
    assert response.status_code == 422, response.text
    detail = response.json()['detail']
    assert detail and all({'loc', 'msg', 'type'} <= item.keys() for item in detail)
    assert all(any('\u4e00' <= ch <= '\u9fff' for ch in item['msg']) for item in detail)
