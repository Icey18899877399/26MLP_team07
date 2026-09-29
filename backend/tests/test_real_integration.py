from importlib.util import find_spec

import pytest

from backend.integration.ml_backend import PackageMLBackend


@pytest.mark.integration
def test_real_ml_core_public_contract_smoke() -> None:
    if find_spec("ml_core") is None:
        pytest.skip("ml_core is not available at the pinned upstream commit")

    backend = PackageMLBackend()

    assert backend.status().available, backend.status().detail
    assert isinstance(backend.list_models(), list)
    assert isinstance(backend.list_datasets(), list)
