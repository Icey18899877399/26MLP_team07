"""Exercise the actual packaged models through the HTTP boundary."""
import pytest
from fastapi.testclient import TestClient
from backend.main import create_app


@pytest.fixture
def client():
    with TestClient(create_app()) as value:
        yield value


def test_live_catalog(client):
    assert client.get('/api/health').json()['ml_backend']['available']
    models = client.get('/api/models').json()
    assert {m['id'] for m in models} == {'kmeans', 'logistic_regression'}
    assert all(m['variants'] == ['optimized'] for m in models)
    assert all(m['compatible_datasets'] for m in models)
    datasets = client.get('/api/datasets').json()
    assert {d['sample_count'] for d in datasets} == {210, 569}


@pytest.mark.parametrize('model,dataset,metric', [
    ('logistic_regression', 'wdbc', 'accuracy'),
    ('kmeans', 'seeds', 'adjusted_rand_index'),
])
def test_real_experiment(client, model, dataset, metric):
    response = client.post('/api/experiments', json={
        'model': model, 'variant': 'optimized', 'dataset': dataset,
        'params': {'max_iter': 20}, 'random_state': 42,
        'test_size': 0.2 if dataset == 'wdbc' else None,
    })
    assert response.status_code == 200, response.text
    assert metric in response.json()['metrics']
    assert response.json()['diagnostics']['run_id']


@pytest.mark.parametrize('extra', [
    {'variant': 'base'}, {'dataset': 'wdbc'},
    {'params': {'unknown': 1}}, {'test_size': 0.2},
])
def test_domain_errors_are_400(client, extra):
    response = client.post('/api/experiments', json={
        'model': 'kmeans', 'dataset': 'seeds', 'variant': 'optimized', **extra,
    })
    assert response.status_code == 400, response.text
