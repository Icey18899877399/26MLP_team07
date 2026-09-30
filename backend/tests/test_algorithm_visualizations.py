"""Every advertised algorithm must train and deliver charts through HTTP."""
import json

import pytest
from fastapi.testclient import TestClient
from backend.main import create_app


CASES = [
    ('logistic_regression', 'wdbc', 'accuracy'),
    ('knn', 'wdbc', 'accuracy'),
    ('gaussian_naive_bayes', 'wdbc', 'accuracy'),
    ('cart_decision_tree', 'wdbc', 'accuracy'),
    ('random_forest', 'wdbc', 'accuracy'),
    ('linear_regression', 'concrete', 'rmse'),
    ('gbdt_regression', 'concrete', 'rmse'),
    ('mlp_regression', 'concrete', 'rmse'),
    ('kmeans', 'seeds', 'adjusted_rand_index'),
    ('dbscan', 'seeds', 'adjusted_rand_index'),
    ('isolation_forest', '6_cardio', 'anomaly_f1'),
    ('one_class_svm', '6_cardio', 'anomaly_f1'),
]


@pytest.mark.integration
@pytest.mark.parametrize('family,dataset,metric', CASES)
def test_default_algorithm_returns_real_chart_series(family, dataset, metric):
    with TestClient(create_app()) as client:
        response = client.post('/api/experiments', json={
            'model': family + '.optimized', 'dataset': dataset,
            'random_state': 42,
        })
    assert response.status_code == 200, response.text
    result = response.json()
    assert metric in result['metrics']
    json.dumps(result, allow_nan=False)
    charts = result['metadata'].get('visualizations', [])
    assert len(charts) >= 2, result['model']
    assert len({chart['id'] for chart in charts}) == len(charts)
    for chart in charts:
        assert chart['title'] and chart['description']
        assert any(series.get('data') for series in chart['option']['series'])
