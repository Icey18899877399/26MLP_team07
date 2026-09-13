"""基础孤立森林最小用例。"""

from pathlib import Path
import sys


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from Models.isolation_forest import IsolationForestScratch


X = [[-0.2], [-0.1], [0.0], [0.1], [0.2], [0.3], [8.0]]
model = IsolationForestScratch(
    n_estimators=120,
    max_samples=7,
    contamination=1.0 / 7.0,
    random_state=7,
)
print(model.fit_predict(X))
