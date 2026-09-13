"""优化孤立森林最小用例。"""

from pathlib import Path
import sys


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from Models.isolation_forest_optimized import OptimizedIsolationForestScratch


X = [[-0.2], [-0.1], [0.0], [0.1], [0.2], [0.3], [8.0]]
model = OptimizedIsolationForestScratch(
    n_estimators=120,
    max_samples=7,
    contamination=1.0 / 7.0,
    n_split_candidates=4,
    random_state=7,
)
print(model.fit_predict(X))
