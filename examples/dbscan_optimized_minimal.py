from pathlib import Path
import sys


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from Models.dbscan_optimized import OptimizedDBSCANScratch


X = [[0.0], [0.1], [0.2], [5.0], [5.1], [5.2], [10.0]]
model = OptimizedDBSCANScratch(
    eps=0.25,
    min_samples=2,
    algorithm="kd_tree",
    standardize=False,
)
print(model.fit_predict(X))
