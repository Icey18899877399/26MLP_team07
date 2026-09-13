import sys
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from Models.kmeans_optimized import OptimizedKMeansScratch


X = [[0.0], [1.0], [9.0], [10.0]]
model = OptimizedKMeansScratch(
    n_clusters=2,
    init="k-means++",
    n_init=5,
    standardize=True,
    random_state=2,
).fit(X)

print(model.labels_)
