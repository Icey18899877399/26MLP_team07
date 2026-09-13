"""KNN 基础版与优化版的最小调用示例。"""

import sys
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from Models import KNNScratch, OptimizedKNNScratch


X = [[0.0], [1.0], [4.0], [5.0]]
y = [0, 0, 1, 1]
samples = [[0.5], [4.5]]

basic_model = KNNScratch(n_neighbors=3).fit(X, y)
optimized_model = OptimizedKNNScratch(
    n_neighbors=3,
    p=2,
    weights="distance",
    standardize=True,
).fit(X, y)

print(basic_model.predict(samples))
print(optimized_model.predict(samples))
