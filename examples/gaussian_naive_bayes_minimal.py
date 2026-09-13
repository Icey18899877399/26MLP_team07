"""高斯朴素贝叶斯基础版与优化版的最小调用示例。"""

import sys
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from Models import GaussianNaiveBayesScratch, OptimizedGaussianNaiveBayesScratch


X = [[0.0], [0.4], [4.8], [5.2], [9.8], [10.2]]
y = ["left", "left", "middle", "middle", "right", "right"]
samples = [[0.2], [5.0], [10.0]]

basic_model = GaussianNaiveBayesScratch(variance_floor=1e-9).fit(X, y)
optimized_model = OptimizedGaussianNaiveBayesScratch(
    var_smoothing=1e-9,
    class_prior=None,
).fit(X, y)

print(basic_model.predict(samples))
print(optimized_model.predict(samples))
