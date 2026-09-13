"""CART 决策树基础版与优化版的最小调用示例。"""

import sys
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from Models import CARTClassifierScratch, OptimizedCARTClassifierScratch


X = [[0.0], [1.0], [2.0], [3.0], [4.0], [5.0]]
y = [0, 0, 1, 1, 0, 0]
samples = [[0.5], [2.5], [4.5]]

basic_model = CARTClassifierScratch(max_depth=None).fit(X, y)
optimized_model = OptimizedCARTClassifierScratch(
    max_depth=3,
    min_samples_leaf=1,
    ccp_alpha=0.0,
    random_state=42,
).fit(X, y)

print(basic_model.predict(samples))
print(optimized_model.predict(samples))
