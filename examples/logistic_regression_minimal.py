"""最简单的逻辑回归调用示例。"""

import sys
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from Models import LogisticRegressionScratch


X = [[-2.0], [-1.0], [1.0], [2.0]]
y = [0, 0, 1, 1]

model = LogisticRegressionScratch(learning_rate=0.2, max_iter=1_000)
model.fit(X, y)

print(model.predict([[-1.5], [1.5]]))
