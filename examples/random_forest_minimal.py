"""随机森林基础版最小调用示例。"""

import sys
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from Models import RandomForestClassifierScratch


X = [[0.0], [0.5], [1.0], [3.0], [3.5], [4.0]]
y = [0, 0, 0, 1, 1, 1]

model = RandomForestClassifierScratch(
    n_estimators=15,
    max_depth=3,
    max_features=None,
    random_state=42,
).fit(X, y)

print(model.predict([[0.2], [3.8]]))

