"""随机森林优化版最小调用示例。"""

import sys
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from Models import OptimizedRandomForestClassifierScratch


def main():
    X = [[0.0], [0.5], [1.0], [3.0], [3.5], [4.0]]
    y = [0, 0, 0, 1, 1, 1]

    model = OptimizedRandomForestClassifierScratch(
        n_estimators=30,
        max_depth=3,
        max_features=None,
        voting="soft",
        oob_score=True,
        n_jobs=2,
        random_state=42,
    ).fit(X, y)

    print(model.predict([[0.2], [3.8]]))
    print(model.oob_score_)


if __name__ == "__main__":
    main()
