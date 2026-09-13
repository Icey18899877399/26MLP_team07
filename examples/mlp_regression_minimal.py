import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from Models.mlp_regression import MLPRegressorScratch


X = [[-2.0], [-1.0], [0.0], [1.0], [2.0]]
y = [4.0, 1.0, 0.0, 1.0, 4.0]

model = MLPRegressorScratch(
    hidden_size=8,
    learning_rate=0.03,
    max_iter=3_000,
    random_state=42,
).fit(X, y)

print(model.predict([[1.5]])[0])
