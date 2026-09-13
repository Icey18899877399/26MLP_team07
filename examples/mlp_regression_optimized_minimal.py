import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from Models.mlp_regression_optimized import OptimizedMLPRegressorScratch


X = [[value / 5.0] for value in range(-10, 11)]
y = [row[0] ** 2 for row in X]

model = OptimizedMLPRegressorScratch(
    hidden_layer_sizes=(12, 6),
    activation="tanh",
    learning_rate=0.02,
    max_iter=1_000,
    batch_size=8,
    random_state=42,
).fit(X, y)

print(model.predict([[1.5]])[0])
