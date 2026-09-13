import sys
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from Models.kmeans import KMeansScratch


X = [[0.0], [1.0], [9.0], [10.0]]
model = KMeansScratch(n_clusters=2, random_state=2).fit(X)

print(model.labels_)
