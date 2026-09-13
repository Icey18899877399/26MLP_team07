"""优化 One-Class SVM 最小用例。"""

from pathlib import Path
import math
import sys


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from Models.one_class_svm_optimized import OptimizedOneClassSVMScratch


normal_X = [
    [math.cos(index * 2.0 * math.pi / 5.0), math.sin(index * 2.0 * math.pi / 5.0)]
    for index in range(5)
]
model = OptimizedOneClassSVMScratch(
    nu=0.1,
    kernel="rbf",
    gamma=2.0,
    standardize=False,
    max_iter=800,
    tol=1e-8,
).fit(normal_X)
print(model.predict(normal_X + [[3.0, 3.0]]))
