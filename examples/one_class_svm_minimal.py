"""基础 One-Class SVM 最小用例。"""

from pathlib import Path
import sys


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from Models.one_class_svm import OneClassSVMScratch


normal_X = [[1.5], [1.8], [2.0], [2.2], [2.5]]
model = OneClassSVMScratch(nu=0.1, max_iter=1200, tol=1e-8).fit(normal_X)
print(model.predict(normal_X + [[-3.0]]))
