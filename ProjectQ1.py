import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.neighbors import KNeighborsClassifier

cancerData = load_breast_cancer()

k = 9
x = cancerData.data[:, :2]
y = cancerData.target

trainingratio = 0.7

trainingDataSize = int(trainingratio * len(x))
X_train = x[0:trainingDataSize, :]
y_train = y[0:trainingDataSize]
X_test = x[trainingDataSize:, :]
y_test = y[trainingDataSize:]

clf = KNeighborsClassifier(k)
clf.fit(x, y)
score = clf.score(x, y)
pred = clf.predict(X_test)

TP = 0
FP = 0
TN = 0
FN = 0

for i in range(len(pred)):
    if pred[i] == y_test[i]:
        if pred[i] == 1:
            TP += 1
        elif pred[i] == 0:
            TN += 1
    elif pred[i] != y_test[i]:
        if pred[i] == 1:
            FP += 1
        elif pred[i] == 0:
            FN += 1

    i = i + 1


print("Results when K value is equal to", k, ":")
ConfusionMatrix = [[TP, FP], [FN, TN]]
print("Confusion Matrix", ConfusionMatrix)
print("Accuracy ", (TP + TN) / (TP + TN + FP + FN) * 100, "%")
print("Precision", TP / (TP + FP) * 100, "%")
print("Recall", TP / (TP + FN) * 100, "%")
