import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import ListedColormap
from sklearn import neighbors, datasets
from sklearn.datasets import load_breast_cancer
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.inspection import DecisionBoundaryDisplay
import numpy as np
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

names = [
    "Nearest Neighbors",
    "Linear SVM",
    "RBF SVM",
    "Gaussian Process",
    "Decision Tree",
    "Random Forest",
    "Neural Net",
    "AdaBoost",
    "Naive Bayes",
    "QDA",
]

classifiers = [
    KNeighborsClassifier(3),
    SVC(kernel="linear", C=0.025),
    SVC(gamma=2, C=1),
    GaussianProcessClassifier(1.0 * RBF(1.0)),
    DecisionTreeClassifier(max_depth=5),
    RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1),
    MLPClassifier(alpha=1, max_iter=1000),
    AdaBoostClassifier(),
    GaussianNB(),
    QuadraticDiscriminantAnalysis(),
]
cancerData = load_breast_cancer()

x = cancerData.data[:, :2]
y = cancerData.target

trainingratio = 0.7

randompermutation = np.random.permutation(len(x))
x = x[randompermutation]
y = y[randompermutation]

input1 = int(input("Enter Radius: "))
input2 = int(input("Enter Texture: "))
X_test = [[input1, input2]]

trainingDataSize = int(trainingratio * len(x))
y_test = y[trainingDataSize:]
X_test1 = x[trainingDataSize:, :]
accuracy_list = []
pred_list = []
for c in range(len(classifiers)):
    clf = classifiers[c]
    clf.fit(x, y)
    pred = clf.predict(X_test)
    pred1 = clf.predict(X_test1)

    if pred == 1:
        print(names[c], ":", 'M')
        pred_list.append('M')
    else:
        print(names[c], ":", 'B')
        pred_list.append('B')

    TP = 0
    FP = 0
    TN = 0
    FN = 0

    for i in range(len(pred1)):
        if pred1[i] == y_test[i]:
            if pred1[i] == 1:
                TP += 1
            elif pred1[i] == 0:
                TN += 1
        elif pred1[i] != y_test[i]:
            if pred1[i] == 1:
                FP += 1
            elif pred1[i] == 0:
                FN += 1

        i = i + 1
    c = c + 1

    newAccuracy = (TP + TN) / (TP + TN + FP + FN) * 100
    accuracy_list.append(newAccuracy)

print("Ensemble Classifier Output: (best three classifiers)")

for i in range(0, 3):
    currentBestAccuracy = 0
    currentBest_name = ""
    currentBest_pred = ''
    for j in range(len(accuracy_list)):

        if accuracy_list[j] > currentBestAccuracy:
            currentBestAccuracy = accuracy_list[j]
            currentBest_name = names[j]
            currentBest_pred = pred_list[j]

    print("Classifier", currentBest_name, "predicted", currentBest_pred)

    accuracy_list.remove(currentBestAccuracy)
    names.remove(currentBest_name)
    pred_list.remove(currentBest_pred)
    i = i + 1