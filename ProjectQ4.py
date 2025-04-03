import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import ListedColormap
from sklearn import neighbors, datasets, metrics
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

k = 3
x = cancerData.data[:, :2]
y = cancerData.target


trainingratio = 0.7

#  randomize the data for every run
randompermutation = np.random.permutation(len(x))
x = x[randompermutation]
y = y[randompermutation]

#  set the figure size to fit all 10 ROC curve plots
figure = plt.figure(figsize=(27, 9))

#  dividing the data into 70% training and 30% testing
trainingDataSize = int(trainingratio * len(x))
X_train = x[0:trainingDataSize, :]
y_train = y[0:trainingDataSize]
X_test = x[trainingDataSize:, :]
y_test = y[trainingDataSize:]

#  for loop to execute every classifier
for c in range(len(classifiers)):
    clf = classifiers[c]
    clf.fit(x, y)
    pred = clf.predict(X_test)

    TP = 0
    FP = 0
    TN = 0
    FN = 0

    #  count the number of TP, FN, TN, and FP
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

    #  generate the roc curve
    FPR, TPR, _ = metrics.roc_curve(y_test, pred)

    print(names[c])
    ConfusionMatrix = [[TP, FP], [FN, TN]]
    print("Confusion Matrix:", ConfusionMatrix)
    print("Accuracy:", (TP + TN) / (TP + TN + FP + FN) * 100, "%")
    print("Precision:", TP / (TP + FP) * 100, "%")
    print("Recall:", TP / (TP + FN) * 100, "%")
    auc_value = metrics.auc(FPR, TPR)
    print("AUC:", auc_value)
    print("===============================================")
    #  subplot was used to fit all plots in a single figure (2 rows, 5 columns)
    plt.subplot(2, 5, c+1)
    x_axis = [0, 1]
    y_axis = x_axis
    plt.plot(x_axis, y_axis)
    plt.plot(FPR, TPR)

    plt.title(names[c])
    c = c + 1

plt.show()