import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import export_text
from sklearn import metrics
# SETTING DATA SET
Bcancer = load_breast_cancer() # Load data set
df = pd.DataFrame(Bcancer.data, columns=Bcancer.feature_names) # Look at the frame
df['class'] = Bcancer.target
print(df.sample(4))
classes = np.unique(Bcancer.target) #Gives number of classes so we have 2 classes (0,1)
print(classes)
classes_names = np.unique(Bcancer.target_names) # Gets classes names
print(classes_names) #gives class 1 and class 2 names specifically
# match what each number represents
classes = dict(zip(classes,classes_names)) # DICTIONARY
print(classes) # After we applied DICTIONARY
df['class'] = df['class'].replace(classes) #Replace class column (0, 1) with benign and malignant. replace every number with its value
print(df.sample(4))
x = df.drop(columns="class")
y = df["class"]
feature_names = x.columns # the feature names column are mean radius, mean texture...) (used for training)
labels = y.unique() # labels are benign and malignant only.
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.3) # (60% training and 40 % testing)
#create model
tree = DecisionTreeClassifier(max_depth=3,
random_state=42) # MAXIMUM DEPTH IS 3 LEVELS ONLY (MINIMUM)
tree.fit(x_train,y_train)
print(export_text(tree, feature_names=list(feature_names)))
# Calculate Accuracy using confusion matrix:
#find predicted t:
y_pred = tree.predict(x_test)
#calculate confusion matrix:
confusionmatrix = metrics.confusion_matrix(y_test,y_pred)
print(confusionmatrix) #classes 0 1 / 0 1
#calculate accuracy using the confusion matrix created Accuracy using TT or Diagnal sum/di = 0.95 Accuracy