# machine-learning
#the_objective_of_over_ML_code_is_to_check_the_quality_of_wine_in_given_.csv_file
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
wine=pd.read_csv("quality.csv", delimiter=';')
wine.head()
wine.isnull().sum()
plt.figure(figsize=(10,5))
sns.countplot(wine['quality'])
plt.show()
plt.figure(figsize=(10,5))
sns.barplot(x='quality',y='alcohol',data=wine,palette='inferno')
plt.show()
plt.figure(figsize=(10,5))
sns.pairplot(wine)
plt.show()
plt.figure(figsize=(10,5))
sns.scatterplot(x='citric acid' , y='pH', data=wine)
plt.show()
plt.figure(figsize=(10,5))
sns.heatmap(wine.corr(), annot=True)
plt.show()
X=wine.drop(['quality'],axis=1)
Y=wine['quality']
X=wine.drop(['quality'],axis=1)
Y=wine['quality']X=wine.drop(['quality'],axis=1)
Y=wine['quality']
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import LinearSVC
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
LR = LogisticRegression()
KNN = KNeighborsClassifier()
NB = GaussianNB()
LSVM = LinearSVC()
NLSVM = SVC(kernel='rbf')
DT = DecisionTreeClassifier()
RF = RandomForestClassifier()
LR_fit = LR.fit(X_train, Y_train)
KNN_fit = KNN.fit(X_train, Y_train)
NB_fit = NB.fit(X_train, Y_train)
LSVM_fit = LSVM.fit(X_train, Y_train)
NLSVM_fit = NLSVM.fit(X_train, Y_train)
DT_fit = DT.fit(X_train, Y_train)
RF_fit = RF.fit(X_train, Y_train)
LR_pred = LR_fit.predict(X_test)
KNN_pred = KNN_fit.predict(X_test)
NB_pred = NB_fit.predict(X_test)
LSVM_pred = LSVM_fit.predict(X_test)
NLSVM_pred = NLSVM_fit.predict(X_test)
DT_pred = DT_fit.predict(X_test)
RF_pred = RF_fit.predict(X_test)
from sklearn.metrics import accuracy_score
print("Logistic Regression is %f percent accurate" % (accuracy_score(LR_pred, Y_test)*100))
print("KNN is %f percent accurate" % (accuracy_score(KNN_pred, Y_test)*100))
print("Naive Bayes is %f percent accurate" % (accuracy_score(NB_pred, Y_test)*100))
print("Linear SVMs is %f percent accurate" % (accuracy_score(LSVM_pred, Y_test)*100))
print("Non Linear SVMs is %f percent accurate" % (accuracy_score(NLSVM_pred, Y_test)*100))
print("Decision Trees is %f percent accurate" % (accuracy_score(DT_pred, Y_test)*100))
print("Random Forests is %f percent accurate" % (accuracy_score(RF_pred, Y_test)*100))
