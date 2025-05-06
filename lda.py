import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

iris = load_iris()
x = iris.data
y = iris.target

sc = StandardScaler()
X = sc.fit_transform(x)

x_train, x_test,y_train, y_test = train_test_split(x, y,test_size=0.2,random_state=42)

lda = LinearDiscriminantAnalysis(n_components=2)
x_train = lda.fit_transform(x_train, y_train)
x_test = lda.transform(x_test)

plt.scatter(
    x_train[:, 0], x_train[:, 1],
    c=y_train,
    cmap='cividis',
    alpha=0.7, edgecolors='k'
)
plt.title('LDA: Training Data')
plt.xlabel('LD1')
plt.ylabel('LD2')
plt.colorbar()
plt.show()

classifier = LinearDiscriminantAnalysis()
classifier.fit(x_train, y_train)
y_pred = classifier.predict(x_test)

accuracy = accuracy_score(y_test, y_pred)
print('Accuracy: ', accuracy)
conf_m = confusion_matrix(y_test, y_pred)
print('Covariance Matrix:\n', conf_m)

