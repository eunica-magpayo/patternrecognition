import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

iris= load_iris()
x= iris.data
y= iris.target

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.20,random_state=23)

clf = LogisticRegression(max_iter=10000, random_state=0)
clf.fit(x_train, y_train)

predictions = clf.predict(x_test)
acc = accuracy_score(y_test, predictions) * 100
print(f"One vs Rest model accuracy: {acc:.2f}%")

clf_softmax = LogisticRegression(multi_class='multinomial', max_iter=10000, random_state=0)

clf_softmax.fit(x_train, y_train)
predictions_softmax = clf_softmax.predict(x_test)
acc_softmax = accuracy_score(y_test, predictions_softmax) * 100
print(f"Softmax (Multinomial) model accuracy: {acc_softmax:.2f}%")


plt.figure(figsize=(8, 6))

sns.scatterplot(x=x_test[:, 2], y=x_test[:, 3], hue=y_test, palette='deep', style=predictions_softmax, markers=['o', 's', 'D'])

plt.title("Logistic Regression")
plt.xlabel("Petal Length")
plt.ylabel("Petal Width")
plt.legend(title='Classes', loc="lower right")
plt.show()
