import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import classification_report, f1_score, roc_curve, auc

iris = load_iris()
x = iris.data
y = iris.target

sc = StandardScaler()
X = sc.fit_transform(x)

y_bin = label_binarize(y, classes=[0,1,2])
n_classes = y_bin.shape[1]

lda = LinearDiscriminantAnalysis()
lr = LogisticRegression(max_iter=200)
nb = GaussianNB()

lda.fit(x,y)
lr.fit(x,y)
nb.fit(x,y)

y_pred_lda = lda.predict(x)
y_pred_lr = lr.predict(x)
y_pred_nb = nb.predict(x)

print("LDA:\n", classification_report(y, y_pred_lda))
print("Logistic Regression:\n", classification_report(y, y_pred_lr))
print("Naive Bayes:\n", classification_report(y, y_pred_nb))

plt.figure()
colors = ['blue', 'red', 'green']
for i in range(n_classes):
    prob_lda = lda.predict_proba(x)[:,i]
    prob_lr = lr.predict_proba(x)[:,i]
    prob_nb = nb.predict_proba(x)[:,i]

    fpr_lda, tpr_lda, _ = roc_curve(y_bin[:,i], prob_lda)
    fpr_lr, tpr_lr, _ = roc_curve(y_bin[:,i], prob_lr)
    fpr_nb, tpr_nb, _ = roc_curve(y_bin[:,i], prob_nb)

    plt.plot(fpr_lda, tpr_lda, color = colors[i], label = f'LDA Class {i} (area = {auc(fpr_lda, tpr_lda):.2f})')
    plt.plot(fpr_lr, tpr_lr, color = colors[i], linestyle = '--', label = f'Logistic Regression Class {i} (area = {auc(fpr_lr, tpr_lr):.2f})')
    plt.plot(fpr_nb, tpr_nb, color = colors[i], linestyle = ':', label = f'Naive Bayes Class {i} (area = {auc(fpr_nb, tpr_nb):.2f})')

plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend(loc='best')
plt.show()
