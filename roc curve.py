import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_curve, auc

iris = load_iris()
x = iris.data[:,2]
y = (iris.target == 2).astype(int)
                            
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.20, random_state=50)

threshold = 4.4
y_predict_train = (x_train > threshold).astype(int)
accuracy = accuracy_score(y_train, y_predict_train)
print("Accuracy:", accuracy)

y_scores = x_test
thresholds = np.arange(1.0, 7.0, 0.1)
tpr = []
fpr = []

for threshold in thresholds:
    y_pred_threshold = (y_scores > threshold).astype(int)
    tp = np.sum((y_pred_threshold == 1) & (y_test == 1))
    fn = np.sum((y_pred_threshold == 0) & (y_test == 1))
    fp = np.sum((y_pred_threshold == 1) & (y_test == 0))
    tn = np.sum((y_pred_threshold == 0) & (y_test == 0))
    
    tpr.append(tp / (tp + fn) if (tp + fn) > 0 else 0)  
    fpr.append(fp / (fp + tn) if (fp + tn) > 0 else 0)

plt.figure()
plt.plot(fpr, tpr, color= 'blue', label='ROC curve')
plt.plot([0, 1], [0, 1], color= 'red',linestyle='--', label='No Skill')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.0])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend()
plt.show()
