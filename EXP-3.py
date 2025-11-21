import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score

# Load example dataset (replace with your dataset)
data = datasets.load_iris()
X = data.data
y = data.target

# Split dataset into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

def evaluate_model(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, average='weighted')
    rec = recall_score(y_true, y_pred, average='weighted')
    return cm, acc, prec, rec

kernels = ['linear', 'poly', 'rbf', 'sigmoid']
results = {}

for kernel in kernels:
    svm = SVC(kernel=kernel)
    svm.fit(X_train, y_train)
    y_pred = svm.predict(X_test)
    cm, acc, prec, rec = evaluate_model(y_test, y_pred)
    results[kernel] = {'confusion_matrix': cm, 'accuracy': acc, 'precision': prec, 'recall': rec}
    print(f"SVM with {kernel} kernel:")
    print("Confusion Matrix:\n", cm)
    print(f"Accuracy: {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall: {rec:.4f}\n")

# Plotting
metrics = ['accuracy', 'precision', 'recall']
x = np.arange(len(kernels))  # label locations
width = 0.25  # width of the bars

fig, ax = plt.subplots(figsize=(10,6))

accuracy_scores = [results[k]['accuracy'] for k in kernels]
precision_scores = [results[k]['precision'] for k in kernels]
recall_scores = [results[k]['recall'] for k in kernels]

rects1 = ax.bar(x - width, accuracy_scores, width, label='Accuracy')
rects2 = ax.bar(x, precision_scores, width, label='Precision')
rects3 = ax.bar(x + width, recall_scores, width, label='Recall')

# Labels and titles
ax.set_ylabel('Scores')
ax.set_title('SVM Kernel Performance Comparison')
ax.set_xticks(x)
ax.set_xticklabels([k.capitalize() for k in kernels])
ax.legend()

# Annotate bars with their values
def autolabel(rects):
    for rect in rects:
        height = rect.get_height()
        ax.annotate(f'{height:.3f}',
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0,3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom')

autolabel(rects1)
autolabel(rects2)
autolabel(rects3)

plt.ylim([0, 1.1])
plt.show()
