import matplotlib.pyplot as plt
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
# Load dataset
iris_data = load_iris()
X = pd.DataFrame(iris_data.data, columns=iris_data.feature_names)
y = iris_data.target
# Split dataset into training and testing
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1, test_size=0.2)
# Train Decision Tree Classifier
clf = DecisionTreeClassifier(random_state=1)
clf.fit(X_train, y_train)
# Predictions
y_pred = clf.predict(X_test)
# Accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")
# Confusion Matrix
disp = ConfusionMatrixDisplay.from_estimator(clf, X_test, y_test,
display_labels=iris_data.target_names, cmap="Blues")
disp.ax_.set_title("Confusion Matrix for Iris Dataset Using Decision Tree")
plt.show()
# Plot Decision Tree
plt.figure(figsize=(12, 8))
plot_tree(clf, feature_names=iris_data.feature_names,
class_names=iris_data.target_names, filled=True)
plt.title("Decision Tree for Iris Dataset")
plt.show()
