# Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix,ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
# Load dataset
df = pd.read_csv('heart_v2.csv')
print(df.head())
# Feature and target variables
X = df.drop('heart disease', axis=1)
y = df['heart disease']
# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
# Shapes of train/test sets
print("\nShape of X_train:", X_train.shape, "\nShape of X_test:", X_test.shape)
print("\nShape of y_train:", y_train.shape, "\nShape of y_test:", y_test.shape)
# Initialize Random Forest
classifier_rf = RandomForestClassifier(
random_state=42,
n_jobs=-1,
n_estimators=100
)
# Train model
classifier_rf.fit(X_train, y_train)
# Predict
y_pred = classifier_rf.predict(X_test)
# Evaluate
accuracy = accuracy_score(y_test, y_pred)
classification_rep = classification_report(y_test, y_pred)
# Print results
print(f"\nAccuracy: {accuracy:.2f}")
print("\nClassification Report:\n", classification_rep)
# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot()
plt.show()


POST LAB - 1

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

n_estimators_list = [10, 50, 100]
accuracies = []

for n in n_estimators_list:
    clf = RandomForestClassifier(n_estimators=n, random_state=42, n_jobs=-1)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    accuracies.append(acc)
    print(f"Random Forest with {n} trees - Accuracy: {acc:.4f}")

# Plot accuracies
plt.figure(figsize=(8,5))
plt.plot(n_estimators_list, accuracies, marker='o')
plt.title('Random Forest Accuracy vs Number of Trees')
plt.xlabel('Number of Trees (n_estimators)')
plt.ylabel('Accuracy')
plt.grid(True)
plt.show()


POST LAB - 2

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

# Initialize Logistic Regression
clf_lr = LogisticRegression(max_iter=1000, random_state=42)
clf_lr.fit(X_train, y_train)

# Predict and evaluate
y_pred_lr = clf_lr.predict(X_test)
acc_lr = accuracy_score(y_test, y_pred_lr)

print(f"Logistic Regression Accuracy: {acc_lr:.4f}")
print("\nClassification Report (Logistic Regression):\n", classification_report(y_test, y_pred_lr))

# Random Forest accuracy and report for comparison (using best n_estimators from above)
best_n = 100
clf_rf = RandomForestClassifier(n_estimators=best_n, random_state=42, n_jobs=-1)
clf_rf.fit(X_train, y_train)
y_pred_rf = clf_rf.predict(X_test)
acc_rf = accuracy_score(y_test, y_pred_rf)

print(f"\nRandom Forest Accuracy ({best_n} trees): {acc_rf:.4f}")
print("\nClassification Report (Random Forest):\n", classification_report(y_test, y_pred_rf))
