# Import necessary libraries
from sklearn.ensemble import AdaBoostClassifier
from sklearn import datasets, metrics
from sklearn.model_selection import train_test_split

# Load Iris dataset
iris = datasets.load_iris()
X, y = iris.data, iris.target

# Display dataset information
print("\nInput Features:", iris.feature_names)
print("Target Classes:", iris.target_names)

# Split dataset into training (70%) and testing (30%)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y)

# Create AdaBoost classifier (default base estimator = DecisionTreeClassifier)
abc = AdaBoostClassifier(n_estimators=50, learning_rate=1, random_state=42)

# Train the AdaBoost Classifier
model = abc.fit(X_train, y_train)

# Predict on the test dataset
y_pred = model.predict(X_test)

# Model Accuracy
accuracy = metrics.accuracy_score(y_test, y_pred)
print("\nAccuracy:", round(accuracy * 100, 2), "%")

# Detailed Classification Report
print("\nClassification Report:\n", metrics.classification_report(
    y_test, y_pred, target_names=iris.target_names))


POST LAB QUESTION - 1

# Import necessary libraries
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn import datasets, metrics
from sklearn.model_selection import train_test_split

# Load Iris dataset
iris = datasets.load_iris()
X, y = iris.data, iris.target

# Split dataset into training and testing sets (70% train, 30% test)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y)

# AdaBoost with n_estimators=50
model_50 = AdaBoostClassifier(
    estimator=DecisionTreeClassifier(max_depth=1),
    n_estimators=50,
    learning_rate=1,
    random_state=42
).fit(X_train, y_train)
y_pred_50 = model_50.predict(X_test)

# AdaBoost with n_estimators=200
model_200 = AdaBoostClassifier(
    estimator=DecisionTreeClassifier(max_depth=1),
    n_estimators=200,
    learning_rate=1,
    random_state=42
).fit(X_train, y_train)
y_pred_200 = model_200.predict(X_test)

# Print results
print("\n--- AdaBoost (n_estimators = 50) ---")
print("Accuracy:", round(metrics.accuracy_score(y_test, y_pred_50) * 100, 2), "%")
print("Classification Report:\n", metrics.classification_report(y_test, y_pred_50, target_names=iris.target_names))

print("\n--- AdaBoost (n_estimators = 200) ---")
print("Accuracy:", round(metrics.accuracy_score(y_test, y_pred_200) * 100, 2), "%")
print("Classification Report:\n", metrics.classification_report(y_test, y_pred_200, target_names=iris.target_names))


POST LAB QUESTION - 2

from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn import datasets, metrics
from sklearn.model_selection import train_test_split
import time

# Load Iris dataset
iris = datasets.load_iris()
X, y = iris.data, iris.target

# Split dataset into training and testing sets (70% train, 30% test)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y)

# AdaBoost with Decision Tree (stump)
start = time.time()
ada_dt = AdaBoostClassifier(
    estimator=DecisionTreeClassifier(max_depth=1),
    n_estimators=50,
    learning_rate=1,
    random_state=42
)
ada_dt.fit(X_train, y_train)
dt_time = time.time() - start
y_pred_dt = ada_dt.predict(X_test)
dt_acc = metrics.accuracy_score(y_test, y_pred_dt)

print("AdaBoost with Decision Tree stump:")
print(f"Training time: {dt_time:.4f} seconds")
print(f"Accuracy: {dt_acc:.4f}")

# AdaBoost with Logistic Regression (note: not ideal, may be slow or unreliable)
try:
    start = time.time()
    ada_lr = AdaBoostClassifier(
        estimator=LogisticRegression(solver='liblinear', max_iter=500),
        n_estimators=50,
        learning_rate=1,
        random_state=42,
        algorithm='SAMME'  # Use SAMME because logistic regression doesn’t support predict_proba well here
    )
    ada_lr.fit(X_train, y_train)
    lr_time = time.time() - start
    y_pred_lr = ada_lr.predict(X_test)
    lr_acc = metrics.accuracy_score(y_test, y_pred_lr)

    print("\nAdaBoost with Logistic Regression:")
    print(f"Training time: {lr_time:.4f} seconds")
    print(f"Accuracy: {lr_acc:.4f}")

except Exception as e:
    print("\nAdaBoost with Logistic Regression failed or unreliable:")
    print(e)

# Logistic Regression alone (baseline)
start = time.time()
lr = LogisticRegression(solver='liblinear', max_iter=500, random_state=42)
lr.fit(X_train, y_train)
lr_only_time = time.time() - start
y_pred_lr_only = lr.predict(X_test)
lr_only_acc = metrics.accuracy_score(y_test, y_pred_lr_only)

print("\nLogistic Regression alone (no boosting):")
print(f"Training time: {lr_only_time:.4f} seconds")
print(f"Accuracy: {lr_only_acc:.4f}")
