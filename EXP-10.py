# 1. Import necessary libraries
import numpy as np
import matplotlib.pyplot as plt
from sklearn.naive_bayes import GaussianNB
from sklearn import metrics
from google.colab import files
# 2. Define helper function
def prepare_person_dataset(fname):
    dataset = []
    with open(fname) as fh:
        for line in fh:
            person = line.strip().split()
            height_weight = (float(person[2]), float(person[3]))
            gender = person[4]
            dataset.append((height_weight, gender))
    return dataset
# 3. Upload training dataset
print("Upload training dataset file (person_data.txt):")
uploaded = files.upload()
trainset = prepare_person_dataset("person_data.txt")
# 4. Upload test dataset
print("Upload test dataset file (person_testset.txt):")
uploaded = files.upload()
testset = prepare_person_dataset("person_testset.txt")
# 5. Prepare features (X) and labels (y)
X_train, y_train = zip(*trainset)
X_test, y_test = zip(*testset)
X_train = np.array(X_train)
y_train = np.array(y_train)
X_test = np.array(X_test)
y_test = np.array(y_test)
# 6. Train Gaussian Naive Bayes model
model = GaussianNB()
model.fit(X_train, y_train)
# 7. Predictions
y_pred = model.predict(X_test)
# 8. Model Evaluation
print("\nClassification Report:\n")
print(metrics.classification_report(y_test, y_pred))
# 9. Confusion Matrix Visualization
metrics.ConfusionMatrixDisplay.from_predictions(y_test, y_pred)
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix")
plt.show()


Post Lab Question - 1


# Q1: Gaussian Naive Bayes - Feature Importance via Accuracy Drop

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score

# Example Dataset
data = {
    'Height': [170, 165, 180, 155, 160, 175, 172, 158, 182, 168],
    'Weight': [70, 60, 80, 50, 55, 75, 72, 52, 85, 58],
    'Gender': [1, 0, 1, 0, 0, 1, 1, 0, 1, 0]  # 1 = Male, 0 = Female
}

df = pd.DataFrame(data)

# Train-test split
X = df[['Height', 'Weight']]
y = df['Gender']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 1️ Model with both Height and Weight
model_both = GaussianNB()
model_both.fit(X_train, y_train)
acc_both = accuracy_score(y_test, model_both.predict(X_test))

# 2️ Model with only Height
model_height = GaussianNB()
model_height.fit(X_train[['Height']], y_train)
acc_height = accuracy_score(y_test, model_height.predict(X_test[['Height']]))

# 3️ Model with only Weight
model_weight = GaussianNB()
model_weight.fit(X_train[['Weight']], y_train)
acc_weight = accuracy_score(y_test, model_weight.predict(X_test[['Weight']]))

print("Accuracy using both Height & Weight: ", round(acc_both, 3))
print("Accuracy using only Height: ", round(acc_height, 3))
print("Accuracy using only Weight: ", round(acc_weight, 3))

# Interpretation
if acc_height > acc_weight:
    print("\nHeight plays a more significant role in predicting gender.")
elif acc_height < acc_weight:
    print("\nWeight plays a more significant role in predicting gender.")
else:
    print("\nBoth features contribute equally.")


Post Lab Question - 2

# Q2: Compare BernoulliNB and GaussianNB

from sklearn.naive_bayes import BernoulliNB
from sklearn.preprocessing import Binarizer

# Reuse train-test split
# Gaussian NB
gnb = GaussianNB()
gnb.fit(X_train, y_train)
y_pred_gnb = gnb.predict(X_test)
acc_gnb = accuracy_score(y_test, y_pred_gnb)

# Bernoulli NB - need to binarize data
binarizer = Binarizer(threshold=X_train.mean().mean())  # Convert to 0/1 based on average threshold
X_train_bin = binarizer.fit_transform(X_train)
X_test_bin = binarizer.transform(X_test)

bnb = BernoulliNB()
bnb.fit(X_train_bin, y_train)
y_pred_bnb = bnb.predict(X_test_bin)
acc_bnb = accuracy_score(y_test, y_pred_bnb)

print("GaussianNB Accuracy:", round(acc_gnb, 3))
print("BernoulliNB Accuracy:", round(acc_bnb, 3))

if acc_gnb > acc_bnb:
    print("\nGaussianNB performs better because it handles continuous features naturally.")
else:
    print("\nBernoulliNB performs better — likely due to dataset nature (binary-friendly).")
