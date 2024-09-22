import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# 1. Load the data
data = pd.read_csv('data_multivar_nb.txt', header=None)
X = data.iloc[:, :-1].values  # Features
y = data.iloc[:, -1].values  # Class labels

# 2. Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 3. Create and train models

# Support Vector Machine (SVM)
svm_model = SVC()
svm_model.fit(X_train, y_train)

# Naive Bayes Classifier (Gaussian Naive Bayes)
nb_model = GaussianNB()
nb_model.fit(X_train, y_train)

# 4. Make predictions on the test data
y_pred_svm = svm_model.predict(X_test)
y_pred_nb = nb_model.predict(X_test)

# 5. Evaluate classification performance

# Function to print the metrics
def print_metrics(y_true, y_pred, model_name):
    print(f"Results for model: {model_name}")
    print(f"Accuracy: {accuracy_score(y_true, y_pred):.3f}")
    print(f"Precision: {precision_score(y_true, y_pred, average='weighted'):.3f}")
    print(f"Recall: {recall_score(y_true, y_pred, average='weighted'):.3f}")
    print(f"F1-Score: {f1_score(y_true, y_pred, average='weighted'):.3f}")
    print(f"Confusion Matrix:\n {confusion_matrix(y_true, y_pred)}\n")

# Evaluation for SVM
print_metrics(y_test, y_pred_svm, "SVM")

# Evaluation for Naive Bayes
print_metrics(y_test, y_pred_nb, "Naive Bayes")


