import numpy as np
from sklearn import preprocessing
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import StandardScaler

# Зчитування та попередня обробка даних
input_file = 'income_data.txt'
X = []
Y = []
count_class1 = 0
count_class2 = 0
max_datapoints = 25000

with open(input_file, 'r') as f:
    for line in f.readlines():
        if count_class1 >= max_datapoints and count_class2 >= max_datapoints:
            break
        if '?' in line: 
            continue

        data = line.strip().split(', ') 

        if data[-1] == '<=50K' and count_class1 < max_datapoints:
            X.append(data)
            Y.append(0)
            count_class1 += 1
        elif data[-1] == '>50K' and count_class2 < max_datapoints:
            X.append(data)
            Y.append(1)
            count_class2 += 1

X = np.array(X)

label_encoder = []
X_encoded = np.empty(X.shape)

for i, item in enumerate(X[0]):
    if item.isdigit():
        X_encoded[:, i] = X[:, i]
    else:
        le = preprocessing.LabelEncoder()
        X_encoded[:, i] = le.fit_transform(X[:, i])
        label_encoder.append(le)

X = X_encoded[:, :-1].astype(int)
Y = np.array(Y)

scaler = StandardScaler()
X = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=5)

metrics = {}

poly_svc = SVC(kernel='poly', degree=2, random_state=0)
poly_svc.fit(X_train, y_train)
y_test_pred = poly_svc.predict(X_test)

accuracy = accuracy_score(y_test, y_test_pred)
precision = precision_score(y_test, y_test_pred, average='weighted')
recall = recall_score(y_test, y_test_pred, average='weighted')
f1 = f1_score(y_test, y_test_pred, average='weighted')

metrics['Polynomial Kernel'] = {
    "Accuracy": accuracy * 100,
    "Precision": precision * 100,
    "Recall": recall * 100,
    "F1 Score": f1 * 100
}

rbf_svc = SVC(kernel='rbf', random_state=0) 
rbf_svc.fit(X_train, y_train)
y_test_pred = rbf_svc.predict(X_test)

accuracy = accuracy_score(y_test, y_test_pred)
precision = precision_score(y_test, y_test_pred, average='weighted')
recall = recall_score(y_test, y_test_pred, average='weighted')
f1 = f1_score(y_test, y_test_pred, average='weighted')

metrics['RBF Kernel'] = {
    "Accuracy": accuracy * 100,
    "Precision": precision * 100,
    "Recall": recall * 100,
    "F1 Score": f1 * 100
}

sigmoid_svc = SVC(kernel='sigmoid', random_state=0)  # Сигмоїдальне ядро
sigmoid_svc.fit(X_train, y_train)
y_test_pred = sigmoid_svc.predict(X_test)

accuracy = accuracy_score(y_test, y_test_pred)
precision = precision_score(y_test, y_test_pred, average='weighted')
recall = recall_score(y_test, y_test_pred, average='weighted')
f1 = f1_score(y_test, y_test_pred, average='weighted')

metrics['Sigmoid Kernel'] = {
    "Accuracy": accuracy * 100,
    "Precision": precision * 100,
    "Recall": recall * 100,
    "F1 Score": f1 * 100
}

for kernel, scores in metrics.items():
    print(f"\n{kernel}:")
    print(f"Accuracy: {scores['Accuracy']:.2f}%")
    print(f"Precision: {scores['Precision']:.2f}%")
    print(f"Recall: {scores['Recall']:.2f}%")
    print(f"F1 Score: {scores['F1 Score']:.2f}%")
