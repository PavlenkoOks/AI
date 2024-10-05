import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC


input_file = 'income_data.txt'
X, Y = [], []
count_class1, count_class2 = 0, 0
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


X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=5)


scaler = preprocessing.StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


models = []
models.append(('Logistic Regression', LogisticRegression(solver='liblinear', multi_class='ovr', max_iter=1000)))
models.append(('Linear Discriminant Analysis', LinearDiscriminantAnalysis()))
models.append(('K-Nearest Neighbors', KNeighborsClassifier()))
models.append(('Decision Tree', DecisionTreeClassifier()))
models.append(('Naive Bayes', GaussianNB()))
models.append(('Support Vector Machine', SVC(gamma='auto')))


results = []
names = []

print("Результати крос-валідації для кожної моделі:")
for name, model in models:

    kfold = StratifiedKFold(n_splits=10, random_state=1, shuffle=True)

    cv_results = cross_val_score(model, X_train, y_train, cv=kfold, scoring='accuracy')
    results.append(cv_results)
    names.append(name)

    print(f'{name}: Mean Accuracy = {cv_results.mean():.4f} ({cv_results.std():.4f})')


plt.figure(figsize=(10, 6))
plt.boxplot(results, labels=names)
plt.title('Порівняння моделей за точністю (accuracy)')
plt.ylabel('Accuracy')
plt.xticks(rotation=45)
plt.grid(True)
plt.show()


best_index = np.argmax([np.mean(result) for result in results])
best_model_name = names[best_index]
print(f'Найкраща модель: {best_model_name} з точністю {np.mean(results[best_index]):.4f}')
