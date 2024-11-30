import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.ensemble import ExtraTreesRegressor

input_file = 'traffic_data.txt'
data = []

with open(input_file, 'r') as f:
    for line in f:
        items = line.strip().split(',')
        data.append(items)

data = np.array(data)

label_encoders = []
X_encoded = np.empty_like(data, dtype=float)

for i, value in enumerate(data[0]):
    if value.isdigit(): 
        X_encoded[:, i] = data[:, i]
    else: 
        encoder = preprocessing.LabelEncoder()
        X_encoded[:, i] = encoder.fit_transform(data[:, i])
        label_encoders.append(encoder)

X = X_encoded[:, :-1].astype(int)
y = X_encoded[:, -1].astype(int)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=5)

params = {'n_estimators': 100, 'max_depth': 4, 'random_state': 0}
regressor = ExtraTreesRegressor(**params)
regressor.fit(X_train, y_train)

y_pred = regressor.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
print(f"Mean absolute error: {mae:.2f}")

test_datapoint = ['Saturday', '10:20', 'Atlanta', 'no']
test_datapoint_encoded = []

for i, value in enumerate(test_datapoint):
    if value.isdigit():
        test_datapoint_encoded.append(int(value))
    else:  
        encoder = label_encoders.pop(0)
        test_datapoint_encoded.append(encoder.transform([value])[0])

test_datapoint_encoded = np.array(test_datapoint_encoded)

predicted_traffic = regressor.predict([test_datapoint_encoded])[0]
print(f"Predicted traffic: {int(predicted_traffic)}")
