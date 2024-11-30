import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import BayesianRidge
from sklearn.metrics import mean_squared_error
from datetime import datetime

data = pd.read_csv('data.txt')
data.dropna(subset=['price'], inplace=True)
data['start_date'] = pd.to_datetime(data['start_date'])
data['day_of_week'] = data['start_date'].dt.dayofweek

encoders = {}
for column in ['origin', 'destination', 'train_type', 'train_class', 'fare']:
    encoder = LabelEncoder()
    data[f'{column}_enc'] = encoder.fit_transform(data[column].str.lower())
    encoders[column] = encoder

X = data[['origin_enc', 'destination_enc', 'train_type_enc', 'train_class_enc', 'fare_enc', 'day_of_week']]
y = data['price']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = BayesianRidge()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print(f"Test Set Mean Squared Error: {mse:.2f}")

def display_options(column_name):
    unique_values = sorted(data[column_name].str.upper().unique())
    print(f"Available {column_name.replace('_', ' ').title()} Options: {', '.join(unique_values)}")

def predict_price(origin, destination, train_type, train_class, fare, start_date):
    try:
        day_of_week = pd.to_datetime(start_date).dayofweek
        input_data = pd.DataFrame([[
            encoders['origin'].transform([origin.lower()])[0],
            encoders['destination'].transform([destination.lower()])[0],
            encoders['train_type'].transform([train_type.lower()])[0],
            encoders['train_class'].transform([train_class.lower()])[0],
            encoders['fare'].transform([fare.lower()])[0],
            day_of_week
        ]], columns=['origin_enc', 'destination_enc', 'train_type_enc', 'train_class_enc', 'fare_enc', 'day_of_week'])
        predicted_price = model.predict(input_data)[0]
        return round(predicted_price, 2)
    except (KeyError, ValueError) as e:
        return f"Invalid input: {e}"

def main():
    print("Train Price Prediction System (Enhanced)")
    print("-" * 40)

    display_options('train_type')
    train_type = input("Enter Train Type: ")

    display_options('train_class')
    train_class = input("Enter Train Class: ")

    display_options('fare')
    fare = input("Enter Fare Type: ")

    origin = input("Enter Origin Station: ")
    destination = input("Enter Destination Station: ")
    start_date = input("Enter Start Date (YYYY-MM-DD): ")

    print("\nProcessing your inputs...")
    predicted_price = predict_price(origin, destination, train_type, train_class, fare, start_date)
    if isinstance(predicted_price, str):
        print(f"Error: {predicted_price}")
    else:
        print(f"The estimated price for your trip is: €{predicted_price}")

if __name__ == "__main__":
    main()
