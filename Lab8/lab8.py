import numpy as np
from linear_regression import LinearRegression
from linear_regression_keras import KerasLinearRegression, generate_data

def train_with_tensorflow():
    n_samples, batch_size, num_steps = 1000, 100, 20000

    X_data = np.random.uniform(0, 1, (n_samples, 1)).astype(np.float32)
    y_data = 2 * X_data + 1 + np.random.normal(0, 0.2, (n_samples, 1)).astype(np.float32)

    lr_model = LinearRegression()
    lr_model.train(X_data, y_data, n_samples, batch_size, num_steps)

def train_with_keras():
    # Keras approach
    n_samples = 1000
    X_data, y_data = generate_data(n_samples)

    lr_model = KerasLinearRegression()
    lr_model.train(X_data, y_data)
    k, b = lr_model.get_parameters()

    print(f"Final parameters: k = {k}, b = {b}")

def main():
    print("Select the training method:")
    print("1. TensorFlow 1.x style")
    print("2. Keras style")
    
    choice = input("Enter 1 or 2: ").strip()

    if choice == '1':
        train_with_tensorflow()
    elif choice == '2':
        train_with_keras()
    else:
        print("Invalid choice! Please enter 1 or 2.")

if __name__ == '__main__':
    main()
