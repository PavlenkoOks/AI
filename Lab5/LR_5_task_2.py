import sys
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from utilities import visualize_classifier

def load_data(input_file):
    data = np.loadtxt(input_file, delimiter=',')
    X, y = data[:, :-1], data[:, -1]
    return X, y

def plot_data(X, y):
    class_0 = X[y == 0]
    class_1 = X[y == 1]

    plt.figure()
    plt.scatter(class_0[:, 0], class_0[:, 1], s=75, c='black', marker='x', label='Class 0')
    plt.scatter(class_1[:, 0], class_1[:, 1], s=75, c='white', edgecolors='black', marker='o', label='Class 1')
    plt.title("Вхідні дані")
    plt.legend()
    plt.show()

def train_and_evaluate(X, y, balance=False):
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=5)

    # Define classifier parameters
    params = {'n_estimators': 100, 'max_depth': 4, 'random_state': 0}
    if balance:
        params['class_weight'] = 'balanced'

    # Train the classifier
    classifier = ExtraTreesClassifier(**params)
    classifier.fit(X_train, y_train)

    # Visualize the training and testing results
    visualize_classifier(classifier, X_train, y_train, 'Тренувальні дані')
    visualize_classifier(classifier, X_test, y_test, 'Тестовий набір даних')

    # Evaluate classifier performance
    class_names = ['Class-0', 'Class-1']
    print("\n" + "#" * 40)
    print("Classifier performance on training dataset\n")
    print(classification_report(y_train, classifier.predict(X_train), target_names=class_names))
    print("#" * 40 + "\n")
    print("Classifier performance on test dataset\n")
    print(classification_report(y_test, classifier.predict(X_test), target_names=class_names))
    print("#" * 40 + "\n")

def main():
    input_file = 'data_imbalance.txt'
    X, y = load_data(input_file)

    # Plot the input data
    plot_data(X, y)

    # Check for optional argument
    balance = len(sys.argv) > 1 and sys.argv[1] == 'balance'

    # Train and evaluate the classifier
    train_and_evaluate(X, y, balance=balance)

if __name__ == "__main__":
    main()
