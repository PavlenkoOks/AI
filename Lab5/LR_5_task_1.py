import argparse
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from utilities import visualize_classifier

def build_arg_parser():
    parser = argparse.ArgumentParser(description='Classify data using ensemble learning techniques')
    parser.add_argument('--classifier-type', dest='classifier_type', required=True, choices=['rf', 'erf'],
                        help="Type of classifier to use: 'rf' (Random Forest) or 'erf' (Extra Random Forest)")
    return parser

def load_data(input_file):
    data = np.loadtxt(input_file, delimiter=',')
    X, y = data[:, :-1], data[:, -1]
    return X, y

def main():
    # Parsing arguments
    args = build_arg_parser().parse_args()
    classifier_type = args.classifier_type

    # Load dataset
    input_file = 'data_random_forests.txt'
    X, y = load_data(input_file)

    # Input data visualization
    plt.figure()
    markers = ['s', 'o', '*']
    for i, label in enumerate(np.unique(y)):
        class_data = X[y == label]
        plt.scatter(class_data[:, 0], class_data[:, 1], s=75, facecolors='white',
                    edgecolors='black', linewidth=1, marker=markers[i], label=f'Class-{int(label)}')
    plt.title('Input Data')
    plt.legend()
    plt.show()

    # Splitting the dataset
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=2)

    # Classifier initialization
    params = {'n_estimators': 100, 'max_depth': 4, 'random_state': 0}
    if classifier_type == 'rf':
        classifier = RandomForestClassifier(**params)
    elif classifier_type == 'erf':
        classifier = ExtraTreesClassifier(**params)
    
    # Training the classifier
    classifier.fit(X_train, y_train)

    # Training dataset visualization
    visualize_classifier(classifier, X_train, y_train, 'Training Dataset')

    # Test dataset visualization
    visualize_classifier(classifier, X_test, y_test, 'Test Dataset')

    # Training performance
    print("\n" + "#" * 40)
    print("\nClassifier performance on training dataset\n")
    print(classification_report(y_train, classifier.predict(X_train), target_names=['Class-0', 'Class-1', 'Class-2']))
    print("#" * 40 + "\n")

    # Test performance
    print("#" * 40)
    print("\nClassifier performance on test dataset\n")
    print(classification_report(y_test, classifier.predict(X_test), target_names=['Class-0', 'Class-1', 'Class-2']))
    print("#" * 40 + "\n")

    # Test data points visualization
    test_datapoints = np.array([[5, 5], [3, 6], [6, 4], [7, 2], [4, 4], [5, 2]])
    print("\nConfidence measure:")
    for datapoint in test_datapoints:
        probabilities = classifier.predict_proba([datapoint])[0]
        predicted_class = f"Class-{np.argmax(probabilities)}"
        print(f"\nDatapoint: {datapoint}")
        print(f"Predicted class: {predicted_class}")

    visualize_classifier(classifier, test_datapoints, [0] * len(test_datapoints), 'Test Data Points')
    plt.show()

if __name__ == '__main__':
    main()
