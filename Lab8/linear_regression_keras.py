import tensorflow as tf
import numpy as np

class KerasLinearRegression:
    def __init__(self, learning_rate=0.001, epochs=20000, batch_size=100):
        self.model = tf.keras.Sequential([
            tf.keras.layers.Dense(1, input_dim=1, use_bias=True,
                                  kernel_initializer='random_normal', bias_initializer='zeros')
        ])
        self.model.compile(optimizer=tf.optimizers.SGD(learning_rate=learning_rate), loss='mean_squared_error')
        self.epochs = epochs
        self.batch_size = batch_size

    def train(self, X_data, y_data):
        self.model.fit(X_data, y_data, epochs=self.epochs, batch_size=self.batch_size, verbose=100)

    def get_parameters(self):
        return self.model.weights[0].numpy()[0][0], self.model.weights[1].numpy()


def generate_data(n_samples):
    X_data = np.random.uniform(0, 1, (n_samples, 1)).astype(np.float32)
    y_data = 2 * X_data + 1 + np.random.normal(0, 0.2, (n_samples, 1)).astype(np.float32)
    return X_data, y_data
