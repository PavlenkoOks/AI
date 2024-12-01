import tensorflow as tf
import numpy as np

class LinearRegression:
    def __init__(self, learning_rate=0.001, stddev=0.01):
        self.k = tf.Variable(tf.random.normal((1, 1), stddev=stddev, dtype=tf.float32), name='slope')
        self.b = tf.Variable(tf.zeros((1,), dtype=tf.float32), name='bias')
        self.optimizer = tf.optimizers.SGD(learning_rate=learning_rate)

    def model(self, X):
        return tf.matmul(X, self.k) + self.b

    def loss_fn(self, y_true, y_pred):
        return tf.reduce_sum(tf.square(y_true - y_pred))

    def train(self, X_data, y_data, n_samples, batch_size, num_steps):
        for step in range(num_steps):
            indices = np.random.choice(n_samples, batch_size)
            X_batch, y_batch = X_data[indices], y_data[indices]

            with tf.GradientTape() as tape:
                y_pred = self.model(X_batch)
                loss = self.loss_fn(y_batch, y_pred)

            gradients = tape.gradient(loss, [self.k, self.b])
            self.optimizer.apply_gradients(zip(gradients, [self.k, self.b]))

            if (step + 1) % 100 == 0:
                print(f"Step {step + 1}: loss = {loss.numpy()}, k = {self.k.numpy()[0][0]}, b = {self.b.numpy()[0]}")
