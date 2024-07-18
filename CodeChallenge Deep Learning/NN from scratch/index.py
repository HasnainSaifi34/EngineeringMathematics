import numpy as np
import pickle
class NeuralNetwork:
    def __init__(self, input_size, hidden_layers, activation_hidden, output_size, activation_output, learning_rate):
        self.input_size = input_size
        self.hidden_layers = hidden_layers
        self.activation_hidden = activation_hidden[0]
        self.activation_hidden_derivative = activation_hidden[1]
        self.activation_output = activation_output[0]
        self.activation_output_derivative = activation_output[1]
        self.output_size = output_size
        self.learning_rate = learning_rate
        self.no_of_neurons = [input_size] + hidden_layers + [output_size]

        self.W = [np.random.randn(self.no_of_neurons[i + 1], self.no_of_neurons[i]) for i in range(len(self.no_of_neurons) - 1)]
        self.b = [np.random.randn(self.no_of_neurons[i + 1], 1) for i in range(len(self.no_of_neurons) - 1)]

    def forward_pass(self, X):
        self.Z = []
        self.A = [X]

        for i in range(len(self.W) - 1):
            Z = self.W[i] @ self.A[-1].T + self.b[i]
            self.Z.append(Z)
            A = self.activation_hidden(Z)
            self.A.append(A.T)
        
        Z = self.W[-1] @ self.A[-1].T + self.b[-1]
        self.Z.append(Z)
        A = self.activation_output(Z)
        self.A.append(A.T)
        
        return self.A[-1]

    def loss(self, y_pred, y_true):
        loss_values = (y_pred - y_true) ** 2
        return 0.5 * np.mean(loss_values)

    def backward_pass(self, y_pred, y_true):
        output_error = y_pred - y_true
        output_delta = output_error * self.activation_output_derivative(self.Z[-1].T)

        dW = output_delta.T @ self.A[-2]
        db = np.sum(output_delta, axis=0).reshape(-1, 1)

        self.W[-1] -= self.learning_rate * dW
        self.b[-1] -= self.learning_rate * db

        delta = output_delta
        for i in range(len(self.W) - 2, -1, -1):
            delta = (self.W[i + 1].T @ delta.T).T * self.activation_hidden_derivative(self.Z[i].T)

            dW = delta.T @ self.A[i]
            db = np.sum(delta, axis=0).reshape(-1, 1)

            self.W[i] -= self.learning_rate * dW
            self.b[i] -= self.learning_rate * db

    def train(self, X_train, y_train, epochs, batch_size):
        n_samples = X_train.shape[0]
        loss_values = []

        for epoch in range(epochs):
            indices = np.arange(n_samples)
            np.random.shuffle(indices)

            for start in range(0, n_samples, batch_size):
                end = min(start + batch_size, n_samples)
                batch_indices = indices[start:end]
                X_batch = X_train[batch_indices]
                y_batch = y_train[batch_indices]

                y_pred = self.forward_pass(X_batch)
                loss = self.loss(y_pred, y_batch)
                loss_values.append(loss)
                self.backward_pass(y_pred, y_batch)

        return loss_values

    def save_model(self, filename):
        with open(filename, 'wb') as f:
            pickle.dump(self.W, f)
            pickle.dump(self.b,f)

    def load_model(self, filename):
        with open(filename, 'rb') as f:
            self.W = pickle.load(f)  # Load weights
            self.b = pickle.load(f)  # Load biases





