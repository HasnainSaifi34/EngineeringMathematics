import numpy as np
class NeuralNetworkUtils:
    @staticmethod
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))

    @staticmethod
    def sigmoid_derivative(x):
        sig = NeuralNetworkUtils.sigmoid(x)
        return sig * (1 - sig)

    @staticmethod
    def tanh(x):
        return np.tanh(x)

    @staticmethod
    def tanh_derivative(x):
        return 1 - np.tanh(x) ** 2

    @staticmethod
    def relu(x):
        return np.maximum(0, x)

    @staticmethod
    def relu_derivative(x):
        return np.where(x > 0, 1, 0)

    @staticmethod
    def leaky_relu(x, alpha=0.01):
        return np.maximum(alpha * x, x)

    @staticmethod
    def leaky_relu_derivative(x, alpha=0.01):
        return np.where(x > 0, 1, alpha)

    @staticmethod
    def softmax(x):
        exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=-1, keepdims=True)

    @staticmethod
    def linear(x):
        return x

    @staticmethod
    def linear_derivative(x):
        return np.ones_like(x)

    @staticmethod
    def mse(y_true, y_pred):
        return 0.5 * np.sum((y_pred - y_true) ** 2)

    @staticmethod
    def mse_derivative(y_true, y_pred):
        return y_pred - y_true

    @staticmethod
    def binary_crossentropy(y_true, y_pred):
        y_pred = np.clip(y_pred, 1e-15, 1 - 1e-15)
        return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))

    @staticmethod
    def binary_crossentropy_derivative(y_true, y_pred):
        y_pred = np.clip(y_pred, 1e-15, 1 - 1e-15)
        return (y_pred - y_true) / (y_pred * (1 - y_pred))

    @staticmethod
    def categorical_crossentropy(y_true, y_pred):
        y_pred = np.clip(y_pred, 1e-15, 1 - 1e-15)
        return -np.sum(y_true * np.log(y_pred), axis=-1)

    @staticmethod
    def categorical_crossentropy_derivative(y_true, y_pred):
        y_pred = np.clip(y_pred, 1e-15, 1 - 1e-15)
        return -y_true / y_pred
    
    @staticmethod
    def elu(x, alpha=1.0):
      return np.where(x > 0, x, alpha * (np.exp(x) - 1))
    @staticmethod
    def elu_derivative(x, alpha=1.0):
      return np.where(x > 0, 1, alpha * np.exp(x))
    @staticmethod
    def get_activation_method(name):
        name = name.lower()
        names = ["sigmoid", "tanh", "relu", "leaky_relu", "softmax", "linear","elu"]
        if name not in names:
            raise ValueError(f"Unsupported activation function: {name}")

        activation_functions = {
            "sigmoid": (NeuralNetworkUtils.sigmoid, NeuralNetworkUtils.sigmoid_derivative),
            "tanh": (NeuralNetworkUtils.tanh, NeuralNetworkUtils.tanh_derivative),
            "relu": (NeuralNetworkUtils.relu, NeuralNetworkUtils.relu_derivative),
            "leaky_relu": (NeuralNetworkUtils.leaky_relu, NeuralNetworkUtils.leaky_relu_derivative),
            "softmax": (NeuralNetworkUtils.softmax, None),  # Softmax typically doesn't need a derivative function in isolation
            "linear": (NeuralNetworkUtils.linear, NeuralNetworkUtils.linear_derivative),
            "elu":(NeuralNetworkUtils.elu , NeuralNetworkUtils.elu_derivative)
        }
        return activation_functions[name]

    @staticmethod
    def get_loss_function(name):
        loss_functions = {
            "mse": (NeuralNetworkUtils.mse, NeuralNetworkUtils.mse_derivative),
            "binary_crossentropy": (NeuralNetworkUtils.binary_crossentropy, NeuralNetworkUtils.binary_crossentropy_derivative),
            "categorical_crossentropy": (NeuralNetworkUtils.categorical_crossentropy, NeuralNetworkUtils.categorical_crossentropy_derivative)
        }
        if name not in loss_functions:
            raise ValueError(f"Unsupported loss function: {name}")
        return loss_functions[name]
