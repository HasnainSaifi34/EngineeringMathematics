import numpy as np;
import pickle
class NeuralNetwork: 
    def __init__(self, input, hidden_layer, activation_hidden, output_layer, ideal_values, activation_output, learning_rate):
        self.input = np.array(input).reshape(1, -1)  # Ensure input is a row vector
        self.hidden_layer = hidden_layer
        self.activation_hidden = activation_hidden[0]
        self.activation_hidden_derivative = activation_hidden[1]
        self.activation_output = activation_output[0]
        self.activation_output_derivative = activation_output[1]
        self.output_layer = output_layer
        self.ideal_values = np.array(ideal_values).reshape(1, -1)
        self.learning_rate = learning_rate
        self.no_of_neurons = [self.input.shape[1]] + hidden_layer + [output_layer]
 
        self.W = [np.random.randn(self.no_of_neurons[i + 1], self.no_of_neurons[i]) for i in range(len(self.no_of_neurons) - 1)]
        self.b = [np.random.randn(self.no_of_neurons[i + 1]) for i in range(len(self.no_of_neurons) - 1)]
        self.A = [self.input]
        self.Z = []

    def forward_pass(self):
        self.Z = []
        self.A = [self.input]

        for i in range(len(self.W) - 1):
            Z = self.W[i] @ self.A[-1].T + self.b[i].reshape(-1, 1)
            self.Z.append(Z)
            A = self.activation_hidden(Z)
            self.A.append(A.T)
        
        Z = self.W[-1] @ self.A[-1].T + self.b[-1].reshape(-1, 1)
        self.Z.append(Z)
        A = self.activation_output(Z)
        self.A.append(A.T)
        
        return self.loss()

    def loss(self):
        if self.ideal_values.shape[1] != self.output_layer:
            raise ValueError("Number of neurons in output layer doesn't match the number of ideal values")
        
        loss_values = (self.A[-1] - self.ideal_values) ** 2
        return (0.5)*np.mean(loss_values)

    def backward_pass(self):
        output_error = self.A[-1] - self.ideal_values #( d(loss)/d(activations))
        output_delta = output_error * self.activation_output_derivative(self.Z[-1].T) #  (  d(loss)/d(activations) * d(activations)/d(z) ) 
        
        dW = output_delta.T @ self.A[-2] #  d(loss)/d(activations) * d(activations)/d(z) * d(z)/d(w) ---> z = wx + b 
        db = np.sum(output_delta, axis=0)
        
        self.W[-1] -= self.learning_rate * dW  # updating weights (output layer)
        self.b[-1] -= self.learning_rate * db  # updating biases  (output layer)
        
        delta = output_delta
        for i in range(len(self.W) - 2, -1, -1):
            delta = (self.W[i + 1].T @ delta.T).T * self.activation_hidden_derivative(self.Z[i].T)
            
            dW = delta.T @ self.A[i]
            db = np.sum(delta, axis=0)
            
            self.W[i] -= self.learning_rate * dW
            self.b[i] -= self.learning_rate * db

    def train(self, epochs):
        loss_values = []
        for _ in range(epochs):
            loss = self.forward_pass()
            loss_values.append(loss)
            self.backward_pass()
        return loss_values
    
    def save_model(self, file_path):
        model_params = {
            'W': self.W,
            'b': self.b
        }
        with open(file_path, 'wb') as f:
            pickle.dump(model_params, f)

    def load_model(self, file_path):
        with open(file_path, 'rb') as f:
            model_params = pickle.load(f)
        
        self.W = model_params['W']
        self.b = model_params['b']    