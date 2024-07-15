import numpy as np

class NeuralNetwork:
    def __init__(self, input, hiddenlayer, activation_hidden, outputlayer, idealValues, activation_output):
        self.input = np.array(input).reshape(1, -1)  # Ensure input is a row vector
        self.hiddenlayer = hiddenlayer
        self.activation_hidden = activation_hidden
        self.activation_output = activation_output
        self.outputlayer = outputlayer
        self.idealValues = idealValues

        self.no_of_neurons = [self.input.shape[1]]

        for neurons in hiddenlayer:
            self.no_of_neurons.append(neurons)

        self.no_of_neurons.append(outputlayer)

        W = []
        b = []

        for i in range(len(self.no_of_neurons) - 1):
            W.append(np.random.randn( self.no_of_neurons[i + 1],self.no_of_neurons[i] ))
            b.append(np.random.randn( self.no_of_neurons[i + 1] ))

        self.W = W
        self.b = b
        self.A = [self.input]
        self.Z = []

    def Forwardpass(self):
        for i in range(len(self.W) - 1):
            Mat_Vec_Mul = self.W[i] @ np.transpose(self.A[i])  # weight activation(L-1) multiplication weight matrix of (L) and (L-1)
            
            shape = Mat_Vec_Mul.shape # to adjust the shape from (i x 1 ) of vectors to  (i,) to reduce errors in computation
            Z = (Mat_Vec_Mul.reshape(shape[0]))+ self.b[i] # Z = wx + b --> weighted sum
            self.Z.append(Z)
            A = self.activation_hidden(Z)
            self.A.append(A)
            
        Mat_Vec_Mul = self.W[-1] @ np.transpose(self.A[-1]) 
        shape = Mat_Vec_Mul.shape; # to adjust the shape from (i x 1 ) --> (i,) to reduce errors in computation
        Z =  (Mat_Vec_Mul.reshape(shape[0]) ) + self.b[-1]
        self.Z.append(Z)
        A = self.activation_output(Z)
        self.A.append(A)

def LeakyRelu(x):
    return np.maximum(0.1 * x, x)

def Tanh(x):
    return np.tanh(x)

# Example usage
NN = NeuralNetwork(
    input=[1, 2, 3],
    hiddenlayer=[2],
    activation_hidden=LeakyRelu,
    outputlayer=3,
    idealValues=[1, 2, 3, 4],
    activation_output=Tanh
)

NN.Forwardpass()

