import numpy as np
import activations
import matplotlib.pyplot as plt
class NeuralNetwork:
    def __init__(self, input, hiddenlayer, activation_hidden, outputlayer, idealValues, activation_output,learning_rate):
        self.input = np.array(input).reshape(1, -1)  # Ensure input is a row vector
        self.hiddenlayer = hiddenlayer
        self.activation_hidden = activation_hidden[0]
        self.activation_hidden_derivative = activation_hidden[1]
        self.activation_output = activation_output[0]
        self.activation_output_derivative = activation_output[1]
        self.outputlayer = outputlayer
        self.idealValues = idealValues
        self.learning_rate = learning_rate
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
            Mat_Vec_Mul = self.W[i] @ np.transpose(self.A[i])  # weight and activation(L-1) multiplication , weight matrix of (L) and (L-1)
            
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
        Loss = self.Loss()
        return Loss;
    

        
    def Loss(self):
      
      if(len(self.idealValues)==self.outputlayer):
  
        activations = self.A[-1]
        loss_values = (activations - self.idealValues) 
        return np.mean(loss_values)
   
      else:
          print("no of neurons in output kayer doesnt match the no of ideal values ")
          
    def Backwardpass(self):
    # Output layer error
      output_error = (self.A[-1] - self.idealValues)
      output_delta = output_error * self.activation_output_derivative(self.Z[-1]) ## delta(l) = Error * d(sigma)/dz = activation_output_derivative(Z(l))
    
    # Reshape for correct dimensions
      output_delta = output_delta.reshape(-1, 1)
      self.A[-2] = self.A[-2].reshape(1, -1)
    
    # Gradients for output layer
      dW = output_delta @ self.A[-2]
      db = np.sum(output_delta, axis=1)
    
      self.W[-1] -= self.learning_rate * dW
      self.b[-1] -= self.learning_rate * db
    
    # Backpropagate through hidden layers
      delta = output_delta
      for i in range(len(self.W) - 2, -1, -1):
        delta = (self.W[i + 1].T @ delta).reshape(-1) * self.activation_hidden_derivative(self.Z[i]) ## W(L+1,L)^T * delta(L) *  d(sigma(L-i))/dz =  activation_hidden_derivative(self.Z[i])
        
        # Reshape for correct dimensions
        delta = delta.reshape(-1, 1)
        self.A[i] = self.A[i].reshape(1, -1)
        
        dW = delta @ self.A[i]
        db = np.sum(delta, axis=1)
        
        self.W[i] -= self.learning_rate * dW
        self.b[i] -= self.learning_rate * db
 
            
            
    def train(self, epochs):
        LossValues = []
        for _ in range(epochs):
            loss = self.Forwardpass()
            LossValues.append(loss)
            self.Backwardpass()
        return LossValues    


