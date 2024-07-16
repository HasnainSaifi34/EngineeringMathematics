
import numpy as np;

def LeakyRelu(x):
    return np.maximum(0.1 * x, x)

def Tanh(x):
    return np.tanh(x)

def regularized_tanh(x):
  return (np.tanh(x) + 1) / 2

def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def elu(x, alpha=1.0):
  return np.where(x >= 0, x, alpha * (np.exp(x) - 1))

def selu(x, alpha=1.67326, lambda_=1.0507):
  return lambda_ * np.where(x >= 0, x, alpha * (np.exp(x) - 1))



def softmax(x):
  exp_x = np.exp(x)
  return exp_x / np.sum(exp_x, axis=-1, keepdims=True)
def softmax_derivative(z):
    s = softmax(z)
    return np.diag(s) - np.outer(s, s)


def LeakyRelu_derivative(x):
    return np.where(x > 0, 1, 0.01)
  
def sigmoid_derivative(x):
    s = sigmoid(x)
    return s * (1 - s)  
  
  
def softmax(z):
        e_z = np.exp(z - np.max(z))
        return e_z / np.sum(e_z)
    
    # Implement softmax derivative
def softmax_derivative(z):
        s = softmax(z)
        return np.diag(s) - np.outer(s, s)