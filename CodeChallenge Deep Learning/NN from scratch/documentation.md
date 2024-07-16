
---

### NeuralNetwork Class

The `NeuralNetwork` class implements a basic feedforward neural network capable of training and making predictions. It accepts the following parameters during initialization:

- **`input`**: Represents the input data or features for the neural network.
- **`hiddenlayer`**: Specifies the number of neurons in each hidden layer as a list of integers.
- **`activation_hidden`**: Tuple containing the activation function and its derivative for the hidden layers.
- **`outputlayer`**: Specifies the number of neurons in the output layer.
- **`idealValues`**: Target output values for training purposes.
- **`activation_output`**: Tuple containing the activation function and its derivative for the output layer.
- **`learning_rate`**: Determines the step size in gradient descent optimization during training.

### Example Usage

```python
# Example instantiation
NN = NeuralNetwork(
    input=[1, 2, 3],
    hiddenlayer=[4],
    activation_hidden=(activations.LeakyRelu, activations.LeakyRelu_derivative),
    outputlayer=2,
    idealValues=[1, 2],
    activation_output=(activations.sigmoid, activations.sigmoid_derivative),
    learning_rate=0.001
)
```

### Purpose

The `NeuralNetwork` class encapsulates the functionality for creating and training a neural network model. It provides methods to initialize the network, perform forward and backward propagation, compute loss, and update model parameters using gradient descent.

<img src="https://eu-images.contentstack.com/v3/assets/blt6b0f74e5591baa03/blt790f1b7ac4e04301/6543ff50fcf447040a6b8dc7/News_Image_(47).png?width=1280&auto=webp&quality=95&format=jpg&disable=upscale"
width=500
heigth=500
/>


---

<img src="https://media.licdn.com/dms/image/D4D22AQH-8ATyCCV_Ww/feedshare-shrink_800/0/1720882385473?e=1724284800&v=beta&t=wc2S7yF2Fo5zIDEjMpC-VdjkvJEg9K4m3mTTHPnknDU"/>


### Forward Pass 

```python
def Forwardpass(self):
    for i in range(len(self.W) - 1):
        Mat_Vec_Mul = self.W[i] @ np.transpose(self.A[i])  # weight and activation(L-1) multiplication , weight matrix of (L) and (L-1)
        
        shape = Mat_Vec_Mul.shape  # to adjust the shape from (i x 1 ) of vectors to (i,) to reduce errors in computation
        Z = (Mat_Vec_Mul.reshape(shape[0])) + self.b[i]  # Z = wx + b --> weighted sum
        self.Z.append(Z)
        A = self.activation_hidden[0](Z)
        self.A.append(A)
        
    Mat_Vec_Mul = self.W[-1] @ np.transpose(self.A[-1]) 
    shape = Mat_Vec_Mul.shape  # to adjust the shape from (i x 1 ) --> (i,) to reduce errors in computation
    Z = (Mat_Vec_Mul.reshape(shape[0])) + self.b[-1]
    self.Z.append(Z)
    A = self.activation_output[0](Z)
    self.A.append(A)
    Loss = self.Loss()
    return Loss
```

### Explanation

#### 1. Initialize the Forward Pass

```python
for i in range(len(self.W) - 1):
```
- **Explanation**: Loop through each layer except the last one (output layer).

#### 2. Weighted Sum Calculation for Hidden Layers

```python
Mat_Vec_Mul = self.W[i] @ np.transpose(self.A[i])
```
- **Explanation**: Compute the weighted sum of inputs for layer `i`. Here, `self.W[i]` is the weight matrix for layer `i`, and `self.A[i]` is the activation from the previous layer.
- **Matrix Equation**: $ Z^{(i)} = W^{(i)} \cdot A^{(i-1)} $
- **Dimensions**:
  - $ W^{(i)} $ is of shape (number of neurons in layer `i`, number of neurons in layer `i-1`)
  - $ A^{(i-1)} $ is of shape (number of neurons in layer `i-1`, 1)

#### 3. Adding Bias and Applying Activation Function

```python
shape = Mat_Vec_Mul.shape
Z = (Mat_Vec_Mul.reshape(shape[0])) + self.b[i]
self.Z.append(Z)
A = self.activation_hidden[0](Z)
self.A.append(A)
```
- **Explanation**: 
  - Reshape the result of the matrix multiplication to match the shape of the bias vector.
  - Add the bias vector to the weighted sum.
  - Apply the activation function to compute the activations for the current layer.
- **Matrix Equation**:
  - $ Z^{(i)} = W^{(i)} \cdot A^{(i-1)} + b^{(i)} $
  - $ A^{(i)} = \sigma(Z^{(i)}) $
- **Dimensions**:
  - $ b^{(i)} $ is of shape (number of neurons in layer `i`, 1)
  - $ Z^{(i)} $ and $ A^{(i)} $ are of shape (number of neurons in layer `i`, 1)

#### 4. Weighted Sum Calculation for Output Layer

```python
Mat_Vec_Mul = self.W[-1] @ np.transpose(self.A[-1])
shape = Mat_Vec_Mul.shape
Z = (Mat_Vec_Mul.reshape(shape[0])) + self.b[-1]
```
- **Explanation**: Compute the weighted sum of inputs for the output layer.
- **Matrix Equation**: $ Z^{(L)} = W^{(L)} \cdot A^{(L-1)} + b^{(L)} $
- **Dimensions**:
  - $ W^{(L)} $ is of shape (number of neurons in output layer, number of neurons in last hidden layer)
  - $ A^{(L-1)} $ is of shape (number of neurons in last hidden layer, 1)

#### 5. Applying Activation Function to Output Layer

```python
self.Z.append(Z)
A = self.activation_output[0](Z)
self.A.append(A)
```
- **Explanation**: Apply the activation function to the weighted sum of the output layer to get the final output.
- **Matrix Equation**: $ \hat{y} = \sigma(Z^{(L)}) $
- **Dimensions**:
  - $ \hat{y} $ is of shape (number of neurons in output layer, 1)

#### 6. Compute Loss

```python
Loss = self.Loss()
return Loss
```
- **Explanation**: Calculate the loss function to measure the difference between the predicted values and the actual values.
- **Loss Function**:
  - If Mean Squared Error (MSE) is used:
  - $ L = \frac{1}{N} \sum_{i=1}^{N} (\hat{y}_i - y_i)^2 $
  - Where $ N $ is the number of outputs, $ \hat{y}_i $ is the predicted value, and $ y_i $ is the actual value.

### Summary

The forward pass involves:
1. Computing the weighted sum $ Z^{(i)} = W^{(i)} \cdot A^{(i-1)} + b^{(i)} $ for each layer.
2. Applying the activation function $ A^{(i)} = \sigma(Z^{(i)}) $ to get the activations for each layer.
3. Storing the intermediate results ($ Z $ and $ A $) for use in backpropagation.
4. Calculating the final output and the loss.

The chain rule is applied during backpropagation, where the gradients are computed and propagated backward through the network, updating the weights and biases to minimize the loss. Each activation becomes the input for the next layer, which is crucial for calculating the gradients using the chain rule.



<img src="https://miro.medium.com/v2/resize:fit:679/0*9lo2ux8ASvt6YJkH.gif"/>


### Backward Pass Explained

```python
def Backwardpass(self):
    # Output layer error
    output_error = (self.A[-1] - self.idealValues)
    output_delta = output_error * self.activation_output_derivative(self.Z[-1])  ## delta(L) = Error * d(sigma(L))/dz = activation_output_derivative(Z(L))

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
        delta = (self.W[i + 1].T @ delta).reshape(-1) * self.activation_hidden_derivative(self.Z[i])  ## W(L+1,L)^T * delta(L) * d(sigma(L-i))/dz = activation_hidden_derivative(self.Z[i])
        
        # Reshape for correct dimensions
        delta = delta.reshape(-1, 1)
        self.A[i] = self.A[i].reshape(1, -1)
        
        dW = delta @ self.A[i]
        db = np.sum(delta, axis=1)
        
        self.W[i] -= self.learning_rate * dW
        self.b[i] -= self.learning_rate * db
```

### Mathematical Explanation and Chain Rule Application

#### 1. Output Layer Error

```python
output_error = (self.A[-1] - self.idealValues)
```
- **Explanation**: The output layer error is the difference between the predicted values (`self.A[-1]`) and the ideal (target) values (`self.idealValues`).
- **Mathematical Notation**: $ E = \hat{y} - y $

#### 2. Output Delta

```python
output_delta = output_error * self.activation_output_derivative(self.Z[-1])
```
- **Explanation**: Multiply the error by the derivative of the activation function of the output layer. This gives the gradient of the loss with respect to the weighted sum $ Z $ of the output layer.
- **Mathematical Notation**: $ \delta^{(L)} = E \cdot \sigma' (Z^{(L)}) $
- **Chain Rule Application**:
  - Error: $ \frac{\partial L}{\partial \hat{y}} $
  - Activation derivative: $ \frac{\partial \hat{y}}{\partial Z^{(L)}} $
  - Combined: $ \delta^{(L)} = \frac{\partial L}{\partial Z^{(L)}} = \frac{\partial L}{\partial \hat{y}} \cdot \frac{\partial \hat{y}}{\partial Z^{(L)}} $

#### 3. Reshape for Correct Dimensions

```python
output_delta = output_delta.reshape(-1, 1)
self.A[-2] = self.A[-2].reshape(1, -1)
```
- **Explanation**: Reshape the delta and the activations for matrix multiplication.

#### 4. Gradients for Output Layer

```python
dW = output_delta @ self.A[-2]
db = np.sum(output_delta, axis=1)
```
- **Explanation**: Calculate the gradients for the weights and biases in the output layer.
- **Mathematical Notation**:
  - Weight gradients: $ \frac{\partial L}{\partial W^{(L)}} = \delta^{(L)} \cdot A^{(L-1)} $
  - Bias gradients: $ \frac{\partial L}{\partial b^{(L)}} = \delta^{(L)} $
- **Chain Rule Application**:
  - For weights: $ \frac{\partial L}{\partial W^{(L)}} = \frac{\partial L}{\partial Z^{(L)}} \cdot \frac{\partial Z^{(L)}}{\partial W^{(L)}} $
  - For biases: $ \frac{\partial L}{\partial b^{(L)}} = \frac{\partial L}{\partial Z^{(L)}} \cdot \frac{\partial Z^{(L)}}{\partial b^{(L)}} $

#### 5. Update Output Layer Weights and Biases

```python
self.W[-1] -= self.learning_rate * dW
self.b[-1] -= self.learning_rate * db
```
- **Explanation**: Update the weights and biases using the calculated gradients and the learning rate.
- **Mathematical Notation**:
  - Weights update: $ W^{(L)} \leftarrow W^{(L)} - \eta \cdot \frac{\partial L}{\partial W^{(L)}} $
  - Biases update: $ b^{(L)} \leftarrow b^{(L)} - \eta \cdot \frac{\partial L}{\partial b^{(L)}} $

#### 6. Backpropagate Through Hidden Layers

```python
delta = output_delta
for i in range(len(self.W) - 2, -1, -1):
    delta = (self.W[i + 1].T @ delta).reshape(-1) * self.activation_hidden_derivative(self.Z[i])
```
- **Explanation**: Backpropagate the delta through each hidden layer.
- **Mathematical Notation**:
  - For layer $ l $: $ \delta^{(l)} = (\delta^{(l+1)} \cdot W^{(l+1)}) \cdot \sigma' (Z^{(l)}) $
- **Chain Rule Application**:
  - For hidden layers: $ \delta^{(l)} = \frac{\partial L}{\partial Z^{(l)}} = \left( \frac{\partial L}{\partial A^{(l+1)}} \cdot \frac{\partial A^{(l+1)}}{\partial Z^{(l+1)}} \cdot \frac{\partial Z^{(l+1)}}{\partial A^{(l)}} \right) \cdot \frac{\partial A^{(l)}}{\partial Z^{(l)}} $

#### 7. Reshape for Correct Dimensions

```python
delta = delta.reshape(-1, 1)
self.A[i] = self.A[i].reshape(1, -1)
```
- **Explanation**: Reshape the delta and the activations for matrix multiplication.

#### 8. Gradients for Hidden Layers

```python
dW = delta @ self.A[i]
db = np.sum(delta, axis=1)
```
- **Explanation**: Calculate the gradients for the weights and biases in the hidden layers.
- **Mathematical Notation**:
  - Weight gradients: $ \frac{\partial L}{\partial W^{(l)}} = \delta^{(l)} \cdot A^{(l-1)} $
  - Bias gradients: $ \frac{\partial L}{\partial b^{(l)}} = \delta^{(l)} $
- **Chain Rule Application**:
  - For weights: $ \frac{\partial L}{\partial W^{(l)}} = \frac{\partial L}{\partial Z^{(l)}} \cdot \frac{\partial Z^{(l)}}{\partial W^{(l)}} $
  - For biases: $ \frac{\partial L}{\partial b^{(l)}} = \frac{\partial L}{\partial Z^{(l)}} \cdot \frac{\partial Z^{(l)}}{\partial b^{(l)}} $

#### 9. Update Hidden Layer Weights and Biases

```python
self.W[i] -= self.learning_rate * dW
self.b[i] -= self.learning_rate * db
```
- **Explanation**: Update the weights and biases using the calculated gradients and the learning rate.
- **Mathematical Notation**:
  - Weights update: $ W^{(l)} \leftarrow W^{(l)} - \eta \cdot \frac{\partial L}{\partial W^{(l)}} $
  - Biases update: $ b^{(l)} \leftarrow b^{(l)} - \eta \cdot \frac{\partial L}{\partial b^{(l)}} $

### Summary

The backward pass involves computing the error and gradients for each layer starting from the output layer and moving backward through the hidden layers. The chain rule is applied to propagate the error gradients backward, allowing the network to update its weights and biases to minimize the loss function. Each line of code in the backward pass implements a step in this gradient descent optimization process.