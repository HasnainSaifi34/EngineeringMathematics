from index import NeuralNetwork;
import matplotlib.pyplot as plt;
import cv2
import activations
import numpy as np

def preprocess_image(image_path):
    # Your preprocessing code (resize, normalize, etc.)
    # Example:
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    image = cv2.resize(image, (28, 28))  # Resize to match MNIST digit size
    image = image.astype('float32') / 255.0  # Normalize to [0, 1]
    image = image.reshape(1, -1)  # Reshape to match input shape (1, 784) for example

    return image


image_of_3 = np.array(preprocess_image("./numbers/3.png")).reshape(-1,1)

image_of_2 =  np.array(preprocess_image("./numbers/2.png")).reshape(-1,1)
idealValues = np.diag([1,1,1,1,1,1,1,1,1,1])


plt.imshow(image_of_2)
NN = NeuralNetwork(
    input=image_of_2,
    hiddenlayer=[16,16],
    activation_hidden=(activations.LeakyRelu,activations.LeakyRelu_derivative),
    outputlayer=10,
    idealValues=idealValues[2],
    activation_output=(activations.LeakyRelu, activations.LeakyRelu_derivative),
    learning_rate=1e-4
)


labels = ["zero","one","two","three","four","five","six","seven","eight","nine","ten"]

epochs = 100000
NN.train(epochs)
Activations = NN.A[-1].copy()

max_index = np.argmax(Activations)
print(f"AI thinks this number is {labels[max_index]}")

