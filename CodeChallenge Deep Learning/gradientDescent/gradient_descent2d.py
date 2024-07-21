import numpy as np
import matplotlib.pyplot as plt




# Define the function
def f(x, y):
    return np.exp(-0.1 * (x**2 + y**2)) * (np.sin(2*x) + np.cos(2*y))

# Define the gradient of the function
def gradient_f(x, y):
    dfdx = np.exp(-0.1 * (x**2 + y**2)) * (2 * np.cos(2*x)) - 0.2 * x * np.exp(-0.1 * (x**2 + y**2)) * (np.sin(2*x) + np.cos(2*y))
    dfdy = np.exp(-0.1 * (x**2 + y**2)) * (-2 * np.sin(2*y)) - 0.2 * y * np.exp(-0.1 * (x**2 + y**2)) * (np.sin(2*x) + np.cos(2*y))
    return np.array([dfdx, dfdy])

# Gradient descent parameters
learning_rate = 0.05
num_iterations = 100
start_point = np.array([3.0, 3.0])

# Perform gradient descent
points = [start_point]
point = start_point
for _ in range(num_iterations):
    grad = gradient_f(point[0], point[1])
    point = point - learning_rate * grad
    points.append(point)

points = np.array(points)

# Create a grid of x and y values
x = np.linspace(-5, 5, 400)
y = np.linspace(-5, 5, 400)
x, y = np.meshgrid(x, y)
z = f(x, y)

# Create a 3D plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(x, y, z, cmap='viridis', alpha=0.8)

# Plot the gradient descent path
ax.plot(points[:, 0], points[:, 1], f(points[:, 0], points[:, 1]), color='r', marker='o', markersize=5, linestyle='-')

# Set labels
ax.set_xlabel('X axis')
ax.set_ylabel('Y axis')
ax.set_zlabel('Z axis')
ax.set_title('3D Plot with Gradient Descent Path')

plt.show()