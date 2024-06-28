import numpy as np
import matplotlib.pyplot as plt

def plot_vectors(vectors, colors=None, labels=None):
    """
    Plot a list of vectors in 2D space.

    Parameters:
    - vectors: list of tuples or numpy arrays representing the vectors, e.g., [(x1, y1), (x2, y2), ...]
    - colors: list of colors for each vector (optional)
    - labels: list of labels for each vector (optional)
    """
    plt.figure()
    ax = plt.gca()
    ax.set_aspect('equal')
    plt.grid(True)
    
    origin = np.array([0, 0])
    
    if colors is None:
        colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k'] * (len(vectors) // 7 + 1)
    
    for i, vector in enumerate(vectors):
        vector = np.array(vector)
        color = colors[i % len(colors)]
        label = labels[i] if labels is not None else None
        ax.quiver(*origin, *vector, color=color, angles='xy', scale_units='xy', scale=1, label=label)
        
        if label is not None:
            ax.text(vector[0], vector[1], f" {label}", color=color, fontsize=12)
    
    max_val = np.max(np.abs(vectors)) * 1.1
    plt.xlim(-max_val, max_val)
    plt.ylim(-max_val, max_val)

    if labels is not None:
        plt.legend()

    plt.show()

# Define the original vector

def EmptyMat(m,n):
   Mat =[]
   for i in range(m):
       Mat.append([])
       for j in range(i):
           Mat[i][j]=0
   return Mat       
