import numpy as np

# Sparsify per-column
def columnWiseOneHot(activations, column_size):
    activations2D = activations.reshape(len(activations) // column_size, column_size)

    indices = np.argmax(activations2D, axis=1)

    result = np.zeros(activations.shape)

    for i in range(len(indices)):
        result[indices[i] + i * column_size] = 1.0

    return result

def sigmoid(x):
    return np.tanh(x * 0.5) * 0.5 + 0.5
