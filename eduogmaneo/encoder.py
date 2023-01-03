import numpy as np 

from .helpers import columnWiseOneHot, sigmoid

class Encoder:
    def __init__(self, num_input_columns, input_column_size, num_hidden_columns, hidden_column_size):
        # Dimensions
        self.num_input_columns = num_input_columns
        self.input_column_size = input_column_size

        self.num_hidden_columns = num_hidden_columns
        self.hidden_column_size = hidden_column_size

        num_input_cells = num_input_columns * input_column_size
        num_hidden_cells = num_hidden_columns * hidden_column_size

        # Parameters
        self.weights = np.random.rand(num_hidden_cells, num_input_cells)

        # State hidden layer (stored dense here, in the actual AOgmaNeo these would be lists of column indices)
        self.hidden_state = np.zeros(num_hidden_cells)

        # Hyperparameters
        self.lr = 0.01

    def step(self, input_state, learn_enabled):
        # Activation
        activations = np.dot(self.weights, input_state)

        self.hidden_state = columnWiseOneHot(activations, self.hidden_column_size)

        # Learning
        if learn_enabled:
            # Reconstruct
            recon_activations = np.dot(self.weights.T, self.hidden_state)

            recon_dense = sigmoid(recon_activations)
            recon_sparse = columnWiseOneHot(recon_activations, self.input_column_size)

            error = input_state - recon_dense

            # If sparse reconstruction doesn't match input, update that reconstruction (per-input-column)
            # We do this here in Pythgn by getting the full error and zeroing out parts of the error where the reconstruction matches the input
            for i in range(self.num_input_columns):
                column_slice = slice(i * self.input_column_size, (i + 1) * self.input_column_size)

                sub_error = input_state[column_slice] - recon_sparse[column_slice]

                if np.all(sub_error == 0.0):
                    error[column_slice] = 0.0

            # Learn
            self.weights += self.lr * np.dot(np.matrix(self.hidden_state).T, np.matrix(error))
