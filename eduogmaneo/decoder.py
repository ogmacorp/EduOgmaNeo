# ----------------------------------------------------------------------------
#  EduOgmaNeo
#  Copyright(c) 2023 Ogma Intelligent Systems Corp. All rights reserved.
#
#  This copy of EduOgmaNeo is licensed to you under the terms described
#  in the EDUOGMANEO_LICENSE.md file included in this distribution.
# ----------------------------------------------------------------------------

import numpy as np 
from copy import copy

from .helpers import columnWiseOneHot, sigmoid

# Simple regression-type decoder
class Decoder:
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

        self.hidden_activations = np.zeros(num_hidden_cells) # Used to keep activations around for t+1 prediction

        self.input_state_prev = np.zeros(num_input_cells) # Used to perform state backup for t+1 prediction

        # Hyperparameters
        self.lr = 0.1

    def step(self, input_state, target_hidden_state, learn_enabled):
        # Learning
        if learn_enabled:
            error = target_hidden_state - sigmoid(self.hidden_activations)

            # Learn
            self.weights += self.lr * np.dot(np.matrix(error).T, np.matrix(self.input_state_prev))

        # Activation
        self.hidden_activations = np.dot(self.weights, input_state)

        self.hidden_state = columnWiseOneHot(self.hidden_activations, self.hidden_column_size)

        # Update prev
        self.input_state_prev = copy(input_state)
