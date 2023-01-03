import numpy as np 

from .encoder import Encoder
from .decoder import Decoder

from copy import copy

# Descriptors
class IO_Desc:
    def __init__(self, num_columns, column_size):
        self.num_columns = num_columns
        self.column_size = column_size

class Layer_Desc:
    def __init__(self, num_hidden_columns, hidden_column_size, ticks_per_update=2, temporal_horizon=2):
        self.num_hidden_columns = num_hidden_columns
        self.hidden_column_size = hidden_column_size
        self.ticks_per_update = ticks_per_update
        self.temporal_horizon = temporal_horizon

# Main class for using SPH
class Hierarchy:
    def __init__(self, io_desc, layer_descs):
        self.io_desc = io_desc
        self.layer_descs = layer_descs

        self.encoders = []
        self.decoders = []
        self.histories = [] # History buffer of input to each layer
        self.ticks = []

        for l in range(len(layer_descs)):
            # First layer is handled separately
            if l == 0:
                enc = Encoder(io_desc.num_columns * layer_descs[l].temporal_horizon, io_desc.column_size, layer_descs[l].num_hidden_columns, layer_descs[l].hidden_column_size)
                self.encoders.append(enc)

                # Two inputs for decoder if has feedback, else 1
                dec = Decoder((2 if l < len(layer_descs) - 1 else 1) * layer_descs[l].num_hidden_columns, layer_descs[l].hidden_column_size, io_desc.num_columns, io_desc.column_size)
                self.decoders.append(dec)

                hist = layer_descs[l].temporal_horizon * [ np.zeros(io_desc.num_columns * io_desc.column_size) ]
                self.histories.append(hist)
            else:
                enc = Encoder(layer_descs[l - 1].num_hidden_columns * layer_descs[l].temporal_horizon, layer_descs[l - 1].hidden_column_size, layer_descs[l].num_hidden_columns, layer_descs[l].hidden_column_size)
                self.encoders.append(enc)

                # Two inputs for decoder if has feedback, else 1
                dec = Decoder((2 if l < len(layer_descs) - 1 else 1) * layer_descs[l].num_hidden_columns, layer_descs[l].hidden_column_size, layer_descs[l - 1].num_hidden_columns * layer_descs[l].ticks_per_update, layer_descs[l - 1].hidden_column_size)
                self.decoders.append(dec)

                hist = layer_descs[l].temporal_horizon * [ np.zeros(layer_descs[l].num_hidden_columns * layer_descs[l].hidden_column_size) ]
                self.histories.append(hist)

            self.ticks.append(0)

    def step(self, input_state, learn_enabled):
        # Add to history
        self.histories[0].insert(0, copy(input_state))
        self.histories[0].pop()

        updates = len(self.encoders) * [ False ]

        # Up pass
        for l in range(len(self.encoders)):
            if l == 0 or self.ticks[l] >= self.layer_descs[l].ticks_per_update:
                self.ticks[l] = 0

                updates[l] = True

                self.encoders[l].step(np.concatenate(self.histories[l]), learn_enabled)

                if l < len(self.encoders) - 1:
                    self.histories[l + 1].insert(0, copy(self.encoders[l].hidden_state))
                    self.histories[l + 1].pop()

                    self.ticks[l + 1] += 1

        # Down pass
        for l in range(len(self.encoders) - 1, -1, -1):
            if updates[l]:
                # Get complete state for decoder (current hidden + feedback if applicable)
                complete_state = None

                # If has feedback
                if l < len(self.encoders) - 1:
                    timeslice = self.layer_descs[l + 1].ticks_per_update - 1 - self.ticks[l + 1] # Need to flip around since we want the newest predictions to be at the front

                    feedback_state = self.decoders[l + 1].hidden_state[timeslice * len(self.encoders[l].hidden_state) : (timeslice + 1) * len(self.encoders[l].hidden_state)]

                    complete_state = np.concatenate((self.encoders[l].hidden_state, feedback_state))
                else:
                    complete_state = self.encoders[l].hidden_state

                # State to predict (from input/layer below)
                target_state = None

                if l == 0:
                    target_state = input_state
                else:
                    target_state = np.concatenate(self.histories[l][:self.layer_descs[l].ticks_per_update])

                self.decoders[l].step(complete_state, target_state, learn_enabled)

        return self.decoders[0].hidden_state # Return prediction


