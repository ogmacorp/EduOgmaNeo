import numpy as np

from eduogmaneo.hierarchy import Hierarchy, IO_Desc, Layer_Desc
from eduogmaneo.helpers import columnWiseOneHot

test_num_input_columns = 4
test_input_column_size = 16

inputs = []

for i in range(10):
    test_inputs = columnWiseOneHot(np.random.rand(test_num_input_columns * test_input_column_size), test_input_column_size)

    inputs.append(test_inputs)

# Add same repeats so memory is required
for i in range(8):
    inputs.append(inputs[-1])

h = Hierarchy(IO_Desc(test_num_input_columns, test_input_column_size), 4 * [ Layer_Desc(4, 16) ])

# Go through in order a couple of times
for episode in range(100):
    correct = 0

    for i in range(len(inputs)):
        # Check if last prediction is equal to new input
        if np.all(inputs[i] == h.decoders[0].hidden_state):
            correct += 1

        h.step(inputs[i], True)

    print(f"Got {correct} out of {len(inputs)}") 


