import numpy as np

from task_1.func import convolve
from task_2.func import conv_layer_im2col

# Example tensors
input_tensor = np.random.rand(1, 4, 4)
filters = np.random.rand(2, 1, 3, 3)

# Convolution with direct method
output_direct = convolve(input_tensor, filters, stride=1, padding=0)

# Convolution with im2col method
output_im2col = conv_layer_im2col(input_tensor, filters, stride=1, padding=0)

# Check if the outputs are close enough
print("Direct output:")
print(output_direct)
print()
print("Im2col output:")
print(output_im2col)
print()
print("Difference:", np.max(np.abs(output_direct - output_im2col)))
print("="*100)