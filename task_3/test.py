import numpy as np

from task_3.func import depthwise_separable_convolution, depthwise_separable_convolution_pytorch

num_iters = 10
# Parameters for the test
input_channels = 3
input_size = 10
depthwise_filter_size = 3
pointwise_output_channels = 8

differences = []

for i in range(num_iters):
    print("Iteration: ", i)
    # Test the function
    inputs = np.random.rand(input_channels, input_size, input_size)
    depthwise_filters = np.random.rand(input_channels, depthwise_filter_size, depthwise_filter_size)
    pointwise_filters = np.random.rand(pointwise_output_channels, input_channels, 1, 1)

    # Get outputs from the custom and PyTorch implementations
    # print("Run custom implementation")
    output_custom = depthwise_separable_convolution(inputs, depthwise_filters, pointwise_filters, stride=1, padding=1)
    # print("Run pytorch implementation")
    output_pytorch = depthwise_separable_convolution_pytorch(inputs, depthwise_filters, pointwise_filters, stride=1, padding=1)

    # Compare the outputs
    difference = np.abs(output_custom - output_pytorch)
    differences.append(difference)
print("Average difference between custom implementation and PyTorch:", np.mean(differences))