import numpy as np
import torch

def depthwise_convolution(inputs, depthwise_filters, stride, padding):
    # Inputs should be of shape (C, H, W)
    # Depthwise filters should be of shape (C, FH, FW) where FH, FW are filter dimensions
    C, H, W = inputs.shape
    _, FH, FW = depthwise_filters.shape
    
    H_out = (H + 2 * padding - FH) // stride + 1
    W_out = (W + 2 * padding - FW) // stride + 1
    output = np.zeros((C, H_out, W_out))
    
    if padding > 0:
        inputs = np.pad(inputs, ((0, 0), (padding, padding), (padding, padding)), 'constant')
    
    for c in range(C):
        for h in range(H_out):
            for w in range(W_out):
                h_start = h * stride
                w_start = w * stride
                h_end = h_start + FH
                w_end = w_start + FW
                output[c, h, w] = np.sum(inputs[c, h_start:h_end, w_start:w_end] * depthwise_filters[c])
    
    return output


def pointwise_convolution(inputs, pointwise_filters):
    # Inputs should be of shape (C, H, W)
    # Pointwise filters should be of shape (M, C, 1, 1) where M is the number of output channels
    C, H, W = inputs.shape
    M, _, _, _ = pointwise_filters.shape
    output = np.zeros((M, H, W))
    
    for m in range(M):
        for h in range(H):
            for w in range(W):
                output[m, h, w] = np.sum(inputs[:, h, w] * pointwise_filters[m, :, 0, 0])
    
    return output


def depthwise_separable_convolution(inputs, depthwise_filters, pointwise_filters, stride, padding):
    # Perform depthwise convolution
    depthwise_output = depthwise_convolution(inputs, depthwise_filters, stride, padding)
    
    # Perform pointwise convolution on the result of depthwise convolution
    output = pointwise_convolution(depthwise_output, pointwise_filters)
    
    return output

def depthwise_separable_convolution_pytorch(inputs, depthwise_filters, pointwise_filters, stride, padding):
    
    # Convert numpy arrays to PyTorch tensors and add batch and channel dimensions as needed
    inputs_tensor = torch.from_numpy(inputs[np.newaxis, :, :, :]).float()
    depthwise_filters_tensor = torch.from_numpy(depthwise_filters[:, np.newaxis, :, :]).float()  # shape: (out_channels, in_channels/groups, H, W)
    pointwise_filters_tensor = torch.from_numpy(pointwise_filters).float()

    # Depthwise convolution
    depth_conv = torch.nn.Conv2d(inputs_tensor.shape[1], depthwise_filters_tensor.shape[0],
                                 kernel_size=depthwise_filters_tensor.shape[2:], 
                                 groups=inputs_tensor.shape[1],  # Groups equal to number of input channels for depthwise
                                 stride=stride, padding=padding)
    depth_conv.weight.data = depthwise_filters_tensor  # Set weights

    # Pointwise convolution
    point_conv = torch.nn.Conv2d(depth_conv.out_channels, pointwise_filters_tensor.shape[0],
                                 kernel_size=(1, 1),
                                 stride=1, padding=0)
    point_conv.weight.data = pointwise_filters_tensor  # Set weights
    point_conv.bias.data.fill_(0)  # Use zero bias

    # Forward pass
    depthwise_output = depth_conv(inputs_tensor)
    output = point_conv(depthwise_output)
    return output.detach().numpy().squeeze()