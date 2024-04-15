import numpy as np

def convolve(inputs, weights, stride=1, padding=0):
    # Inputs: input data tensor (C, H, W)
    # Weights: filters tensor (M, C, R, S)
    # Stride: the number of steps to move the filter on the input
    # Padding: number of zero-padding rows and columns on each side of input
    
    C, H, W = inputs.shape
    M, _, R, S = weights.shape
    
    # Calculate the dimensions of the output
    H_out = (H + 2 * padding - R) // stride + 1
    W_out = (W + 2 * padding - S) // stride + 1
    
    # Initialize the output tensor
    output = np.zeros((M, H_out, W_out))
    
    # Apply padding to the input tensor
    if padding > 0:
        inputs_padded = np.pad(inputs, ((0, 0), (padding, padding), (padding, padding)), mode='constant', constant_values=0)
    else:
        inputs_padded = inputs
    
    # Perform the convolution operation
    for m in range(M):  # over each filter
        for h in range(H_out):  # over each output row
            for w in range(W_out):  # over each output column
                h_start = h * stride
                w_start = w * stride
                h_end = h_start + R
                w_end = w_start + S
                
                # Element-wise multiplication and sum
                output[m, h, w] = np.sum(inputs_padded[:, h_start:h_end, w_start:w_end] * weights[m, :, :, :])
    
    return output

def check_signs(array1, array2):
    # Ensure that array1 and array2 have the same shape
    if array1.shape != array2.shape:
        print("not equal by shape")
        return False
    
    # Calculate the signs of each element in the arrays
    sign_array1 = np.sign(array1)
    sign_array2 = np.sign(array2)
    
    # Check if all elements have the same sign or are both zero
    return np.array_equal(sign_array1, sign_array2)