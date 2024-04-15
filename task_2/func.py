import numpy as np

def im2col(input_data, kernel_height, kernel_width, stride, padding):
    # Add padding around the input
    C, H, W = input_data.shape
    H_padded, W_padded = H + 2 * padding, W + 2 * padding
    input_padded = np.pad(input_data, ((0, 0), (padding, padding), (padding, padding)), 'constant')

    # Calculate the size of the output height and width
    out_height = (H_padded - kernel_height) // stride + 1
    out_width = (W_padded - kernel_width) // stride + 1

    # Create an array to hold the columns
    cols = np.zeros((C * kernel_height * kernel_width, out_height * out_width))

    # Fill the array with patches from the input
    col_idx = 0
    for y in range(0, H_padded - kernel_height + 1, stride):
        for x in range(0, W_padded - kernel_width + 1, stride):
            patch = input_padded[:, y:y + kernel_height, x:x + kernel_width].reshape(C * kernel_height * kernel_width)
            cols[:, col_idx] = patch
            col_idx += 1
    
    return cols

# Test im2col with a small example
input_data = np.random.rand(1, 4, 4)  # 1 channel, 4x4 input
kernel_height, kernel_width, stride, padding = 3, 3, 1, 0
cols = im2col(input_data, kernel_height, kernel_width, stride, padding)
print("Columns shape:", cols.shape)

def conv_layer_im2col(inputs, weights, stride, padding):
    # Get dimensions
    C, H, W = inputs.shape
    M, _, R, S = weights.shape

    # Transform input to columns
    cols = im2col(inputs, R, S, stride, padding)
    
    # Reshape weights to rows
    weights_reshaped = weights.reshape(M, -1)
    
    # Perform matrix multiplication
    output_reshaped = np.dot(weights_reshaped, cols)
    
    # Reshape the output
    output_height = (H + 2 * padding - R) // stride + 1
    output_width = (W + 2 * padding - S) // stride + 1
    output = output_reshaped.reshape(M, output_height, output_width)
    
    return output