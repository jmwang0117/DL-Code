import numpy as np

def conv2d(input, kernel, stride=1, padding=0):
    """
    2D Convolution without bias term.

    Parameters:
    - input: 2D array of shape (height, width).
    - kernel: 2D convolution kernel of shape (k_height, k_width).
    - stride: stride of the convolution.
    - padding: zero-padding added to both sides of the input.

    Returns:
    - output: 2D array of convolved data.
    """
    # Add padding to the input image
    input_padded = np.pad(input, padding, mode='constant', constant_values=0)
    
    # Calculate output dimensions
    output_height = ((input_padded.shape[0] - kernel.shape[0]) // stride) + 1
    output_width = ((input_padded.shape[1] - kernel.shape[1]) // stride) + 1
    
    # Initialize the output with zeros
    output = np.zeros((output_height, output_width))
    
    # Perform convolution
    for y in range(0, output_height):
        for x in range(0, output_width):
            output[y, x] = np.sum(input_padded[y*stride:y*stride+kernel.shape[0], x*stride:x*stride+kernel.shape[1]] * kernel)
    
    return output

def test_conv2d():
    # Define a simple input array
    input = np.array([
        [1, 2, 3, 4],
        [5, 6, 7, 8],
        [9, 10, 11, 12],
        [13, 14, 15, 16]
    ])
    
    # Define a simple kernel
    kernel = np.array([
        [1, 0],
        [0, -1]
    ])
    
    # Perform convolution
    output = conv2d(input, kernel, stride=1, padding=0)
    
    # Expected output calculated manually or using a reference method
    expected_output = np.array([
        [1, 2, 3],
        [5, 6, 7],
        [9, 10, 11]
    ])
    
    # Validate the output
    assert np.allclose(output, expected_output), "The output of the convolution does not match the expected output."
    print("Test passed! Output matches the expected output.")

test_conv2d()