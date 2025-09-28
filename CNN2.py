import torch
import torch.nn as nn

# -----------------------------
# Helper function: Apply Conv2D
# -----------------------------
def comp_conv2d(conv2d, X):
    """
    Apply a 2D convolution layer to input X.

    Args:
        conv2d (nn.Conv2d): A Conv2D layer
        X (Tensor): Input tensor of shape (H, W)

    Returns:
        Tensor: Output tensor of shape (H_out, W_out)
    """
    # Add batch and channel dimensions: (1, 1, H, W)
    X = X.reshape((1, 1) + X.shape)
    Y = conv2d(X)
    # Remove batch and channel dimensions -> (H_out, W_out)
    return Y.reshape(Y.shape[2:])

# -----------------------------
# Example usage
# -----------------------------
if __name__ == "__main__":
    # Create a random input tensor of shape (8, 8)
    X = torch.rand(size=(8, 8))

    # 1. Padding = 1, Kernel = 3x3 (output same size as input)
    conv2d = nn.Conv2d(1, 1, kernel_size=3, padding=1)
    print("Kernel=3x3, Padding=1 ->", comp_conv2d(conv2d, X).shape)

    # 2. Padding = (2,1), Kernel = (5,3)
    conv2d = nn.Conv2d(1, 1, kernel_size=(5, 3), padding=(2, 1))
    print("Kernel=(5,3), Padding=(2,1) ->", comp_conv2d(conv2d, X).shape)

    # 3. Stride = 2, Kernel = 3x3, Padding = 1
    conv2d = nn.Conv2d(1, 1, kernel_size=3, padding=1, stride=2)
    print("Kernel=3x3, Padding=1, Stride=2 ->", comp_conv2d(conv2d, X).shape)

    # 4. Stride = (3,4), Kernel = (3,5), Padding = (0,1)
    conv2d = nn.Conv2d(1, 1, kernel_size=(3, 5), padding=(0, 1), stride=(3, 4))
    print("Kernel=(3,5), Padding=(0,1), Stride=(3,4) ->", comp_conv2d(conv2d, X).shape)
