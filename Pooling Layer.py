import torch
from torch import nn
from d2l import torch as d2l


def main():
    # Select device (use GPU if available, otherwise CPU)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # ---------------- Example 1: Average Pooling ----------------
    # Input: a 3x3 matrix, reshaped to (1, 1, 3, 3) for batch and channel dimensions
    X = torch.tensor([
        [0.0, 1.0, 2.0],
        [3.0, 4.0, 5.0],
        [6.0, 7.0, 8.0]
    ]).unsqueeze(0).unsqueeze(0).to(device)

    # Define 2D average pooling with kernel size 2x2 and stride 1
    pool2d_mean = nn.AvgPool2d(kernel_size=2, stride=1)
    Y = pool2d_mean(X)
    print("Example 1: AvgPool2d Output")
    print(Y.squeeze())  # Remove dimensions of size 1 for readability
    print("-" * 50)

    # ---------------- Example 2: Max Pooling with padding ----------------
    # Input: 4x4 matrix with values from 0 to 15, reshaped to (1, 1, 4, 4)
    X = torch.arange(16, dtype=torch.float32).reshape((1, 1, 4, 4)).to(device)

    # Define 2D max pooling with kernel size (2,3), stride (2,3), and padding (0,1)
    pool2d = nn.MaxPool2d(kernel_size=(2, 3), stride=(2, 3), padding=(0, 1))
    Y = pool2d(X)
    print("Example 2: MaxPool2d Output with padding")
    print(Y.squeeze())
    print("-" * 50)

    # ---------------- Example 3: Max Pooling with multiple channels ----------------
    # Concatenate X and X+1 along the channel dimension -> shape becomes (1, 2, 4, 4)
    X = torch.cat((X, X + 1), dim=1).to(device)

    # Define 2D max pooling with kernel size 3x3, stride 2, and padding 1
    pool2d = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
    Y = pool2d(X)
    print("Example 3: MaxPool2d Output with multiple channels")
    print(Y.squeeze())  # Output keeps the same number of channels (2)
    print("-" * 50)


if __name__ == "__main__":
    main()
