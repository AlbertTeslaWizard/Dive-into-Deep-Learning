import torch
import torch.nn.functional as F


def corr2d(X: torch.Tensor, K: torch.Tensor) -> torch.Tensor:
    """
    Single-channel 2D cross-correlation using conv2d.

    Args:
        X (torch.Tensor): Input tensor of shape (H, W).
        K (torch.Tensor): Kernel tensor of shape (h, w).

    Returns:
        torch.Tensor: Output tensor of shape (H-h+1, W-w+1).
    """
    X = X.unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)
    K = K.unsqueeze(0).unsqueeze(0)  # (1, 1, h, w)
    Y = F.conv2d(X, K)
    return Y.squeeze(0).squeeze(0)   # (H-h+1, W-w+1)


def corr2d_multi_in(X: torch.Tensor, K: torch.Tensor) -> torch.Tensor:
    """
    Multi-input, single-output 2D cross-correlation.

    Args:
        X (torch.Tensor): Input tensor of shape (C_in, H, W).
        K (torch.Tensor): Kernel tensor of shape (C_in, kH, kW).

    Returns:
        torch.Tensor: Output tensor of shape (H-kH+1, W-kW+1).
    """
    X = X.unsqueeze(0)  # (1, C_in, H, W)
    K = K.unsqueeze(0)  # (1, C_in, kH, kW)
    Y = F.conv2d(X, K)
    return Y.squeeze(0).squeeze(0)   # (H-kH+1, W-kW+1)


def corr2d_multi_in_out(X: torch.Tensor, K: torch.Tensor) -> torch.Tensor:
    """
    Multi-input, multi-output 2D cross-correlation.

    Args:
        X (torch.Tensor): Input tensor of shape (C_in, H, W).
        K (torch.Tensor): Kernel tensor of shape (C_out, C_in, kH, kW).

    Returns:
        torch.Tensor: Output tensor of shape (C_out, H-kH+1, W-kW+1).
    """
    X = X.unsqueeze(0)  # (1, C_in, H, W)
    Y = F.conv2d(X, K)  # (1, C_out, H-kH+1, W-kW+1)
    return Y.squeeze(0) # (C_out, H-kH+1, W-kW+1)


def corr2d_multi_in_out_1x1(X: torch.Tensor, K: torch.Tensor) -> torch.Tensor:
    """
    Special case: 1x1 convolution (multi-in, multi-out).

    Args:
        X (torch.Tensor): Input tensor of shape (C_in, H, W).
        K (torch.Tensor): Kernel tensor of shape (C_out, C_in, 1, 1).

    Returns:
        torch.Tensor: Output tensor of shape (C_out, H, W).
    """
    return corr2d_multi_in_out(X, K)  # 1x1 is just a special conv2d


# =============================
# Example usage
# =============================

# Single-channel example
X = torch.tensor([[0.0, 1.0, 2.0],
                  [3.0, 4.0, 5.0],
                  [6.0, 7.0, 8.0]])
K = torch.tensor([[0.0, 1.0],
                  [2.0, 3.0]])
print("Single-channel corr2d result:")
print(corr2d(X, K))

# Multi-input example
X = torch.tensor([[[0.0, 1.0, 2.0],
                   [3.0, 4.0, 5.0],
                   [6.0, 7.0, 8.0]],
                  [[1.0, 2.0, 3.0],
                   [4.0, 5.0, 6.0],
                   [7.0, 8.0, 9.0]]])
K = torch.tensor([[[0.0, 1.0],
                   [2.0, 3.0]],
                  [[1.0, 2.0],
                   [3.0, 4.0]]])
print("\nMulti-input corr2d result:")
print(corr2d_multi_in(X, K))

# Multi-input & multi-output example
K_multi_out = torch.stack((K, K + 1, K + 2), dim=0)
print("\nMulti-in, multi-out corr2d result:")
print(corr2d_multi_in_out(X, K_multi_out))

# 1x1 convolution example
X = torch.normal(0, 1, (3, 3, 3))      # (C_in=3, H=3, W=3)
K = torch.normal(0, 1, (2, 3, 1, 1))   # (C_out=2, C_in=3, kH=1, kW=1)
print("\n1x1 convolution result:")
print(corr2d_multi_in_out_1x1(X, K))
