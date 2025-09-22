import torch
import torch.nn as nn
import torch.nn.functional as F


class MLP(nn.Module):
    """A simple Multi-Layer Perceptron with weight initialization."""

    def __init__(self, input_dim=20, hidden_dim=256, output_dim=10):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )
        # Apply custom weight initialization
        self.net.apply(self._init_weights)

    def forward(self, x):
        return self.net(x)

    @staticmethod
    def _init_weights(m):
        """Initialize linear layers with Normal(0, 0.01)."""
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, std=0.01)


class MySequential(nn.Module):
    """Custom implementation of Sequential container."""

    def __init__(self, *args):
        super().__init__()
        for idx, module in enumerate(args):
            self._modules[str(idx)] = module

    def forward(self, x):
        for block in self._modules.values():
            x = block(x)
        return x


class FixedHiddenMLP(nn.Module):
    """MLP with a fixed random hidden transformation."""

    def __init__(self, input_dim=20, hidden_dim=20):
        super().__init__()
        # Fixed random weight (not trainable)
        self.rand_weight = torch.rand((hidden_dim, hidden_dim), requires_grad=False)
        self.linear = nn.Linear(input_dim, hidden_dim)

    def forward(self, x):
        x = self.linear(x)
        # Apply fixed random transformation with ReLU
        x = F.relu(torch.mm(x, self.rand_weight) + 1)
        x = self.linear(x)
        # Scale down if values get too large
        while x.abs().sum() > 1:
            x /= 2
        return x.sum()


class NestedMLP(nn.Module):
    """An MLP with a nested Sequential sub-network."""

    def __init__(self, input_dim=20, hidden1=64, hidden2=32, output_dim=16):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden1),
            nn.ReLU(),
            nn.Linear(hidden1, hidden2),
            nn.ReLU()
        )
        self.linear = nn.Linear(hidden2, output_dim)

    def forward(self, x):
        return self.linear(self.net(x))


if __name__ == "__main__":
    # Example: combining different modules into one model
    X = torch.rand(2, 20)
    model = nn.Sequential(
        NestedMLP(),
        nn.Linear(16, 20),
        FixedHiddenMLP()
    )
    print(model(X))
