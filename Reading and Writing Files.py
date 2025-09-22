import torch
from torch import nn
from torch.nn import functional as F

# =========================
# Tensor save/load examples
# =========================

# Save and load a single tensor
x = torch.arange(4)
torch.save(x, 'x-file.pt')  # Save tensor to disk
x2 = torch.load('x-file.pt')  # Load tensor from disk
print("Single tensor:", x2)

# Save and load multiple tensors as a list
y = torch.zeros(4)
torch.save([x, y], 'x-files.pt')
x2, y2 = torch.load('x-files.pt')
print("Multiple tensors:", x2, y2)

# Save and load tensors as a dictionary
mydict = {'x': x, 'y': y}
torch.save(mydict, 'mydict.pt')
mydict2 = torch.load('mydict.pt')
print("Dictionary of tensors:", mydict2)

# =========================
# Define a simple MLP model
# =========================

class MLP(nn.Module):
    def __init__(self, input_size=20, hidden_size=256, output_size=10):
        super().__init__()
        self.hidden = nn.Linear(input_size, hidden_size)
        self.output = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = F.relu(self.hidden(x))
        return self.output(x)

# Instantiate the model and create dummy input
net = MLP()
X = torch.randn(2, 20)
Y = net(X)
print("Original model output:", Y)

# Save model parameters
torch.save(net.state_dict(), 'mlp.params')

# Load model parameters into a new model
clone = MLP()
clone.load_state_dict(torch.load('mlp.params'))
clone.eval()  # Set model to evaluation mode
print("Cloned model output:", clone(X))
