import torch.nn as nn

# Define Q-network
class DQN(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(DQN, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(state_dim, 3072, bias=False),
            nn.ReLU(),
            nn.Linear(3072, 3072, bias=False),
            nn.ReLU(),
            nn.Linear(3072, action_dim, bias=False)
        )
        self.initialize_parameters()

    def initialize_parameters(self):
        for layer in self.fc:
            if isinstance(layer, nn.Linear):
                nn.init.uniform_(layer.weight, a=-0.001, b=0.001)

    def forward(self, x):
        return x + self.fc(x)