import torch.nn as nn
import torch

# since my network isn't using input_ids, it's not batching anything...
class QNetwork(nn.Module):
    def __init__(self):
        super(QNetwork, self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        rands = torch.randn(1, 1, 128256)
        rands[0, 0, torch.randint(0, 128255, size=(1,)).item()] = 13.0
        dist = torch.softmax(rands, dim=-1) # Exaggerate differences
        # state is going to be log probability
        self.state = torch.log(dist/dist.sum(dim=-1)).to(self.device)
        self.bias = nn.Parameter(torch.zeros(1, 1, 128256)).to(self.device)
    
    def forward(self, input_ids=None):
        # simplest version possible, bias terms only
        q_value = torch.softmax(self.state + self.bias, dim=-1)
        return q_value

def train(env, q_network, target_network, optimizer):
    if len(env.memory) < 1:
        return  # Not enough samples

    states, actions, log_p_a, next_states, dones = env.sample_experiences()

    actions = actions.view(-1, 1, 1)
    # network outputs probability of each action, q-value is transformation of that
    q_values = torch.log(q_network(states).gather(2, actions).squeeze(1))

    # Compute target Q-values using the target network
    with torch.no_grad():
        log_p_b = torch.log(target_network(next_states).max(-1)[0])
        target_q_values = log_p_a + log_p_b * (1 - dones.float())
        print(nn.MSELoss()(q_values, target_q_values))

    # Compute loss and update network
    loss = nn.MSELoss()(q_values, target_q_values)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

# Define Q-network
# how do I want it to look in prod?
class DQN(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=3072):
        super(DQN, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(state_dim, hidden_dim, bias=False),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim, bias=False),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim, bias=False)
        )
        self.initialize_parameters()

    def initialize_parameters(self):
        for layer in self.fc:
            if isinstance(layer, nn.Linear):
                nn.init.uniform_(layer.weight, a=-0.001, b=0.001)

    def forward(self, x):
        return x + self.fc(x)