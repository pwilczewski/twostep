import torch.nn as nn
from environment import LlamaEnv
from training_data import config_dataset
import os
from dotenv import load_dotenv
import torch
import torch.optim as optim

load_dotenv()
HF_TOKEN = os.getenv('HF_TOKEN')

class LogitsOutput:
    def __init__(self, logits):
        self.logits = logits

class QNetwork(nn.Module):
    def __init__(self):
        super(QNetwork, self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.state = torch.abs(torch.randn(1, 1, 128256) ** 10)  # Exaggerate differences
        self.logits = torch.logit(self.state/self.state.sum(dim=-1)).to(self.device)
        # self.logits = self.logits*5 # even more extreme!
    
    def forward(self, input_ids):
        # Return logits directly instead of in a dict to match expected output
        return LogitsOutput(self.logits)

def train(env, q_network, target_network, optimizer):
    if len(env.memory) < 32:
        return  # Not enough samples

    states, actions, rewards, next_states, dones = env.sample_experiences()

    # Compute Q-values for selected actions
    q_values = q_network(states).gather(1, actions.unsqueeze(1)).squeeze(1)

    # Compute target Q-values using the target network
    with torch.no_grad():
        next_q_values = target_network(next_states).max(1)[0]
        target_q_values = rewards + 0.99 * next_q_values * (1 - dones)

    # Compute loss and update network
    loss = nn.MSELoss()(q_values, target_q_values)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

model_name = "meta-llama/Llama-3.2-1B"
q_network = QNetwork()
target_network = QNetwork()

print(q_network.parameters())

target_network.load_state_dict(q_network.state_dict())  # Copy weights
target_network.eval()  # Target network does not train

optimizer = optim.Adam(q_network.parameters(), lr=0.001)

dataset = config_dataset(model_name, HF_TOKEN)
env = LlamaEnv(target_network, dataset)

num_steps = 0
for episode in range(10): # Number of episodes
    done = False
    
    while not done:
        num_steps += 1
        action = torch.argmax(target_network.logits, dim=-1)
        sarsd = env.step(action)
        
        # Train every UPDATE_FREQUENCY steps
        if num_steps % 4 == 0:
            train(env, q_network, target_network, optimizer)
        
        # Update target network
        if num_steps % 1000 == 0:
            target_network.load_state_dict(q_network.state_dict())

# a trained network will transform its internal random state, to the random state of another network