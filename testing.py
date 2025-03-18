from environment import LlamaEnv
from data import config_dataset
import os
from dotenv import load_dotenv
import torch
import torch.optim as optim
from model import QNetwork, train

load_dotenv()
HF_TOKEN = os.getenv('HF_TOKEN')

model_name = "meta-llama/Llama-3.2-1B"
q_network, target_network = QNetwork(), QNetwork()

target_network.load_state_dict(q_network.state_dict())  # Copy weights
target_network.eval()  # Target network does not train

optimizer = optim.Adam(q_network.parameters(), lr=0.001)

dataset = config_dataset(model_name, HF_TOKEN)
env = LlamaEnv(target_network, dataset)

num_steps = 0
for episode in range(20): # Number of episodes

    print(f"Episode {episode}")

    num_steps += 1
    with torch.no_grad():
        action_probs = target_network()
        action = torch.multinomial(action_probs[0][0], num_samples=1).item()
        sarsd = env.step(action)
        print(sarsd)
    
    # Train every UPDATE_FREQUENCY steps
    if num_steps % 2 == 0:
        train(env, q_network, target_network, optimizer)
    
    # Update target network
    if num_steps % 1000 == 0:
        target_network.load_state_dict(q_network.state_dict())

# how do I know if it's training?