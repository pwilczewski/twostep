import torch
from collections import deque
import random

class LlamaEnv:
    """Lightweight custom environment with large action and observation spaces"""

    def _select_random_context(self):
      fixed_data = self.fixed_data
      start_index = random.randint(0, fixed_data['input_ids'].size(0) - 1)

      return {'input_ids': fixed_data['input_ids'][0:start_index].unsqueeze(0), 
              'attention_mask': fixed_data['attention_mask'][0:start_index].unsqueeze(0)}
    
    def __init__(self, model, dataset):
        self.action_size = 128256
        self.state_size = 128256
        MEMORY_SIZE = 10000

        self.memory = deque(maxlen=MEMORY_SIZE)

        self.model = model
        dataset = iter(dataset) # tokenized dataset, created using load_dataset
        self.fixed_data = next(dataset)
        
        self.context = self._select_random_context()  # size = (1, max_length)
        self.state = self.model()  # size = (1, max_length, state_size)
    
    def step(self, action):
        with torch.no_grad():
          # Calculate reward as log probability of the selected action state
          log_probs = torch.log_softmax(self.state, dim=-1)
          reward = log_probs[0, -1, action].item()

          # Update context with chosen action
          self.context["input_ids"] = torch.cat((self.context["input_ids"], torch.tensor([[action]])), dim=1)
          self.context["attention_mask"] = torch.cat((self.context["attention_mask"], torch.tensor([[1]])), dim=1)

          # Update state using model logits with new context
          next_state = self.model() # not yet actually using the new context
          done = self.context["input_ids"].shape[1] == 512 or action == 128001

          sarsd = (self.state, action, reward, next_state, done)
          self.memory.append(sarsd)

          if done:
            self.reset()
          else:
            self.state = next_state
        
        return sarsd
    
    def sample_experiences(self, BATCH_SIZE=1):
        batch = random.sample(self.memory, BATCH_SIZE)
        states, actions, rewards, next_states, dones = zip(*batch)
        
        # Convert tuple of tensors into batched tensors
        states = torch.cat([s.squeeze(0) for s in states]).unsqueeze(1)  # Shape: (batch, 1, state_size)
        actions = torch.tensor(actions)  # Shape: (batch,)
        rewards = torch.tensor(rewards)  # Shape: (batch,)
        next_states = torch.cat([ns.squeeze(0) for ns in next_states]).unsqueeze(1)  # Shape: (batch, 1, state_size)
        dones = torch.tensor(dones)  # Shape: (batch,)
        
        return states, actions, rewards, next_states, dones

    def reset(self):
        """Reset the environment"""
        with torch.no_grad():
          self.context = self._select_random_context()
          self.state = self.model()

        return self.state
    
    def close(self):
        pass 