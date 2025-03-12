import torch
from collections import deque
import random

class LlamaEnv:
    """Lightweight custom environment with large action and observation spaces"""

    def _select_random_context(self):
      first_item = next(self.dataset)
      
      attention_mask = first_item['attention_mask']
      valid_indices = torch.where(attention_mask == 1)[0]
      selected_index = random.choice(valid_indices.tolist())

      input_ids = first_item['input_ids']
      input_ids[selected_index + 1:] = 128001
      attention_mask[selected_index + 1:] = 0

      return {'input_ids': input_ids.unsqueeze(0), 'attention_mask': attention_mask.unsqueeze(0)}
    
    def __init__(self, model, dataset):
        self.action_size = 128256
        self.state_size = 128256
        MEMORY_SIZE = 10000

        self.memory = deque(maxlen=MEMORY_SIZE)

        self.model = model
        self.dataset = iter(dataset) # tokenized dataset, created using load_dataset
        
        self.context = self._select_random_context()  # size = (1, max_length)
        self.state = self.model(input_ids=self.context['input_ids']).logits  # size = (1, max_length, state_size)
    
    # everything should be batched eventually, doing one at a time for now
    def step(self, action):
        with torch.no_grad():
          # Calculate reward as log probability of the selected action state
          log_probs = torch.log_softmax(self.state, dim=-1)
          reward = log_probs[0, -1, action].item()  # Get log prob of selected action
          
          padding_positions = (self.context["attention_mask"] == 0).to(dtype=torch.int)  # Convert to int for argmax
          first_padding_idx = torch.argmax(padding_positions, dim=1)  # First 0 per batch
          batch_indices = torch.arange(1)

          # Replace the first padding token in input_ids
          self.context["input_ids"][batch_indices, first_padding_idx] = action

          # Update attention mask
          self.context["attention_mask"][batch_indices, first_padding_idx] = 1

          # Update state using model logits with new context
          outputs = self.model(input_ids=self.context['input_ids'])
          next_state = outputs.logits
          done = first_padding_idx == 511 or action == 128001

          sarsd = (self.state, action, reward, next_state, done)
          self.memory.append(sarsd)

          if done:
            self.reset()
          else:
            self.state = next_state
        
        return sarsd
    
    def sample_experiences(self, BATCH_SIZE=32):
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
          self.state = self.model(input_ids=self.context['input_ids']).logits

        return self.state
    
    def close(self):
        pass 