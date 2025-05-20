import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import random
from collections import deque

class PinballDQN(nn.Module):
    def __init__(self, input_size, output_size):
        super(PinballDQN, self).__init__()
        
        # Extract dimensions
        self.base_features = 5  # Number of numeric state features
        self.image_features = input_size - self.base_features  # Screen pixels
        
        # Assuming 40x36 binary image (downsampled from 160x144)
        self.screen_width = 40
        self.screen_height = 36
        
        # Efficient CNN with reduced parameters
        self.conv1 = nn.Conv2d(1, 8, kernel_size=4, stride=2, padding=1)
        self.conv2 = nn.Conv2d(8, 16, kernel_size=4, stride=2, padding=1)
        
        # Calculate the size after convolutions with padding (larger strides reduce dimensions faster)
        def conv_output_size(size, kernel_size, stride, padding):
            return (size + 2*padding - kernel_size) // stride + 1
        
        conv1_width = conv_output_size(self.screen_width, 4, 2, 1)
        conv1_height = conv_output_size(self.screen_height, 4, 2, 1)
        
        conv2_width = conv_output_size(conv1_width, 4, 2, 1)
        conv2_height = conv_output_size(conv1_height, 4, 2, 1)
        
        self.conv_output_size = conv2_width * conv2_height * 16
        
        # Simple feature fusion
        self.base_fc = nn.Linear(self.base_features, 16)
        
        # Dueling DQN architecture - with smaller hidden layer
        self.combined_fc = nn.Linear(self.conv_output_size + 16, 64)
        self.value_stream = nn.Linear(64, 1)
        self.advantage_stream = nn.Linear(64, output_size)
        
    def forward(self, x):
        # Split input into base features and image
        base_x = x[:, :self.base_features]
        image_x = x[:, self.base_features:]
        
        # Process base features
        base_features = F.relu(self.base_fc(base_x))
        
        # Process image features
        batch_size = x.size(0)
        image_x = image_x.view(batch_size, 1, self.screen_height, self.screen_width)
        
        image_features = F.relu(self.conv1(image_x))
        image_features = F.relu(self.conv2(image_features))
        image_features = image_features.view(batch_size, -1)
        
        # Combine features
        combined = torch.cat((base_features, image_features), dim=1)
        
        # Shared representation
        features = F.relu(self.combined_fc(combined))
        
        # Dueling architecture
        value = self.value_stream(features)
        advantages = self.advantage_stream(features)
        
        # Combine value and advantages (Q = V + A - mean(A))
        q_values = value + (advantages - advantages.mean(dim=1, keepdim=True))
        
        return q_values

class PinballAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        
        # Learning parameters
        self.gamma = 0.99
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.9995
        self.learning_rate = 0.001
        
        # Create neural networks: main and target
        self.model = PinballDQN(state_size, action_size)
        self.target_model = PinballDQN(state_size, action_size)
        self.update_target_model()
        
        # Standard optimizer
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        
        # Regular replay buffer (more memory efficient)
        self.memory = deque(maxlen=5000)
        
        # Simple memory sample weights (lightweight approximation of prioritized replay)
        self.sample_weights = deque(maxlen=5000)
        
        # Counter for updating target network
        self.update_counter = 0
        self.target_update_freq = 200
        
    def update_target_model(self):
        """
        Copy weights from main network to target network
        """
        self.target_model.load_state_dict(self.model.state_dict())
    
    def remember(self, state, action, reward, next_state, done):
        """
        Store experience in memory with initial weight of 1.0
        """
        self.memory.append((state, action, reward, next_state, done))
        self.sample_weights.append(1.0)  # Start with uniform weights
    
    def act(self, state):
        """
        Select action using epsilon-greedy policy
        """
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            act_values = self.model(state_tensor)
            return torch.argmax(act_values).item()
    
    def sample_batch(self, batch_size):
        """
        Sample batch with a simple weighted approach
        """
        # Normalize weights to get probabilities
        weights_sum = sum(self.sample_weights)
        probs = [w/weights_sum for w in self.sample_weights]
        
        # Sample indices based on probabilities
        indices = np.random.choice(len(self.memory), batch_size, p=probs)
        
        # Get samples
        samples = [self.memory[idx] for idx in indices]
        
        return samples, indices
    
    def update_sample_weights(self, indices, errors):
        """
        Update sample weights based on TD errors (simplified approach)
        """
        for i, idx in enumerate(indices):
            # Simple weighting: larger errors get slightly higher weights
            # But we cap it to avoid excessive focus on outliers
            self.sample_weights[idx] = min(abs(errors[i]) + 0.1, 2.0)
    
    def replay(self, batch_size):
        """
        Train the model using semi-weighted experience replay
        """
        if len(self.memory) < batch_size:
            return None
        
        # Sample batch with simple weighting
        minibatch, indices = self.sample_batch(batch_size)
        
        states = torch.FloatTensor([t[0] for t in minibatch])
        actions = torch.LongTensor([[t[1]] for t in minibatch])
        rewards = torch.FloatTensor([[t[2]] for t in minibatch])
        next_states = torch.FloatTensor([t[3] for t in minibatch])
        dones = torch.FloatTensor([[t[4]] for t in minibatch])
        
        # Current Q values
        curr_q_values = self.model(states).gather(1, actions)
        
        # Calculate TD target with Double Q-learning (efficient version)
        with torch.no_grad():
            # Select actions using online network, but evaluate with target
            next_q_values = self.target_model(next_states).gather(
                1, self.model(next_states).max(1, keepdim=True)[1]
            )
            target_q_values = rewards + (self.gamma * next_q_values * (1 - dones))
        
        # TD errors for updating sample weights
        td_errors = (target_q_values - curr_q_values).detach().squeeze().tolist()
        
        # Calculate loss (standard Huber loss)
        loss = F.smooth_l1_loss(curr_q_values, target_q_values)
        
        # Update model
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # Update sample weights for future sampling
        self.update_sample_weights(indices, td_errors)
        
        # Update epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        
        # Periodically update target network
        self.update_counter += 1
        if self.update_counter % self.target_update_freq == 0:
            self.update_target_model()
        
        return loss.item()