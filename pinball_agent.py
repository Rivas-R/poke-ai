"""
Pokemon Pinball - Agent Implementation (DQN) - Optimized
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import random
from collections import deque

# Define a more efficient neural network architecture for the agent
class PinballDQN(nn.Module):
    def __init__(self, input_size, output_size):
        super(PinballDQN, self).__init__()
        
        # Extract dimensions
        self.base_features = 3  # Number of numeric state features
        self.image_features = input_size - self.base_features  # Screen pixels
        
        # Process image features with optimized convolutional layers if input includes image
        if self.image_features > 0:
            # Assuming 40x36 binary image (downsampled from 160x144)
            self.screen_width = 40
            self.screen_height = 36
            
            # Smaller kernels and fewer filters for faster processing
            self.conv1 = nn.Conv2d(1, 8, kernel_size=3, stride=2)
            self.conv2 = nn.Conv2d(8, 16, kernel_size=3, stride=1)
            
            # Calculate the size after convolutions
            def conv_output_size(size, kernel_size, stride):
                return (size - kernel_size) // stride + 1
            
            conv1_width = conv_output_size(self.screen_width, 3, 2)
            conv1_height = conv_output_size(self.screen_height, 3, 2)
            
            conv2_width = conv_output_size(conv1_width, 3, 1)
            conv2_height = conv_output_size(conv1_height, 3, 1)
            
            self.conv_output_size = conv2_width * conv2_height * 16
            
            # FC layer for processing base features
            self.base_fc = nn.Linear(self.base_features, 16)
            
            # Combine conv features with base features - smaller layers
            self.fc1 = nn.Linear(self.conv_output_size + 16, 64)
            self.fc2 = nn.Linear(64, output_size)
        else:
            # If no image features, use simple fully connected architecture
            self.fc1 = nn.Linear(input_size, 32)
            self.fc2 = nn.Linear(32, output_size)
        
    def forward(self, x):
        if hasattr(self, 'conv1'):
            # Split input into base features and image
            base_x = x[:, :self.base_features]
            image_x = x[:, self.base_features:]
            
            # Process base features
            base_features = F.relu(self.base_fc(base_x))
            
            # Reshape and process image features
            batch_size = x.size(0)
            image_x = image_x.view(batch_size, 1, self.screen_height, self.screen_width)
            image_features = F.relu(self.conv1(image_x))
            image_features = F.relu(self.conv2(image_features))
            image_features = image_features.view(batch_size, -1)
            
            # Combine features
            combined = torch.cat((base_features, image_features), dim=1)
            
            # Simplified fully connected path
            x = F.relu(self.fc1(combined))
            return self.fc2(x)
        else:
            # Simple fully connected path
            x = F.relu(self.fc1(x))
            return self.fc2(x)

# Reinforcement learning agent class
class PinballAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        
        # Learning parameters
        self.gamma = 0.99    # Discount factor
        self.epsilon = 1.0   # Initial exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.9995
        self.learning_rate = 0.001
        
        # Create neural networks: main and target
        self.model = PinballDQN(state_size, action_size)
        self.target_model = PinballDQN(state_size, action_size)
        self.update_target_model()
        
        # Optimizer
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        
        # Experience memory - slightly reduced size for memory efficiency
        self.memory = deque(maxlen=5000)
        
        # Counter for updating target network
        self.update_counter = 0
        
    def update_target_model(self):
        """
        Copy weights from main network to target network
        """
        self.target_model.load_state_dict(self.model.state_dict())
    
    def remember(self, state, action, reward, next_state, done):
        """
        Store experience in memory
        """
        self.memory.append((state, action, reward, next_state, done))
    
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
    
    def replay(self, batch_size):
        """
        Train the model using experience replay
        """
        if len(self.memory) < batch_size:
            return None
        
        # Sample random batch from memory
        minibatch = random.sample(self.memory, batch_size)
        
        states = torch.FloatTensor([t[0] for t in minibatch])
        actions = torch.LongTensor([[t[1]] for t in minibatch])
        rewards = torch.FloatTensor([[t[2]] for t in minibatch])
        next_states = torch.FloatTensor([t[3] for t in minibatch])
        dones = torch.FloatTensor([[t[4]] for t in minibatch])
        
        # Current Q values
        curr_q_values = self.model(states).gather(1, actions)
        
        # Future Q values using target network
        with torch.no_grad():
            next_q_values = self.target_model(next_states).max(1, keepdim=True)[0]
        
        # Calculate target Q values
        target_q_values = rewards + (self.gamma * next_q_values * (1 - dones))
        
        # Calculate loss and update model
        loss = F.smooth_l1_loss(curr_q_values, target_q_values)
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # Update epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        
        # Periodically update target network
        self.update_counter += 1
        if self.update_counter % 100 == 0:
            self.update_target_model()
        
        return loss.item()