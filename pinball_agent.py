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

class PinballDQN(nn.Module):
    def __init__(self, input_size, output_size, dropout_rate=0.2):
        super(PinballDQN, self).__init__()
        
        # Fixed dimensions based on our knowledge
        self.base_features = 5  # Number of numeric state features
        
        # Assume 40x36 binary image (downsampled from 160x144)
        self.screen_width = 40
        self.screen_height = 36
        
        # Enhanced CNN architecture with batch normalization and efficient filters
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=1)
        self.bn1 = nn.BatchNorm2d(16)
        self.pool1 = nn.MaxPool2d(2, 2)  # Add pooling for better feature extraction
        
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.pool2 = nn.MaxPool2d(2, 2)
        
        # Add a third conv layer for better feature extraction
        self.conv3 = nn.Conv2d(32, 32, kernel_size=3, stride=1)
        self.bn3 = nn.BatchNorm2d(32)
        
        # Calculate the size after convolutions and pooling
        # After conv1 + pool1: (40-3+1)/1 = 38, then 38/2 = 19
        # After conv2 + pool2: (19-3+1)/1 = 17, then 17/2 = 8
        # After conv3: (8-3+1)/1 = 6
        conv_width = 6
        conv_height = 5  # Similar calculation for height starting from 36
        
        self.conv_output_size = conv_width * conv_height * 32
        
        # FC layer for processing base features - increase capacity
        self.base_fc = nn.Linear(self.base_features, 32)
        self.base_bn = nn.BatchNorm1d(32)
        
        # Combine conv features with base features - larger layers for more capacity
        self.fc1 = nn.Linear(self.conv_output_size + 32, 128)
        self.fc1_bn = nn.BatchNorm1d(128)
        self.dropout1 = nn.Dropout(dropout_rate)
        
        self.fc2 = nn.Linear(128, 64)
        self.fc2_bn = nn.BatchNorm1d(64)
        self.dropout2 = nn.Dropout(dropout_rate)
        
        # Output layer - no activation needed for Q-values
        self.fc3 = nn.Linear(64, output_size)
        
        # Initialize weights with improved method
        self._initialize_weights()
        
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
        
    def forward(self, x):
        # Split input into base features and image
        base_x = x[:, :self.base_features]
        image_x = x[:, self.base_features:]
        
        # Process base features with batch normalization
        base_features = F.relu(self.base_bn(self.base_fc(base_x)))
        
        # Reshape and process image features
        batch_size = x.size(0)
        image_x = image_x.view(batch_size, 1, self.screen_height, self.screen_width)
        
        # Enhanced convolutional pipeline with batch norm and pooling
        image_features = F.relu(self.bn1(self.conv1(image_x)))
        image_features = self.pool1(image_features)
        
        image_features = F.relu(self.bn2(self.conv2(image_features)))
        image_features = self.pool2(image_features)
        
        image_features = F.relu(self.bn3(self.conv3(image_features)))
        image_features = image_features.view(batch_size, -1)
        
        # Combine features
        combined = torch.cat((base_features, image_features), dim=1)
        
        # Enhanced fully connected path with batch norm and dropout
        x = F.relu(self.fc1_bn(self.fc1(combined)))
        x = self.dropout1(x)
        
        x = F.relu(self.fc2_bn(self.fc2(x)))
        x = self.dropout2(x)
        
        return self.fc3(x)

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