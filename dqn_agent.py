"""
Double Deep Q-Network (DDQN) Agent with Bidirectional LSTM

This module implements the core RL agent that learns optimal lockdown policies.
The agent uses:
- Bidirectional LSTM to process temporal sequences of epidemic states
- Double DQN to reduce overestimation bias
- Experience replay for sample efficiency
- Target network for stable learning
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from collections import deque
import os
import json

from replay_buffer import create_replay_buffer

class LSTMQNetwork(nn.Module):
    """
    Q-Network architecture with Bidirectional LSTM.
    
    Architecture:
    1. Bidirectional LSTM processes temporal sequences
    2. Fully connected layers map LSTM output to Q-values
    3. Dropout for regularization
    
    The bidirectional LSTM helps capture:
    - Temporal patterns in epidemic dynamics (e.g., exponential growth)
    - Delayed effects of interventions (lockdowns take time to show results)
    - Awareness-disease co-evolution patterns
    
    Input: [batch_size, sequence_length, state_dim]
    Output: [batch_size, action_dim] Q-values for each action
    """
    
    def __init__(self, config):
        """
        Initialize Q-Network.
        
        Args:
            config (DQNConfig): Configuration object with network parameters
        """
        super(LSTMQNetwork, self).__init__()
        
        self.state_dim = config.state_dim
        self.action_dim = config.action_dim
        self.lstm_hidden_size = config.lstm_hidden_size
        self.lstm_num_layers = config.lstm_num_layers
        self.bidirectional = config.lstm_bidirectional
        
        # === LSTM Layer ===
        # Bidirectional LSTM: processes sequence in both forward and backward directions
        self.lstm = nn.LSTM(
            input_size=self.state_dim,
            hidden_size=self.lstm_hidden_size,
            num_layers=self.lstm_num_layers,
            batch_first=True,  # Input shape: [batch, seq, features]
            dropout=config.lstm_dropout if self.lstm_num_layers > 1 else 0,
            bidirectional=self.bidirectional
        )
        
        # Calculate LSTM output size (doubled if bidirectional)
        lstm_output_size = self.lstm_hidden_size * (2 if self.bidirectional else 1)
        
        # === Fully Connected Layers ===
        # Build FC network dynamically based on config
        fc_layers = []
        input_size = lstm_output_size
        
        for hidden_size in config.fc_hidden_sizes:
            fc_layers.append(nn.Linear(input_size, hidden_size))
            fc_layers.append(nn.ReLU())
            fc_layers.append(nn.Dropout(config.fc_dropout))
            input_size = hidden_size
        
        # Final layer outputs Q-values for each action
        fc_layers.append(nn.Linear(input_size, self.action_dim))
        
        self.fc_network = nn.Sequential(*fc_layers)
        
        # Initialize weights (helps with training stability)
        self._initialize_weights()
        
    def _initialize_weights(self):
        """
        Initialize network weights using Xavier/Kaiming initialization.
        """
        for module in self.modules():
            if isinstance(module, nn.Linear):
                # Kaiming initialization for ReLU activations
                nn.init.kaiming_normal_(module.weight, mode='fan_in', nonlinearity='relu')
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.LSTM):
                # Xavier initialization for LSTM weights
                for name, param in module.named_parameters():
                    if 'weight' in name:
                        nn.init.xavier_normal_(param)
                    elif 'bias' in name:
                        nn.init.constant_(param, 0)
    
    def forward(self, x):
        """
        Forward pass through the network.
        
        Args:
            x (torch.Tensor): Input state sequences [batch_size, seq_len, state_dim]
            
        Returns:
            torch.Tensor: Q-values for each action [batch_size, action_dim]
        """
        # Pass through LSTM
        # lstm_out: [batch_size, seq_len, lstm_output_size]
        # hidden: final hidden state (we don't use this)
        lstm_out, (hidden, cell) = self.lstm(x)
        
        # Take the last timestep's output (most recent state representation)
        # Shape: [batch_size, lstm_output_size]
        last_output = lstm_out[:, -1, :]
        
        # Pass through fully connected network to get Q-values
        # Shape: [batch_size, action_dim]
        q_values = self.fc_network(last_output)
        
        return q_values
    
    def get_lstm_representation(self, x):
        """
        Get LSTM representation of state sequence (useful for analysis).
        
        Args:
            x (torch.Tensor): Input state sequences [batch_size, seq_len, state_dim]
            
        Returns:
            torch.Tensor: LSTM output [batch_size, lstm_output_size]
        """
        with torch.no_grad():
            lstm_out, _ = self.lstm(x)
            return lstm_out[:, -1, :]


class DoubleDQNAgent:
    """
    Double Deep Q-Network Agent.
    
    Implements the Double DQN algorithm which addresses overestimation bias
    in standard DQN by decoupling action selection from action evaluation.
    
    Key components:
    - Q-Network: Estimates Q-values and selects actions
    - Target Network: Provides stable targets for training
    - Replay Buffer: Stores and samples experiences
    - Epsilon-Greedy: Balances exploration vs exploitation
    
    Training process:
    1. Observe state, select action (ε-greedy)
    2. Execute action, observe reward and next state
    3. Store experience in replay buffer
    4. Sample batch from buffer and update Q-network
    5. Periodically update target network
    """
    
    def __init__(self, config):
        """
        Initialize Double DQN Agent.
        
        Args:
            config (DQNConfig): Configuration object
        """
        self.config = config
        self.device = config.device
        
        # === Initialize Networks ===
        # Q-Network: actively trained
        self.q_network = LSTMQNetwork(config).to(self.device)
        # Target Network: slowly updated, provides stable targets
        self.target_network = LSTMQNetwork(config).to(self.device)
        # Start with same weights
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.target_network.eval()  # Target network is not trained directly
        
        # === Optimizer ===
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=config.learning_rate)
        
        # === Replay Buffer ===
        self.replay_buffer = create_replay_buffer(config)
        
        # === Exploration Parameters ===
        self.epsilon = config.epsilon_start
        self.epsilon_decay = config.epsilon_decay
        self.epsilon_end = config.epsilon_end
        
        # === State Management ===
        # Store recent states for sequence processing
        self.state_buffer = deque(maxlen=config.sequence_length)
        self.sequence_length = config.sequence_length
        
        # === Training Statistics ===
        self.steps = 0
        self.episodes = 0
        self.losses = []
        
        print(f"\n{'='*60}")
        print("DOUBLE DQN AGENT INITIALIZED")
        print(f"{'='*60}")
        print(f"Q-Network Parameters: {sum(p.numel() for p in self.q_network.parameters()):,}")
        print(f"Device: {self.device}")
        print(f"Replay Buffer Type: {'Prioritized' if config.use_prioritized_replay else 'Standard'}")
        print(f"{'='*60}\n")
    
    def reset_state_buffer(self, initial_state):
        """
        Reset the state buffer at the start of a new episode.
        
        We pad with the initial state to create a full sequence.
        
        Args:
            initial_state (np.array): First state of episode [state_dim]
        """
        self.state_buffer.clear()
        # Fill buffer with initial state (creates [seq_len, state_dim])
        for _ in range(self.sequence_length):
            self.state_buffer.append(initial_state)
    
    def add_state_to_buffer(self, state):
        """
        Add new state to buffer (automatically removes oldest if full).
        
        Args:
            state (np.array): New state to add [state_dim]
        """
        self.state_buffer.append(state)
    
    def get_state_sequence(self):
        """
        Get current state sequence from buffer.
        
        Returns:
            np.array: State sequence [seq_len, state_dim]
        """
        return np.array(list(self.state_buffer))
    
    def select_action(self, state_sequence, epsilon=None):
        """
        Select action using epsilon-greedy policy.
        
        With probability ε: random action (exploration)
        With probability 1-ε: action with highest Q-value (exploitation)
        
        Args:
            state_sequence (np.array): State sequence [seq_len, state_dim]
            epsilon (float, optional): Exploration rate. If None, uses self.epsilon
            
        Returns:
            int: Selected action (0, 1, or 2)
        """
        if epsilon is None:
            epsilon = self.epsilon
        
        # Exploration: random action
        if np.random.rand() < epsilon:
            return np.random.randint(self.config.action_dim)
        
        # Exploitation: best action according to Q-network
        with torch.no_grad():
            # Add batch dimension and convert to tensor
            state_tensor = torch.FloatTensor(state_sequence).unsqueeze(0).to(self.device)
            
            # Get Q-values
            q_values = self.q_network(state_tensor)
            
            # Select action with highest Q-value
            action = q_values.argmax(dim=1).item()
            
        return action
    
    def store_experience(self, state_sequence, action, reward, next_state_sequence, done):
        """
        Store experience in replay buffer.
        
        Args:
            state_sequence (np.array): Current state sequence
            action (int): Action taken
            reward (float): Reward received
            next_state_sequence (np.array): Next state sequence
            done (bool): Whether episode ended
        """
        self.replay_buffer.push(state_sequence, action, reward, next_state_sequence, done)
    
    def train_step(self):
        """
        Perform one training step (update Q-network).
        
        This implements the Double DQN update:
        1. Sample batch from replay buffer
        2. Compute target Q-values using target network
        3. Compute current Q-values using Q-network
        4. Calculate loss and update Q-network
        
        Returns:
            float: Loss value (None if buffer too small)
        """
        # Don't train until we have enough experiences
        min_required = max(self.config.min_buffer_size, self.config.batch_size)
        if len(self.replay_buffer) < min_required:
            return None
        
        # === Sample Batch ===
        if self.config.use_prioritized_replay:
            states, actions, rewards, next_states, dones, indices, weights = \
                self.replay_buffer.sample(self.config.batch_size)
        else:
            states, actions, rewards, next_states, dones = \
                self.replay_buffer.sample(self.config.batch_size)
            weights = torch.ones(self.config.batch_size).to(self.device)
            indices = None
        
        # === Compute Current Q-Values ===
        # Q(s_t, a_t) - what we predicted
        current_q_values = self.q_network(states)
        current_q = current_q_values.gather(1, actions.unsqueeze(1)).squeeze(1)
        
        # === Compute Target Q-Values (Double DQN) ===
        with torch.no_grad():
            # Step 1: Use Q-network to SELECT best action for next state
            next_q_values_online = self.q_network(next_states)
            next_actions = next_q_values_online.argmax(dim=1)
            
            # Step 2: Use target network to EVALUATE that action
            next_q_values_target = self.target_network(next_states)
            next_q = next_q_values_target.gather(1, next_actions.unsqueeze(1)).squeeze(1)
            
            # Compute target: r + γ * Q_target(s', argmax Q(s', a'))
            # If done, next_q should be 0 (no future reward)
            target_q = rewards + (1 - dones) * self.config.gamma * next_q
        
        # === Compute Loss ===
        # TD-error for each sample
        td_errors = target_q - current_q
        
        # Weighted MSE loss (weights are 1.0 for standard replay)
        loss = (weights * (td_errors ** 2)).mean()
        
        # === Update Priorities (if using PER) ===
        if self.config.use_prioritized_replay and indices is not None:
            self.replay_buffer.update_priorities(indices, td_errors.detach().cpu().numpy())
        
        # === Backpropagation ===
        self.optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping (prevents exploding gradients)
        if self.config.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), self.config.grad_clip)
        
        self.optimizer.step()
        
        # === Update Statistics ===
        self.steps += 1
        self.losses.append(loss.item())
        
        return loss.item()
    
    def update_target_network(self):
        """
        Update target network with Q-network weights.
        
        Two update strategies:
        1. Hard update: Copy weights directly (default)
        2. Soft update: Slowly blend weights (smoother but slower)
        """
        if self.config.soft_update:
            # Soft update: θ_target = τ*θ_q + (1-τ)*θ_target
            for target_param, q_param in zip(self.target_network.parameters(), 
                                            self.q_network.parameters()):
                target_param.data.copy_(
                    self.config.tau * q_param.data + (1 - self.config.tau) * target_param.data
                )
        else:
            # Hard update: θ_target = θ_q
            self.target_network.load_state_dict(self.q_network.state_dict())
    
    def update_epsilon(self):
        """
        Decay epsilon (reduce exploration over time).
        
        Two decay strategies:
        1. Exponential decay: ε = ε * decay_rate (default)
        2. Linear decay: ε decreases linearly over episodes
        """
        if self.config.use_linear_epsilon_decay:
            # Linear decay
            decay_per_episode = (self.config.epsilon_start - self.config.epsilon_end) / \
                              self.config.epsilon_decay_episodes
            self.epsilon = max(self.epsilon_end, self.epsilon - decay_per_episode)
        else:
            # Exponential decay
            self.epsilon = max(self.config.epsilon_end, self.epsilon * self.config.epsilon_decay)
    
    def update_per_beta(self):
        """
        Anneal beta for prioritized replay (if used).
        
        Beta increases from start to end over training to gradually
        correct for the bias introduced by prioritized sampling.
        """
        if self.config.use_prioritized_replay:
            progress = min(1.0, self.episodes / self.config.per_beta_decay_episodes)
            new_beta = self.config.per_beta_start + progress * \
                      (self.config.per_beta_end - self.config.per_beta_start)
            self.replay_buffer.update_beta(new_beta)
    
    def save_model(self, filepath, episode=None, metrics=None):
        """
        Save model checkpoint.
        
        Saves:
        - Q-network weights
        - Target network weights
        - Optimizer state
        - Training statistics
        - Episode number and metrics (if provided)
        
        Args:
            filepath (str): Path to save checkpoint
            episode (int, optional): Current episode number
            metrics (dict, optional): Additional metrics to save
        """
        checkpoint = {
            'episode': episode,
            'q_network_state_dict': self.q_network.state_dict(),
            'target_network_state_dict': self.target_network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'steps': self.steps,
            'episodes': self.episodes,
            'config': self.config.to_dict(),
            'metrics': metrics
        }
        
        torch.save(checkpoint, filepath)
        print(f"Model saved to: {filepath}")
        
        # Also save metrics as JSON for easy access
        if metrics is not None:
            json_path = filepath.replace('.pth', '_metrics.json')
            with open(json_path, 'w') as f:
                json.dump(metrics, f, indent=2)
    
    def load_model(self, filepath, load_optimizer=True):
        """
        Load model checkpoint.
        
        Args:
            filepath (str): Path to checkpoint
            load_optimizer (bool): Whether to load optimizer state
            
        Returns:
            dict: Checkpoint dictionary with metrics and episode info
        """
        checkpoint = torch.load(filepath, map_location=self.device)
        
        # Load network weights
        self.q_network.load_state_dict(checkpoint['q_network_state_dict'])
        self.target_network.load_state_dict(checkpoint['target_network_state_dict'])
        
        # Load optimizer (optional - may not want this for evaluation)
        if load_optimizer and 'optimizer_state_dict' in checkpoint:
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        # Load training state
        self.epsilon = checkpoint.get('epsilon', self.config.epsilon_end)
        self.steps = checkpoint.get('steps', 0)
        self.episodes = checkpoint.get('episodes', 0)
        
        print(f"Model loaded from: {filepath}")
        print(f"  Episode: {checkpoint.get('episode', 'Unknown')}")
        print(f"  Epsilon: {self.epsilon:.4f}")
        
        return checkpoint
    
    def get_statistics(self):
        """
        Get training statistics.
        
        Returns:
            dict: Statistics including loss, epsilon, buffer size
        """
        stats = {
            'episodes': self.episodes,
            'steps': self.steps,
            'epsilon': self.epsilon,
            'buffer_size': len(self.replay_buffer),
            'avg_loss_last_100': np.mean(self.losses[-100:]) if len(self.losses) > 0 else 0,
        }
        
        if self.config.use_prioritized_replay:
            stats['per_beta'] = self.replay_buffer.beta
        
        return stats
    
    def set_train_mode(self):
        """Set networks to training mode."""
        self.q_network.train()
    
    def set_eval_mode(self):
        """Set networks to evaluation mode (disables dropout, etc.)."""
        self.q_network.eval()


# === Utility Functions ===

# def test_agent_creation():
#     """
#     Test function to verify agent creation works correctly.
#     Run this to validate installation and configuration.
#     """
#     print("Testing DQN Agent Creation...")
    
#     from config import DQNConfig
    
#     # Create config
#     config = DQNConfig()
#     print(config)
    
#     # Create agent
#     agent = DoubleDQNAgent(config)
    
#     # Test forward pass
#     dummy_sequence = np.random.randn(config.sequence_length, config.state_dim).astype(np.float32)
#     action = agent.select_action(dummy_sequence, epsilon=1.0)  # Force random
#     print(f"\nSelected action: {action}")
    
#     # Test storing experience
#     next_sequence = np.random.randn(config.sequence_length, config.state_dim).astype(np.float32)
#     agent.store_experience(dummy_sequence, action, 1.0, next_sequence, False)
#     print(f"Experience stored. Buffer size: {len(agent.replay_buffer)}")
    
#     print("\n✓ Agent creation test passed!")
    
#     return agent


if __name__ == "__main__":
    # Run test when file is executed directly
    print("Double DQN Agent module loaded. To test agent creation, uncomment the test function call.")
    # test_agent_creation()