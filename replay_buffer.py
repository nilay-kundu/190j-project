"""
Experience Replay Buffer for DQN Agent

Implements two types of replay buffers:
1. Standard Uniform Replay Buffer - samples experiences uniformly
2. Prioritized Experience Replay (PER) - samples important experiences more often
"""

import numpy as np
import random
from collections import deque, namedtuple
import torch


# Define the experience tuple structure
Experience = namedtuple('Experience', 
                       ['state_sequence', 'action', 'reward', 'next_state_sequence', 'done'])


class ReplayBuffer:
    """
    Standard uniform experience replay buffer.
    
    This buffer stores agent experiences (state, action, reward, next_state, done)
    and allows random sampling for training. Using a replay buffer:
    - Breaks correlation between consecutive experiences
    - Allows multiple updates from the same experience
    - Improves sample efficiency
    
    Attributes:
        buffer_size (int): Maximum number of experiences to store
        buffer (deque): Circular buffer storing experiences
        device (torch.device): Device to load tensors onto
    """
    
    def __init__(self, buffer_size, device='cpu'):
        """
        Initialize replay buffer.
        
        Args:
            buffer_size (int): Maximum capacity of buffer
            device (str or torch.device): Device for tensor operations
        """
        self.buffer_size = buffer_size
        self.buffer = deque(maxlen=buffer_size)
        self.device = device
        
    def push(self, state_sequence, action, reward, next_state_sequence, done):
        """
        Add a new experience to the buffer.
        
        Args:
            state_sequence (np.array): Sequence of states [seq_len, state_dim]
            action (int): Action taken
            reward (float): Reward received
            next_state_sequence (np.array): Next sequence of states
            done (bool): Whether episode terminated
        """
        experience = Experience(state_sequence, action, reward, next_state_sequence, done)
        self.buffer.append(experience)
    
    def sample(self, batch_size):
        """
        Randomly sample a batch of experiences from buffer.
        
        Args:
            batch_size (int): Number of experiences to sample
            
        Returns:
            tuple: Batched tensors of (states, actions, rewards, next_states, dones)
                - states: [batch_size, seq_len, state_dim]
                - actions: [batch_size]
                - rewards: [batch_size]
                - next_states: [batch_size, seq_len, state_dim]
                - dones: [batch_size]
        """
        # Randomly sample experiences
        experiences = random.sample(self.buffer, batch_size)
        
        # Unpack experiences into separate lists
        state_sequences = [e.state_sequence for e in experiences]
        actions = [e.action for e in experiences]
        rewards = [e.reward for e in experiences]
        next_state_sequences = [e.next_state_sequence for e in experiences]
        dones = [e.done for e in experiences]
        
        # Convert to tensors and move to device
        states = torch.FloatTensor(np.array(state_sequences)).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(np.array(next_state_sequences)).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)
        
        return states, actions, rewards, next_states, dones
    
    def __len__(self):
        """Return current size of buffer."""
        return len(self.buffer)
    
    def clear(self):
        """Clear all experiences from buffer."""
        self.buffer.clear()


class PrioritizedReplayBuffer:
    """
    Prioritized Experience Replay (PER) buffer.
    
    Samples experiences based on their TD-error (how surprising they were).
    Experiences with high TD-error are sampled more frequently, which helps
    the agent learn faster from important experiences.
    
    Key concepts:
    - Priority: Importance of each experience (based on TD-error)
    - Alpha (α): How much to prioritize (0 = uniform, 1 = fully prioritized)
    - Beta (β): Importance sampling correction (0 = no correction, 1 = full correction)
    
    Attributes:
        buffer_size (int): Maximum capacity
        alpha (float): Priority exponent
        beta (float): Importance sampling exponent
        epsilon (float): Small constant to ensure non-zero priorities
        priorities (np.array): Priority values for each experience
        buffer (list): Storage for experiences
        position (int): Current write position in buffer
    """
    
    def __init__(self, buffer_size, alpha=0.6, beta=0.4, epsilon=1e-6, device='cpu'):
        """
        Initialize prioritized replay buffer.
        
        Args:
            buffer_size (int): Maximum capacity of buffer
            alpha (float): Priority exponent (0-1)
            beta (float): Importance sampling exponent (0-1)
            epsilon (float): Small constant to avoid zero priorities
            device (str or torch.device): Device for tensor operations
        """
        self.buffer_size = buffer_size
        self.alpha = alpha
        self.beta = beta
        self.epsilon = epsilon
        self.device = device
        
        # Storage
        self.buffer = []
        self.priorities = np.zeros(buffer_size, dtype=np.float32)
        self.position = 0
        
    def push(self, state_sequence, action, reward, next_state_sequence, done):
        """
        Add a new experience with maximum priority.
        
        New experiences get max priority to ensure they're sampled at least once.
        
        Args:
            state_sequence (np.array): Sequence of states
            action (int): Action taken
            reward (float): Reward received
            next_state_sequence (np.array): Next sequence of states
            done (bool): Whether episode terminated
        """
        # Get maximum priority (or 1.0 for first experience)
        max_priority = self.priorities.max() if len(self.buffer) > 0 else 1.0
        
        # Create experience
        experience = Experience(state_sequence, action, reward, next_state_sequence, done)
        
        # Add to buffer (overwrite old experiences if full)
        if len(self.buffer) < self.buffer_size:
            self.buffer.append(experience)
        else:
            self.buffer[self.position] = experience
        
        # Set priority
        self.priorities[self.position] = max_priority
        
        # Update position (circular)
        self.position = (self.position + 1) % self.buffer_size
    
    def sample(self, batch_size):
        """
        Sample batch of experiences based on priorities.
        
        Returns experiences, their indices, and importance sampling weights.
        
        Args:
            batch_size (int): Number of experiences to sample
            
        Returns:
            tuple: (states, actions, rewards, next_states, dones, indices, weights)
                - states: [batch_size, seq_len, state_dim]
                - actions: [batch_size]
                - rewards: [batch_size]
                - next_states: [batch_size, seq_len, state_dim]
                - dones: [batch_size]
                - indices: [batch_size] - buffer indices for updating priorities
                - weights: [batch_size] - importance sampling weights
        """
        # Get valid priorities (only for filled buffer positions)
        buffer_len = len(self.buffer)
        priorities = self.priorities[:buffer_len]
        
        # Calculate sampling probabilities
        # P(i) = p_i^α / Σ p_j^α
        probs = priorities ** self.alpha
        probs /= probs.sum()
        
        # Sample indices based on probabilities
        indices = np.random.choice(buffer_len, batch_size, p=probs, replace=False)
        
        # Get experiences
        experiences = [self.buffer[idx] for idx in indices]
        
        # Calculate importance sampling weights
        # w_i = (N * P(i))^(-β) / max_j w_j
        total = len(self.buffer)
        weights = (total * probs[indices]) ** (-self.beta)
        weights /= weights.max()  # Normalize by max weight
        
        # Unpack experiences
        state_sequences = [e.state_sequence for e in experiences]
        actions = [e.action for e in experiences]
        rewards = [e.reward for e in experiences]
        next_state_sequences = [e.next_state_sequence for e in experiences]
        dones = [e.done for e in experiences]
        
        # Convert to tensors
        states = torch.FloatTensor(np.array(state_sequences)).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(np.array(next_state_sequences)).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)
        weights = torch.FloatTensor(weights).to(self.device)
        
        return states, actions, rewards, next_states, dones, indices, weights
    
    def update_priorities(self, indices, td_errors):
        """
        Update priorities for sampled experiences based on TD-errors.
        
        Higher TD-error = more surprising = higher priority.
        
        Args:
            indices (np.array): Buffer indices of experiences
            td_errors (np.array or torch.Tensor): TD-errors for each experience
        """
        # Convert to numpy if tensor
        if isinstance(td_errors, torch.Tensor):
            td_errors = td_errors.detach().cpu().numpy()
        
        # Update priorities: p_i = |TD_error_i| + ε
        for idx, td_error in zip(indices, td_errors):
            self.priorities[idx] = abs(td_error) + self.epsilon
    
    def update_beta(self, beta):
        """
        Update beta parameter (typically annealed during training).
        
        Beta should increase from ~0.4 to 1.0 over training to fully correct
        for the bias introduced by prioritized sampling.
        
        Args:
            beta (float): New beta value
        """
        self.beta = beta
    
    def __len__(self):
        """Return current size of buffer."""
        return len(self.buffer)
    
    def clear(self):
        """Clear all experiences from buffer."""
        self.buffer.clear()
        self.priorities = np.zeros(self.buffer_size, dtype=np.float32)
        self.position = 0


def create_replay_buffer(config):
    """
    Factory function to create appropriate replay buffer based on config.
    
    Args:
        config (DQNConfig): Configuration object
        
    Returns:
        ReplayBuffer or PrioritizedReplayBuffer: Initialized replay buffer
    """
    if config.use_prioritized_replay:
        print("Creating Prioritized Experience Replay buffer")
        return PrioritizedReplayBuffer(
            buffer_size=config.buffer_size,
            alpha=config.per_alpha,
            beta=config.per_beta_start,
            epsilon=config.per_epsilon,
            device=config.device
        )
    else:
        print("Creating standard Replay buffer")
        return ReplayBuffer(
            buffer_size=config.buffer_size,
            device=config.device
        )