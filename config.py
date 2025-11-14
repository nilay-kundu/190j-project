"""
Configuration File for DQN Agent
Centralizes all hyperparameters for easy tuning and experimentation
"""

import torch
import os

class DQNConfig:
    """
    Configuration class containing all hyperparameters for the DQN agent.
    
    This centralized configuration makes it easy to:
    - Experiment with different hyperparameters
    - Save/load complete experiment settings
    - Ensure consistency across training runs
    """
    
    def __init__(self):
        # ============================================================
        # ENVIRONMENT PARAMETERS
        # ============================================================
        self.state_dim = 7  # [active_cases, new_infections, deaths, recoveries, R_eff, rho_A, economy]
        self.action_dim = 3  # [No lockdown (0), Partial lockdown (1), Full lockdown (2)]
        self.max_episode_length = 365  # Maximum days per episode
        
        # ============================================================
        # NETWORK ARCHITECTURE
        # ============================================================
        # LSTM parameters - helps capture temporal dependencies in epidemic dynamics
        self.lstm_hidden_size = 128  # Number of LSTM hidden units
        self.lstm_num_layers = 2  # Number of stacked LSTM layers
        self.lstm_bidirectional = True  # Use bidirectional LSTM for better context
        self.lstm_dropout = 0.2  # Dropout between LSTM layers (prevent overfitting)
        
        # Fully connected layers after LSTM
        self.fc_hidden_sizes = [256, 128]  # Hidden layer sizes for FC network
        self.fc_dropout = 0.3  # Dropout in FC layers
        
        # Sequence processing
        self.sequence_length = 14  # Number of days to look back (2 weeks of history)
        
        # ============================================================
        # DOUBLE DQN PARAMETERS
        # ============================================================
        self.gamma = 0.99  # Discount factor for future rewards (0.99 = value future highly)
        self.learning_rate = 1e-4  # Adam optimizer learning rate
        self.target_update_frequency = 10  # Update target network every N episodes
        self.soft_update = False  # If True, use soft updates (tau); if False, hard copy
        self.tau = 0.005  # Soft update parameter (only used if soft_update=True)
        
        # ============================================================
        # EXPLORATION PARAMETERS (Epsilon-Greedy)
        # ============================================================
        self.epsilon_start = 1.0  # Initial exploration rate (100% random)
        self.epsilon_end = 0.05  # Final exploration rate (5% random)
        self.epsilon_decay = 0.995  # Decay rate per episode
        # Alternative: Linear decay
        self.use_linear_epsilon_decay = False
        self.epsilon_decay_episodes = 500  # Episodes to decay from start to end (if linear)
        
        # ============================================================
        # EXPERIENCE REPLAY BUFFER
        # ============================================================
        self.buffer_size = 50000  # Maximum number of experiences to store
        self.batch_size = 64  # Number of experiences to sample per training step
        self.min_buffer_size = 1000  # Minimum experiences before training starts
        
        # Prioritized Experience Replay (PER) - optional enhancement
        self.use_prioritized_replay = False  # Set True to enable PER
        self.per_alpha = 0.6  # Priority exponent (how much to prioritize)
        self.per_beta_start = 0.4  # Importance sampling exponent (start)
        self.per_beta_end = 1.0  # Importance sampling exponent (end)
        self.per_beta_decay_episodes = 1000  # Episodes to anneal beta
        self.per_epsilon = 1e-6  # Small constant to avoid zero priority
        
        # ============================================================
        # TRAINING PARAMETERS
        # ============================================================
        self.num_episodes = 1000  # Total training episodes
        self.train_frequency = 1  # Train every N steps (1 = train every step)
        self.grad_clip = 1.0  # Gradient clipping threshold (prevent exploding gradients)
        
        # ============================================================
        # DEVICE CONFIGURATION
        # ============================================================
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        
        # ============================================================
        # SAVING AND LOGGING
        # ============================================================
        self.checkpoint_dir = "checkpoints"
        self.log_dir = "logs"
        self.save_frequency = 50  # Save model every N episodes
        self.eval_frequency = 25  # Evaluate agent every N episodes
        self.eval_episodes = 5  # Number of episodes for evaluation
        
        # Create directories if they don't exist
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        os.makedirs(self.log_dir, exist_ok=True)
        
        # ============================================================
        # RANDOM SEEDS (for reproducibility)
        # ============================================================
        self.seed = 42
        
    def to_dict(self):
        """
        Convert configuration to dictionary for saving.
        
        Returns:
            dict: Configuration as a dictionary
        """
        return {k: v for k, v in self.__dict__.items() if not k.startswith('_')}
    
    def __str__(self):
        """
        Pretty print configuration.
        
        Returns:
            str: Formatted configuration string
        """
        config_str = "\n" + "="*60 + "\n"
        config_str += "DQN AGENT CONFIGURATION\n"
        config_str += "="*60 + "\n"
        
        config_str += "\nEnvironment:\n"
        config_str += f"  State Dimension: {self.state_dim}\n"
        config_str += f"  Action Dimension: {self.action_dim}\n"
        config_str += f"  Max Episode Length: {self.max_episode_length}\n"
        
        config_str += "\nLSTM Architecture:\n"
        config_str += f"  Hidden Size: {self.lstm_hidden_size}\n"
        config_str += f"  Num Layers: {self.lstm_num_layers}\n"
        config_str += f"  Bidirectional: {self.lstm_bidirectional}\n"
        config_str += f"  Sequence Length: {self.sequence_length}\n"
        
        config_str += "\nDQN Parameters:\n"
        config_str += f"  Gamma: {self.gamma}\n"
        config_str += f"  Learning Rate: {self.learning_rate}\n"
        config_str += f"  Target Update Freq: {self.target_update_frequency}\n"
        
        config_str += "\nExploration:\n"
        config_str += f"  Epsilon Start: {self.epsilon_start}\n"
        config_str += f"  Epsilon End: {self.epsilon_end}\n"
        config_str += f"  Epsilon Decay: {self.epsilon_decay}\n"
        
        config_str += "\nReplay Buffer:\n"
        config_str += f"  Buffer Size: {self.buffer_size}\n"
        config_str += f"  Batch Size: {self.batch_size}\n"
        config_str += f"  Prioritized Replay: {self.use_prioritized_replay}\n"
        
        config_str += "\nTraining:\n"
        config_str += f"  Num Episodes: {self.num_episodes}\n"
        config_str += f"  Device: {self.device}\n"
        
        config_str += "="*60 + "\n"
        
        return config_str


# Predefined configurations for different scenarios
class FastTrainingConfig(DQNConfig):
    """
    Configuration for fast training / debugging.
    Smaller networks, less episodes, more exploration.
    """
    def __init__(self):
        super().__init__()
        self.lstm_hidden_size = 64
        self.fc_hidden_sizes = [128, 64]
        self.num_episodes = 200
        self.buffer_size = 10000
        self.target_update_frequency = 5


class HighPerformanceConfig(DQNConfig):
    """
    Configuration for high-performance training.
    Larger networks, more episodes, prioritized replay.
    """
    def __init__(self):
        super().__init__()
        self.lstm_hidden_size = 256
        self.lstm_num_layers = 3
        self.fc_hidden_sizes = [512, 256, 128]
        self.num_episodes = 2000
        self.use_prioritized_replay = True
        self.buffer_size = 100000


# Factory function to get configurations
def get_config(config_name="default"):
    """
    Factory function to get different configurations.
    
    Args:
        config_name (str): Name of configuration ('default', 'fast', 'high_performance')
        
    Returns:
        DQNConfig: Configuration object
    """
    configs = {
        "default": DQNConfig,
        "fast": FastTrainingConfig,
        "high_performance": HighPerformanceConfig
    }
    
    if config_name not in configs:
        print(f"Unknown config '{config_name}', using 'default'")
        config_name = "default"
    
    return configs[config_name]()