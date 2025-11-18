"""
Logging Utilities for Training
Supports console output and TensorBoard
"""

import os
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter


class Logger:
    """
    Handles logging to console and TensorBoard.
    
    Features:
    - Console output with timestamps
    - TensorBoard scalar logging
    - Automatic log file creation
    """
    
    def __init__(self, log_dir, experiment_name):
        """
        Initialize logger.
        
        Args:
            log_dir (str): Directory for logs
            experiment_name (str): Name of experiment
        """
        self.log_dir = log_dir
        self.experiment_name = experiment_name
        
        # Create experiment directory
        self.experiment_dir = os.path.join(log_dir, experiment_name)
        os.makedirs(self.experiment_dir, exist_ok=True)
        
        # Create log file
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.log_file = os.path.join(self.experiment_dir, f'training_log_{timestamp}.txt')
        
        # Initialize TensorBoard writer
        tensorboard_dir = os.path.join(self.experiment_dir, 'tensorboard')
        self.writer = SummaryWriter(tensorboard_dir)
        
        self.log(f"Logger initialized: {experiment_name}")
        self.log(f"Log file: {self.log_file}")
        self.log(f"TensorBoard logs: {tensorboard_dir}")
        self.log(f"View with: tensorboard --logdir {tensorboard_dir}")
    
    def log(self, message):
        """
        Log message to console and file.
        
        Args:
            message (str): Message to log
        """
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        log_line = f"[{timestamp}] {message}"
        
        # Print to console
        print(log_line)
        
        # Write to file
        with open(self.log_file, 'a') as f:
            f.write(log_line + '\n')
    
    def log_metrics(self, metrics_dict, step):
        """
        Log metrics to TensorBoard.
        
        Args:
            metrics_dict (dict): Dictionary of metric_name: value
            step (int): Global step (usually episode number)
        """
        for metric_name, value in metrics_dict.items():
            # Skip non-numeric values
            if not isinstance(value, (int, float)):
                continue
            
            # Organize metrics into groups for TensorBoard
            if 'eval' in metric_name:
                group = 'Evaluation'
            elif 'loss' in metric_name:
                group = 'Training'
            elif 'action' in metric_name:
                group = 'Actions'
            elif metric_name in ['episode_reward', 'episode_length']:
                group = 'Episode'
            elif metric_name in ['total_deaths', 'total_infections', 'peak_active']:
                group = 'Epidemic'
            elif metric_name in ['epsilon', 'buffer_size']:
                group = 'Agent'
            else:
                group = 'Other'
            
            # Log to TensorBoard with group prefix
            self.writer.add_scalar(f'{group}/{metric_name}', value, step)
    
    def log_histogram(self, tag, values, step):
        """
        Log histogram to TensorBoard.
        
        Args:
            tag (str): Name of histogram
            values (array-like): Values to plot
            step (int): Global step
        """
        self.writer.add_histogram(tag, values, step)
    
    def close(self):
        """Close TensorBoard writer."""
        self.writer.close()
        self.log("Logger closed")


class ConsoleFormatter:
    """
    Utilities for formatted console output.
    """
    
    @staticmethod
    def progress_bar(current, total, bar_length=40, prefix='', suffix=''):
        """
        Create a text progress bar.
        
        Args:
            current (int): Current progress
            total (int): Total items
            bar_length (int): Length of progress bar
            prefix (str): Text before bar
            suffix (str): Text after bar
            
        Returns:
            str: Formatted progress bar
        """
        percent = 100 * (current / float(total))
        filled_length = int(bar_length * current // total)
        bar = '█' * filled_length + '-' * (bar_length - filled_length)
        
        return f'\r{prefix} |{bar}| {percent:.1f}% {suffix}'
    
    @staticmethod
    def format_time(seconds):
        """
        Format seconds into human-readable time.
        
        Args:
            seconds (float): Time in seconds
            
        Returns:
            str: Formatted time string
        """
        if seconds < 60:
            return f"{seconds:.1f}s"
        elif seconds < 3600:
            minutes = seconds / 60
            return f"{minutes:.1f}m"
        else:
            hours = seconds / 3600
            return f"{hours:.2f}h"
    
    @staticmethod
    def colored_metric(value, threshold, higher_is_better=True):
        """
        Add color to metric based on threshold (for terminals that support ANSI).
        
        Args:
            value (float): Metric value
            threshold (float): Threshold for color change
            higher_is_better (bool): Whether higher values are better
            
        Returns:
            str: Colored string
        """
        # ANSI color codes
        GREEN = '\033[92m'
        RED = '\033[91m'
        YELLOW = '\033[93m'
        RESET = '\033[0m'
        
        if higher_is_better:
            if value >= threshold:
                color = GREEN
            elif value >= threshold * 0.8:
                color = YELLOW
            else:
                color = RED
        else:
            if value <= threshold:
                color = GREEN
            elif value <= threshold * 1.2:
                color = YELLOW
            else:
                color = RED
        
        return f"{color}{value:.2f}{RESET}"


def setup_logging(log_dir, experiment_name):
    """
    Setup logging infrastructure.
    
    Args:
        log_dir (str): Base directory for logs
        experiment_name (str): Name of experiment
        
    Returns:
        Logger: Configured logger instance
    """
    # Create directories
    os.makedirs(log_dir, exist_ok=True)
    
    # Create logger
    logger = Logger(log_dir, experiment_name)
    
    return logger