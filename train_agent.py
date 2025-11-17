"""
Training Loop for DQN Agent on Epidemic Control
Automated training with metrics, logging, and evaluation
"""

import numpy as np
import pickle
import os
import json
import time
import argparse
from datetime import datetime
from tqdm import tqdm
import torch

from config import DQNConfig, get_config
from dqn_agent import DoubleDQNAgent
from epidemic_simulator import EpidemicSimulator, SCENARIOS
from utils.metrics import MetricsTracker, plot_training_curves
from utils.logger import Logger

# ==============================================================================
# TRAINING FUNCTIONS
# ==============================================================================

def train_episode(agent, simulator, epsilon, max_steps=500):
    """
    Run one training episode.
    
    Args:
        agent (DoubleDQNAgent)
        simulator (EpidemicSimulator)
        epsilon (float): Exploration rate
        max_steps (int): Maximum steps per episode
        
    Returns:
        dict: Episode statistics
    """
    # Reset environment
    stats = simulator.reset()
    initial_state = simulator.get_state_vector(stats)
    agent.reset_state_buffer(initial_state)
    
    # Episode tracking
    episode_reward = 0
    episode_losses = []
    actions_taken = {0: 0, 1: 0, 2: 0}  # Count each action
    
    # Run episode
    for step in range(max_steps):
        state_sequence = agent.get_state_sequence()
        
        # Select action
        action = agent.select_action(state_sequence, epsilon)
        actions_taken[action] += 1
        
        # Execute action in environment
        stats, done = simulator.step(action)
        reward = simulator.get_reward(stats)
        next_state = simulator.get_state_vector(stats)
        
        # Update state buffer
        agent.add_state_to_buffer(next_state)
        next_sequence = agent.get_state_sequence()
        agent.store_experience(state_sequence, action, reward, next_sequence, done)
     
        min_required = max(agent.config.min_buffer_size, agent.config.batch_size)
        if len(agent.replay_buffer) >= min_required:
            loss = agent.train_step()
            if loss is not None:
                episode_losses.append(loss)
        
        episode_reward += reward
        
        if done:
            break
    
    # Get final statistics
    final_stats = simulator.history[-1] if simulator.history else stats
    
    # Compile episode results
    episode_info = {
        'episode_reward': episode_reward,
        'episode_length': len(simulator.history),
        'total_deaths': final_stats['D'],
        'total_infections': final_stats['Total_Ever_Infected'],
        'peak_active': max([s['Active'] for s in simulator.history]),
        'final_economy': final_stats['adjusted_economy'],
        'avg_awareness': np.mean([s['rho_A'] for s in simulator.history]),
        'avg_loss': np.mean(episode_losses) if episode_losses else 0,
        'actions_no_lockdown': actions_taken[0],
        'actions_partial': actions_taken[1],
        'actions_full': actions_taken[2],
        'buffer_size': len(agent.replay_buffer)
    }
    return episode_info

def evaluate_agent(agent, simulator, num_episodes=5, max_steps=500):
    """
    Evaluate agent without exploration in completely greedy alg.
    
    Args:
        agent (DoubleDQNAgent)
        simulator (EpidemicSimulator)
        num_episodes (int)
        max_steps (int): Maximum steps per episode
        
    Returns:
        dict: Evaluation statistics
    """
    agent.set_eval_mode()  # Disable dropout, etc.
    
    eval_results = []
    
    for _ in range(num_episodes):
        # Reset
        stats = simulator.reset()
        initial_state = simulator.get_state_vector(stats)
        agent.reset_state_buffer(initial_state)
        
        episode_reward = 0
        
        # Run episode 
        for step in range(max_steps):
            state_sequence = agent.get_state_sequence()
            action = agent.select_action(state_sequence, epsilon=0.0)  # Greedy
            
            stats, done = simulator.step(action)
            reward = simulator.get_reward(stats)
            next_state = simulator.get_state_vector(stats)
            
            agent.add_state_to_buffer(next_state)
            episode_reward += reward
            
            if done:
                break
        
        # Record results
        final_stats = simulator.history[-1]
        eval_results.append({
            'reward': episode_reward,
            'deaths': final_stats['D'],
            'infections': final_stats['Total_Ever_Infected'],
            'peak_active': max([s['Active'] for s in simulator.history]),
            'episode_length': len(simulator.history)
        })
    
    agent.set_train_mode()  # Re-enable training mode
    
    # Aggregate statistics
    avg_metrics = {
        'eval_avg_reward': np.mean([r['reward'] for r in eval_results]),
        'eval_std_reward': np.std([r['reward'] for r in eval_results]),
        'eval_avg_deaths': np.mean([r['deaths'] for r in eval_results]),
        'eval_std_deaths': np.std([r['deaths'] for r in eval_results]),
        'eval_avg_infections': np.mean([r['infections'] for r in eval_results]),
        'eval_avg_peak_active': np.mean([r['peak_active'] for r in eval_results]),
        'eval_avg_length': np.mean([r['episode_length'] for r in eval_results])
    }
    
    return avg_metrics


# ==============================================================================
# CURRICULUM LEARNING 
# ==============================================================================

def get_curriculum_params(episode, total_episodes):
    """
    Increase difficulty during training.
    
    Args:
        episode (int): Current episode number
        total_episodes (int): Total training episodes
        
    Returns:
        dict: Epidemic parameters for this episode
    """
    progress = episode / total_episodes
    
    if progress < 0.3:
        # Easy: Low transmission
        scenario = 'long_duration'
    elif progress < 0.7:
        # Medium: Realistic scenario
        scenario = 'realistic'
    else:
        # Hard: Keep realistic or use 'extreme'
        scenario = 'realistic'
    
    return SCENARIOS[scenario].copy()


# ==============================================================================
# MAIN TRAINING LOOP
# ==============================================================================

def train(agent, network_data, config, args):
    """
    Main training loop.
    
    Args:
        agent (DoubleDQNAgent)
        network_data (dict): Network structure
        config (DQNConfig): Configuration
        args (Namespace): Command-line arguments
    """
    # Initialize logger and metrics tracker
    logger = Logger(config.log_dir, args.experiment_name)
    metrics_tracker = MetricsTracker()
    
    # Create simulator with initial parameters
    initial_params = SCENARIOS[args.scenario].copy()
    simulator = EpidemicSimulator(network_data, initial_params)
    
    # Training settings
    start_episode = 0
    best_reward = -float('inf')
    best_deaths = float('inf')
    
    # Resume from checkpoint if specified
    if args.resume:
        checkpoint = agent.load_model(args.resume)
        start_episode = checkpoint.get('episode', 0)
        best_reward = checkpoint.get('best_reward', best_reward)
        logger.log(f"Resumed from episode {start_episode}")
    
    logger.log("="*60)
    logger.log(f"TRAINING STARTED: {args.experiment_name}")
    logger.log("="*60)
    logger.log(f"Scenario: {args.scenario}")
    logger.log(f"Total Episodes: {args.episodes}")
    logger.log(f"Curriculum Learning: {args.curriculum}")
    logger.log(f"Device: {config.device}")
    logger.log("="*60)
    
    # Training loop
    training_start_time = time.time()
    
    for episode in tqdm(range(start_episode, args.episodes), desc="Training"):
        episode_start_time = time.time()
        
        # Update simulator parameters 
        if args.curriculum:
            current_params = get_curriculum_params(episode, args.episodes)
            simulator = EpidemicSimulator(network_data, current_params)
        
        # Train one episode
        episode_info = train_episode(agent, simulator, agent.epsilon, max_steps=args.max_steps)
        
        # Update agent
        agent.episodes += 1
        agent.update_epsilon()
        
        if args.use_per:
            agent.update_per_beta()
        
        # Update target network
        if episode % config.target_update_frequency == 0:
            agent.update_target_network()
        
        # Record metrics
        episode_time = time.time() - episode_start_time
        episode_info['episode'] = episode
        episode_info['epsilon'] = agent.epsilon
        episode_info['episode_time'] = episode_time
        metrics_tracker.add_episode(episode_info)
        
        # Log to tensorboard
        logger.log_metrics(episode_info, episode)
        
        # Periodic evaluation
        if (episode + 1) % config.eval_frequency == 0:
            eval_metrics = evaluate_agent(agent, simulator, num_episodes=config.eval_episodes)
            metrics_tracker.add_evaluation(episode, eval_metrics)
            logger.log_metrics(eval_metrics, episode)
            
            # Console output
            logger.log(f"\n[Episode {episode+1}/{args.episodes}] Evaluation:")
            logger.log(f"  Avg Reward: {eval_metrics['eval_avg_reward']:.2f} "
                      f"(±{eval_metrics['eval_std_reward']:.2f})")
            logger.log(f"  Avg Deaths: {eval_metrics['eval_avg_deaths']:.1f} "
                      f"(±{eval_metrics['eval_std_deaths']:.1f})")
            logger.log(f"  Avg Infections: {eval_metrics['eval_avg_infections']:.1f}")
            logger.log(f"  Epsilon: {agent.epsilon:.4f}")
            
            # Check for best model
            if eval_metrics['eval_avg_reward'] > best_reward:
                best_reward = eval_metrics['eval_avg_reward']
                best_deaths = eval_metrics['eval_avg_deaths']
                
                # Save best model
                save_path = os.path.join(config.checkpoint_dir, 'best_model.pth')
                agent.save_model(
                    save_path,
                    episode=episode,
                    metrics={
                        'eval_avg_reward': best_reward,
                        'eval_avg_deaths': best_deaths,
                        'episode': episode
                    }
                )
                logger.log(f"  New best model saved! (Reward: {best_reward:.2f})")
        
        # Periodic checkpointing
        if (episode + 1) % config.save_frequency == 0:
            save_path = os.path.join(config.checkpoint_dir, f'model_episode_{episode+1}.pth')
            agent.save_model(
                save_path,
                episode=episode,
                metrics={'episode_reward': episode_info['episode_reward']}
            )
            logger.log(f"[Episode {episode+1}] Checkpoint saved")
        
        # Periodic console output
        if (episode + 1) % 10 == 0:
            recent_rewards = metrics_tracker.get_recent_average('episode_reward', window=10)
            recent_deaths = metrics_tracker.get_recent_average('total_deaths', window=10)
            logger.log(f"[Episode {episode+1}] "
                      f"Reward: {recent_rewards:.2f}, "
                      f"Deaths: {recent_deaths:.1f}, "
                      f"ε: {agent.epsilon:.4f}, "
                      f"Loss: {episode_info['avg_loss']:.4f}")
    
    # Training complete
    training_time = time.time() - training_start_time
    logger.log("\n" + "="*60)
    logger.log("TRAINING COMPLETE")
    logger.log("="*60)
    logger.log(f"Total Time: {training_time/3600:.2f} hours")
    logger.log(f"Best Reward: {best_reward:.2f}")
    logger.log(f"Best Deaths: {best_deaths:.1f}")
    logger.log("="*60)
    
    # Save final model
    final_save_path = os.path.join(config.checkpoint_dir, 'final_model.pth')
    agent.save_model(final_save_path, episode=args.episodes-1)
    logger.log(f"Final model saved to: {final_save_path}")
    
    # Save metrics
    metrics_path = os.path.join(config.log_dir, 'training_metrics.json')
    metrics_tracker.save(metrics_path)
    logger.log(f"Metrics saved to: {metrics_path}")
    
    # Create training plots
    plot_path = os.path.join(config.log_dir, 'training_curves.png')
    plot_training_curves(metrics_tracker, save_path=plot_path)
    logger.log(f"Training curves saved to: {plot_path}")
    
    logger.close()
    
    return agent, metrics_tracker

# ==============================================================================
# COMMAND-LINE INTERFACE
# ==============================================================================

def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description='Train DQN agent for epidemic control')
    
    # Training parameters
    parser.add_argument('--episodes', type=int, default=1000, help='Number of training episodes')
    parser.add_argument('--max-steps', type=int, default=500, help='Maximum steps per episode')
    parser.add_argument('--scenario', type=str, default='realistic', choices=['realistic', 'extreme', 'long_duration'], help='Epidemic scenario')
    
    # Agent configuration
    parser.add_argument('--config', type=str, default='default', choices=['default', 'fast', 'high_performance'], help='Agent configuration preset')
    parser.add_argument('--use-per', action='store_true', help='Use prioritized experience replay')
    
    # Curriculum learning
    parser.add_argument('--curriculum', action='store_true', help='Use curriculum learning (easy to hard)')
    
    # Checkpointing
    parser.add_argument('--resume', type=str, default=None,
                       help='Path to checkpoint to resume from')
    parser.add_argument('--experiment-name', type=str, default=None,
                       help='Name for this training run')
    
    # Evaluation mode
    parser.add_argument('--evaluate', action='store_true',
                       help='Evaluate a trained model instead of training')
    parser.add_argument('--model', type=str, default=None,
                       help='Path to model for evaluation')
    
    # Network
    parser.add_argument('--network-path', type=str, 
                       default='networks/multilayer_network.pkl',
                       help='Path to network data')
    
    return parser.parse_args()


def main():
    """Main entry point."""
    args = parse_args()
    
    # Generate name if not provided
    if args.experiment_name is None:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        args.experiment_name = f"train_{args.scenario}_{timestamp}"
    
    # Load network
    print(f"Loading network from: {args.network_path}")
    with open(args.network_path, 'rb') as f:
        network_data = pickle.load(f)
    print(f"Network loaded: {network_data['N']} nodes\n")
    
    # Create configuration
    config = get_config(args.config)
    if args.use_per:
        config.use_prioritized_replay = True
    
    print(config)
    
    # Create agent
    agent = DoubleDQNAgent(config)
    
    # Evaluation mode
    if args.evaluate:
        if args.model is None:
            print("Error: --model must be specified for evaluation")
            return
        
        print(f"\n{'='*60}")
        print("EVALUATION MODE")
        print(f"{'='*60}")
        
        # Load model
        agent.load_model(args.model)
        
        # Create simulator
        params = SCENARIOS[args.scenario].copy()
        simulator = EpidemicSimulator(network_data, params)
        
        # Run evaluation
        print(f"Running {config.eval_episodes} evaluation episodes...")
        eval_results = evaluate_agent(agent, simulator, num_episodes=config.eval_episodes)
        
        print(f"\n{'='*60}")
        print("EVALUATION RESULTS")
        print(f"{'='*60}")
        print(f"Average Reward: {eval_results['eval_avg_reward']:.2f} "
              f"(±{eval_results['eval_std_reward']:.2f})")
        print(f"Average Deaths: {eval_results['eval_avg_deaths']:.1f} "
              f"(±{eval_results['eval_std_deaths']:.1f})")
        print(f"Average Infections: {eval_results['eval_avg_infections']:.1f}")
        print(f"Average Peak Active: {eval_results['eval_avg_peak_active']:.1f}")
        print(f"Average Episode Length: {eval_results['eval_avg_length']:.1f} days")
        print(f"{'='*60}\n")
        
        return
    
    # Training mode
    train(agent, network_data, config, args)

if __name__ == "__main__":
    main()