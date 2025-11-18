"""
Test Script for Branch 2: Training Loop
Validates training infrastructure
"""

import numpy as np
import sys
import os
import shutil
import pickle

from config import DQNConfig, get_config
from dqn_agent import DoubleDQNAgent
from epidemic_simulator import EpidemicSimulator, SCENARIOS
from utils.metrics import MetricsTracker, plot_training_curves, print_summary
from utils.logger import Logger


def test_metrics_tracker():
    """Test metrics tracking functionality."""
    print("\n" + "="*60)
    print("TEST 1: Metrics Tracker")
    print("="*60)
    
    tracker = MetricsTracker()
    
    # Add some episode data
    for i in range(10):
        tracker.add_episode({
            'episode': i,
            'episode_reward': 100 + np.random.randn() * 10,
            'total_deaths': 50 + np.random.randint(-5, 5),
            'total_infections': 200 + np.random.randint(-20, 20),
            'episode_length': 150 + np.random.randint(-10, 10),
            'avg_loss': 0.1 + np.random.rand() * 0.05
        })
    
    print(f"  Added 10 episodes to tracker")
    
    # Add evaluation data
    tracker.add_evaluation(5, {
        'eval_avg_reward': 105.0,
        'eval_avg_deaths': 48.0,
        'eval_avg_infections': 195.0
    })
    
    print(f"  Added evaluation at episode 5")
    
    # Test retrieval
    rewards = tracker.get_metric('episode_reward')
    assert len(rewards) == 10
    print(f"  Retrieved {len(rewards)} reward values")
    
    # Test moving average
    recent_avg = tracker.get_recent_average('episode_reward', window=5)
    print(f"  Recent average reward: {recent_avg:.2f}")
    
    # Test save/load
    os.makedirs('test_logs', exist_ok=True)
    tracker.save('test_logs/test_metrics.json')
    
    new_tracker = MetricsTracker()
    new_tracker.load('test_logs/test_metrics.json')
    
    assert len(new_tracker.get_metric('episode_reward')) == 10
    print(f"  ✓ Save/load works correctly")
    
    # Cleanup
    shutil.rmtree('test_logs')
    
    print("\n✓ Metrics tracker test passed!")


def test_logger():
    """Test logging functionality."""
    print("\n" + "="*60)
    print("TEST 2: Logger")
    print("="*60)
    
    os.makedirs('test_logs', exist_ok=True)
    
    logger = Logger('test_logs', 'test_experiment')
    
    # Test console logging
    logger.log("Test message 1")
    logger.log("Test message 2")
    print("  ✓ Console logging works")
    
    # Test metrics logging
    test_metrics = {
        'episode_reward': 100.5,
        'total_deaths': 50,
        'avg_loss': 0.15,
        'epsilon': 0.9
    }
    logger.log_metrics(test_metrics, step=1)
    print("  ✓ TensorBoard logging works")
    
    # Close logger
    logger.close()
    print("  ✓ Logger closed successfully")
    
    # Cleanup
    shutil.rmtree('test_logs')
    
    print("\n✓ Logger test passed!")


def test_training_episode():
    """Test single training episode."""
    print("\n" + "="*60)
    print("TEST 3: Training Episode")
    print("="*60)
    
    # Create small network for testing
    print("  Creating test network...")
    from network_construction import MultiLayerSimplicialNetwork
    
    network = MultiLayerSimplicialNetwork(N=100, k1=3, k2=1, er_prob=0.05)
    network.build_adjacency_structures()
    
    network_data = {
        'N': network.N,
        'k1': network.k1,
        'k2': network.k2,
        'er_prob': network.er_prob,
        'physical_layer': network.physical_layer,
        'info_layer_graph': network.info_layer_graph,
        'simplices_2': network.simplices_2,
        'matrix_physical': network.matrix_physical,
        'matrix_info': network.matrix_info,
        'adjacency_list_physical': network.adjacency_list_physical,
        'adjacency_list_info': network.adjacency_list_info,
        'adjacency_triangles_list': network.adjacency_triangles_list
    }
    
    # Create agent and simulator
    config = get_config('fast')
    config.min_buffer_size = 10
    config.batch_size = 8
    
    agent = DoubleDQNAgent(config)
    
    params = SCENARIOS['realistic'].copy()
    simulator = EpidemicSimulator(network_data, params)
    
    print("  Running training episode...")
    
    # Import training function
    from train_agent import train_episode
    
    episode_info = train_episode(agent, simulator, epsilon=0.5, max_steps=50)
    
    print(f"\n  Episode Results:")
    print(f"    Reward: {episode_info['episode_reward']:.2f}")
    print(f"    Deaths: {episode_info['total_deaths']}")
    print(f"    Infections: {episode_info['total_infections']}")
    print(f"    Length: {episode_info['episode_length']} days")
    print(f"    Actions: No={episode_info['actions_no_lockdown']}, "
          f"Partial={episode_info['actions_partial']}, Full={episode_info['actions_full']}")
    
    assert episode_info['episode_length'] > 0
    assert episode_info['episode_reward'] != 0
    print("\n✓ Training episode test passed!")


def test_evaluation():
    """Test agent evaluation."""
    print("\n" + "="*60)
    print("TEST 4: Agent Evaluation")
    print("="*60)
    
    # Create small network
    from network_construction import MultiLayerSimplicialNetwork
    
    network = MultiLayerSimplicialNetwork(N=100, k1=3, k2=1, er_prob=0.05)
    network.build_adjacency_structures()
    
    network_data = {
        'N': network.N,
        'k1': network.k1,
        'k2': network.k2,
        'er_prob': network.er_prob,
        'physical_layer': network.physical_layer,
        'info_layer_graph': network.info_layer_graph,
        'simplices_2': network.simplices_2,
        'matrix_physical': network.matrix_physical,
        'matrix_info': network.matrix_info,
        'adjacency_list_physical': network.adjacency_list_physical,
        'adjacency_list_info': network.adjacency_list_info,
        'adjacency_triangles_list': network.adjacency_triangles_list
    }
    
    # Create agent and simulator
    config = get_config('fast')
    agent = DoubleDQNAgent(config)
    
    params = SCENARIOS['realistic'].copy()
    simulator = EpidemicSimulator(network_data, params)
    
    print("  Running evaluation (3 episodes, greedy policy)...")
    
    from train_agent import evaluate_agent
    
    eval_results = evaluate_agent(agent, simulator, num_episodes=3, max_steps=50)
    
    print(f"\n  Evaluation Results:")
    print(f"    Avg Reward: {eval_results['eval_avg_reward']:.2f} "
          f"(±{eval_results['eval_std_reward']:.2f})")
    print(f"    Avg Deaths: {eval_results['eval_avg_deaths']:.1f} "
          f"(±{eval_results['eval_std_deaths']:.1f})")
    print(f"    Avg Infections: {eval_results['eval_avg_infections']:.1f}")
    
    assert 'eval_avg_reward' in eval_results
    assert 'eval_avg_deaths' in eval_results
    print("\n✓ Evaluation test passed!")


def test_full_training():
    """Test complete training loop (very short)."""
    print("\n" + "="*60)
    print("TEST 5: Full Training Loop (5 episodes)")
    print("="*60)
    
    # Create small network
    from network_construction import MultiLayerSimplicialNetwork
    
    print("  Creating test network...")
    network = MultiLayerSimplicialNetwork(N=100, k1=3, k2=1, er_prob=0.05)
    network.build_adjacency_structures()
    
    network_data = {
        'N': network.N,
        'k1': network.k1,
        'k2': network.k2,
        'er_prob': network.er_prob,
        'physical_layer': network.physical_layer,
        'info_layer_graph': network.info_layer_graph,
        'simplices_2': network.simplices_2,
        'matrix_physical': network.matrix_physical,
        'matrix_info': network.matrix_info,
        'adjacency_list_physical': network.adjacency_list_physical,
        'adjacency_list_info': network.adjacency_list_info,
        'adjacency_triangles_list': network.adjacency_triangles_list
    }
    
    # Configure for quick test
    config = get_config('fast')
    config.min_buffer_size = 10
    config.batch_size = 8
    config.target_update_frequency = 2
    config.eval_frequency = 3
    config.eval_episodes = 2
    config.save_frequency = 10  # Won't save during test
    
    agent = DoubleDQNAgent(config)
    
    print("  Running 5 training episodes...")
    
    from train_agent import train_episode, evaluate_agent
    from epidemic_simulator import SCENARIOS
    
    params = SCENARIOS['realistic'].copy()
    simulator = EpidemicSimulator(network_data, params)
    
    # Create metrics tracker
    tracker = MetricsTracker()
    
    for episode in range(5):
        # Train episode
        episode_info = train_episode(agent, simulator, agent.epsilon, max_steps=30)
        episode_info['episode'] = episode
        episode_info['epsilon'] = agent.epsilon
        
        tracker.add_episode(episode_info)
        
        # Update agent
        agent.episodes += 1
        agent.update_epsilon()
        
        if episode % config.target_update_frequency == 0:
            agent.update_target_network()
        
        # Evaluate
        if episode == 3:  # Eval on episode 3
            eval_results = evaluate_agent(agent, simulator, num_episodes=2, max_steps=30)
            tracker.add_evaluation(episode, eval_results)
            print(f"\n  [Episode {episode}] Evaluation: "
                  f"Reward={eval_results['eval_avg_reward']:.2f}, "
                  f"Deaths={eval_results['eval_avg_deaths']:.1f}")
        
        if (episode + 1) % 2 == 0:
            print(f"  Episode {episode + 1}/5 complete: "
                  f"Reward={episode_info['episode_reward']:.2f}, "
                  f"Deaths={episode_info['total_deaths']}")
    
    print(f"\n  Training complete!")
    print(f"    Episodes: {len(tracker.get_metric('episode'))}")
    print(f"    Evaluations: {len(tracker.evaluation_episodes)}")
    
    # Test plotting (save to temp file)
    os.makedirs('test_results', exist_ok=True)
    plot_training_curves(tracker, save_path='test_results/test_training_curves.png')
    
    assert os.path.exists('test_results/test_training_curves.png')
    print("  ✓ Training curves generated")
    
    # Cleanup
    shutil.rmtree('test_results')
    
    print("\n✓ Full training loop test passed!")


def run_all_tests():
    """Run all Branch 2 tests."""
    print("\n" + "="*70)
    print(" "*20 + "BRANCH 2 TEST SUITE")
    print("="*70)
    print("Testing Training Loop Components")
    print("="*70)
    
    try:
        # Test 1: Metrics Tracker
        test_metrics_tracker()
        
        # Test 2: Logger
        test_logger()
        
        # Test 3: Training Episode
        test_training_episode()
        
        # Test 4: Evaluation
        test_evaluation()
        
        # Test 5: Full Training
        test_full_training()
        
        # Success!
        print("\n" + "="*70)
        print(" "*20 + "✓ ALL TESTS PASSED!")
        print("="*70)
        print("\nBranch 2 implementation is ready!")
        print("Next steps:")
        print("  1. Run full training: python train_agent.py --episodes 1000")
        print("  2. Monitor with TensorBoard: tensorboard --logdir logs/")
        print("  3. Evaluate trained model: python train_agent.py --evaluate --model checkpoints/best_model.pth")
        print("="*70 + "\n")
        
        return True
        
    except Exception as e:
        print("\n" + "="*70)
        print(" "*20 + "✗ TEST FAILED!")
        print("="*70)
        print(f"\nError: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)