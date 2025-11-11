"""
Test Script for DQN Agent Implementation
Tests all components to ensure they work correctly together before integrating with actual simulator
"""

import numpy as np
import torch
import sys

from config import DQNConfig, get_config
from replay_buffer import ReplayBuffer, PrioritizedReplayBuffer, create_replay_buffer
from dqn_agent import DoubleDQNAgent, LSTMQNetwork


def test_config():
    """Test configuration system."""
    print("\n" + "="*60)
    print("TEST 1: Configuration System")
    print("="*60)
    
    # Test default config
    config = DQNConfig()
    print(config)
    
    # Test config dictionary conversion
    config_dict = config.to_dict()
    assert 'state_dim' in config_dict
    assert 'action_dim' in config_dict
    print("✓ Config dictionary conversion works")
    
    # Test config factory
    fast_config = get_config('fast')
    assert fast_config.num_episodes == 200
    print("✓ Config factory works")
    
    print("\n✓ All configuration tests passed!")
    return config


def test_replay_buffer(config):
    """Test replay buffer functionality."""
    print("\n" + "="*60)
    print("TEST 2: Replay Buffer")
    print("="*60)
    
    # Test standard buffer
    print("\nTesting Standard Replay Buffer...")
    buffer = ReplayBuffer(buffer_size=100, device=config.device)
    
    # Add experiences
    for i in range(50):
        state_seq = np.random.randn(config.sequence_length, config.state_dim).astype(np.float32)
        next_state_seq = np.random.randn(config.sequence_length, config.state_dim).astype(np.float32)
        buffer.push(state_seq, action=i % 3, reward=1.0, next_state_sequence=next_state_seq, done=False)
    
    print(f"  Added 50 experiences. Buffer size: {len(buffer)}")
    assert len(buffer) == 50
    
    # Test sampling
    states, actions, rewards, next_states, dones = buffer.sample(batch_size=16)
    assert states.shape == (16, config.sequence_length, config.state_dim)
    assert actions.shape == (16,)
    print(f"  Sampled batch. States shape: {states.shape}")
    print("✓ Standard replay buffer works")
    
    # Test prioritized buffer
    print("\nTesting Prioritized Replay Buffer...")
    per_buffer = PrioritizedReplayBuffer(buffer_size=100, device=config.device)
    
    # Add experiences
    for i in range(50):
        state_seq = np.random.randn(config.sequence_length, config.state_dim).astype(np.float32)
        next_state_seq = np.random.randn(config.sequence_length, config.state_dim).astype(np.float32)
        per_buffer.push(state_seq, action=i % 3, reward=1.0, next_state_sequence=next_state_seq, done=False)
    
    # Test sampling
    states, actions, rewards, next_states, dones, indices, weights = per_buffer.sample(batch_size=16)
    assert weights.shape == (16,)
    print(f"  Sampled batch. Weights shape: {weights.shape}")
    
    # Test priority update
    td_errors = np.random.randn(16)
    per_buffer.update_priorities(indices, td_errors)
    print("✓ Prioritized replay buffer works")
    
    print("\n✓ All replay buffer tests passed!")


def test_network_architecture(config):
    """Test LSTM Q-Network architecture."""
    print("\n" + "="*60)
    print("TEST 3: LSTM Q-Network Architecture")
    print("="*60)
    
    # Create network
    network = LSTMQNetwork(config).to(config.device)
    
    # Count parameters
    num_params = sum(p.numel() for p in network.parameters())
    print(f"\nNetwork created with {num_params:,} parameters")
    
    # Test forward pass
    batch_size = 8
    dummy_input = torch.randn(batch_size, config.sequence_length, config.state_dim).to(config.device)
    
    q_values = network(dummy_input)
    assert q_values.shape == (batch_size, config.action_dim)
    print(f"Forward pass successful. Output shape: {q_values.shape}")
    
    # Test LSTM representation extraction
    representation = network.get_lstm_representation(dummy_input)
    expected_size = config.lstm_hidden_size * (2 if config.lstm_bidirectional else 1)
    assert representation.shape == (batch_size, expected_size)
    print(f"LSTM representation shape: {representation.shape}")
    
    print("\n✓ Network architecture tests passed!")


def test_agent(config):
    """Test DQN agent functionality."""
    print("\n" + "="*60)
    print("TEST 4: Double DQN Agent")
    print("="*60)
    
    # Create agent
    agent = DoubleDQNAgent(config)
    
    # Test state buffer management
    print("\nTesting state buffer...")
    initial_state = np.random.randn(config.state_dim).astype(np.float32)
    agent.reset_state_buffer(initial_state)
    assert len(agent.state_buffer) == config.sequence_length
    print(f"  State buffer initialized with length: {len(agent.state_buffer)}")
    
    # Test action selection
    print("\nTesting action selection...")
    state_sequence = agent.get_state_sequence()
    
    # Test exploration (random action)
    random_action = agent.select_action(state_sequence, epsilon=1.0)
    assert 0 <= random_action < config.action_dim
    print(f"  Random action (ε=1.0): {random_action}")
    
    # Test exploitation (best action)
    best_action = agent.select_action(state_sequence, epsilon=0.0)
    assert 0 <= best_action < config.action_dim
    print(f"  Best action (ε=0.0): {best_action}")
    
    # Test experience storage
    print("\nTesting experience storage...")
    next_state = np.random.randn(config.state_dim).astype(np.float32)
    agent.add_state_to_buffer(next_state)
    next_sequence = agent.get_state_sequence()
    
    agent.store_experience(state_sequence, random_action, 1.0, next_sequence, False)
    print(f"  Experience stored. Buffer size: {len(agent.replay_buffer)}")
    
    # Fill buffer with more experiences
    print("\nFilling replay buffer...")
    for i in range(config.min_buffer_size):
        state_seq = np.random.randn(config.sequence_length, config.state_dim).astype(np.float32)
        next_seq = np.random.randn(config.sequence_length, config.state_dim).astype(np.float32)
        agent.store_experience(state_seq, i % 3, np.random.randn(), next_seq, i % 100 == 0)
    
    print(f"  Buffer filled. Size: {len(agent.replay_buffer)}/{config.buffer_size}")
    
    # Test training step
    print("\nTesting training step...")
    loss = agent.train_step()
    assert loss is not None
    print(f"  Training step executed. Loss: {loss:.4f}")
    
    # Test target network update
    print("\nTesting target network update...")
    old_params = list(agent.target_network.parameters())[0].clone()
    agent.update_target_network()
    new_params = list(agent.target_network.parameters())[0]
    params_changed = not torch.equal(old_params, new_params)
    print(f"  Target network updated: {params_changed}")
    
    # Test epsilon decay
    print("\nTesting epsilon decay...")
    old_epsilon = agent.epsilon
    agent.update_epsilon()
    print(f"  Epsilon: {old_epsilon:.4f} → {agent.epsilon:.4f}")
    
    # Test statistics
    print("\nTesting statistics...")
    stats = agent.get_statistics()
    print(f"  Statistics: {stats}")
    
    print("\n✓ All agent tests passed!")
    return agent


def test_save_load(agent, config):
    """Test model saving and loading."""
    print("\n" + "="*60)
    print("TEST 5: Model Save/Load")
    print("="*60)
    
    # Save model
    import os
    os.makedirs('test_checkpoints', exist_ok=True)
    save_path = 'test_checkpoints/test_model.pth'
    
    metrics = {
        'test_reward': 100.0,
        'test_deaths': 50,
        'test_episode': 10
    }
    
    agent.save_model(save_path, episode=10, metrics=metrics)
    print(f"✓ Model saved to: {save_path}")
    
    # Create new agent and load
    new_agent = DoubleDQNAgent(config)
    checkpoint = new_agent.load_model(save_path)
    
    assert checkpoint['episode'] == 10
    assert checkpoint['metrics']['test_reward'] == 100.0
    print("✓ Model loaded successfully")
    
    # Verify weights match
    old_param = list(agent.q_network.parameters())[0]
    new_param = list(new_agent.q_network.parameters())[0]
    assert torch.equal(old_param, new_param)
    print("✓ Network weights match")
    
    # Clean up
    import shutil
    if os.path.exists('test_checkpoints'):
        shutil.rmtree('test_checkpoints')
    
    print("\n✓ Save/load tests passed!")


def test_integration():
    """Integration test: Simulate a few training steps."""
    print("\n" + "="*60)
    print("TEST 6: Integration Test")
    print("="*60)
    
    # Use fast config for quick testing
    config = get_config('fast')
    config.min_buffer_size = 32  # Lower for testing
    config.batch_size = 16  # Smaller batch size for testing
    config.num_episodes = 3
    
    agent = DoubleDQNAgent(config)
    
    print("\nSimulating 3 training episodes...")
    
    for episode in range(3):
        # Reset for new episode
        initial_state = np.random.randn(config.state_dim).astype(np.float32)
        agent.reset_state_buffer(initial_state)
        
        episode_reward = 0
        
        # Simulate episode
        for step in range(20):  # Short episodes for testing
            # Get current state sequence
            state_sequence = agent.get_state_sequence()
            
            # Select action
            action = agent.select_action(state_sequence)
            
            # Simulate environment step
            reward = np.random.randn()
            next_state = np.random.randn(config.state_dim).astype(np.float32)
            done = (step == 19)
            
            # Add next state to buffer
            agent.add_state_to_buffer(next_state)
            next_sequence = agent.get_state_sequence()
            
            # Store experience
            agent.store_experience(state_sequence, action, reward, next_sequence, done)
            
            # Train (only if we have enough experiences for a full batch)
            if len(agent.replay_buffer) >= max(config.min_buffer_size, config.batch_size):
                loss = agent.train_step()
            
            episode_reward += reward
        
        # Update target network
        if episode % config.target_update_frequency == 0:
            agent.update_target_network()
        
        # Decay epsilon
        agent.update_epsilon()
        agent.episodes += 1
        
        stats = agent.get_statistics()
        print(f"  Episode {episode+1}: Reward={episode_reward:.2f}, "
              f"ε={stats['epsilon']:.4f}, Buffer={stats['buffer_size']}")
    
    print("\n✓ Integration test passed!")


def run_all_tests():
    """Run all tests."""
    print("\n" + "="*70)
    print(" "*15 + "BRANCH 1 TEST SUITE")
    print("="*70)
    print("Testing DQN Agent Implementation Components")
    print("="*70)
    
    try:
        # Test 1: Configuration
        config = test_config()
        
        # Test 2: Replay Buffer
        test_replay_buffer(config)
        
        # Test 3: Network Architecture
        test_network_architecture(config)
        
        # Test 4: Agent
        agent = test_agent(config)
        
        # Test 5: Save/Load
        test_save_load(agent, config)
        
        # Test 6: Integration
        test_integration()
        
        # Success!
        print("\n" + "="*70)
        print(" "*20 + "✓ ALL TESTS PASSED!")
        print("="*70)
        print("\nBranch 1 implementation is ready for integration!")
        print("Next steps:")
        print("  1. Integrate with epidemic_simulator.py")
        print("  2. Create Monte Carlo training loop")
        print("  3. Test on actual epidemic scenarios")
        print("="*70 + "\n")
        
        return True
        
    except Exception as e:
        print("\n" + "="*70)
        print(" "*20 + "TEST FAILED!")
        print("="*70)
        print(f"\nError: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)