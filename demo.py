'''
SAMPLE DEMO (VERY SMALL) FOR DOUBLE DQN AGENT IN EPIDEMIC SIMULATION
'''
import pickle
from config import DQNConfig
from dqn_agent import DoubleDQNAgent
from epidemic_simulator import EpidemicSimulator

# Load the pre-built network
with open('networks/multilayer_network.pkl', 'rb') as f:
    network_data = pickle.load(f)

# Define epidemic parameters
params = {
    'beta0': 0.15,          # Base infection rate
    'mu': 0.4,              # E -> I incubation rate
    'gamma_r': 1/21,        # I -> R recovery rate
    'mortality_rate': 0.02, # I -> D death probability
    'gamma_a': 0.3,         # Awareness protection factor
    'lambda_u': 0.12,       # 1-simplex awareness spread
    'lambda_delta': 0.6,    # 2-simplex awareness spread
    'delta': 0.8            # Forgetting rate
}

# Setup
config = DQNConfig()
agent = DoubleDQNAgent(config)
simulator = EpidemicSimulator(network_data, params)

# Episode
stats = simulator.reset()
initial_state = simulator.get_state_vector(stats)
agent.reset_state_buffer(initial_state)

for day in range(365):
    # Agent decides
    state_seq = agent.get_state_sequence()
    action = agent.select_action(state_seq)
    
    # Environment responds
    stats, done = simulator.step(action)
    reward = simulator.get_reward(stats)
    next_state = simulator.get_state_vector(stats)
    
    # Learn
    agent.add_state_to_buffer(next_state)
    next_seq = agent.get_state_sequence()
    agent.store_experience(state_seq, action, reward, next_seq, done)
    
    if len(agent.replay_buffer) >= config.min_buffer_size:
        agent.train_step()
    
    if done:
        break

# Update networks
agent.update_target_network()
agent.update_epsilon()