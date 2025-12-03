"""
Epidemic Simulator - SEIRD + UAU on Multilayer Simplicial Network
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pickle

sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (15, 10)


# ==============================================================================
# PARAMETERS
# ==============================================================================

SCENARIOS = {
    'extreme': {
        'beta0': 1.0, 'mu': 0.1, 'gamma_r': 1/40, 'mortality_rate': 0.5,
        'gamma_a': 0.0, 'lambda_u': 0.1, 'lambda_delta': 0.25, 'delta': 0.7,
        'waning_rate': 0.00725, 'lockdown_action': 0
    },
    'realistic': {
        'beta0': 0.10, 'mu': 0.2, 'gamma_r': 1/10, 'mortality_rate': 0.02,
        'gamma_a': 0.3, 'lambda_u': 0.2, 'lambda_delta': 0.4, 'delta': 0.6,
        'waning_rate': 0.00725, 'lockdown_action': 0
    },
    'long_duration': {
        'beta0': 0.04, 'mu': 0.1, 'gamma_r': 1/14, 'mortality_rate': 0.0165,
        'gamma_a': 0.6, 'lambda_u': 0.15, 'lambda_delta': 0.25, 'delta': 0.45,
        'waning_rate': 0.00725, 'lockdown_action': 2
    }
}

LOCKDOWN_MULTIPLIERS = {0: 1.0, 1: 0.65, 2: 0.4}
ECONOMY_LEVELS = {0: 1.0, 1: 0.65, 2: 0.4}


# ==============================================================================
# NODE STATE
# ==============================================================================

class NodeState:
    def __init__(self, N):
        self.N = N
        self.disease_states = np.full(N, 'S', dtype='<U1')
        self.awareness_states = np.full(N, 'U', dtype='<U1')
        
        # Initialize 1% aware
        initial_aware = np.random.choice(N, max(1, int(N * 0.01)), replace=False)
        self.awareness_states[initial_aware] = 'A'
        
        # Seed 15 exposed
        unaware = np.where(self.awareness_states == 'U')[0]
        seeds = np.random.choice(unaware, min(15, len(unaware)), replace=False)
        self.disease_states[seeds] = 'E'
        
        self.incubation_timer = np.zeros(N, dtype=int)
        self.infection_timer = np.zeros(N, dtype=int)
        
        for node in seeds:
            self.incubation_timer[node] = np.random.randint(1, 6)
        
        self.update_counts()
    
    def update_counts(self):
        self.counts = {
            'S': np.sum(self.disease_states == 'S'),
            'E': np.sum(self.disease_states == 'E'),
            'I': np.sum(self.disease_states == 'I'),
            'R': np.sum(self.disease_states == 'R'),
            'D': np.sum(self.disease_states == 'D'),
            'U': np.sum(self.awareness_states == 'U'),
            'A': np.sum(self.awareness_states == 'A'),
        }
        
        self.counts['AS'] = np.sum((self.awareness_states == 'A') & (self.disease_states == 'S'))
        self.counts['US'] = np.sum((self.awareness_states == 'U') & (self.disease_states == 'S'))
        self.counts['AI'] = np.sum((self.awareness_states == 'A') & (self.disease_states == 'I'))
        self.counts['UI'] = np.sum((self.awareness_states == 'U') & (self.disease_states == 'I'))
        self.counts['Active'] = self.counts['I']
        self.counts['Aware_Density'] = self.counts['A'] / self.N


# ==============================================================================
# EPIDEMIC SIMULATOR
# ==============================================================================

class EpidemicSimulator:
    def __init__(self, network_data, params):
        self.N = network_data['N']
        self.network_data = network_data
        
        # Disease parameters
        self.beta0 = params['beta0']
        self.mu = params['mu']
        self.gamma_r = params['gamma_r']
        self.mortality_rate = params['mortality_rate']
        self.gamma_a = params['gamma_a']
        self.waning_rate = params['waning_rate']
        
        # Awareness parameters
        self.lambda_u = params['lambda_u']
        self.lambda_delta = params['lambda_delta']
        self.delta = params['delta']
        
        # Policy parameters
        self.lockdown_multipliers = LOCKDOWN_MULTIPLIERS
        self.economy_levels = ECONOMY_LEVELS
        self.current_lockdown_level = 0
        
        # Network structure
        self.adj_phys = network_data['adjacency_list_physical']
        self.adj_info = network_data['adjacency_list_info']
        self.adj_triangles = network_data['adjacency_triangles_list']
        
        # Calculate λ*
        total_triangles = len(network_data.get('simplices_2', []))
        k2_avg = 3 * total_triangles / self.N
        self.lambda_star = self.lambda_delta * self.delta / k2_avg if k2_avg > 0 else 0
        
        # State tracking
        self.state = NodeState(self.N)
        self.day = 0
        self.history = []
        self.cumulative_deaths = 0
        self.cumulative_infections = 0
    
    def _update_waning_immunity(self):
        R_nodes = np.where(self.state.disease_states == 'R')[0]
        for i in R_nodes:
            if np.random.rand() < self.waning_rate:
                self.state.disease_states[i] = 'S'
    
    def _update_disease_spread(self, transmission_multiplier):
        new_infections = 0
        
        # E → I
        E_nodes = np.where(self.state.disease_states == 'E')[0]
        for i in E_nodes:
            self.state.incubation_timer[i] -= 1
            if self.state.incubation_timer[i] <= 0:
                self.state.disease_states[i] = 'I'
                avg_recovery_time = 1 / self.gamma_r
                self.state.infection_timer[i] = max(1, np.random.randint(
                    int(avg_recovery_time * 0.7), int(avg_recovery_time * 1.3) + 1))
                if self.state.awareness_states[i] != 'R':
                    self.state.awareness_states[i] = 'A'
        
        # S → E
        S_nodes = np.where(self.state.disease_states == 'S')[0]
        beta_U = self.beta0 * transmission_multiplier
        beta_A = self.gamma_a * beta_U
        
        for i in S_nodes:
            I_neighbors = [j for j in self.adj_phys[i] if self.state.disease_states[j] == 'I']
            if not I_neighbors:
                continue
            
            beta_eff = beta_A if self.state.awareness_states[i] == 'A' else beta_U
            prob_infection = min(1 - (1 - beta_eff) ** len(I_neighbors), 0.9)
            
            if np.random.rand() < prob_infection:
                self.state.disease_states[i] = 'E'
                self.state.incubation_timer[i] = np.random.randint(1, 6)
                new_infections += 1
        
        self.cumulative_infections += new_infections
        return new_infections
    
    def _update_mortality(self):
        I_nodes = np.where(self.state.disease_states == 'I')[0]
        if len(I_nodes) == 0:
            return 0, 0
        
        deaths_today = recoveries_today = 0
        for i in I_nodes:
            self.state.infection_timer[i] -= 1
            if self.state.infection_timer[i] <= 0:
                if np.random.rand() < self.mortality_rate:
                    self.state.disease_states[i] = 'D'
                    self.state.awareness_states[i] = 'R'
                    deaths_today += 1
                else:
                    self.state.disease_states[i] = 'R'
                    recoveries_today += 1
                self.state.infection_timer[i] = 0
        
        self.cumulative_deaths += deaths_today
        return deaths_today, recoveries_today
    
    def _update_awareness_spread(self):
        # A → U (Forgetting)
        A_nodes = np.where(self.state.awareness_states == 'A')[0]
        for i in A_nodes:
            if self.state.disease_states[i] != 'D' and np.random.rand() < self.delta:
                self.state.awareness_states[i] = 'U'
        
        # U → A (Awareness spread)
        U_nodes = np.where(self.state.awareness_states == 'U')[0]
        for i in U_nodes:
            if self.state.disease_states[i] == 'D':
                continue
            
            became_aware = False
            
            # 1-Simplex
            A_neighbors = [j for j in self.adj_info[i] if self.state.awareness_states[j] == 'A']
            if A_neighbors:
                prob_not_informed = (1 - self.lambda_u) ** len(A_neighbors)
                if np.random.rand() > prob_not_informed:
                    became_aware = True
            
            # 2-Simplex
            if not became_aware and self.lambda_star > 0:
                for j, k in self.adj_triangles[i]:
                    if (self.state.awareness_states[j] == 'A' and 
                        self.state.awareness_states[k] == 'A'):
                        if np.random.rand() < self.lambda_star:
                            became_aware = True
                            break
            
            if became_aware:
                self.state.awareness_states[i] = 'A'
    
    def step(self, action):
        self.day += 1
        self.current_lockdown_level = action
        transmission_multiplier = self.lockdown_multipliers[action]
        
        self._update_waning_immunity()
        self._update_awareness_spread()
        new_infections = self._update_disease_spread(transmission_multiplier)
        deaths_today, recoveries_today = self._update_mortality()
        self.state.update_counts()
        
        # Calculate R_eff
        S_count = self.state.counts['S']
        if S_count > 0:
            AS_count = self.state.counts['AS']
            US_count = self.state.counts['US']
            beta_avg = ((AS_count * self.gamma_a * self.beta0 * transmission_multiplier) + 
                       (US_count * self.beta0 * transmission_multiplier)) / S_count
            R_eff = beta_avg * (S_count / self.N) / self.gamma_r
        else:
            beta_avg = R_eff = 0
        
        # Calculate economy
        lockdown_capacity = self.economy_levels[action]
        workforce_available = self.N - self.state.counts['D'] - self.state.counts['I']
        adjusted_economy = lockdown_capacity * workforce_available / self.N
        
        stats = {
            'day': self.day, 'action': action, 'lockdown_level': action,
            'S': self.state.counts['S'], 'E': self.state.counts['E'],
            'I': self.state.counts['I'], 'R': self.state.counts['R'],
            'D': self.state.counts['D'], 'Active': self.state.counts['I'],
            'Total_Ever_Infected': (self.state.counts['E'] + self.state.counts['I'] + 
                                   self.state.counts['R'] + self.state.counts['D']),
            'new_infections': new_infections, 'new_deaths': deaths_today,
            'new_recoveries': recoveries_today, 'A': self.state.counts['A'],
            'U': self.state.counts['U'], 'rho_A': self.state.counts['Aware_Density'],
            'AS': self.state.counts['AS'], 'US': self.state.counts['US'],
            'AI': self.state.counts['AI'], 'R_eff': R_eff, 'beta_avg': beta_avg,
            'base_economy': lockdown_capacity, 'adjusted_economy': adjusted_economy,
            'workforce_available': workforce_available,
            'transmission_multiplier': transmission_multiplier,
            'cumulative_deaths': self.cumulative_deaths,
            'cumulative_infections': self.cumulative_infections
        }
        
        self.history.append(stats)
        done = (self.state.counts['I'] == 0 and self.state.counts['E'] == 0)
        return stats, done
    
    def reset(self):
        self.state = NodeState(self.N)
        self.day = 0
        self.history = []
        self.cumulative_deaths = 0
        self.cumulative_infections = 0
        self.current_lockdown_level = 0
        initial_stats, _ = self.step(action=0)
        return initial_stats
    
    def get_state_vector(self, stats):
        """
        Extract normalized state vector for RL agent.
        
        Args:
            stats (dict): Current statistics dictionary
            
        Returns:
            np.array: Normalized state vector [7 features]
                [active_cases, new_infections, deaths, recoveries, R_eff, rho_A, economy]
        """
        state_vec = np.array([
            stats['Active'] / self.N,                    # Active cases (normalized)
            stats['new_infections'] / self.N,            # New infections (normalized)
            stats['D'] / self.N,                         # Deaths (normalized)
            stats['R'] / self.N,                         # Recoveries (normalized)
            min(stats['R_eff'] / 5.0, 1.0),             # R_eff (capped at 5)
            stats['rho_A'],                              # Awareness density (already 0-1)
            stats['adjusted_economy']                     # Economy (already 0-1)
        ], dtype=np.float32)
        
        return state_vec
    
    def get_reward(self, stats):
        """
        Calculate RL reward (Ohi et al. structure).
        
        Formula: R(t) = Economy(t) × exp(-8 × Active(t)/N) - 5 × Deaths(t)
        
        Args:
            stats (dict): Current statistics dictionary
            
        Returns:
            float: Reward value
        """
        Active_norm = stats['Active'] / self.N
        Deaths_today = stats['new_deaths']
        Economy_t = stats['adjusted_economy']
        
        reward = Economy_t * np.exp(-8 * Active_norm) - 5 * Deaths_today
        
        return reward
    
    def get_history_df(self):
        return pd.DataFrame(self.history)


# ==============================================================================
# VISUALIZATION
# ==============================================================================

def visualize_simulation(df, save_path):
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Epidemic Simulation Results', fontsize=16, fontweight='bold')
    days = df['day']
    
    # 1. SEIRD Model Dynamics
    axes[0,0].plot(days, df['S'], label='Susceptible', color='blue', linewidth=2)
    axes[0,0].plot(days, df['E'], label='Exposed', color='orange', linewidth=2)
    axes[0,0].plot(days, df['I'], label='Infectious', color='red', linewidth=2)
    axes[0,0].plot(days, df['R'], label='Recovered', color='green', linewidth=2)
    axes[0,0].plot(days, df['D'], label='Dead', color='black', linewidth=2)
    axes[0,0].set_xlabel('Day')
    axes[0,0].set_ylabel('Population')
    axes[0,0].set_title('SEIRD Model Dynamics', fontweight='bold')
    axes[0,0].legend()
    axes[0,0].grid(True, alpha=0.3)
    
    # 2. Active Cases Over Time
    axes[0,1].fill_between(days, df['Active'], alpha=0.3, color='red')
    axes[0,1].plot(days, df['Active'], color='red', linewidth=2)
    axes[0,1].axhline(df['Active'].max(), color='red', linestyle='--', 
                      label=f'Peak: {df["Active"].max()}')
    axes[0,1].set_xlabel('Day')
    axes[0,1].set_ylabel('Active Infectious Cases')
    axes[0,1].set_title('Active Cases (I) Over Time', fontweight='bold')
    axes[0,1].legend()
    axes[0,1].grid(True, alpha=0.3)
    
    # 3. Daily New Infections
    axes[0,2].bar(days, df['new_infections'], alpha=0.6, color='orange', edgecolor='black')
    axes[0,2].plot(days, df['new_infections'].rolling(7).mean(), 
                   color='red', linewidth=2, label='7-day avg')
    axes[0,2].set_xlabel('Day')
    axes[0,2].set_ylabel('New Infections (S→E)')
    axes[0,2].set_title('Daily New Infections', fontweight='bold')
    axes[0,2].legend()
    axes[0,2].grid(True, alpha=0.3)
    
    # 4. Total Deaths
    axes[1,0].plot(days, df['D'], color='black', linewidth=2)
    axes[1,0].fill_between(days, df['D'], alpha=0.2, color='black')
    axes[1,0].set_xlabel('Day')
    axes[1,0].set_ylabel('Cumulative Deaths')
    axes[1,0].set_title(f'Total Deaths: {df["D"].max()}', fontweight='bold')
    axes[1,0].grid(True, alpha=0.3)
    
    # 5. Combined Awareness-Disease States
    axes[1,1].plot(days, df['AS'], label='Aware-Susceptible', color='green', linewidth=2)
    axes[1,1].plot(days, df['US'], label='Unaware-Susceptible', color='orange', linewidth=2)
    axes[1,1].plot(days, df['AI'], label='Aware-Infectious', color='red', linewidth=2)
    axes[1,1].set_xlabel('Day')
    axes[1,1].set_ylabel('Population')
    axes[1,1].set_title('Combined Awareness-Disease States', fontweight='bold')
    axes[1,1].legend()
    axes[1,1].grid(True, alpha=0.3)
    
    # 6. Economic Impact
    axes[1,2].plot(days, df['adjusted_economy'], color='darkgreen', linewidth=2)
    axes[1,2].fill_between(days, df['adjusted_economy'], alpha=0.3, color='green')
    axes[1,2].set_xlabel('Day')
    axes[1,2].set_ylabel('Economic Activity (Adjusted)')
    axes[1,2].set_title('Economic Impact (Lockdown + Mortality)', fontweight='bold')
    axes[1,2].grid(True, alpha=0.3)
    axes[1,2].set_ylim([0, 1.1])
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✓ Visualization saved to: {save_path}")
    plt.close()


# ==============================================================================
# RUNNER
# ==============================================================================

def run_simulation(network_data, scenario_name='realistic', max_days=365):
    params = SCENARIOS[scenario_name].copy()
    lockdown_action = params.get('lockdown_action')
    
    simulator = EpidemicSimulator(network_data, params)
    simulator.reset()
    
    print(f"\nRunning '{scenario_name}' with lockdown level {lockdown_action}")
    print(f"{'Day':<6} {'Active':<8} {'New_Inf':<8} {'Deaths':<8} {'R_eff':<8} {'Economy':<8}")
    print("-" * 54)
    
    done = False
    for day in range(2, max_days + 1):
        if done:
            print(f"\nSimulation ended on day {day-1}")
            break
        
        stats, done = simulator.step(action=lockdown_action)
        
        if day % 50 == 0 or day == 2:
            print(f"{day:<6} {stats['Active']:<8} {stats['new_infections']:<8} "
                  f"{stats['D']:<8} {stats['R_eff']:<8.2f} {stats['adjusted_economy']:<8.2f}")
    
    # Save results with lockdown level in filename
    df = simulator.get_history_df()
    csv_path = f'results/simulation_{scenario_name}_level_{lockdown_action}.csv'
    png_path = f'results/epidemic_simulation_{scenario_name}_level_{lockdown_action}.png'
    
    df.to_csv(csv_path, index=False)
    print(f"\n✓ Results saved to: {csv_path}")
    
    visualize_simulation(df, png_path)
    
    # Print summary
    total_infected = df['Total_Ever_Infected'].max()
    print(f"\n{'='*60}")
    print(f"Total days: {len(df)} | Peak active: {df['Active'].max()} | "
          f"Total deaths: {df['D'].max()} | Attack rate: {total_infected/simulator.N*100:.1f}%")
    print(f"{'='*60}\n")
    
    return simulator, df


# ==============================================================================
# MAIN
# ==============================================================================

def main():
    with open('networks/multilayer_network.pkl', 'rb') as f:
        network_data = pickle.load(f)
    
    scenario = 'long_duration'  # Options: 'extreme', 'realistic', 'long_duration'
    max_days = 720
    
    run_simulation(network_data, scenario, max_days)


if __name__ == "__main__":
    main()