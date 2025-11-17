"""
Step 2: Epidemic Simulator (Refactored)
Implements SEIRD + UAU dynamics on multilayer simplicial network

Key improvements:
- Organized parameter management
- Stochastic incubation and recovery periods
- Waning immunity (SEIRS model)
- Realistic transmission probabilities
- Comprehensive visualization

Runtime: ~5-10 minutes for 300-day simulation
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
from collections import defaultdict

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (15, 10)


# ==============================================================================
# PARAMETER CONFIGURATION
# ==============================================================================

class SimulationParams:
    """Centralized parameter configuration"""
    
    # Available scenarios
    SCENARIOS = {
        'extreme': {
            'beta0': 1.0,
            'mu': 0.1,
            'gamma_r': 1/40,
            'mortality_rate': 0.5,
            'gamma_a': 0.0,
            'lambda_u': 0.1,
            'lambda_delta': 0.25,
            'delta': 0.7,
            'waning_rate': 0.00725,
            'lockdown_action': 0
        },
        'realistic': {
            'beta0': 0.10,
            'mu': 0.2,
            'gamma_r': 1/10,
            'mortality_rate': 0.02,
            'gamma_a': 0.3,
            'lambda_u': 0.2,
            'lambda_delta': 0.4,
            'delta': 0.6,
            'waning_rate': 0.00725,
            'lockdown_action': 0
        },
        'long_duration': {
            'beta0': 0.04,
            'mu': 0.1,
            'gamma_r': 1/14,
            'mortality_rate': 0.0165,
            'gamma_a': 0.6,
            'lambda_u': 0.15,
            'lambda_delta': 0.25,
            'delta': 0.45,
            'waning_rate': 0.00725,
            'lockdown_action': 1
        }
    }
    
    # Lockdown settings
    LOCKDOWN_MULTIPLIERS = {
        0: 1.0,   # No restrictions
        1: 0.75,  # Social distancing (25% reduction)
        2: 0.3    # Full lockdown (70% reduction)
    }
    
    ECONOMY_LEVELS = {
        0: 1.0,   # Full economy
        1: 0.75,  # 75% economy
        2: 0.30   # 30% economy
    }
    
    # Initial conditions
    INITIAL_AWARE_PERCENT = 0.01  # 1% initially aware
    INITIAL_EXPOSED_COUNT = 15     # Number of seed infections
    
    @classmethod
    def get_scenario(cls, name='realistic'):
        """Get parameter set for a scenario"""
        if name not in cls.SCENARIOS:
            raise ValueError(f"Unknown scenario: {name}. Available: {list(cls.SCENARIOS.keys())}")
        return cls.SCENARIOS[name].copy()


# ==============================================================================
# NODE AND STATE MANAGEMENT
# ==============================================================================

class NodeState:
    """Tracks disease and awareness states for all nodes"""
    
    def __init__(self, N):
        self.N = N
        
        # Disease states: S, E, I, R, D
        self.disease_states = np.full(N, 'S', dtype='<U1')
        
        # Awareness states: U (Unaware), A (Aware), R (Removed/Dead)
        self.awareness_states = np.full(N, 'U', dtype='<U1')
        
        # Initialize aware population
        initial_aware_count = max(1, int(N * SimulationParams.INITIAL_AWARE_PERCENT))
        initial_aware_nodes = np.random.choice(N, initial_aware_count, replace=False)
        self.awareness_states[initial_aware_nodes] = 'A'
        
        # Seed initial infections (exposed, unaware)
        unaware_nodes = np.where(self.awareness_states == 'U')[0]
        seed_count = min(SimulationParams.INITIAL_EXPOSED_COUNT, len(unaware_nodes))
        seed_nodes = np.random.choice(unaware_nodes, size=seed_count, replace=False)
        self.disease_states[seed_nodes] = 'E'
        
        # Timers for disease progression
        self.incubation_timer = np.zeros(N, dtype=int)  # Days in E state
        self.infection_timer = np.zeros(N, dtype=int)   # Days in I state
        
        # Set random incubation periods (1-5 days)
        for node in seed_nodes:
            self.incubation_timer[node] = np.random.randint(1, 6)
        
        self.update_counts()
    
    def update_counts(self):
        """Update all population counts"""
        # Basic disease states
        self.counts = {
            'S': np.sum(self.disease_states == 'S'),
            'E': np.sum(self.disease_states == 'E'),
            'I': np.sum(self.disease_states == 'I'),
            'R': np.sum(self.disease_states == 'R'),
            'D': np.sum(self.disease_states == 'D'),
        }
        
        # Awareness states
        self.counts['U'] = np.sum(self.awareness_states == 'U')
        self.counts['A'] = np.sum(self.awareness_states == 'A')
        
        # Combined states
        self.counts['AS'] = np.sum((self.awareness_states == 'A') & (self.disease_states == 'S'))
        self.counts['US'] = np.sum((self.awareness_states == 'U') & (self.disease_states == 'S'))
        self.counts['AI'] = np.sum((self.awareness_states == 'A') & (self.disease_states == 'I'))
        self.counts['UI'] = np.sum((self.awareness_states == 'U') & (self.disease_states == 'I'))
        
        # Derived metrics
        self.counts['Active'] = self.counts['I']  # Only infectious counted as active
        self.counts['Aware_Density'] = self.counts['A'] / self.N


# ==============================================================================
# EPIDEMIC SIMULATOR CORE
# ==============================================================================

class EpidemicSimulator:
    """Simulates epidemic dynamics on multilayer simplicial network"""
    
    def __init__(self, network_data, params):
        self.N = network_data['N']
        self.network_data = network_data
        
        # === DISEASE PARAMETERS ===
        self.beta0 = params['beta0']                    # Base infection rate
        self.mu = params['mu']                          # Incubation rate (E→I)
        self.gamma_r = params['gamma_r']                # Recovery rate (I→R)
        self.mortality_rate = params['mortality_rate']  # Death probability
        self.gamma_a = params['gamma_a']                # Awareness protection factor
        self.waning_rate = params['waning_rate']        # Immunity waning rate (R→S)
        
        # === AWARENESS PARAMETERS ===
        self.lambda_u = params['lambda_u']              # 1-simplex awareness spread
        self.lambda_delta = params['lambda_delta']      # 2-simplex awareness spread
        self.delta = params['delta']                    # Forgetting rate
        
        # === POLICY PARAMETERS ===
        self.lockdown_multipliers = SimulationParams.LOCKDOWN_MULTIPLIERS
        self.economy_levels = SimulationParams.ECONOMY_LEVELS
        self.current_lockdown_level = 0
        
        # === NETWORK STRUCTURE ===
        self.adj_phys = network_data['adjacency_list_physical']
        self.adj_info = network_data['adjacency_list_info']
        self.adj_triangles = network_data['adjacency_triangles_list']
        
        # Calculate effective λ* for 2-simplices (Fan et al. Eq 2.3)
        total_triangles = len(network_data.get('simplices_2', []))
        k2_avg = 3 * total_triangles / self.N
        self.lambda_star = self.lambda_delta * self.delta / k2_avg if k2_avg > 0 else 0
        
        # === STATE TRACKING ===
        self.state = NodeState(self.N)
        self.day = 0
        self.history = []
        self.cumulative_deaths = 0
        self.cumulative_infections = 0
        
        self._print_initialization()
    
    def _print_initialization(self):
        """Print simulation configuration"""
        print(f"\n{'='*60}")
        print("EPIDEMIC SIMULATOR INITIALIZED")
        print(f"{'='*60}")
        print(f"Population: {self.N}")
        
        print(f"\nDisease Parameters:")
        print(f"  β₀ (base transmission): {self.beta0:.3f}")
        print(f"  μ (incubation rate): {self.mu:.3f} → avg {1/self.mu:.1f} days")
        print(f"  γᵣ (recovery rate): {self.gamma_r:.4f} → avg {1/self.gamma_r:.1f} days")
        print(f"  Mortality rate: {self.mortality_rate:.1%}")
        print(f"  Waning immunity: {self.waning_rate:.5f} per day → avg {1/self.waning_rate:.0f} days")
        print(f"  γₐ (awareness protection): {self.gamma_a:.2f} → {(1-self.gamma_a)*100:.0f}% reduction")
        
        print(f"\nAwareness Parameters:")
        print(f"  λ (1-simplex spread): {self.lambda_u:.3f}")
        print(f"  λ_δ (2-simplex param): {self.lambda_delta:.3f}")
        print(f"  λ* (effective 2-simplex): {self.lambda_star:.4f}")
        print(f"  δ (forgetting rate): {self.delta:.3f} → avg {1/self.delta:.1f} days")
        
        print(f"\nPolicy Options:")
        print(f"  Lockdown levels: {list(self.lockdown_multipliers.keys())}")
        print(f"  Contact reductions: {[f'{(1-v)*100:.0f}%' for v in self.lockdown_multipliers.values()]}")
        print(f"{'='*60}\n")
    
    # ==========================================================================
    # DISEASE DYNAMICS
    # ==========================================================================
    
    def _update_waning_immunity(self):
        """Handle R → S transitions (waning immunity)"""
        R_nodes = np.where(self.state.disease_states == 'R')[0]
        
        for i in R_nodes:
            if np.random.rand() < self.waning_rate:
                self.state.disease_states[i] = 'S'
    
    def _update_disease_spread(self, transmission_multiplier):
        """Handle S→E and E→I transitions"""
        new_infections = 0
        
        # === E → I (Incubation completion) ===
        E_nodes = np.where(self.state.disease_states == 'E')[0]
        
        for i in E_nodes:
            self.state.incubation_timer[i] -= 1
            
            if self.state.incubation_timer[i] <= 0:
                self.state.disease_states[i] = 'I'
                
                # Set infectious duration (stochastic around mean)
                avg_recovery_time = 1 / self.gamma_r
                self.state.infection_timer[i] = max(1, np.random.randint(
                    int(avg_recovery_time * 0.7), 
                    int(avg_recovery_time * 1.3) + 1
                ))
                
                # Infected individuals become spontaneously aware
                if self.state.awareness_states[i] != 'R':
                    self.state.awareness_states[i] = 'A'
        
        # === S → E (New infections) ===
        S_nodes = np.where(self.state.disease_states == 'S')[0]
        
        # Calculate effective transmission rates
        beta_U = self.beta0 * transmission_multiplier  # Unaware susceptible
        beta_A = self.gamma_a * beta_U                 # Aware susceptible
        
        for i in S_nodes:
            # Find infectious neighbors in physical layer
            I_neighbors = [j for j in self.adj_phys[i] 
                          if self.state.disease_states[j] == 'I']
            
            if not I_neighbors:
                continue
            
            # Select transmission rate based on awareness
            beta_eff = beta_A if self.state.awareness_states[i] == 'A' else beta_U
            
            # Probability of infection from any neighbor
            prob_infection = 1 - (1 - beta_eff) ** len(I_neighbors)
            prob_infection = min(prob_infection, 0.9)  # Cap at 90%
            
            if np.random.rand() < prob_infection:
                self.state.disease_states[i] = 'E'
                self.state.incubation_timer[i] = np.random.randint(1, 6)
                new_infections += 1
        
        self.cumulative_infections += new_infections
        return new_infections
    
    def _update_mortality(self):
        """Handle I→R and I→D transitions"""
        I_nodes = np.where(self.state.disease_states == 'I')[0]
        
        if len(I_nodes) == 0:
            return 0, 0
        
        deaths_today = 0
        recoveries_today = 0
        
        for i in I_nodes:
            self.state.infection_timer[i] -= 1
            
            if self.state.infection_timer[i] <= 0:
                if np.random.rand() < self.mortality_rate:
                    # Death
                    self.state.disease_states[i] = 'D'
                    self.state.awareness_states[i] = 'R'  # Removed
                    deaths_today += 1
                else:
                    # Recovery
                    self.state.disease_states[i] = 'R'
                    recoveries_today += 1
                
                self.state.infection_timer[i] = 0
        
        self.cumulative_deaths += deaths_today
        return deaths_today, recoveries_today
    
    # ==========================================================================
    # AWARENESS DYNAMICS
    # ==========================================================================
    
    def _update_awareness_spread(self):
        """Handle U→A (awareness spread) and A→U (forgetting)"""
        
        # === A → U (Forgetting) ===
        A_nodes = np.where(self.state.awareness_states == 'A')[0]
        
        for i in A_nodes:
            if self.state.disease_states[i] != 'D':  # Only living can forget
                if np.random.rand() < self.delta:
                    self.state.awareness_states[i] = 'U'
        
        # === U → A (Awareness spread) ===
        U_nodes = np.where(self.state.awareness_states == 'U')[0]
        
        for i in U_nodes:
            if self.state.disease_states[i] == 'D':  # Skip dead
                continue
            
            became_aware = False
            
            # 1-Simplex (pairwise) transmission
            A_neighbors = [j for j in self.adj_info[i] 
                          if self.state.awareness_states[j] == 'A']
            
            if A_neighbors:
                prob_not_informed = (1 - self.lambda_u) ** len(A_neighbors)
                if np.random.rand() > prob_not_informed:
                    became_aware = True
            
            # 2-Simplex (group) transmission
            if not became_aware and self.lambda_star > 0:
                for j, k in self.adj_triangles[i]:
                    if (self.state.awareness_states[j] == 'A' and 
                        self.state.awareness_states[k] == 'A'):
                        if np.random.rand() < self.lambda_star:
                            became_aware = True
                            break
            
            if became_aware:
                self.state.awareness_states[i] = 'A'
    
    # ==========================================================================
    # SIMULATION STEP
    # ==========================================================================
    
    def step(self, action):
        """Execute one simulation day"""
        self.day += 1
        self.current_lockdown_level = action
        transmission_multiplier = self.lockdown_multipliers[action]
        
        # 1. Waning immunity
        self._update_waning_immunity()
        
        # 2. Awareness dynamics
        self._update_awareness_spread()
        
        # 3. Disease spread
        new_infections = self._update_disease_spread(transmission_multiplier)
        
        # 4. Recovery and mortality
        deaths_today, recoveries_today = self._update_mortality()
        
        # 5. Update counts
        self.state.update_counts()
        
        # 6. Calculate R_eff
        S_count = self.state.counts['S']
        if S_count > 0:
            AS_count = self.state.counts['AS']
            US_count = self.state.counts['US']
            
            beta_avg = ((AS_count * self.gamma_a * self.beta0 * transmission_multiplier) + 
                       (US_count * self.beta0 * transmission_multiplier)) / S_count
            
            R_eff = beta_avg * (S_count / self.N) / self.gamma_r
        else:
            beta_avg = 0
            R_eff = 0
        
        # 7. Calculate economy
        lockdown_capacity = self.economy_levels[action]
        workforce_available = self.N - self.state.counts['D'] - self.state.counts['I']
        adjusted_economy = lockdown_capacity * workforce_available / self.N
        
        # 8. Record statistics
        stats = {
            # Metadata
            'day': self.day,
            'action': action,
            'lockdown_level': action,
            
            # Disease states
            'S': self.state.counts['S'],
            'E': self.state.counts['E'],
            'I': self.state.counts['I'],
            'R': self.state.counts['R'],
            'D': self.state.counts['D'],
            
            # Key metrics
            'Active': self.state.counts['I'],
            'Total_Ever_Infected': (self.state.counts['E'] + self.state.counts['I'] + 
                                   self.state.counts['R'] + self.state.counts['D']),
            'new_infections': new_infections,
            'new_deaths': deaths_today,
            'new_recoveries': recoveries_today,
            
            # Awareness
            'A': self.state.counts['A'],
            'U': self.state.counts['U'],
            'rho_A': self.state.counts['Aware_Density'],
            
            # Combined states
            'AS': self.state.counts['AS'],
            'US': self.state.counts['US'],
            'AI': self.state.counts['AI'],
            
            # Epidemiological metrics
            'R_eff': R_eff,
            'beta_avg': beta_avg,
            
            # Economic
            'base_economy': lockdown_capacity,
            'adjusted_economy': adjusted_economy,
            'workforce_available': workforce_available,
            'transmission_multiplier': transmission_multiplier,
            
            # Cumulative
            'cumulative_deaths': self.cumulative_deaths,
            'cumulative_infections': self.cumulative_infections
        }
        
        self.history.append(stats)
        
        # Check termination
        done = (self.state.counts['I'] == 0 and self.state.counts['E'] == 0)
        
        return stats, done
    
    def reset(self):
        """Reset simulator for new episode"""
        self.state = NodeState(self.N)
        self.day = 0
        self.history = []
        self.cumulative_deaths = 0
        self.cumulative_infections = 0
        self.current_lockdown_level = 0
        
        initial_stats, _ = self.step(action=0)
        return initial_stats
    
    # ==========================================================================
    # UTILITY METHODS
    # ==========================================================================
    
    def get_reward(self, stats):
        """Calculate RL reward (Ohi et al. structure)"""
        Active_norm = stats['Active'] / self.N
        Deaths_today = stats['new_deaths']
        Economy_t = stats['adjusted_economy']
        
        reward = Economy_t * np.exp(-8 * Active_norm) - 5 * Deaths_today
        return reward
    
    def get_state_vector(self, stats):
        """Extract normalized state vector for RL agent"""
        state_vec = np.array([
            stats['Active'] / self.N,
            stats['new_infections'] / self.N,
            stats['D'] / self.N,
            stats['R'] / self.N,
            min(stats['R_eff'] / 5.0, 1.0),
            stats['rho_A'],
            stats['adjusted_economy']
        ], dtype=np.float32)
        
        return state_vec
    
    def get_history_df(self):
        """Convert history to pandas DataFrame"""
        return pd.DataFrame(self.history)
    
    def save_results(self, filepath='results/simulation_results.csv'):
        """Save simulation history"""
        df = self.get_history_df()
        df.to_csv(filepath, index=False)
        print(f"Results saved to: {filepath}")
        return df


# ==============================================================================
# VISUALIZATION AND REPORTING
# ==============================================================================

def visualize_simulation(df, save_path='results/epidemic_simulation.png'):
    """Create comprehensive visualization of simulation results"""
    print(f"\n{'='*60}")
    print("CREATING VISUALIZATIONS")
    print(f"{'='*60}")
    
    fig, axes = plt.subplots(3, 3, figsize=(20, 15))
    fig.suptitle('Epidemic Simulation Results', fontsize=16, fontweight='bold')
    
    days = df['day']
    
    # 1. SEIRD Compartments
    print("[1/9] Plotting SEIRD compartments...")
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
    
    # 2. Active Cases
    print("[2/9] Plotting active cases...")
    axes[0,1].fill_between(days, df['Active'], alpha=0.3, color='red')
    axes[0,1].plot(days, df['Active'], color='red', linewidth=2)
    axes[0,1].axhline(df['Active'].max(), color='red', linestyle='--', 
                      label=f'Peak: {df["Active"].max()}')
    axes[0,1].set_xlabel('Day')
    axes[0,1].set_ylabel('Active Infectious Cases')
    axes[0,1].set_title('Active Cases (I) Over Time', fontweight='bold')
    axes[0,1].legend()
    axes[0,1].grid(True, alpha=0.3)
    
    # 3. New Infections per Day
    print("[3/9] Plotting daily new infections...")
    axes[0,2].bar(days, df['new_infections'], alpha=0.6, color='orange', edgecolor='black')
    axes[0,2].plot(days, df['new_infections'].rolling(7).mean(), 
                   color='red', linewidth=2, label='7-day avg')
    axes[0,2].set_xlabel('Day')
    axes[0,2].set_ylabel('New Infections (S→E)')
    axes[0,2].set_title('Daily New Infections', fontweight='bold')
    axes[0,2].legend()
    axes[0,2].grid(True, alpha=0.3)
    
    # 4. Cumulative Deaths
    print("[4/9] Plotting cumulative deaths...")
    axes[1,0].plot(days, df['D'], color='black', linewidth=2)
    axes[1,0].fill_between(days, df['D'], alpha=0.2, color='black')
    axes[1,0].set_xlabel('Day')
    axes[1,0].set_ylabel('Cumulative Deaths')
    axes[1,0].set_title(f'Total Deaths: {df["D"].max()}', fontweight='bold')
    axes[1,0].grid(True, alpha=0.3)
    
    # 5. R_eff over time
    print("[5/9] Plotting R_eff...")
    axes[1,1].plot(days, df['R_eff'], color='purple', linewidth=2)
    axes[1,1].axhline(1.0, color='red', linestyle='--', linewidth=2, label='R=1 threshold')
    axes[1,1].fill_between(days, df['R_eff'], 1, 
                          where=df['R_eff']>1, alpha=0.3, color='red', label='R>1 (Growing)')
    axes[1,1].fill_between(days, df['R_eff'], 1, 
                          where=df['R_eff']<=1, alpha=0.3, color='green', label='R<1 (Declining)')
    axes[1,1].set_xlabel('Day')
    axes[1,1].set_ylabel('Effective Reproduction Number')
    axes[1,1].set_title('R_eff Over Time', fontweight='bold')
    axes[1,1].legend()
    axes[1,1].grid(True, alpha=0.3)
    axes[1,1].set_ylim(bottom=0)
    
    # 6. Awareness Density
    print("[6/9] Plotting awareness density...")
    axes[1,2].plot(days, df['rho_A'], color='blue', linewidth=2)
    axes[1,2].fill_between(days, df['rho_A'], alpha=0.3, color='blue')
    axes[1,2].axhline(df['rho_A'].mean(), color='red', linestyle='--', 
                      label=f'Mean: {df["rho_A"].mean():.2f}')
    axes[1,2].set_xlabel('Day')
    axes[1,2].set_ylabel('Awareness Density (ρ_A)')
    axes[1,2].set_title('Population Awareness Over Time', fontweight='bold')
    axes[1,2].legend()
    axes[1,2].grid(True, alpha=0.3)
    axes[1,2].set_ylim([0, 1])
    
    # 7. Awareness vs Disease States
    print("[7/9] Plotting combined states...")
    axes[2,0].plot(days, df['AS'], label='Aware-Susceptible', color='green', linewidth=2)
    axes[2,0].plot(days, df['US'], label='Unaware-Susceptible', color='orange', linewidth=2)
    axes[2,0].plot(days, df['AI'], label='Aware-Infectious', color='red', linewidth=2)
    axes[2,0].set_xlabel('Day')
    axes[2,0].set_ylabel('Population')
    axes[2,0].set_title('Combined Awareness-Disease States', fontweight='bold')
    axes[2,0].legend()
    axes[2,0].grid(True, alpha=0.3)
    
    # 8. Economy
    print("[8/9] Plotting economic impact...")
    axes[2,1].plot(days, df['adjusted_economy'], color='darkgreen', linewidth=2)
    axes[2,1].fill_between(days, df['adjusted_economy'], alpha=0.3, color='green')
    axes[2,1].set_xlabel('Day')
    axes[2,1].set_ylabel('Economic Activity (Adjusted)')
    axes[2,1].set_title('Economic Impact (Lockdown + Mortality)', fontweight='bold')
    axes[2,1].grid(True, alpha=0.3)
    axes[2,1].set_ylim([0, 1.1])
    
    # 9. Phase Diagram: R_eff vs Awareness
    print("[9/9] Creating phase diagram...")
    scatter = axes[2,2].scatter(df['rho_A'], df['R_eff'], 
                               c=days, cmap='viridis', 
                               s=30, alpha=0.6, edgecolors='black', linewidths=0.5)
    axes[2,2].axhline(1.0, color='red', linestyle='--', linewidth=2, label='R=1')
    axes[2,2].set_xlabel('Awareness Density (ρ_A)')
    axes[2,2].set_ylabel('R_eff')
    axes[2,2].set_title('Phase Diagram: R_eff vs Awareness', fontweight='bold')
    axes[2,2].legend()
    axes[2,2].grid(True, alpha=0.3)
    cbar = plt.colorbar(scatter, ax=axes[2,2])
    cbar.set_label('Day')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\n✓ Visualization saved to: {save_path}")
    plt.show()


def print_summary_statistics(df, N):
    """Print comprehensive summary of simulation results"""
    print(f"\n{'='*60}")
    print("SIMULATION SUMMARY STATISTICS")
    print(f"{'='*60}")
    
    total_ever_infected = df['Total_Ever_Infected'].max()
    total_deaths = df['D'].max()
    
    print("\nDisease Outcomes:")
    print(f"  • Total simulation days: {len(df)}")
    print(f"  • Peak active cases (I): {df['Active'].max()} ({df['Active'].max()/N*100:.1f}%)")
    print(f"  • Peak day: {df['Active'].idxmax()}")
    print(f"  • Total ever infected: {total_ever_infected} ({total_ever_infected/N*100:.1f}%)")
    print(f"  • Total deaths: {total_deaths} ({total_deaths/N*100:.1f}% of population)")
    print(f"  • Total recovered: {df['R'].max()}")
    print(f"  • Final susceptible: {df['S'].iloc[-1]}")
    
    print(f"\nEpidemiological Metrics:")
    print(f"  • Average R_eff: {df['R_eff'].mean():.2f}")
    print(f"  • Max R_eff: {df['R_eff'].max():.2f}")
    print(f"  • Days with R>1: {(df['R_eff'] > 1).sum()} ({(df['R_eff'] > 1).sum()/len(df)*100:.1f}%)")
    print(f"  • Attack rate: {(total_ever_infected/N*100):.1f}%")
    if total_ever_infected > 0:
        print(f"  • Case fatality rate: {(total_deaths / total_ever_infected * 100):.2f}%")
    
    print(f"\nAwareness Dynamics:")
    print(f"  • Average awareness density: {df['rho_A'].mean():.2f}")
    print(f"  • Peak awareness: {df['rho_A'].max():.2f}")
    print(f"  • Final awareness: {df['rho_A'].iloc[-1]:.2f}")
    
    print(f"\nEconomic Impact:")
    print(f"  • Average economic activity: {df['adjusted_economy'].mean():.2f}")
    print(f"  • Final economic capacity: {df['adjusted_economy'].iloc[-1]:.2f}")
    print(f"  • Economic loss from workforce: {(1 - df['workforce_available'].iloc[-1]/df['workforce_available'].iloc[0]) * 100:.1f}%")
    
    print(f"\n{'='*60}\n")


# ==============================================================================
# SIMULATION RUNNER
# ==============================================================================

def run_simulation(network_data, scenario_name='realistic', max_days=365):
    """
    Run epidemic simulation with specified scenario
    
    Args:
        network_data: Network structure from pickle file
        scenario_name: Name of scenario from SimulationParams.SCENARIOS
        max_days: Maximum simulation days
    """
    # Get parameters for scenario
    params = SimulationParams.get_scenario(scenario_name)
    lockdown_action = params.get('lockdown_action', 0)
    
    # Initialize simulator
    simulator = EpidemicSimulator(network_data, params)
    current_stats = simulator.reset()
    
    print(f"\nStarting '{scenario_name}' simulation...")
    print(f"Lockdown policy: Level {lockdown_action}")
    print(f"\n{'Day':<6} {'Active(I)':<10} {'New_Inf':<10} {'Deaths(D)':<10} {'R_eff':<8} {'ρ_A':<8} {'Adj_Econ':<10}")
    print("-" * 72)
    
    # Run simulation
    done = False
    for day in range(2, max_days + 1):
        if done:
            print(f"\nSimulation ended early on day {day-1}. Disease contained.")
            break
        
        current_stats, done = simulator.step(action=lockdown_action)
        
        # Print progress every 25 days
        if day % 25 == 0 or day == 2:
            print(f"{day:<6} "
                  f"{current_stats['Active']:<10} "
                  f"{current_stats['new_infections']:<10} "
                  f"{current_stats['D']:<10} "
                  f"{current_stats['R_eff']:<8.2f} "
                  f"{current_stats['rho_A']:<8.2f} "
                  f"{current_stats['adjusted_economy']:<10.2f}")
    
    # Save and visualize results
    df = simulator.save_results(filepath=f'results/simulation_{scenario_name}_output.csv')
    print_summary_statistics(df, simulator.N)
    visualize_simulation(df, save_path=f'results/epidemic_simulation_{scenario_name}.png')
    
    return simulator, df


# ==============================================================================
# MAIN EXECUTION
# ==============================================================================

def main():
    """Main execution function"""
    
    # Load network
    print("Loading network data...")
    with open('networks/multilayer_network.pkl', 'rb') as f:
        network_data = pickle.load(f)
    print(f"✓ Network loaded: {network_data['N']} nodes\n")
    
    # Run simulation with specified scenario
    scenario = 'long_duration'  # Options: 'extreme', 'realistic', 'long_duration'
    max_days = 720
    
    simulator, df = run_simulation(
        network_data=network_data,
        scenario_name=scenario,
        max_days=max_days
    )
    
    print("\n" + "="*60)
    print("SIMULATION COMPLETE")
    print("="*60)
    print(f"Final statistics saved to: results/simulation_{scenario}_output.csv")
    print(f"Visualizations saved to: results/epidemic_simulation_{scenario}.png")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()