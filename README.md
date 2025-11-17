# Epidemic Control via Deep Reinforcement Learning

> **Optimizing lockdown policies through co-evolution of disease transmission and public awareness on multilayer simplicial networks**

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
---

## Overview

The COVID-19 pandemic revealed that epidemic dynamics are fundamentally shaped by the **co-evolution of disease transmission and public awareness**. Traditional epidemic models fail to capture group-level social structures that characterize real-world contact patterns.

This project combines:
- **Multilayer Simplicial Complexes**: Model higher-order interactions (groups, not just pairs)
- **SEIRD Epidemic Model**: Susceptible-Exposed-Infectious-Recovered-Dead dynamics
- **UAU Awareness Model**: Unaware-Aware-Unaware information spread
- **Deep Reinforcement Learning**: Double DQN with Bidirectional LSTM to learn optimal lockdown policies

---

## Quick Start

### Installation
```bash
# Clone repository
git clone <your-repo-url>
cd epidemic-rl-project

# Install dependencies
pip install -r requirements.txt
```

### Run a Simulation
```bash
# 1. Build the network (if not already created)
python network_construction.py

# 2. Run baseline simulation
python epidemic_simulator.py

# 3. Train RL agent (Branch 2)
python train_agent.py --episodes 1000 --scenario realistic
```

---

## Project Structure

```
190j-project/
│
├── README.md                         
├── requirements.txt                   # Dependencies
│
├── config.py                         # RL agent configuration
├── dqn_agent.py                      # Double DQN with LSTM
├── replay_buffer.py                  # Experience replay
│
├── network_construction.py           # Build multilayer network
├── epidemic_simulator.py             # SEIRD + UAU simulator
│
├── train_agent.py                    # Training loop
├── utils/                            # Training helpers and visualizers
│
├── networks/                          # Saved network structures
├── checkpoints/                       # Trained models
├── logs/                              # Training logs
└── results/                           # Simulation outputs & plots
```

---

## Model Architecture

### 1. Multilayer Simplicial Network

**Physical Layer** (Disease Transmission):
- Erdős-Rényi random graph
- N = 1000 nodes, p = 0.005 connection probability
- Models close-contact disease spread

**Information Layer** (Awareness Propagation):
- Random simplicial complex
- **1-simplices** (edges): Pairwise information sharing (k₁ = 4)
- **2-simplices** (triangles): Group consensus effects (k₂ = 1)
- Models social media, news, community discussions

### 2. Epidemic Dynamics (SEIRD)

```
S → E → I → R
        ↓
        D
```

- **S**: Susceptible (healthy, can be infected)
- **E**: Exposed (infected but not yet infectious, 1-5 day incubation)
- **I**: Infectious (can spread disease, ~10-14 days)
- **R**: Recovered (immune, but immunity wanes over time)
- **D**: Dead (removed from both layers)

### 3. Awareness Dynamics (UAU)

```
U ⇄ A → U
```

- **U**: Unaware (no protective behavior, full infection risk)
- **A**: Aware (protective behavior, 70% reduced infection risk)

**Spread Mechanisms**:
1. **Pairwise** (1-simplex): Friends informing each other
2. **Group** (2-simplex): Community/family consensus effects
3. **Spontaneous**: Infected individuals become aware

**Forgetting**: Aware individuals gradually forget (A → U)

### 4. RL Agent Architecture

**Neural Network**:
```
State Sequence (14 days × 7 features)
    ↓
Bidirectional LSTM (128 hidden × 2 layers)
    ↓
Fully Connected (256 → 128)
    ↓
Output (3 actions: No/Partial/Full Lockdown)
```

**Algorithm**: Double Deep Q-Network (DDQN)
- Reduces overestimation bias
- Experience replay for sample efficiency
- Target network for stable learning

**State Vector** (7 features):
1. Active infectious cases (normalized)
2. New infections (normalized)
3. Cumulative deaths (normalized)
4. Cumulative recoveries (normalized)
5. Effective reproduction number (R_eff)
6. Awareness density (ρ_A)
7. Adjusted economic activity

**Actions** (3 lockdown levels):
- **0**: No restrictions (economy = 1.0, transmission = 1.0)
- **1**: Partial lockdown (economy = 0.65, transmission = 0.65)
- **2**: Full lockdown (economy = 0.4, transmission = 0.4)

**Reward Function**:
```
R(t) = Economy(t) × exp(-8 × Active(t)/N) - 5 × Deaths(t)
```
Balances economic activity against infection control and mortality.

---

## Research Questions

1. **Can RL discover non-obvious policies?** (e.g., early aggressive lockdown → minimal long-term restriction)
2. **How does awareness propagation affect optimal policy?** (strong awareness → less lockdown needed?)
3. **Impact of higher-order interactions?** (2-simplices vs pairwise-only)
4. **Trade-off frontiers**: Deaths vs economic loss under different scenarios

---

## Results

### Baseline (No Lockdown)
| Scenario | Peak Active | Total Deaths | Attack Rate | Avg R_eff |
|----------|-------------|--------------|-------------|-----------|
| Realistic | ~180 | 20-40 | 45-65% | 1.2-1.8 |
| Extreme | ~500 | 400+ | 90%+ | 3.5+ |

### RL Agent (After Training)
*Results will be updated.

---

## References

### Network Science
1. **Fan et al.** - "Epidemics on multilayer simplicial complexes"
2. **Boccaletti et al.** - "The structure and dynamics of networks with higher order interactions"

### Reinforcement Learning
3. **Mnih et al. (2015)** - "Human-level control through deep reinforcement learning" (DQN)
4. **Van Hasselt et al. (2016)** - "Deep Reinforcement Learning with Double Q-learning" (DDQN)
5. **Schaul et al. (2015)** - "Prioritized Experience Replay"

### Epidemic Control with RL
6. **Ohi et al.** - "Exploring Optimal Control of Epidemic Spread Using Reinforcement Learning"
7. **Kompella et al.** - "Reinforcement Learning for Optimization of COVID-19 Mitigation policies"