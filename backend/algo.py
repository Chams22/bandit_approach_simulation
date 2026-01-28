import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm  # For the progress bar (pip install tqdm)

# -----------------------------------------------------------------------------
# PART 1: THE ALGORITHM (Jamieson & Jain - Algo 1)
# -----------------------------------------------------------------------------
class JamiesonJainAlgo:
    """
    Implementation of Algorithm 1 for TPR maximization under FDR constraint.
    Sources: Jamieson & Jain NeurIPS [1-3]
    """
    def __init__(self, n_arms, mu_0, delta):
        self.n = n_arms
        self.mu_0 = mu_0
        self.delta = delta
        
        # Statistics per arm
        self.counts = np.zeros(n_arms, dtype=int)
        self.emp_means = np.zeros(n_arms, dtype=float)
        self.time = 0
        
        # Set of discoveries (S_t)
        self.S_t = set()

    def phi(self, t, delta_val):
        """
        "Anytime" Confidence Interval.
        Formula used in the Experiments section of the paper [4].
        """
        if t == 0: return float('inf')
        # Precise Kaufmann/Jamieson formula for simulations [4]
        num = 2 * np.log(1/delta_val) + 6 * np.log(np.log(1/delta_val) + 1e-10) + \
              3 * np.log(np.log(np.e * t / 2) + 1e-10)
        return np.sqrt(num / t)

    def select_arm(self, mode='adaptive'):
        """
        Choose which arm to pull.
        mode='adaptive': Jamieson Strategy (UCB on undiscovered arms) [2]
        mode='uniform' : Baseline Strategy (Round-Robin) [5]
        """
        # Initialization: pull each arm at least once
        if self.time < self.n:
            return self.time

        if mode == 'uniform':
            # Uniform sampling (Step 1 of your notes)
            return self.time % self.n
        
        else:
            # Adaptive sampling (Step 2 of your notes)
            # We only pull from arms that are NOT yet discovered (not in S_t)
            candidates = [i for i in range(self.n) if i not in self.S_t]
            
            if not candidates:
                # If everything is discovered, pull randomly (or stop)
                return np.random.choice(range(self.n))

            # UCB Strategy to maximize discoveries (xi_t = 1 for TPR) [1]
            best_ucb = -float('inf')
            selected = candidates[0]
            
            for i in candidates:
                # UCB = Mean + Confidence Interval
                ucb = self.emp_means[i] + self.phi(self.counts[i], self.delta)
                if ucb > best_ucb:
                    best_ucb = ucb
                    selected = i
            return selected

    def update(self, arm_idx, reward):
        """Update empirical means and the discovery set (BH)."""
        # 1. Standard empirical mean update
        n = self.counts[arm_idx] # Current number of pulls for this arm
        self.emp_means[arm_idx] = (self.emp_means[arm_idx] * n + reward) / (n + 1) # New mean
        self.counts[arm_idx] += 1
        self.time += 1
        
        # 2. Anytime Benjamini-Hochberg Procedure to update S_t [2, 6]
        # We look for the largest k such that k arms pass the test
        # Test: LCB_i >= mu_0 with an adjusted delta (delta * k / n)
        k_hat = 0
        current_St = set()
        
        # Check k from n down to 1
        for k in range(self.n, 0, -1):
            effective_delta = self.delta * k / self.n
            passing_arms = []
            
            for i in range(self.n):
                lcb = self.emp_means[i] - self.phi(self.counts[i], effective_delta)
                if lcb >= self.mu_0:
                    passing_arms.append(i)
            
            if len(passing_arms) >= k:
                k_hat = k
                current_St = set(passing_arms)
                break
        
        # Update the discovery set (union with previous because it's anytime)
        self.S_t.update(current_St)

# -----------------------------------------------------------------------------
# PART 2: SIMULATION ENGINE (Step 3: Normal Distributions)
# -----------------------------------------------------------------------------

def run_experiment(true_means, horizon, mode, n_simulations=20):
    """
    Runs n_simulations Monte-Carlo to smooth the results.
    """
    n_arms = len(true_means)
    # Identify "True" positives (H1) to calculate TPR
    true_positives = [i for i, m in enumerate(true_means) if m > mu_0]
    
    # Array to store TPR over time
    tpr_history = np.zeros(horizon)
    
    print(f"Simulation Mode: {mode.upper()} ({n_simulations} runs)")
    
    for _ in tqdm(range(n_simulations)):
        # Create a new instance for each run
        algo = JamiesonJainAlgo(n_arms, mu_0, delta)
        
        run_tpr = []
        
        for t in range(horizon):
            # 1. The algo chooses an arm
            arm = algo.select_arm(mode=mode)
            
            # 2. The environment generates a reward (Normal Distribution) [7]
            reward = np.random.normal(loc=true_means[arm], scale=1.0)
            
            # 3. The algo updates itself
            algo.update(arm, reward)
            
            # 4. Calculate TPR (True Positive Rate) at time t
            # How many true positives are currently in S_t?
            nb_found = len(algo.S_t.intersection(true_positives))
            current_tpr = nb_found / len(true_positives) if true_positives else 1.0
            run_tpr.append(current_tpr)
            
        tpr_history += np.array(run_tpr)
        
    return tpr_history / n_simulations

# -----------------------------------------------------------------------------
# PART 3: CONFIGURATION AND EXECUTION
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    # Scenario parameters (Inspired by the Experiments in paper [7])
    n_arms = 20
    mu_0 = 0.0          # Baseline (e.g., zero efficacy)
    delta = 0.05        # FDR Risk (5%)
    horizon = 3000      # Total budget
    
    # Definition of "True Means" (H1 and H0)
    # 4 effective arms (0.5), 16 ineffective arms (0.0)
    true_means = np.array([0.5]*4 + [0.0]*16)
    
    # 1. Run UNIFORM simulation (Step 1 Notes)
    tpr_uniform = run_experiment(true_means, horizon, mode='uniform', n_simulations=30)
    
    # 2. Run ADAPTIVE simulation (Step 2 Notes)
    tpr_adaptive = run_experiment(true_means, horizon, mode='adaptive', n_simulations=30)
    
    # 3. Plot the Comparative Graph (Expected Result)
    plt.figure(figsize=(10, 6))
    
    # Curves
    plt.plot(tpr_adaptive, label='Jamieson Algo (Adaptive)', color='#ff7f0e', linewidth=2)
    plt.plot(tpr_uniform, label='Uniform Sampling', color='#1f77b4', linestyle='--')
    
    # Targets and legends
    plt.axhline(y=1.0, color='gray', linestyle=':', alpha=0.5, label='Target (100% discovered)')
    plt.title(f"Comparison: Uniform vs Adaptive (Avg over 30 simulations)\n{n_arms} arms, 4 positives")
    plt.xlabel("Number of Samples (t)")
    plt.ylabel("True Positive Rate (TPR)")
    plt.legend(loc='lower right')
    plt.grid(True, alpha=0.3)
    
    plt.show()
    print("Simulation finished. The graph should appear.")