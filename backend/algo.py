import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

# -----------------------------------------------------------------------------
# PART 1: THE ALGORITHM
# -----------------------------------------------------------------------------
class JamiesonJainAlgo:
    def __init__(self, n_arms, mu_0, delta):
        self.n = n_arms
        self.mu_0 = mu_0
        self.delta = delta
        
        self.counts = np.zeros(n_arms, dtype=int)
        self.emp_means = np.zeros(n_arms, dtype=float)
        self.time = 0
        self.S_t = set()
        
        # History for visualization
        # Initialized with zeros for t=0
        self.counts_evolution = [np.zeros(n_arms, dtype=int)]

    def phi(self, t, delta_val):
        if t == 0: return float('inf')
        num = 2 * np.log(1/delta_val) + 6 * np.log(np.log(1/delta_val) + 1e-10) + \
              3 * np.log(np.log(np.e * t / 2) + 1e-10)
        return np.sqrt(num / t)

    def select_arm(self):
        if self.time < self.n:
            return self.time
        
        candidates = [i for i in range(self.n) if i not in self.S_t]
        
        if not candidates:
            return "stop"

        best_ucb = -float('inf')
        selected = candidates[0]
        
        for i in candidates:
            ucb = self.emp_means[i] + self.phi(self.counts[i], self.delta)
            if ucb > best_ucb:
                best_ucb = ucb
                selected = i
        return selected

    def update(self, arm_idx, observation):
        n = self.counts[arm_idx]
        self.emp_means[arm_idx] = (self.emp_means[arm_idx] * n + observation) / (n + 1)
        self.counts[arm_idx] += 1
        self.time += 1
        self.counts_evolution.append(self.counts.copy()) 
        
        k_hat = 0
        current_St = set()
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
        self.S_t.update(current_St)

class UniformAlgo:
    def __init__(self, n_arms, mu_0, delta):
        self.n = n_arms
        self.mu_0 = mu_0
        self.delta = delta
        
        self.counts = np.zeros(n_arms, dtype=int)
        self.emp_means = np.zeros(n_arms, dtype=float)
        self.time = 0
        self.S_t = set()
        
        # History for visualization
        # Initialized with zeros for t=0
        self.counts_evolution = [np.zeros(n_arms, dtype=int)]

    def phi(self, t, delta_val):
        if t == 0: return float('inf')
        num = 2 * np.log(1/delta_val) + 6 * np.log(np.log(1/delta_val) + 1e-10) + \
              3 * np.log(np.log(np.e * t / 2) + 1e-10)
        return np.sqrt(num / t)

    def select_arm(self):
        if self.time < self.n:
            return self.time
        return self.time % self.n

    def update(self, arm_idx, observation):
        n = self.counts[arm_idx]
        self.emp_means[arm_idx] = (self.emp_means[arm_idx] * n + observation) / (n + 1)
        self.counts[arm_idx] += 1
        self.time += 1
        self.counts_evolution.append(self.counts.copy()) 
        
        k_hat = 0
        current_St = set()
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
        self.S_t.update(current_St)


# -----------------------------------------------------------------------------
# PART 2: SIMULATION ENGINE
# -----------------------------------------------------------------------------

def run_experiment(true_means, horizon, mode, n_simulations=20):
    n_arms = len(true_means)
    all_data=[[] for i in range(n_arms)]
    true_positives = [i for i, m in enumerate(true_means) if m > mu_0]
    
    tpr_history_sum = np.zeros(horizon)
    tpr_list = []
    
    # Store the AVERAGE number of pulls at each time t
    counts_evolution_sum = np.zeros((horizon + 1, n_arms))
    counts_list=[]

    print(f"Simulation Mode: {mode.upper()} ({n_simulations} runs)")
    
    for no_sim in tqdm(range(n_simulations)):
        if mode=='adaptive':
            algo = JamiesonJainAlgo(n_arms, mu_0, delta)
        elif mode=='uniform':
            algo = UniformAlgo(n_arms, mu_0, delta)
        else:
            raise ValueError("Algorithm name not detected, choose between uniform and adaptive")
        
        run_tpr = []
        
        for t in range(horizon):
            arm = algo.select_arm()
            
            if arm == "stop":
                # If we stop before the end, we fill the lists with the last value
                # so that arrays have the correct size (horizon)
                last_tpr = run_tpr[-1] if run_tpr else 1.0
                remaining_steps = horizon - len(run_tpr)
                run_tpr.extend([last_tpr] * remaining_steps)
                
                # For counts, repeat the last known row until the end
                last_counts = algo.counts_evolution[-1]
                for _ in range(remaining_steps):
                     algo.counts_evolution.append(last_counts.copy())
                break
            
            else:
                observation = np.random.normal(loc=true_means[arm], scale=1.0)
                all_data[arm].append(observation)
                algo.update(arm, observation)
                
                nb_found = len(algo.S_t.intersection(true_positives))
                current_tpr = nb_found / len(true_positives) if true_positives else 1.0
                run_tpr.append(current_tpr)

        # TPR aggregation
        tpr_i = np.array(run_tpr)
        tpr_list.append(tpr_i)
        tpr_history_sum += tpr_i
        
        # Counts aggregation (ensure we take the first 'horizon+1' elements)
        # (horizon + 1 because there is the initial state at t=0)
        counts_arr = np.array(algo.counts_evolution)[:horizon+1]
        counts_list.append(counts_arr)
        counts_evolution_sum += counts_arr

    tpr_history_mean = tpr_history_sum / n_simulations
    counts_history_mean = counts_evolution_sum / n_simulations

    return tpr_history_mean, tpr_list, counts_history_mean, counts_list

# -----------------------------------------------------------------------------
# PART 3: CONFIGURATION AND EXECUTION
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    from pathlib import Path

    def find_git_root(start: Path | None = None) -> Path:
        p = (start or Path(__file__)).resolve()
        for parent in [p, *p.parents]:
            git_entry = parent / ".git"
            if git_entry.is_dir() or git_entry.is_file():  # support worktree (.git file)
                return parent
        raise RuntimeError("Git root not found (no .git in parents)")

    git_root = find_git_root()
    plt.close('all')
    
    # Scenario: 2 good arms (0, 1) and 2 bad ones (2, 3)
    mu_0 = 0.0
    delta = 0.05
    horizon = 800
    n_sims = 20
    
    true_means = np.array([0.5, 0.5, 0.35, 0.35, 0.0, 0.0])
    n_arms = len(true_means)
    
    # 1. Run Simulations
    tpr_unif, _, counts_unif_mean, counts_unif_list, all_data = run_experiment(true_means, horizon, 'uniform', n_sims)
    tpr_adapt, _, counts_adapt_mean, counts_adapt_list, all_data = run_experiment(true_means, horizon, 'adaptive', n_sims)
    
    # --- PLOT 1: TPR ---
    plt.figure(1, figsize=(10, 5))
    plt.plot(tpr_adapt, label='Adaptive', color='#ff7f0e', linewidth=2)
    plt.plot(tpr_unif, label='Uniform', color='#1f77b4', linestyle='--')
    plt.axhline(y=1.0, color='gray', linestyle=':')
    plt.title("Discovery speed (TPR)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(git_root / "figure_real_time/figure1.png", dpi=300, bbox_inches="tight")


    # --- PLOT 2: PULL EVOLUTION ---
    plt.figure(2, figsize=(12, 6))
    
    # Subplot 1: Uniform
    plt.subplot(1, 2, 1)
    plt.title("Uniform: Number of pulls per arm")
    for arm_idx in range(n_arms):
        label = f"Arm {arm_idx} ($mu$={true_means[arm_idx]})"
        plt.plot(counts_unif_mean[:, arm_idx], label=label, linewidth=2)
    plt.xlabel("Time (t)")
    plt.ylabel("Number of pulls ($T_i(t)$)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Subplot 2: Adaptive
    plt.subplot(1, 2, 2)
    plt.title("Adaptive: Number of pulls per arm")
    for arm_idx in range(n_arms):
        linestyle = '-' if true_means[arm_idx] > mu_0 else '--'
        label = f"Arm {arm_idx} ($mu$={true_means[arm_idx]})"
        plt.plot(counts_adapt_mean[:, arm_idx], label=label, linewidth=2, linestyle=linestyle)
        
    plt.xlabel("Time (t)")
    plt.ylabel("Number of pulls ($T_i(t)$)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(git_root / "figure_real_time/figure2.png", dpi=300, bbox_inches="tight")


    # --- PLOT 3: PULL EVOLUTION (SPAGHETTI PLOT) ---
    plt.figure(3, figsize=(14, 6))
    plt.title(f"Adaptive: Number of pulls per arm ({n_sims} simulations)")
    
    for arm_idx in range(n_arms):
        color = f'C{arm_idx}' 
        linestyle = '-' if true_means[arm_idx] > mu_0 else '--'
        label = f"Arm {arm_idx} ($mu$={true_means[arm_idx]})"
        
        for sim_counts in counts_adapt_list:
            plt.plot(sim_counts[:, arm_idx], 
                     color=color, 
                     alpha=0.15,
                     linewidth=0.8,
                     linestyle=linestyle)

        plt.plot(counts_adapt_mean[:, arm_idx], 
                 label=label, 
                 color=color, 
                 linewidth=2.5,
                 linestyle=linestyle)
        
    plt.xlabel("Time (t)")
    plt.ylabel("Number of pulls ($T_i(t)$)")
    plt.legend(loc='upper left')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(git_root / "figure_real_time/figure3.png", dpi=300, bbox_inches="tight")