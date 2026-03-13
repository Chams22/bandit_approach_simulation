import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy.optimize import brentq
np.random.seed(1234)

# -----------------------------------------------------------------------------
# PART 1: THE ALGORITHM
# -----------------------------------------------------------------------------
class JamiesonJainAlgo:
    def __init__(self, n_arms, mu_0, delta, init_pulls=1):
        """
        Initializes the adaptive bandit algorithm.

        Parameters
        ----------
        n_arms : int
            The total number of arms (distributions) available.
        mu_0 : float
            The baseline threshold. We want to identify arms with mean > mu_0.
        delta : float
            The confidence level / False Discovery Rate (FDR) parameter (e.g., 0.05).
        init_pulls : int, optional
            Number of initial pulls per arm before running adaptive phase (default=1).
        """
        self.n = n_arms
        self.mu_0 = mu_0
        self.delta = delta
        self.init_pulls = init_pulls
        
        self.counts = np.zeros(n_arms, dtype=int)
        self.emp_means = np.zeros(n_arms, dtype=float)
        self.time = 0
        self.S_t = set()
        
        # History for visualization
        # Initialized with zeros for t=0
        self.counts_evolution = [np.zeros(n_arms, dtype=int)]

    def phi(self, t, delta_val):
        """
        Calculates the "Anytime" Confidence Interval width.
        
        This bound is based on the Law of the Iterated Logarithm (LIL), 
        ensuring the confidence interval remains valid at all time steps.

        Parameters
        ----------
        t : int
            Number of times the specific arm has been pulled.
        delta_val : float
            The confidence level (or p-value during BH procedure).

        Returns
        -------
        float
            The width of the confidence interval.
        """
        if t == 0: 
            return float('inf')
        
        # Calculate the numerator based on the LIL concentration inequality
        num = 2 * np.log(1/delta_val) + 6 * np.log(np.log(1/delta_val) + 1e-10) + \
              3 * np.log(np.log(np.e * t / 2) + 1e-10)
              
        # SAFETY FIX: Prevent the numerator from becoming negative.
        # This occurs when delta_val approaches 1.0 (e.g., during p-value root-finding),
        # as the logarithmic terms can result in a negative sum.
        num = max(0.0, num)
        
        return np.sqrt(num / t)

    def select_arm(self):
        """
        Determines which arm to pull next based on the UCB strategy.

        Strategy:
        1. During initialization phase, cycle through arms.
        2. Identify candidate arms (those NOT yet in the discovery set S_t).
        3. If no candidates remain, return "stop".
        4. Otherwise, select the candidate with the highest Upper Confidence Bound (UCB).

        Returns
        -------
        int or str
            The index of the arm to pull, or "stop" if all arms are discovered.
        """
        # Initialization phase: cycle through arms
        init_phase_end = self.n * self.init_pulls
        if self.time < init_phase_end:
            return self.time % self.n
        
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

    def bh_update(self, arm_idx, observation):
        """
        Updates the algorithm's state with a new observation and runs the decision procedure.

        Functionality:
        1. Updates the empirical mean and pull count for the pulled arm.
        2. Saves the current pull counts to history.
        3. Runs the Benjamini-Hochberg (BH) procedure using "Anytime" p-values (via LCB)
           to determine which arms can be added to the discovery set S_t.

        Parameters
        ----------
        arm_idx : int
            The index of the arm that was pulled.
        observation : float
            The reward/value observed from the arm.
        """
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

    def bh_update_optimized(self, arm_idx, observation):
        """
        Updates the algorithm's state with a new observation and runs the decision procedure.
        
        Optimized version: Uses sorted anytime p-values to run the Benjamini-Hochberg
        procedure in O(n log n) time instead of O(n^2).
        
        Parameters
        ----------
        arm_idx : int
            The index of the arm that was pulled.
        observation : float
            The reward/value observed from the arm.
        """
        # 1. Mise à jour des statistiques du bras tiré
        n_pulls = self.counts[arm_idx]
        self.emp_means[arm_idx] = (self.emp_means[arm_idx] * n_pulls + observation) / (n_pulls + 1)
        self.counts[arm_idx] += 1
        self.time += 1
        self.counts_evolution.append(self.counts.copy()) 
        
        # 2. Calcul des p-values anytime pour tous les bras
        # On stocke des tuples : (p_value, index_du_bras)
        p_values_with_idx = [(self.get_anytime_pvalue(i), i) for i in range(self.n)]
        
        # 3. Tri des p-values par ordre croissant (Complexité : O(n log n))
        p_values_with_idx.sort(key=lambda x: x[0])
        
        # 4. Procédure de Benjamini-Hochberg classique
        # On cherche le plus grand k tel que p_(k) <= delta * k / n
        current_St = set()
        
        for k in range(self.n, 0, -1):
            # Attention : les listes Python commencent à l'index 0
            # Le k-ième élément est donc à l'index k - 1
            p_val_k = p_values_with_idx[k - 1][0]
            effective_delta = self.delta * k / self.n
            
            # Dès qu'on trouve le plus grand k qui valide la condition
            if p_val_k <= effective_delta:
                # On ajoute tous les bras du rang 1 au rang k à notre ensemble
                for rank in range(k):
                    discovered_arm_idx = p_values_with_idx[rank][1]
                    current_St.add(discovered_arm_idx)
                break # On a trouvé le max k, on arrête la boucle
                
        # 5. Mise à jour de l'ensemble global des découvertes
        self.S_t.update(current_St)

    def get_anytime_pvalue(self, arm_idx):
        """
        Calculates the anytime p-value for a given arm.
        Solves the equation: emp_mean - phi(count, p) = mu_0

        Parameters
        ----------
        arm_idx : int
            The index of the arm for which to calculate the p-value.

        Returns
        -------
        float
            The calculated anytime p-value.
        """
        t = self.counts[arm_idx]
        mean = self.emp_means[arm_idx]
        
        # 1. If the arm hasn't been played yet, p-value = 1.0 (total uncertainty)
        if t == 0:
            return 1.0
        
        # 2. If the mean is less than or equal to mu_0, we cannot reject H0 (looking for positive effects)
        # The p-value is therefore 1.0
        diff = mean - self.mu_0
        if diff <= 0:
            return 1.0

        # 3. Function to solve: phi(t, p) - (mean - mu_0) = 0
        # We look for p such that the confidence interval width equals the distance to mu_0
        def objective(p):
            # Note: p must not be 0 or 1 to avoid log errors
            if p <= 0: return float('inf') 
            if p >= 1: return -float('inf')
            return self.phi(t, p) - diff

        try:
            # Search for p between a very small value (e.g., 1e-12) and 0.9999
            p_value = brentq(objective, 1e-12, 0.9999)
            return p_value
        except ValueError:
            # If brentq fails (rare edge cases), return 1.0 as a precaution
            return 1.0

class UniformAlgo:
    def __init__(self, n_arms, mu_0, delta):
        """
        Initializes the Uniform (Round-Robin) sampling algorithm.
        
        Parameters
        ----------
        n_arms : int
            The total number of arms.
        mu_0 : float
            The baseline threshold.
        delta : float
            The confidence parameter.
        """
        self.n = n_arms
        self.mu_0 = mu_0
        self.delta = delta
        
        self.counts = np.zeros(n_arms, dtype=int)
        self.emp_means = np.zeros(n_arms, dtype=float)
        self.time = 0
        self.S_t = set()
        
        # Historique pour visualisation
        # On initialise avec des zéros pour t=0
        self.counts_evolution = [np.zeros(n_arms, dtype=int)]

    def phi(self, t, delta_val):
        """
        Calculates the "Anytime" Confidence Interval width.
        
        This bound is based on the Law of the Iterated Logarithm (LIL), 
        ensuring the confidence interval remains valid at all time steps.

        Parameters
        ----------
        t : int
            Number of times the specific arm has been pulled.
        delta_val : float
            The confidence level (or p-value during BH procedure).

        Returns
        -------
        float
            The width of the confidence interval.
        """
        if t == 0: 
            return float('inf')
        
        # Calculate the numerator based on the LIL concentration inequality
        num = 2 * np.log(1/delta_val) + 6 * np.log(np.log(1/delta_val) + 1e-10) + \
              3 * np.log(np.log(np.e * t / 2) + 1e-10)
              
        # SAFETY FIX: Prevent the numerator from becoming negative.
        # This occurs when delta_val approaches 1.0 (e.g., during p-value root-finding),
        # as the logarithmic terms can result in a negative sum.
        num = max(0.0, num)
        
        return np.sqrt(num / t)

    def select_arm(self):
            """
            Selects the next arm uniformly at random.
            
            This avoids periodic sampling biases (unlike deterministic Round-Robin)
            and simulates a standard randomized controlled trial.
            
            Returns
            -------
            int
                The index of the arm to pull.
            """
            return np.random.randint(self.n)
    
    def bh_update_optimized(self, arm_idx, observation):
        """
        Updates the algorithm's state with a new observation and runs the decision procedure.
        
        Optimized version: Uses sorted anytime p-values to run the Benjamini-Hochberg
        procedure in O(n log n) time instead of O(n^2).
        
        Parameters
        ----------
        arm_idx : int
            The index of the arm that was pulled.
        observation : float
            The reward/value observed from the arm.
        """
        # 1. Mise à jour des statistiques du bras tiré
        n_pulls = self.counts[arm_idx]
        self.emp_means[arm_idx] = (self.emp_means[arm_idx] * n_pulls + observation) / (n_pulls + 1)
        self.counts[arm_idx] += 1
        self.time += 1
        self.counts_evolution.append(self.counts.copy()) 
        
        # 2. Calcul des p-values anytime pour tous les bras
        # On stocke des tuples : (p_value, index_du_bras)
        p_values_with_idx = [(self.get_anytime_pvalue(i), i) for i in range(self.n)]
        
        # 3. Tri des p-values par ordre croissant (Complexité : O(n log n))
        p_values_with_idx.sort(key=lambda x: x[0])
        
        # 4. Procédure de Benjamini-Hochberg classique
        # On cherche le plus grand k tel que p_(k) <= delta * k / n
        current_St = set()
        
        for k in range(self.n, 0, -1):
            # Attention : les listes Python commencent à l'index 0
            # Le k-ième élément est donc à l'index k - 1
            p_val_k = p_values_with_idx[k - 1][0]
            effective_delta = self.delta * k / self.n
            
            # Dès qu'on trouve le plus grand k qui valide la condition
            if p_val_k <= effective_delta:
                # On ajoute tous les bras du rang 1 au rang k à notre ensemble
                for rank in range(k):
                    discovered_arm_idx = p_values_with_idx[rank][1]
                    current_St.add(discovered_arm_idx)
                break # On a trouvé le max k, on arrête la boucle
                
        # 5. Mise à jour de l'ensemble global des découvertes
        self.S_t.update(current_St)

    def bh_update(self, arm_idx, observation):
        """
        Updates the state and checks for discoveries.
        
        Note:
        Although the sampling is uniform (dumb), the update/inference rule is 
        intelligent and identical to the adaptive algorithm (Benjamini-Hochberg 
        with Anytime bounds) to ensure a fair comparison of False Discovery Rate control.
        
        Parameters
        ----------
        arm_idx : int
            Index of the pulled arm.
        observation : float
            Observed reward.
        """
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

    def get_anytime_pvalue(self, arm_idx):
        """
        Calculates the anytime p-value for a given arm.
        Solves the equation: emp_mean - phi(count, p) = mu_0

        Parameters
        ----------
        arm_idx : int
            The index of the arm for which to calculate the p-value.

        Returns
        -------
        float
            The calculated anytime p-value.
        """
        t = self.counts[arm_idx]
        mean = self.emp_means[arm_idx]
        
        # 1. If the arm hasn't been played yet, p-value = 1.0 (total uncertainty)
        if t == 0:
            return 1.0
        
        # 2. If the mean is less than or equal to mu_0, we cannot reject H0 (looking for positive effects)
        # The p-value is therefore 1.0
        diff = mean - self.mu_0
        if diff <= 0:
            return 1.0

        # 3. Function to solve: phi(t, p) - (mean - mu_0) = 0
        # We look for p such that the confidence interval width equals the distance to mu_0
        def objective(p):
            # Note: p must not be 0 or 1 to avoid log errors
            if p <= 0: return float('inf') 
            if p >= 1: return -float('inf')
            return self.phi(t, p) - diff

        try:
            # Search for p between a very small value (e.g., 1e-12) and 0.9999
            p_value = brentq(objective, 1e-12, 0.9999)
            return p_value
        except ValueError:
            # If brentq fails (rare edge cases), return 1.0 as a precaution
            return 1.0



# -----------------------------------------------------------------------------
# PART 2: SIMULATION ENGINE
# -----------------------------------------------------------------------------

def run_experiment(true_means, horizon, mode, n_simulations=20, init_pulls=1): #attention mettre seed pour stabiliser
    """
    Runs Monte-Carlo simulations to evaluate the performance of the bandit algorithm.

    This function executes `n_simulations` independent runs of the specified algorithm 
    (adaptive or uniform) over a fixed time horizon. It tracks the True Positive Rate (TPR), 
    the number of pulls for each arm, and the raw observations collected.

    Parameters
    ----------
    true_means : array-like
        The true expected values (means) of the reward distributions for each arm. 
        Used to determine true positives (H1) where mean > mu_0.
    horizon : int
        The total budget or maximum number of time steps for the experiment.
    mode : str
        The sampling strategy to use: 'adaptive' (Jamieson & Jain) or 'uniform'.
    n_simulations : int, optional
        The number of independent Monte-Carlo simulations to run for averaging results 
        (default is 20).
    init_pulls : int, optional
        Number of initial pulls per arm (default is 1).

    Returns
    -------
    tpr_history_mean : ndarray
        The average True Positive Rate over time across all simulations.
    tpr_list : list of ndarray
        A list containing the TPR history for each individual simulation.
    counts_history_mean : ndarray
        The average number of pulls for each arm at each time step across all simulations.
    counts_list : list of ndarray
        A list containing the pull count history for each individual simulation.
    all_data_sim : list of list
        A nested list containing the raw observations (rewards) collected for each arm 
        during each simulation. Structure: [simulation_index][arm_index] -> list of values.
    """
    n_arms = len(true_means)
    all_data_sim=[]
    true_positives = [i for i, m in enumerate(true_means) if m > mu_0]
    
    tpr_history_sum = np.zeros(horizon)
    tpr_list = []
    
    # Store the AVERAGE number of pulls at each time t
    counts_evolution_sum = np.zeros((horizon + 1, n_arms))
    counts_list=[]

    print(f"Simulation Mode: {mode.upper()} ({n_simulations} runs, init_pulls={init_pulls})")
    
    for no_sim in tqdm(range(n_simulations)):
        all_data=[[] for i in range(n_arms)]

        if mode=='adaptive':
            algo = JamiesonJainAlgo(n_arms, mu_0, delta, init_pulls=init_pulls)
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
                #algo.bh_update(arm, observation)
                algo.bh_update_optimized(arm, observation)
                
                nb_found = len(algo.S_t.intersection(true_positives))
                current_tpr = nb_found / len(true_positives) if true_positives else 1.0
                run_tpr.append(current_tpr)
        all_data_sim.append(all_data)

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

    return tpr_history_mean, tpr_list, counts_history_mean, counts_list, all_data_sim

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
    
    # Test with 30 initial pulls
    print("\n" + "="*70)
    print("Testing with 30 initial pulls")
    print("="*70)
    tpr_adapt_30, _, counts_adapt_mean_30, counts_adapt_list_30, all_data = run_experiment(true_means, horizon, 'adaptive', n_sims, init_pulls=30)
    
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
    
    # --- PLOT 4: COMPARISON init_pulls=1 vs init_pulls=30 ---
    plt.figure(4, figsize=(12, 5))
    plt.plot(tpr_adapt, label='init_pulls=1', color='#1f77b4', linewidth=2)
    plt.plot(tpr_adapt_30, label='init_pulls=30', color='#ff7f0e', linewidth=2)
    plt.axhline(y=1.0, color='gray', linestyle=':', alpha=0.5)
    plt.title("Impact of Initialization: init_pulls=1 vs init_pulls=30")
    plt.xlabel("Time (t)")
    plt.ylabel("True Positive Rate (TPR)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(git_root / "figure_real_time/figure4_init_comparison.png", dpi=300, bbox_inches="tight")
    print("\nFigure 4 saved: figure4_init_comparison.png")
    
    # PART D: Test different proportions of true positives
    print("\n" + "="*70)
    print("PART D: Testing different proportions of true positives")
    print("="*70)
    
    # Define different proportions: (n_good, n_bad)
    proportions = [(1, 9), (9, 1), (5, 5), (4, 6), (6, 4)]
    results_by_proportion = {}
    
    for n_good, n_bad in proportions:
        print(f"\n--- Testing with {n_good} good arms and {n_bad} bad arms ---")
        
        # Create true_means: good arms with effect size 0.5, bad arms with 0
        total_arms = n_good + n_bad
        true_means_prop = np.concatenate([
            np.ones(n_good) * 0.5,    # good arms
            np.zeros(n_bad)            # bad arms
        ])
        
        tpr_adapt_prop, _, counts_adapt_prop, _, _ = run_experiment(
            true_means_prop, horizon, 'adaptive', n_sims
        )
        
        results_by_proportion[(n_good, n_bad)] = {
            'tpr': tpr_adapt_prop,
            'counts': counts_adapt_prop,
            'true_means': true_means_prop
        }
    
    # --- PLOT 5: TPR for different proportions ---
    plt.figure(5, figsize=(12, 7))
    
    colors = plt.cm.viridis(np.linspace(0, 1, len(proportions)))
    
    for idx, (n_good, n_bad) in enumerate(proportions):
        tpr = results_by_proportion[(n_good, n_bad)]['tpr']
        label = f'{n_good}G/{n_bad}B ({n_good}/{n_good+n_bad} positive)'
        plt.plot(tpr, label=label, color=colors[idx], linewidth=2)
    
    plt.axhline(y=1.0, color='gray', linestyle=':', alpha=0.5)
    plt.title("Impact of Proportion: Adaptive Algorithm TPR for Different Ratios")
    plt.xlabel("Time (t)")
    plt.ylabel("True Positive Rate (TPR)")
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(git_root / "figure_real_time/figure5_proportions.png", dpi=300, bbox_inches="tight")
    print("Figure 5 saved: figure5_proportions.png")
    
    # --- PLOT 6: Convergence time vs proportion ---
    plt.figure(6, figsize=(10, 6))
    
    convergence_times = []
    proportion_labels = []
    
    for n_good, n_bad in proportions:
        tpr = results_by_proportion[(n_good, n_bad)]['tpr']
        # Find time to reach TPR >= 0.95
        idx_95 = np.where(tpr >= 0.95)[0]
        time_95 = idx_95[0] if len(idx_95) > 0 else horizon
        convergence_times.append(time_95)
        proportion_labels.append(f'{n_good}G/{n_bad}B')
    
    bars = plt.bar(range(len(proportions)), convergence_times, color=colors, edgecolor='black', linewidth=1.5)
    plt.xticks(range(len(proportions)), proportion_labels, fontsize=11)
    plt.xlabel("Proportion (G=Good, B=Bad)", fontsize=12)
    plt.ylabel("Time to reach TPR ≥ 0.95 (steps)", fontsize=12)
    plt.title("Convergence Speed vs Proportion of True Positives")
    plt.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for bar, time in zip(bars, convergence_times):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(time)}', ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(git_root / "figure_real_time/figure6_convergence_by_proportion.png", dpi=300, bbox_inches="tight")
    print("Figure 6 saved: figure6_convergence_by_proportion.png")
    
    # --- Summary table ---
    print("\n" + "="*70)
    print("SUMMARY: Impact of Proportion on Algorithm Performance")
    print("="*70)
    print(f"{'Proportion':<15} {'Time to TPR≥95%':<20} {'Final TPR':<15}")
    print("-"*70)
    
    for n_good, n_bad in proportions:
        tpr = results_by_proportion[(n_good, n_bad)]['tpr']
        idx_95 = np.where(tpr >= 0.95)[0]
        time_95 = idx_95[0] if len(idx_95) > 0 else horizon
        final_tpr = tpr[-1]
        
        proportion_str = f'{n_good}G/{n_bad}B'
        print(f"{proportion_str:<15} {time_95:<20} {final_tpr:<15.4f}")
    
    print("="*70 + "\n")