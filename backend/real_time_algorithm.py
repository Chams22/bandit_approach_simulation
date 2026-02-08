import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy.optimize import brentq

# -----------------------------------------------------------------------------
# PART 1: THE ALGORITHM
# -----------------------------------------------------------------------------
class JamiesonJainAlgo:
    def __init__(self, n_arms, mu_0, delta):
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
        """
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
        """
        Calculates the "Anytime" Confidence Interval width.

        This function implements the specific bound based on the Law of the Iterated Logarithm 
        cited in the paper (Kaufmann et al.), ensuring the interval is valid for all time steps.

        Parameters
        ----------
        t : int
            The number of times the specific arm has been pulled.
        delta_val : float
            The specific confidence level to use (which may vary during BH procedure).

        Returns
        -------
        float
            The width of the confidence interval. Returns infinity if t=0 to force exploration.
        """
        if t == 0: return float('inf')
        num = 2 * np.log(1/delta_val) + 6 * np.log(np.log(1/delta_val) + 1e-10) + \
              3 * np.log(np.log(np.e * t / 2) + 1e-10)
        return np.sqrt(num / t)

    def select_arm(self):
        """
        Determines which arm to pull next based on the UCB strategy.

        Strategy:
        1. If t < n, pull every arm once for initialization.
        2. Identify candidate arms (those NOT yet in the discovery set S_t).
        3. If no candidates remain, return "stop".
        4. Otherwise, select the candidate with the highest Upper Confidence Bound (UCB).

        Returns
        -------
        int or str
            The index of the arm to pull, or "stop" if all arms are discovered.
        """
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

    def get_anytime_pvalue(self, arm_idx):
        """
        Calcule la p-value anytime pour un bras donné.
        Résout l'équation : emp_mean - phi(count, p) = mu_0
        """
        t = self.counts[arm_idx]
        mean = self.emp_means[arm_idx]
        
        # 1. Si on n'a pas encore joué le bras, p-value = 1.0 (incertitude totale)
        if t == 0:
            return 1.0
        
        # 2. Si la moyenne est inférieure à mu_0, on ne peut pas rejeter H0 (on cherche des effets positifs)
        # La p-value est donc 1.0 (ou très proche)
        diff = mean - self.mu_0
        if diff <= 0:
            return 1.0

        # 3. Fonction à résoudre : phi(t, p) - (mean - mu_0) = 0
        # On cherche p tel que la largeur de l'intervalle soit égale à la distance à mu_0
        def objective(p):
            # Attention : p ne doit pas être 0 ou 1 pour éviter les erreurs de log
            if p <= 0: return float('inf') 
            if p >= 1: return -float('inf')
            return self.phi(t, p) - diff

        try:
            # On cherche p entre une valeur très petite (ex: 1e-10) et 1.0
            p_value = brentq(objective, 1e-12, 0.9999)
            return p_value
        except ValueError:
            # Si brentq échoue (cas rares aux limites), on renvoie 1.0 par prudence
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
        
        Even for the uniform algorithm, we use the same statistical confidence bound 
        to validate discoveries (Inference Rule).
        
        Parameters
        ----------
        t : int
            Number of pulls.
        delta_val : float
            Confidence level.
            
        Returns
        -------
        float
            Confidence interval width.
        """
        if t == 0: return float('inf')
        num = 2 * np.log(1/delta_val) + 6 * np.log(np.log(1/delta_val) + 1e-10) + \
              3 * np.log(np.log(np.e * t / 2) + 1e-10)
        return np.sqrt(num / t)

    def select_arm(self):
        """
        Selects the next arm using a Round-Robin strategy.
        
        Logic:
        Simply rotates through arms 0, 1, 2, ..., n-1, 0, ...
        
        Returns
        -------
        int
            The index of the arm to pull.
        """
        if self.time < self.n:
            return self.time
        return self.time % self.n

    def update(self, arm_idx, observation):
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
        Calcule la p-value anytime pour un bras donné.
        Résout l'équation : emp_mean - phi(count, p) = mu_0
        """
        t = self.counts[arm_idx]
        mean = self.emp_means[arm_idx]
        
        # 1. Si on n'a pas encore joué le bras, p-value = 1.0 (incertitude totale)
        if t == 0:
            return 1.0
        
        # 2. Si la moyenne est inférieure à mu_0, on ne peut pas rejeter H0 (on cherche des effets positifs)
        # La p-value est donc 1.0 (ou très proche)
        diff = mean - self.mu_0
        if diff <= 0:
            return 1.0

        # 3. Fonction à résoudre : phi(t, p) - (mean - mu_0) = 0
        # On cherche p tel que la largeur de l'intervalle soit égale à la distance à mu_0
        def objective(p):
            # Attention : p ne doit pas être 0 ou 1 pour éviter les erreurs de log
            if p <= 0: return float('inf') 
            if p >= 1: return -float('inf')
            return self.phi(t, p) - diff

        try:
            # On cherche p entre une valeur très petite (ex: 1e-10) et 1.0
            p_value = brentq(objective, 1e-12, 0.9999)
            return p_value
        except ValueError:
            # Si brentq échoue (cas rares aux limites), on renvoie 1.0 par prudence
            return 1.0


# -----------------------------------------------------------------------------
# PART 2: SIMULATION ENGINE
# -----------------------------------------------------------------------------

def run_experiment(true_means, horizon, mode, n_simulations=20):
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

    print(f"Simulation Mode: {mode.upper()} ({n_simulations} runs)")
    
    for no_sim in tqdm(range(n_simulations)):
        all_data=[[] for i in range(n_arms)]

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