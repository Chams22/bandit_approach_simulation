import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy.optimize import brentq

np.random.seed(1234)

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
mu_0 = 0.0
delta = 0.05 
horizon = 800

def prepare_experiment(true_means, horizon, n_sims, scale):
    """
    Pre-generates all random observations for the experiment to ensure consistency.

    This function creates a 3D dataset of random rewards based on the true means.
    Structure: [simulation_index][arm_index][time_step]

    Parameters
    ----------
    true_means : array-like
        The true expected values (means) for each arm.
    horizon : int
        The total number of time steps (budget).
    n_sims : int
        The number of simulations to run.

    Returns
    -------
    list
        A nested list containing pre-generated observations for every simulation, 
        arm, and time step.
    """
    n_arms = len(true_means)
    all_arm_data_by_sim=[]
    for sim in range(n_sims):
        all_arm_data=[]   
        for arm in range(n_arms):
            result_arm=[]
            for t in range(horizon):
                observation = np.random.normal(loc=true_means[arm], scale=scale) # set the seed?
                result_arm.append(observation)
            all_arm_data.append(result_arm)
        all_arm_data_by_sim.append(all_arm_data)
    return all_arm_data_by_sim
        
def run_experiment(true_means, horizon, mode, all_arm_data, n_simulations=20, mu_0=0.0):
    """
    Runs the bandit experiment using pre-generated data for consistency.

    This function executes the specified algorithm (Adaptive or Uniform) over a set of 
    simulations. Crucially, instead of generating random rewards on the fly, it consumes 
    pre-generated observations from `all_arm_data`. This ensures that both algorithms 
    are tested against the exact same sequence of random events (variance reduction), 
    making the comparison strictly fair.

    Parameters
    ----------
    true_means : array-like
        The true expected values (means) for each arm.
    horizon : int
        The total budget (max time steps) for each simulation.
    mode : str
        The algorithm to run: 'adaptive' or 'uniform'.
    all_arm_data : list of list of list
        A 3D structure containing pre-generated rewards.
        Shape: [simulation_index][arm_index][pull_count]
        This allows the function to "replay" specific random draws.
    n_simulations : int, default=20
        The number of independent runs to execute.

    Returns
    -------
    tpr_history_mean : ndarray
        Average True Positive Rate over time.
    tpr_list : list
        History of TPR for each individual simulation.
    counts_history_mean : ndarray
        Average pull counts per arm over time.
    counts_list : list
        History of pull counts for each individual simulation.
    """
    n_arms = len(true_means)
    true_positives = [i for i, m in enumerate(true_means) if m > mu_0]
    
    tpr_history_sum = np.zeros(horizon)
    tpr_list = []
    
    # Store the AVERAGE number of pulls at each time t
    counts_evolution_sum = np.zeros((horizon + 1, n_arms))
    counts_list=[]

    print(f"Simulation Mode: {mode.upper()} ({n_simulations} runs)")
    
    for no_sim in tqdm(range(n_simulations)):
        # Track how many times we have pulled each arm *in this specific simulation*
        # This is necessary to fetch the correct next value from the pre-generated list.
        all_arm_counts = [0 for _ in range(n_arms)]

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
                # Fetch the next pre-generated observation for this specific arm in this simulation
                observation = all_arm_data[no_sim][arm][all_arm_counts[arm]]
                
                # print(mode, "arm:", arm, "count:", all_arm_counts[arm])
                
                # Increment the local counter so the next pull gets the next value
                all_arm_counts[arm]+=1
                
                #algo.bh_update(arm, observation)
                algo.bh_update_optimized(arm, observation)
                
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
""" from pathlib import Path
from typing import Optional

def find_git_root(start: Optional[Path] = None) -> Path:
    p = (start or Path(__file__ if "__file__" in locals() else Path.cwd())).resolve()
    for parent in [p, *p.parents]:
        git_entry = parent / ".git"
        if git_entry.is_dir() or git_entry.is_file():
            return parent
    return p  # Retourne le dossier actuel si .git n'est pas trouvé

if __name__ == "__main__":
    git_root = find_git_root()
    save_dir = git_root / "figure_reconstitutive"
    save_dir.mkdir(parents=True, exist_ok=True)

    # Liste de vos scénarios (on fait varier n_sims ici)
    scenarios = [1000, 100, 20]
    scale=0.1
    mu_0 = 0.0
    delta = 0.05
    horizon = 800
    for sims in scenarios:
        print(f"Running scenario with n_sims = {sims}...")
        
        # 1. Préparation et Simulation
        true_means = np.array([0.5, 0.5, 0.35, 0.35, 0.0, 0.0])
        n_arms = len(true_means)
        horizon = 800
        mu_0 = 0.0
        
        all_arm_data = prepare_experiment(true_means, horizon, sims, scale)
        tpr_unif, _, counts_unif_mean, counts_unif_list = run_experiment(true_means, horizon, 'uniform', all_arm_data, sims)
        tpr_adapt, _, counts_adapt_mean, counts_adapt_list = run_experiment(true_means, horizon, 'adaptive', all_arm_data, sims)

        # --- PLOT 1: TPR ---
        plt.figure(figsize=(10, 5)) # Pas de numéro fixe pour éviter les conflits
        plt.plot(tpr_adapt, label='Adaptive', color='#ff7f0e', linewidth=2)
        plt.plot(tpr_unif, label='Uniform', color='#1f77b4', linestyle='--')
        plt.title(f"Discovery speed (TPR) - n_sims: {sims}")
        plt.legend()
        # On ajoute {sims} dans le nom du fichier
        plt.savefig(save_dir / f"tpr_sims_{sims}.png")
        plt.close() # TRÈS IMPORTANT : ferme la figure pour libérer la mémoire

        # --- PLOT 2: PULLS ---
        
        plt.figure(figsize=(12, 6))
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
            
            plt.savefig(save_dir / f"pulls_sims_{sims}.png")
            plt.close()

        # --- PLOT 3: SPAGHETTI ---
        plt.figure(3, figsize=(14, 6))
        plt.title(f"Adaptive: Number of pulls per arm ({sims} simulations)")
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
            
            print("Displaying plots...")
            plt.savefig(save_dir / f"spaghetti_sims_{sims}.png")
            plt.close()
            
            
# parameters
delta_base = 0.05
n_sims = 100
horizon = 800

# different scenarios
configs = [
    {"name": "Delta_Sigma1",   "delta": delta_base,      "sigma": 1.0},
    {"name": "DeltaSur4_Sigma1", "delta": delta_base / 4, "sigma": 1.0},
    {"name": "Delta_Sigma2",   "delta": delta_base,      "sigma": 1.414}, # sqrt(2) bc N(mu, 2)
    {"name": "DeltaSur4_Sigma2", "delta": delta_base / 4, "sigma": 1.414}
]


# Dossier de sauvegarde
git_root = find_git_root()
save_dir = git_root / "figure_reconstitutive" / "task_normal_params"
save_dir.mkdir(parents=True, exist_ok=True)


for conf in configs:
    print(f"\n>>> Test en cours : {conf['name']} (Delta={conf['delta']:.4f}, Sigma={conf['sigma']:.3f})")
    
    # 1. Ajustement des moyennes (Signal)
    # On définit 2 bras "top" (0.5 + delta), 2 bras "moyens" (0.5), et 2 bras "nuls" (0.0)
    current_means = np.array([0.5 + conf['delta'], 0.5 + conf['delta'], 0.5, 0.5, 0.0, 0.0])
    n_arms = len(current_means)
    
    # 2. Préparation des données (Bruit)
    # On passe bien la 'scale' ici pour simuler la variance de la loi normale
    all_arm_data = prepare_experiment(current_means, horizon, n_sims, scale=conf['sigma'])
    
    # 3. Exécution des simulations
    tpr_unif, _, counts_unif_mean, _ = run_experiment(current_means, horizon, 'uniform', all_arm_data, n_sims)
    tpr_adapt, _, counts_adapt_mean, counts_adapt_list = run_experiment(current_means, horizon, 'adaptive', all_arm_data, n_sims)
    
    # --- GRAPHIQUE 1 : TPR (Vitesse de découverte) ---
    plt.figure(figsize=(10, 5))
    plt.plot(tpr_adapt, label='Adaptive', color='#ff7f0e', linewidth=2)
    plt.plot(tpr_unif, label='Uniform', color='#1f77b4', linestyle='--')
    plt.axhline(y=1.0, color='gray', linestyle=':')
    plt.title(f"TPR - Scénario: {conf['name']}")
    plt.xlabel("Temps")
    plt.ylabel("Taux de Vrais Positifs")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(save_dir / f"tpr_{conf['name']}.png")
    plt.close()

    # --- GRAPHIQUE 2 : PULLS (Répartition des tirages) ---
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.title(f"Uniform ({conf['name']})")
    for i in range(n_arms):
        plt.plot(counts_unif_mean[:, i], label=f"Arm {i} (μ={current_means[i]:.3f})")
    
    plt.subplot(1, 2, 2)
    plt.title(f"Adaptive ({conf['name']})")
    for i in range(n_arms):
        linestyle = '-' if current_means[i] > mu_0 else '--'
        plt.plot(counts_adapt_mean[:, i], label=f"Arm {i}", linestyle=linestyle)
    
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(save_dir / f"pulls_{conf['name']}.png")
    plt.close()

print(f"\n Les fichiers sont dans : {save_dir}")
 """
#-------------------------------------------------------------------------------------------
# task 1 et 2
#-------------------------------------------------------------------------------------------


import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Optional

def find_git_root(start: Optional[Path] = None) -> Path:
    p = (start or Path(__file__ if "__file__" in locals() else Path.cwd())).resolve()
    for parent in [p, *p.parents]:
        git_entry = parent / ".git"
        if git_entry.is_dir() or git_entry.is_file():
            return parent
    return p 

def run_task_1():
    print("Exécution de la Tâche 1...")
    git_root = find_git_root()
    save_dir = git_root / "figure_reconstitutive"
    save_dir.mkdir(parents=True, exist_ok=True)

    scenarios = [1000, 100, 20]
    scale = 0.1
    mu_0 = 0.0
    delta = 0.05
    horizon = 800

    for sims in scenarios:
        print(f"Running scenario with n_sims = {sims}...")
        true_means = np.array([0.5, 0.5, 0.35, 0.35, 0.0, 0.0])
        n_arms = len(true_means)
        
        all_arm_data = prepare_experiment(true_means, horizon, sims, scale)
        tpr_unif, _, counts_unif_mean, _ = run_experiment(true_means, horizon, 'uniform', all_arm_data, sims)
        tpr_adapt, _, counts_adapt_mean, counts_adapt_list = run_experiment(true_means, horizon, 'adaptive', all_arm_data, sims)

        # --- PLOT 1: TPR ---
        plt.figure(figsize=(10, 5))
        plt.plot(tpr_adapt, label='Adaptive', color='#ff7f0e', linewidth=2)
        plt.plot(tpr_unif, label='Uniform', color='#1f77b4', linestyle='--')
        plt.title(f"TPR - n_sims: {sims}")
        plt.legend()
        plt.savefig(save_dir / f"tpr_sims_{sims}.png")
        plt.close()

        # --- PLOT 2: PULLS ---
        plt.figure(figsize=(12, 6))
        plt.subplot(1, 2, 1)
        for arm_idx in range(n_arms):
            plt.plot(counts_unif_mean[:, arm_idx], label=f"Arm {arm_idx}", linewidth=2)
        plt.legend()
        
        plt.subplot(1, 2, 2)
        for arm_idx in range(n_arms):
            linestyle = '-' if true_means[arm_idx] > mu_0 else '--'
            plt.plot(counts_adapt_mean[:, arm_idx], label=f"Arm {arm_idx}", linewidth=2, linestyle=linestyle)
        plt.legend()
        plt.savefig(save_dir / f"pulls_sims_{sims}.png")
        plt.close()

        # --- PLOT 3: SPAGHETTI ---
        plt.figure(figsize=(14, 6))
        for arm_idx in range(n_arms):
            color = f'C{arm_idx}'
            linestyle = '-' if true_means[arm_idx] > mu_0 else '--'
            for sim_counts in counts_adapt_list:
                plt.plot(sim_counts[:, arm_idx], color=color, alpha=0.1, linewidth=0.5, linestyle=linestyle)
            plt.plot(counts_adapt_mean[:, arm_idx], color=color, linewidth=2, linestyle=linestyle)
        plt.savefig(save_dir / f"spaghetti_sims_{sims}.png")
        plt.close()

n_sims = 100
def run_task_2():
    print(f" Simultation lois normales")
    
    # Configuration des dossiers
    git_root = find_git_root()
    save_dir = git_root / "figure_reconstitutive" / "task_normal_params"
    save_dir.mkdir(parents=True, exist_ok=True)

    # Les 4 scénarios demandés
    # N(mu, 2) signifie Variance = 2, donc Sigma = sqrt(2) ≈ 1.414
    configs = [
        {"name": "Delta_Sigma1",      "d": delta,     "s": 1.0},
        {"name": "DeltaSur4_Sigma1",  "d": delta / 4, "s": 1.0},
        {"name": "Delta_Sigma2",      "d": delta,     "s": 2.0},
        {"name": "DeltaSur4_Sigma2",  "d": delta / 4, "s": 2.0},
        {"name": "DeltaSur2_Sigma0.02",  "d": delta / 2, "s": 0.02},
        {"name": "Delta_Sigma0.02",  "d": delta , "s": 0.02}
    ]

    for conf in configs:
        print(f"\n--- Scénario : {conf['name']} (Sigma={conf['s']:.3f}) ---")
        
        # Définition des moyennes pour ce scénario
        # 2 bras gagnants, 2 bras neutres, 2 bras nuls
        current_means = np.array([ conf['d'], conf['d'], 0.0, 0.0])
        n_arms = len(current_means)
        
        # Génération des données avec la 'scale' (sigma) spécifique
        all_arm_data = prepare_experiment(current_means, horizon, n_sims, scale=conf['s'])
        
        # Exécution des algos
        # IMPORTANT : On récupère bien 'counts_adapt_list' pour le spaghetti plot
        _, _, counts_unif_mean, _ = run_experiment(current_means, horizon, 'uniform', all_arm_data, n_sims)
        tpr_adapt, _, counts_adapt_mean, counts_adapt_list = run_experiment(current_means, horizon, 'adaptive', all_arm_data, n_sims)
        
        # --- GRAPHIQUE 1 : TPR ---
        plt.figure(figsize=(10, 5))
        plt.plot(tpr_adapt, label='Adaptive', color='#ff7f0e', linewidth=2)
        plt.axhline(y=1.0, color='gray', linestyle=':', alpha=0.5)
        plt.title(f"TPR - {conf['name']}")
        plt.legend()
        plt.savefig(save_dir / f"tpr_{conf['name']}.png")
        plt.close()

        # --- GRAPHIQUE 2 : PULLS MOYENS ---
        plt.figure(figsize=(12, 6))
        plt.subplot(1, 2, 2)
        plt.title(f"Adaptive Mean Pulls ({conf['name']})")
        for i in range(n_arms):
            ls = '-' if current_means[i] > mu_0 else '--'
            plt.plot(counts_adapt_mean[:, i], label=f"Arm {i} (μ={current_means[i]:.3f})", linestyle=ls)
        plt.legend(fontsize='small')
        plt.savefig(save_dir / f"pulls_{conf['name']}.png")
        plt.close()

        # --- GRAPHIQUE 3 : SPAGHETTI PLOT (Celui que tu voulais ajouter) ---
        plt.figure(figsize=(14, 6))
        plt.title(f"Adaptive: Individual Simulations ({conf['name']})")
        
        for arm_idx in range(n_arms):
            color = f'C{arm_idx}' 
            ls = '-' if current_means[arm_idx] > mu_0 else '--'
            label = f"Arm {arm_idx} (μ={current_means[arm_idx]:.3f})"
            
            # Tracé des lignes individuelles (fines et transparentes)
            for sim_counts in counts_adapt_list:
                plt.plot(sim_counts[:, arm_idx], color=color, alpha=0.1, linewidth=0.5, linestyle=ls)

            # Tracé de la moyenne (épaisse)
            plt.plot(counts_adapt_mean[:, arm_idx], label=label, color=color, linewidth=2.5, linestyle=ls)
            
        plt.xlabel("Time (t)")
        plt.ylabel("Cumulative Pulls")
        plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(save_dir / f"spaghetti_{conf['name']}.png", dpi=300, bbox_inches="tight")
        plt.close()

    print(f"\n Terminé ! Résultats dans : {save_dir}")

if __name__ == "__main__":
    # run_task_1() 
    run_task_2()