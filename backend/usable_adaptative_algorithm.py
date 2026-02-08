import numpy as np
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

def prepare_experiment(true_means, horizon, n_sims):
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
                observation = np.random.normal(loc=true_means[arm], scale=1.0) # set the seed?
                result_arm.append(observation)
            all_arm_data.append(result_arm)
        all_arm_data_by_sim.append(all_arm_data)
    return all_arm_data_by_sim
        
def run_experiment(arms, mu_0, delta, horizon, mode, all_arm_data, n_simulations):
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
    pnb_history_mean : ndarray
        Average True Positive Rate over time.
    pnb_list : list
        History of pr for each individual simulation.
    counts_history_mean : ndarray
        Average pull counts per arm over time.
    counts_list : list
        History of pull counts for each individual simulation.
    """
    n_arms = len(arms)
    
    pnb_history_sum = np.zeros(horizon)
    pnb_list = []
    
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
        
        run_pr = []
        
        for t in range(horizon):
            arm = algo.select_arm()
            
            if arm == "stop":
                # If we stop before the end, we fill the lists with the last value
                # so that arrays have the correct size (horizon)
                last_pr = run_pr[-1] if run_pr else 1.0
                remaining_steps = horizon - len(run_pr)
                run_pr.extend([last_pr] * remaining_steps)
                
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
                
                algo.update(arm, observation)
                
                nb_found = len(algo.S_t)
                current_pr = nb_found
                run_pr.append(current_pr)

        # pr aggregation
        pnb_i = np.array(run_pr)
        pnb_list.append(pnb_i)
        pnb_history_sum += pnb_i
        
        # Counts aggregation (ensure we take the first 'horizon+1' elements)
        # (horizon + 1 because there is the initial state at t=0)
        counts_arr = np.array(algo.counts_evolution)[:horizon+1]
        counts_list.append(counts_arr)
        counts_evolution_sum += counts_arr

    pnb_history_mean = pnb_history_sum / n_simulations
    counts_history_mean = counts_evolution_sum / n_simulations

    return pnb_history_mean, pnb_list, counts_history_mean, counts_list

