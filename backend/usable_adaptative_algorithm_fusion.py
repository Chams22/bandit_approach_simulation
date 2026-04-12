import numpy as np
from tqdm import tqdm

# -----------------------------------------------------------------------------
# PART 1: THE ALGORITHM
# -----------------------------------------------------------------------------
class JamiesonJainAlgo:
    def __init__(self, n_arms, mu_0, delta, rho=1.0):
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
        rho : float
            Tuning parameter for the Normal Mixture confidence sequence (prior variance).
            Acts as a floor on interval width at small t. Default: 1.0.
        """
        self.n = n_arms
        self.mu_0 = mu_0
        self.delta = delta
        self.rho = rho
        
        self.counts = np.zeros(n_arms, dtype=int)
        self.emp_means = np.zeros(n_arms, dtype=float)
        self.emp_vars = np.zeros(n_arms, dtype=float)  # V_hat: sum of squared prediction errors
        self.time = 0
        self.S_t = set()
        
        # History for visualization
        # Initialized with zeros for t=0
        self.counts_evolution = [np.zeros(n_arms, dtype=int)]  # nb of draw of each arms

    def phi(self, t, delta_val, V_hat):
        """
        Normal Mixture confidence sequence radius (Howard et al., 2021).

        Valid for unbounded data with unknown variance. Uses the running
        sum of squared prediction errors V_hat (computed via Welford).

        Parameters
        ----------
        t : int
            Number of times the specific arm has been pulled.
        delta_val : float
            The confidence level (or p-value during BH procedure).
        V_hat : float
            Running sum of squared prediction errors: sum of (X_i - X_bar_{i-1})^2.

        Returns
        -------
        float
            The confidence radius phi_NM.
        """
        if t == 0:
            return float('inf')

        log_term = np.log(np.sqrt((V_hat + self.rho) / self.rho) / delta_val)
        log_term = max(0.0, log_term)

        return np.sqrt(2 * (V_hat + self.rho) * log_term) / t

    def init_process(self, data):
        """
        Initializes the algorithm with pre-collected data for each arm.
        Updates statistics observation-by-observation (Welford) and runs BH afterwards.

        Parameters
        ----------
        data : list of list
            data[arm_idx] = list of initial observations for that arm.
        """
        for arm_idx, arm_data in enumerate(data):
            for obs in arm_data:
                # Welford update: V_hat = sum of squared prediction errors
                old_mean = self.emp_means[arm_idx]
                self.emp_means[arm_idx] = (old_mean * self.counts[arm_idx] + obs) / (self.counts[arm_idx] + 1)
                self.emp_vars[arm_idx] += (obs - old_mean) ** 2
                self.counts[arm_idx] += 1
                self.time += 1
                self.counts_evolution.append(self.counts.copy())
        
        # --- BH après init ---
        p_values_with_idx = [(self.get_anytime_pvalue(i), i) for i in range(self.n)]
        p_values_with_idx.sort(key=lambda x: x[0])
        for k in range(self.n, 0, -1):
            if p_values_with_idx[k-1][0] <= self.delta * k / self.n:
                for rank in range(k):
                    self.S_t.add(p_values_with_idx[rank][1])
                break
        print(f"DEBUG INIT: emp_means={self.emp_means}, p_values={[pv for pv,_ in sorted(p_values_with_idx, key=lambda x: x[1])]}, S_t={self.S_t}")

    def select_arm(self):
        """
        Determines which arm to pull next based on the UCB strategy.

        Strategy:
        1. If any arm has not been sampled, pull it first.
        2. Identify candidate arms (those NOT yet in the discovery set S_t).
        3. If no candidates remain, return "stop".
        4. Otherwise, select the candidate with the highest Upper Confidence Bound (UCB).

        Returns
        -------
        int or str
            The index of the arm to pull, or "stop" if all arms are discovered.
        """
        # S'assurer que chaque bras a été tiré au moins une fois
        unsampled = [i for i in range(self.n) if self.counts[i] == 0]
        if unsampled:
            return unsampled[0]
        
        candidates = [i for i in range(self.n) if i not in self.S_t]
        if not candidates:
            return "stop"

        best_ucb = -float('inf')
        selected = candidates[0]
        
        for i in candidates:
            ucb = self.emp_means[i] + self.phi(self.counts[i], self.delta, self.emp_vars[i])
            if ucb > best_ucb:
                best_ucb = ucb
                selected = i
        return selected

    def bh_update(self, arm_idx, observation):
        """
        Updates the algorithm's state with a new observation and runs the decision procedure.

        Uses LCB-based BH procedure: for each candidate k from n down to 1,
        checks if at least k arms have LCB >= mu_0 at level delta*k/n.

        Parameters
        ----------
        arm_idx : int
            The index of the arm that was pulled.
        observation : float
            The reward/value observed from the arm.
        """
        # Welford update: V_hat = sum of squared prediction errors
        n_pulls = self.counts[arm_idx]
        old_mean = self.emp_means[arm_idx]
        self.emp_means[arm_idx] = (old_mean * n_pulls + observation) / (n_pulls + 1)
        self.emp_vars[arm_idx] += (observation - old_mean) ** 2
        
        self.counts[arm_idx] += 1
        self.time += 1
        self.counts_evolution.append(self.counts.copy()) 
        
        current_St = set()
        for k in range(self.n, 0, -1):
            effective_delta = self.delta * k / self.n
            passing_arms = []
            for i in range(self.n):
                lcb = self.emp_means[i] - self.phi(self.counts[i], effective_delta, self.emp_vars[i])
                if lcb >= self.mu_0:
                    passing_arms.append(i)
            if len(passing_arms) >= k:
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
        # 1. Welford update: V_hat = sum of squared prediction errors
        n_pulls = self.counts[arm_idx]
        old_mean = self.emp_means[arm_idx]
        self.emp_means[arm_idx] = (old_mean * n_pulls + observation) / (n_pulls + 1)
        self.emp_vars[arm_idx] += (observation - old_mean) ** 2

        self.counts[arm_idx] += 1
        self.time += 1
        self.counts_evolution.append(self.counts.copy()) 
        
        # 2. Calcul des p-values NM en forme fermée — O(n) sans brentq
        p_values_with_idx = [(self.get_anytime_pvalue(i), i) for i in range(self.n)]
        p_values = [pv for pv, _ in sorted(p_values_with_idx, key=lambda x: x[1])]

        # 3. Tri des p-values par ordre croissant (Complexité : O(n log n))
        p_values_with_idx.sort(key=lambda x: x[0])
        
        # 4. Procédure de Benjamini-Hochberg classique
        # On cherche le plus grand k tel que p_(k) <= delta * k / n
        current_St = set()
        
        for k in range(self.n, 0, -1):
            p_val_k = p_values_with_idx[k - 1][0]
            effective_delta = self.delta * k / self.n
            
            if p_val_k <= effective_delta:
                for rank in range(k):
                    discovered_arm_idx = p_values_with_idx[rank][1]
                    current_St.add(discovered_arm_idx)
                break
                
        # 5. Mise à jour de l'ensemble global des découvertes
        self.S_t.update(current_St)
        return(p_values)

    def get_anytime_pvalue(self, arm_idx):
        """
        Closed-form anytime p-value from the Normal Mixture confidence sequence.

        Derived by inverting phi_NM = diff analytically:
            p_NM = sqrt((V_hat + rho) / rho) * exp(-diff^2 * t^2 / (2 * (V_hat + rho)))

        Complexity: O(1) — no numerical root-finding needed.
        """
        t = self.counts[arm_idx]
        if t == 0:
            return 1.0
        
        diff = self.emp_means[arm_idx] - self.mu_0
        if diff <= 0:
            return 1.0

        V_hat = self.emp_vars[arm_idx]
        p_value = np.sqrt((V_hat + self.rho) / self.rho) * np.exp(-diff ** 2 * t ** 2 / (2 * (V_hat + self.rho)))
        return float(np.clip(p_value, 1e-300, 1.0))

        
class UniformAlgo:
    def __init__(self, n_arms, mu_0, delta, rho=1.0):
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
        rho : float
            Tuning parameter for the Normal Mixture confidence sequence (prior variance).
            Acts as a floor on interval width at small t. Default: 1.0.
        """
        self.n = n_arms
        self.mu_0 = mu_0
        self.delta = delta
        self.rho = rho
        
        self.counts = np.zeros(n_arms, dtype=int)
        self.emp_means = np.zeros(n_arms, dtype=float)
        self.emp_vars = np.zeros(n_arms, dtype=float)  # V_hat: sum of squared prediction errors
        self.time = 0
        self.S_t = set()
        
        # Historique pour visualisation
        # On initialise avec des zéros pour t=0
        self.counts_evolution = [np.zeros(n_arms, dtype=int)]

    def phi(self, t, delta_val, V_hat):
        """
        Normal Mixture confidence sequence radius (Howard et al., 2021).

        Parameters
        ----------
        t : int
            Number of times the specific arm has been pulled.
        delta_val : float
            The confidence level (or p-value during BH procedure).
        V_hat : float
            Running sum of squared prediction errors: sum of (X_i - X_bar_{i-1})^2.

        Returns
        -------
        float
            The confidence radius phi_NM.
        """
        if t == 0:
            return float('inf')

        log_term = np.log(np.sqrt((V_hat + self.rho) / self.rho) / delta_val)
        log_term = max(0.0, log_term)

        return np.sqrt(2 * (V_hat + self.rho) * log_term) / t
    
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
        # 1. Welford update: V_hat = sum of squared prediction errors
        n_pulls = self.counts[arm_idx]
        old_mean = self.emp_means[arm_idx]
        self.emp_means[arm_idx] = (old_mean * n_pulls + observation) / (n_pulls + 1)
        self.emp_vars[arm_idx] += (observation - old_mean) ** 2

        self.counts[arm_idx] += 1
        self.time += 1
        self.counts_evolution.append(self.counts.copy()) 
        
        # 2. Calcul des p-values NM en forme fermée — O(n) sans brentq
        p_values_with_idx = [(self.get_anytime_pvalue(i), i) for i in range(self.n)]
        p_values = [pv for pv, _ in sorted(p_values_with_idx, key=lambda x: x[1])]

        # 3. Tri des p-values par ordre croissant (Complexité : O(n log n))
        p_values_with_idx.sort(key=lambda x: x[0])
        
        # 4. Procédure de Benjamini-Hochberg classique
        current_St = set()
        
        for k in range(self.n, 0, -1):
            p_val_k = p_values_with_idx[k - 1][0]
            effective_delta = self.delta * k / self.n
            
            if p_val_k <= effective_delta:
                for rank in range(k):
                    discovered_arm_idx = p_values_with_idx[rank][1]
                    current_St.add(discovered_arm_idx)
                break
                
        # 5. Mise à jour de l'ensemble global des découvertes
        self.S_t.update(current_St)
        return(p_values)

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
        # Welford update: V_hat = sum of squared prediction errors
        n_pulls = self.counts[arm_idx]
        old_mean = self.emp_means[arm_idx]
        self.emp_means[arm_idx] = (old_mean * n_pulls + observation) / (n_pulls + 1)
        self.emp_vars[arm_idx] += (observation - old_mean) ** 2
        
        self.counts[arm_idx] += 1
        self.time += 1
        self.counts_evolution.append(self.counts.copy()) 
        
        current_St = set()
        for k in range(self.n, 0, -1):
            effective_delta = self.delta * k / self.n
            passing_arms = []
            for i in range(self.n):
                lcb = self.emp_means[i] - self.phi(self.counts[i], effective_delta, self.emp_vars[i])
                if lcb >= self.mu_0:
                    passing_arms.append(i)
            if len(passing_arms) >= k:
                current_St = set(passing_arms)
                break
        self.S_t.update(current_St)

    def get_anytime_pvalue(self, arm_idx):
        """
        Closed-form anytime p-value from the Normal Mixture confidence sequence.

        Derived by inverting phi_NM = diff analytically:
            p_NM = sqrt((V_hat + rho) / rho) * exp(-diff^2 * t^2 / (2 * (V_hat + rho)))

        Complexity: O(1) — no numerical root-finding needed.
        """
        t = self.counts[arm_idx]
        if t == 0:
            return 1.0
        
        diff = self.emp_means[arm_idx] - self.mu_0
        if diff <= 0:
            return 1.0

        V_hat = self.emp_vars[arm_idx]
        p_value = np.sqrt((V_hat + self.rho) / self.rho) * np.exp(-diff ** 2 * t ** 2 / (2 * (V_hat + self.rho)))
        return float(np.clip(p_value, 1e-300, 1.0))
            

def _run_single_simulation(algo, no_sim, all_arm_data, horizon, mode,
                            control_arm, init_nb, init_choice, variable_mu_choice, n_arms, is_true_mean, true_positives):
    """
    Runs a single simulation for a given algorithm instance.
    Common logic shared between 'adaptive' and 'uniform' modes.
    """
    p_values_list = []
    all_arm_counts = [0 for _ in range(n_arms)]
    run_pr = []

    # --- Init (adaptive only) ---
    if mode == 'adaptive' and init_choice == True:
        if init_nb > 0:
            data_init = []
            for arm in range(n_arms):
                data_init_arm = all_arm_data[no_sim][arm][0:init_nb]
                data_init.append(data_init_arm)
            algo.init_process(data_init)
            all_arm_counts = [init_nb for _ in range(n_arms)]
    
    

    # --- Main loop ---
    for t in range(0, horizon):

        # Select arm
        if mode == 'adaptive' and variable_mu_choice == True:
            algo.mu_0 = algo.emp_means[control_arm]
            arm = algo.select_arm()
            # we force the draw of the control arm
            if all_arm_counts[control_arm] < max(all_arm_counts):
                arm = control_arm
        else:
            arm = algo.select_arm()

        if arm == "stop":
            # If we stop before the end, we fill the lists with the last value
            # so that arrays have the correct size (horizon)
            print("stop triggered")
            
            remaining_steps = horizon - len(run_pr)
            
            # 1. Rattrapage pour run_pr
            last_pr = run_pr[-1] if run_pr else 0
            run_pr.extend([last_pr] * remaining_steps)

            # 2. Rattrapage pour counts_evolution
            last_counts = algo.counts_evolution[-1]
            for _ in range(remaining_steps):
                algo.counts_evolution.append(last_counts.copy())
                
            # 3. Rattrapage pour p_values_list
            last_p_values = p_values_list[-1] if p_values_list else [1.0 for _ in range(n_arms)]
            for _ in range(remaining_steps):
                p_values_list.append(list(last_p_values))

            break

        else:
            # Fetch the next pre-generated observation for this specific arm in this simulation
            len_arm = len(all_arm_data[no_sim][arm])
            # if we are at the end of the arm we start at zero again
            if all_arm_counts[arm] >= len_arm:
                observation = all_arm_data[no_sim][arm][all_arm_counts[arm] - len_arm]
                print("time", t, "all_arm_counts[arm]", all_arm_counts[arm], "len_arm", len_arm)
                print("arm ended, recycling data for this arm: ", arm)
            else:
                observation = all_arm_data[no_sim][arm][all_arm_counts[arm]]

            # Increment the local counter so the next pull gets the next value
            all_arm_counts[arm] += 1

            p_values_t = algo.bh_update_optimized(arm, observation)
            p_values_list.append(p_values_t)  # register the p value of the iteration t

            if is_true_mean:
                nb_found = len(algo.S_t.intersection(true_positives))
                current_tpr = nb_found / len(true_positives) if true_positives else 1.0
                run_pr.append(current_tpr)
            else : 
                # adding the number of arm found as positive in this turn
                nb_found = len(algo.S_t)
                run_pr.append(nb_found)  # number of positive in the simulation by draw
    return run_pr, p_values_list


def run_experiment(arms, mu_0, delta, horizon, mode, all_arm_data, n_simulations, control_arm, init_nb, init_choice, variable_mu_choice, is_true_mean):
    """
    Runs the bandit experiment using pre-generated data for consistency.

    This function executes the specified algorithm (Adaptive or Uniform) over a set of 
    simulations. Crucially, instead of generating random rewards on the fly, it consumes 
    pre-generated observations from `all_arm_data`. This ensures that both algorithms 
    are tested against the exact same sequence of random events (variance reduction), 
    making the comparison strictly fair.

    Parameters
    ----------
    arms : array-like
        The arms of the bandit.
    mu_0 : float
        The null hypothesis mean.
    delta : float
        The significance level.
    horizon : int
        The total budget (max time steps) for each simulation.
    mode : str
        The algorithm to run: 'adaptive' or 'uniform'.
    all_arm_data : list of list of list
        A 3D structure containing pre-generated rewards.
        Shape: [simulation_index][arm_index][pull_count]
        This allows the function to "replay" specific random draws.
    n_simulations : int
        The number of independent runs to execute.
    control_arm : int
        Index of the control arm.
    init_nb : int
        Number of initial pulls per arm before the main loop.
    init_choice : bool
        Whether to use the initialization process.
    variable_mu_choice : bool
        Whether to use a variable mu_0 from the empirical mean of the control arm.
    is_true_mean : bool
        Whether to compute True Positive Rate (requires known arm means).

    Returns
    -------
    pnb_history_mean : ndarray
        Average number of arms found positive over time.
    pnb_list : list
        History of pr for each individual simulation.
    counts_history_mean : ndarray
        Average pull counts per arm over time.
    counts_list : list
        History of pull counts for each individual simulation.
    np_p_values_list_by_sim : ndarray
        P-values for each simulation, arm, and time step.
    np_p_values_mean : ndarray
        Average p-values over time.
    list_positive : list
        List of positive arms found at the end of each simulation.
    """
    print("EXECUTION RUN EXP")
    n_arms = len(arms)
    if is_true_mean:
        true_positives = [i for i, m in enumerate(arms) if m > mu_0]
    else: 
        true_positives = None

    pnb_list = []
    counts_list = []
    p_values_list_by_sim = []
    list_positive = []
    counts_evolution_sum = np.zeros((horizon + 1, n_arms))

    # --- Algo factory ---
    algo_factory = {
        'adaptive': lambda: JamiesonJainAlgo(n_arms, mu_0, delta),
        'uniform':  lambda: UniformAlgo(n_arms, mu_0, delta),
    }

    if mode not in algo_factory:
        raise ValueError("Algorithm name not detected, choose between uniform and adaptive")

    if variable_mu_choice and mode == "adaptive":
        print(f"Simulation Mode: {mode.upper()} VAR ({n_simulations} runs)")
    else:
        print(f"Simulation Mode: {mode.upper()} ({n_simulations} runs)")

    # --- Simulation loop ---
    for no_sim in tqdm(range(n_simulations)):

        algo = algo_factory[mode]()

        run_pr, p_values_list = _run_single_simulation(
            algo, no_sim, all_arm_data, horizon, mode,
            control_arm, init_nb, init_choice, variable_mu_choice, n_arms, is_true_mean, true_positives
        )

        # save the list of the positive at the end of the simulation
        list_positive.append(algo.S_t)

        # --- pr aggregation ---
        pnb_i = np.array(run_pr)
        pnb_list.append(pnb_i)  # save the number of positive by draw for each simulation

        # --- Counts aggregation ---
        # (horizon + 1 because there is the initial state at t=0)
        counts_arr = np.array(algo.counts_evolution)
        counts_list.append(counts_arr)
        counts_evolution_sum += counts_arr
        p_values_list_by_sim.append(p_values_list)
    # --- Compute pnb_history_mean ---
    # Toutes les simulations ont exactement 'horizon' entrées (garantie par le padding dans _run_single_simulation)
    pnb_history_mean = np.mean(np.array(pnb_list), axis=0)

    counts_history_mean = counts_evolution_sum / n_simulations

    # --- Compute np_p_values_mean ---
    # 1. On trouve le nombre maximum d'itérations parmi toutes les simulations
    max_length = max(len(arr) for arr in p_values_list_by_sim)
    n_sims_total = len(p_values_list_by_sim)
    # 2. On crée une grande matrice 3D vide remplie de NaN (Not a Number)
    padded_array = np.full((n_sims_total, max_length, n_arms), np.nan)
    # 3. On insère chaque simulation dans cette matrice
    for i, arr in enumerate(p_values_list_by_sim):
        arr_np = np.array(arr)
        padded_array[i, :arr_np.shape[0], :] = arr_np
    # 4. On calcule la moyenne en ignorant les "trous" (NaN)
    np_p_values_mean = np.nanmean(padded_array, axis=0)
    np_p_values_list_by_sim = padded_array
    return pnb_history_mean, pnb_list, counts_history_mean, counts_list, np_p_values_list_by_sim, np_p_values_mean, list_positive