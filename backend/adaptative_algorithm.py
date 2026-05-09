import numpy as np
from tqdm import tqdm
from scipy.optimize import brentq
from statistics import mean

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
        self.emp_vars = np.zeros(n_arms, dtype=float)  # Welford M2 : somme des carrés des écarts
        self.time = 0
        self.S_t = set()
        self.p_values = np.ones(n_arms, dtype=float)
        self._p_values_ready = False
        self._p_values_mu_0 = mu_0
        
        # History for visualization
        # Initialized with zeros for t=0
        self.counts_evolution = [np.zeros(n_arms, dtype=int)] # nb of draw of each arms

    def phi(self, t, delta_val, sigma=1.0):
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
        sigma : float
            The estimated standard deviation of the arm's distribution.

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
        
        return sigma * np.sqrt(num / t)

    def _refresh_p_values(self, changed_arm_idx=None):
        recalc_all = (
            not self._p_values_ready
            or changed_arm_idx is None
            or self._p_values_mu_0 != self.mu_0
        )

        if recalc_all:
            for i in range(self.n):
                self.p_values[i] = self.get_anytime_pvalue(i)
        else:
            self.p_values[changed_arm_idx] = self.get_anytime_pvalue(changed_arm_idx)

        self._p_values_ready = True
        self._p_values_mu_0 = self.mu_0
        return self.p_values

    def _bh_from_p_values(self):
        p_values_with_idx = [(self.p_values[i], i) for i in range(self.n)]
        p_values_with_idx.sort(key=lambda x: x[0])

        current_St = set()
        for k in range(self.n, 0, -1):
            p_val_k = p_values_with_idx[k - 1][0]
            effective_delta = self.delta * k / self.n

            if p_val_k <= effective_delta:
                for rank in range(k):
                    current_St.add(p_values_with_idx[rank][1])
                break

        self.S_t.update(current_St)
        return self.p_values.tolist()
    
    def init_process(self, data):
        for arm_idx, arm_data in enumerate(data):
            for obs in arm_data:
                # Welford update observation par observation
                n = self.counts[arm_idx]
                old_mean = self.emp_means[arm_idx]
                self.emp_means[arm_idx] = (old_mean * n + obs) / (n + 1)
                new_mean = self.emp_means[arm_idx]
                self.emp_vars[arm_idx] += (obs - old_mean) * (obs - new_mean)
                self.counts[arm_idx] += 1
                self.time += 1
                self.counts_evolution.append(self.counts.copy())
        
        # --- BH après init ---
        self._refresh_p_values()
        self._bh_from_p_values()
        # print(f"DEBUG INIT: emp_means={self.emp_means}, p_values={[pv for pv,_ in sorted(p_values_with_idx, key=lambda x: x[1])]}, S_t={self.S_t}")



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
            sigma = np.sqrt(self.emp_vars[i] / self.counts[i]) if self.counts[i] > 1 else 1.0
            ucb = self.emp_means[i] + self.phi(self.counts[i], self.delta, sigma)
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
        # Welford update
        n_pulls = self.counts[arm_idx]
        old_mean = self.emp_means[arm_idx]
        self.emp_means[arm_idx] = (old_mean * n_pulls + observation) / (n_pulls + 1)
        new_mean = self.emp_means[arm_idx]
        self.emp_vars[arm_idx] += (observation - old_mean) * (observation - new_mean)
        
        self.counts[arm_idx] += 1
        self.time += 1
        self.counts_evolution.append(self.counts.copy()) 
        
        k_hat = 0
        current_St = set()
        for k in range(self.n, 0, -1):
            effective_delta = self.delta * k / self.n
            passing_arms = []
            for i in range(self.n):
                sigma = np.sqrt(self.emp_vars[i] / (self.counts[i] -1 )) if self.counts[i] > 1 else 1.0
                lcb = self.emp_means[i] - self.phi(self.counts[i], effective_delta, sigma)
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
        # 1. Mise à jour des statistiques du bras tiré (Welford)
        n_pulls = self.counts[arm_idx]
        old_mean = self.emp_means[arm_idx]
        self.emp_means[arm_idx] = (old_mean * n_pulls + observation) / (n_pulls + 1)
        new_mean = self.emp_means[arm_idx]
        self.emp_vars[arm_idx] += (observation - old_mean) * (observation - new_mean)

        self.counts[arm_idx] += 1
        self.time += 1
        self.counts_evolution.append(self.counts.copy()) 

        self._refresh_p_values(changed_arm_idx=arm_idx)
        return self._bh_from_p_values()

    def get_anytime_pvalue(self, arm_idx):
        t = self.counts[arm_idx]
        emp_mean = self.emp_means[arm_idx]
        
        if t == 0:
            return 1.0
        
        diff = emp_mean - self.mu_0
        if diff <= 0:
            return 1.0

        sigma = np.sqrt(self.emp_vars[arm_idx] / (t-1)) if t > 1 else 1.0

        objective_low = self.phi(t, 1e-12, sigma) - diff
        if objective_low <= 0:
            return 1e-15

        objective_high = self.phi(t, 0.9999, sigma) - diff
        if objective_high >= 0:
            return 1.0

        def objective(p):
            if p <= 0: return float('inf') 
            if p >= 1: return -float('inf')
            return self.phi(t, p, sigma) - diff

        try:
            p_value = brentq(objective, 1e-12, 0.9999)
            return p_value
        except ValueError:
            # brentq échoue = pas de changement de signe entre les bornes
            # On vérifie quel cas on est :
            if objective_low <= 0:
                # phi(t, 1e-12, sigma) < diff
                # → l'évidence dépasse même le bound le plus strict
                # → p-value extrêmement petite
                return 1e-15
            else:
                # phi(t, 0.9999, sigma) > diff (très rare)
                # → même le test le plus laxiste ne rejette pas
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
        self.emp_vars = np.zeros(n_arms, dtype=float)  # Welford M2 : somme des carrés des écarts
        self.time = 0
        self.S_t = set()
        self.p_values = np.ones(n_arms, dtype=float)
        self._p_values_ready = False
        self._p_values_mu_0 = mu_0
        
        # Historique pour visualisation
        # On initialise avec des zéros pour t=0
        self.counts_evolution = [np.zeros(n_arms, dtype=int)]

    def init_process(self, data):
        for arm_idx, arm_data in enumerate(data):
            for obs in arm_data:
                n = self.counts[arm_idx]
                old_mean = self.emp_means[arm_idx]
                self.emp_means[arm_idx] = (old_mean * n + obs) / (n + 1)
                new_mean = self.emp_means[arm_idx]
                self.emp_vars[arm_idx] += (obs - old_mean) * (obs - new_mean)
                self.counts[arm_idx] += 1
                self.time += 1
                self.counts_evolution.append(self.counts.copy())

        # BH après init
        self._refresh_p_values()
        self._bh_from_p_values()

    def phi(self, t, delta_val, sigma=1.0):
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
        sigma : float
            The estimated standard deviation of the arm's distribution.

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
        
        return sigma * np.sqrt(num / t)

    def _refresh_p_values(self, changed_arm_idx=None):
        recalc_all = (
            not self._p_values_ready
            or changed_arm_idx is None
            or self._p_values_mu_0 != self.mu_0
        )

        if recalc_all:
            for i in range(self.n):
                self.p_values[i] = self.get_anytime_pvalue(i)
        else:
            self.p_values[changed_arm_idx] = self.get_anytime_pvalue(changed_arm_idx)

        self._p_values_ready = True
        self._p_values_mu_0 = self.mu_0
        return self.p_values

    def _bh_from_p_values(self):
        p_values_with_idx = [(self.p_values[i], i) for i in range(self.n)]
        p_values_with_idx.sort(key=lambda x: x[0])

        current_St = set()
        for k in range(self.n, 0, -1):
            p_val_k = p_values_with_idx[k - 1][0]
            effective_delta = self.delta * k / self.n

            if p_val_k <= effective_delta:
                for rank in range(k):
                    current_St.add(p_values_with_idx[rank][1])
                break

        self.S_t.update(current_St)
        return self.p_values.tolist()
    
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
        # 1. Mise à jour des statistiques du bras tiré (Welford)
        n_pulls = self.counts[arm_idx]
        old_mean = self.emp_means[arm_idx]
        self.emp_means[arm_idx] = (old_mean * n_pulls + observation) / (n_pulls + 1)
        new_mean = self.emp_means[arm_idx]
        self.emp_vars[arm_idx] += (observation - old_mean) * (observation - new_mean)

        self.counts[arm_idx] += 1
        self.time += 1
        self.counts_evolution.append(self.counts.copy()) 

        self._refresh_p_values(changed_arm_idx=arm_idx)
        return self._bh_from_p_values()

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
        # Welford update
        n_pulls = self.counts[arm_idx]
        old_mean = self.emp_means[arm_idx]
        self.emp_means[arm_idx] = (old_mean * n_pulls + observation) / (n_pulls + 1)
        new_mean = self.emp_means[arm_idx]
        self.emp_vars[arm_idx] += (observation - old_mean) * (observation - new_mean)
        
        self.counts[arm_idx] += 1
        self.time += 1
        self.counts_evolution.append(self.counts.copy()) 
        
        k_hat = 0
        current_St = set()
        for k in range(self.n, 0, -1):
            effective_delta = self.delta * k / self.n
            passing_arms = []
            for i in range(self.n):
                sigma = np.sqrt(self.emp_vars[i] / (self.counts[i] -1)) if self.counts[i] > 1 else 1.0
                lcb = self.emp_means[i] - self.phi(self.counts[i], effective_delta, sigma)
                if lcb >= self.mu_0:
                    passing_arms.append(i)
            if len(passing_arms) >= k:
                k_hat = k
                current_St = set(passing_arms)
                break
        self.S_t.update(current_St)

    def get_anytime_pvalue(self, arm_idx):
        t = self.counts[arm_idx]
        emp_mean = self.emp_means[arm_idx]
        
        if t == 0:
            return 1.0
        
        diff = emp_mean - self.mu_0
        if diff <= 0:
            return 1.0

        sigma = np.sqrt(self.emp_vars[arm_idx] / (t-1) )if t > 1 else 1.0

        objective_low = self.phi(t, 1e-12, sigma) - diff
        if objective_low <= 0:
            return 1e-15

        objective_high = self.phi(t, 0.9999, sigma) - diff
        if objective_high >= 0:
            return 1.0

        def objective(p):
            if p <= 0: return float('inf') 
            if p >= 1: return -float('inf')
            return self.phi(t, p, sigma) - diff

        try:
            p_value = brentq(objective, 1e-12, 0.9999)
            return p_value
        except ValueError:
            # brentq échoue = pas de changement de signe entre les bornes
            # On vérifie quel cas on est :
            if objective_low <= 0:
                # phi(t, 1e-12, sigma) < diff
                # → l'évidence dépasse même le bound le plus strict
                # → p-value extrêmement petite
                return 1e-15
            else:
                # phi(t, 0.9999, sigma) > diff (très rare)
                # → même le test le plus laxiste ne rejette pas
                return 1.0
            

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
    if init_choice:
        if init_nb > 0:
            data_init = []
            for arm in range(n_arms):
                data_init_arm = all_arm_data[no_sim][arm][0:init_nb]
                data_init.append(data_init_arm)
            algo.init_process(data_init)
            all_arm_counts = [init_nb for _ in range(n_arms)]
            algo.counts_evolution = [algo.counts.copy()]
    
    # --- Main loop ---
    for t in range(0, horizon):

        # Select arm
        if variable_mu_choice == True:
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
            if is_true_mean:
                nb_found = len(algo.S_t.intersection(true_positives))
                last_pr = nb_found / len(true_positives) if true_positives else 1.0
            else:
                last_pr = len(algo.S_t)
            run_pr.extend([last_pr] * remaining_steps)

            # 2. Rattrapage pour counts_evolution
            last_counts = algo.counts_evolution[-1]
            for _ in range(remaining_steps):
                algo.counts_evolution.append(last_counts.copy())
                
            # 3. Rattrapage pour p_values_list (LA CORRECTION)
            # On récupère la dernière liste de p-values calculée (ou on met 1.0 par défaut si vide)
            last_p_values = p_values_list[-1] if p_values_list else [1.0 for _ in range(n_arms)]
            for _ in range(remaining_steps):
                # On utilise list() pour créer une copie indépendante à chaque itération
                p_values_list.append(list(last_p_values))
            break

        else:
            # Fetch the next pre-generated observation for this specific arm in this simulation
            len_arm = len(all_arm_data[no_sim][arm])
            # if we are at the end of the arm we start at zero again
            if all_arm_counts[arm] >= len_arm:
                observation = np.random.choice(all_arm_data[no_sim][arm])
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
        if variable_mu_choice:
            true_positives = [i for i, m in enumerate(arms)
                              if i != control_arm and m > arms[control_arm]]
        else:
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

    mode_label = f"{mode.upper()} VAR" if variable_mu_choice else mode.upper()
    print(f"Simulation Mode: {mode_label} ({n_simulations} runs)")

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
