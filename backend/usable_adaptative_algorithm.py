import numpy as np
from tqdm import tqdm
from scipy.optimize import brentq
from statistics import mean


# -----------------------------------------------------------------------------
# PART 1: THE ALGORITHM
# -----------------------------------------------------------------------------
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
    
    def init_process(self, data):
        for arm_idx, arm_data in enumerate(data):
            self.emp_means[arm_idx] = mean(arm_data)

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
        p_values = [self.get_anytime_pvalue(i) for i in range(self.n)]

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
        return(p_values)


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
        p_values = [self.get_anytime_pvalue(i) for i in range(self.n)]

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


   
def run_experiment(arms, mu_0, delta, horizon, mode, all_arm_data, n_simulations, control_arm, init_nb, init_choice, variable_mu_choice):
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
    print("EXECUTION RUN EXP")
    n_arms = len(arms)
    
    pnb_history_sum = np.zeros(horizon)
    pnb_list = []
    
    # Store the AVERAGE number of pulls at each time t
    counts_evolution_sum = np.zeros((horizon + 1, n_arms))
    counts_list=[]
    p_values_list_by_sim=[]


    print(f"Simulation Mode: {mode.upper()} ({n_simulations} runs)")
    

    if mode=='adaptive':
        counts_evolution_sum = np.zeros((horizon + 1 - init_nb, n_arms))
        for no_sim in tqdm(range(n_simulations)):
            algo = JamiesonJainAlgo(n_arms, mu_0, delta)
            p_values_list=[]
            # Track how many times we have pulled each arm *in this specific simulation*
            # This is necessary to fetch the correct next value from the pre-generated list.
            all_arm_counts = [0 for _ in range(n_arms)]
            run_pr = []

            if init_choice==True: 
                if init_nb>0:
                    data_init=[]
                    for arm in range(n_arms):
                        data_init_arm=all_arm_data[no_sim][arm][0:init_nb]
                        data_init.append(data_init_arm)
                    algo.init_process(data_init)
                    all_arm_counts = [init_nb for _ in range(n_arms)]
                else:
                    init_nb=0
            
            for t in range(init_nb, horizon):
                if variable_mu_choice==True:
                    algo.mu_0=algo.emp_means[control_arm]
                    arm = algo.select_arm()
                    if all_arm_counts[control_arm]<max(all_arm_counts):
                        # print("pas max")
                        arm=control_arm
                else:
                    arm = algo.select_arm()

                
                if arm == "stop":
                    # If we stop before the end, we fill the lists with the last value
                    # so that arrays have the correct size (horizon)
                    print("stop triggered")
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
                    p_values_t=algo.bh_update_optimized(arm, observation)
                    p_values_list.append(p_values_t) #register the p value of the iteration t in the p_value_list for the simulation no_sim
                    # print(len(p_values_t))

                    nb_found = len(algo.S_t)
                    current_pr = nb_found
                    run_pr.append(current_pr)

            # pr aggregation
            pnb_i = np.array(run_pr)
            pnb_list.append(pnb_i)
            # Counts aggregation (ensure we take the first 'horizon+1' elements)
            # (horizon + 1 because there is the initial state at t=0)
            counts_arr = np.array(algo.counts_evolution)[:horizon+1]
            counts_list.append(counts_arr)
            counts_evolution_sum += counts_arr
            # print(len(p_values_list))
            p_values_list_by_sim.append(p_values_list)    
        # 1. Trouver la longueur de la simulation la plus longue
        max_length = max(len(pnb) for pnb in pnb_list)
        n_sims_total = len(pnb_list)
        # 2. Créer une matrice vide de la bonne taille (n_simulations x max_length)
        pnb_matrix = np.zeros((n_sims_total, max_length))
        # 3. Remplir la matrice
        for i, pnb_i in enumerate(pnb_list):
            length = len(pnb_i)
            # On insère les vraies données de la simulation
            pnb_matrix[i, :length] = pnb_i
            # Si la simulation est plus courte que max_length, on répète la dernière valeur
            if length < max_length:
                last_val = pnb_i[-1] if length > 0 else 0.0
                pnb_matrix[i, length:] = last_val
        # 4. Calculer la moyenne finale proprement
        pnb_history_mean = np.mean(pnb_matrix, axis=0)
        counts_history_mean = counts_evolution_sum / n_simulations
        # np_p_values_list_by_sim=np.array(p_values_list_by_sim)
        # np_p_values_mean = np.mean(np_p_values_list_by_sim, axis=0)
        # 1. On trouve le nombre maximum d'itérations parmi toutes les simulations
        max_length = max(len(arr) for arr in p_values_list_by_sim)
        n_arms = np.array(p_values_list_by_sim[0]).shape[1]
        n_sims_total = len(p_values_list_by_sim)
        # 2. On crée une grande matrice 3D vide remplie de NaN (Not a Number)
        padded_array = np.full((n_sims_total, max_length, n_arms), np.nan)
        # 3. On insère chaque simulation dans cette matrice
        for i, arr in enumerate(p_values_list_by_sim):
            arr_np = np.array(arr)
            padded_array[i, :arr_np.shape[0], :] = arr_np
        # 4. On calcule la moyenne en ignorant les "trous" (NaN)
        np_p_values_mean = np.nanmean(padded_array, axis=0)
        # Tu peux aussi garder padded_array si tu as besoin de renvoyer la liste complète
        np_p_values_list_by_sim = padded_array

    elif mode=='uniform':
        for no_sim in tqdm(range(n_simulations)):
            algo = UniformAlgo(n_arms, mu_0, delta)
            p_values_list=[]
            all_arm_counts = [0 for _ in range(n_arms)]
            run_pr = []
            
            for t in range(horizon):
                arm = algo.select_arm()
                
                if arm == "stop":
                    # If we stop before the end, we fill the lists with the last value
                    # so that arrays have the correct size (horizon)
                    print("stop triggered")
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
                    
                    p_values_t=algo.bh_update_optimized(arm, observation)
                    p_values_list.append(p_values_t) #register the p value of the iteration t in the p_value_list for the simulation no_sim
                    # print(len(p_values_t))

                    nb_found = len(algo.S_t)
                    current_pr = nb_found
                    run_pr.append(current_pr)

            # pr aggregation
            pnb_i = np.array(run_pr)
            pnb_list.append(pnb_i)
            # Counts aggregation (ensure we take the first 'horizon+1' elements)
            # (horizon + 1 because there is the initial state at t=0)
            counts_arr = np.array(algo.counts_evolution)[:horizon+1]
            counts_list.append(counts_arr)
            counts_evolution_sum += counts_arr
            # print(len(p_values_list))
            p_values_list_by_sim.append(p_values_list)    
        # 1. Trouver la longueur de la simulation la plus longue
        max_length = max(len(pnb) for pnb in pnb_list)
        n_sims_total = len(pnb_list)
        # 2. Créer une matrice vide de la bonne taille (n_simulations x max_length)
        pnb_matrix = np.zeros((n_sims_total, max_length))
        # 3. Remplir la matrice
        for i, pnb_i in enumerate(pnb_list):
            length = len(pnb_i)
            # On insère les vraies données de la simulation
            pnb_matrix[i, :length] = pnb_i
            # Si la simulation est plus courte que max_length, on répète la dernière valeur
            if length < max_length:
                last_val = pnb_i[-1] if length > 0 else 0.0
                pnb_matrix[i, length:] = last_val
        # 4. Calculer la moyenne finale proprement
        pnb_history_mean = np.mean(pnb_matrix, axis=0)
        counts_history_mean = counts_evolution_sum / n_simulations
        # np_p_values_list_by_sim=np.array(p_values_list_by_sim)
        # np_p_values_mean = np.mean(np_p_values_list_by_sim, axis=0)
        # 1. On trouve le nombre maximum d'itérations parmi toutes les simulations
        max_length = max(len(arr) for arr in p_values_list_by_sim)
        n_arms = np.array(p_values_list_by_sim[0]).shape[1]
        n_sims_total = len(p_values_list_by_sim)
        # 2. On crée une grande matrice 3D vide remplie de NaN (Not a Number)
        padded_array = np.full((n_sims_total, max_length, n_arms), np.nan)
        # 3. On insère chaque simulation dans cette matrice
        for i, arr in enumerate(p_values_list_by_sim):
            arr_np = np.array(arr)
            padded_array[i, :arr_np.shape[0], :] = arr_np
        # 4. On calcule la moyenne en ignorant les "trous" (NaN)
        np_p_values_mean = np.nanmean(padded_array, axis=0)
        # Tu peux aussi garder padded_array si tu as besoin de renvoyer la liste complète
        np_p_values_list_by_sim = padded_array

    else:
        raise ValueError("Algorithm name not detected, choose between uniform and adaptive")
        
    return pnb_history_mean, pnb_list, counts_history_mean, counts_list, np_p_values_list_by_sim, np_p_values_mean