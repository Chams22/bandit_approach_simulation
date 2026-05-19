import numpy as np
import hashlib
from tqdm import tqdm
from scipy.optimize import brentq
from statistics import mean


def _deterministic_bootstrap_observation(arm_data, bootstrap_key, no_sim, arm, pull_index):
    raw = f"{bootstrap_key}|{no_sim}|{arm}|{pull_index}".encode("utf-8")
    idx = int.from_bytes(hashlib.blake2b(raw, digest_size=8).digest(), "big") % len(arm_data)
    return arm_data[idx]


# -----------------------------------------------------------------------------
# PART 1: THE ALGORITHM
# -----------------------------------------------------------------------------
class JamiesonJainAlgo:
    def __init__(self, n_arms, mu_0, delta, control_arm_idx=None):
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
        self.control_arm_idx = control_arm_idx
        
        self.counts = np.zeros(n_arms, dtype=int)
        self.emp_means = np.zeros(n_arms, dtype=float)
        self.emp_vars = np.zeros(n_arms, dtype=float)  # Welford M2: sum of squared deviations
        self.time = 0
        self.S_t = set()
        self.p_values = np.ones(n_arms, dtype=float)
        self._p_values_ready = False
        self._p_values_mu_0 = mu_0
        
        # History for visualization
        # Initialized with zeros for t=0
        self.counts_evolution = [np.zeros(n_arms, dtype=int)] # nb of draw of each arms

    def _test_indices(self):
        if self.control_arm_idx is None:
            return list(range(self.n))
        return [i for i in range(self.n) if i != self.control_arm_idx]

    def _sigma_hat(self, arm_idx):
        return np.sqrt(self.emp_vars[arm_idx] / (self.counts[arm_idx] - 1)) if self.counts[arm_idx] > 1 else 1.0

    def _gap(self, arm_idx):
        if self.control_arm_idx is None:
            return self.emp_means[arm_idx] - self.mu_0
        return self.emp_means[arm_idx] - self.emp_means[self.control_arm_idx]

    def _combined_phi(self, arm_idx, delta_val):
        phi_arm = self.phi(self.counts[arm_idx], delta_val, self._sigma_hat(arm_idx))
        if self.control_arm_idx is None:
            return phi_arm
        phi_control = self.phi(
            self.counts[self.control_arm_idx],
            delta_val,
            self._sigma_hat(self.control_arm_idx),
        )
        return phi_arm + phi_control

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
            or changed_arm_idx == self.control_arm_idx
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
        test_indices = self._test_indices()
        n_tests = len(test_indices)
        if n_tests == 0:
            return self.p_values.tolist()

        p_values_with_idx = [(self.p_values[i], i) for i in test_indices]
        p_values_with_idx.sort(key=lambda x: x[0])

        current_St = set()
        for k in range(n_tests, 0, -1):
            p_val_k = p_values_with_idx[k - 1][0]
            effective_delta = self.delta * k / n_tests

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
        
        # --- BH after init ---
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
        # Make sure each arm has been pulled at least once
        unsampled = [i for i in range(self.n) if self.counts[i] == 0]
        if unsampled:
            return unsampled[0]
        
        candidates = [i for i in self._test_indices() if i not in self.S_t]
        if not candidates:
            return "stop"

        best_ucb = -float('inf')
        selected = candidates[0]
        
        for i in candidates:
            if self.control_arm_idx is None:
                ucb = self.emp_means[i] + self._combined_phi(i, self.delta)
            else:
                ucb = self._gap(i) + self._combined_phi(i, self.delta)
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
        test_indices = self._test_indices()
        n_tests = len(test_indices)
        for k in range(n_tests, 0, -1):
            effective_delta = self.delta * k / n_tests
            passing_arms = []
            for i in test_indices:
                if self.control_arm_idx is None:
                    lcb = self.emp_means[i] - self._combined_phi(i, effective_delta)
                    is_passing = lcb >= self.mu_0
                else:
                    gap_lcb = self._gap(i) - self._combined_phi(i, effective_delta)
                    is_passing = gap_lcb >= 0
                if is_passing:
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
        # 1. Update statistics for the pulled arm (Welford)
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
        
        if self.control_arm_idx is not None and arm_idx == self.control_arm_idx:
            return 1.0

        if t == 0:
            return 1.0
        
        if self.control_arm_idx is not None and self.counts[self.control_arm_idx] == 0:
            return 1.0

        diff = self._gap(arm_idx)
        if diff <= 0:
            return 1.0

        objective_low = self._combined_phi(arm_idx, 1e-12) - diff
        if objective_low <= 0:
            return 1e-15

        objective_high = self._combined_phi(arm_idx, 0.9999) - diff
        if objective_high >= 0:
            return 1.0

        def objective(p):
            if p <= 0: return float('inf') 
            if p >= 1: return -float('inf')
            return self._combined_phi(arm_idx, p) - diff

        try:
            p_value = brentq(objective, 1e-12, 0.9999)
            return p_value
        except ValueError:
            # brentq fails when there is no sign change between bounds
            # Check which case applies:
            if objective_low <= 0:
                # combined phi at 1e-12 is below diff
                # -> evidence exceeds even the strictest bound
                # -> extremely small p-value
                return 1e-15
            else:
                # combined phi at 0.9999 is above diff (very rare)
                # -> even the loosest test does not reject
                return 1.0
        
class UniformAlgo:
    def __init__(self, n_arms, mu_0, delta, control_arm_idx=None):
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
        self.control_arm_idx = control_arm_idx
        
        self.counts = np.zeros(n_arms, dtype=int)
        self.emp_means = np.zeros(n_arms, dtype=float)
        self.emp_vars = np.zeros(n_arms, dtype=float)  # Welford M2: sum of squared deviations
        self.time = 0
        self.S_t = set()
        self.p_values = np.ones(n_arms, dtype=float)
        self._p_values_ready = False
        self._p_values_mu_0 = mu_0
        
        # Historique pour visualisation
        # Initialize with zeros for t=0
        self.counts_evolution = [np.zeros(n_arms, dtype=int)]

    def _test_indices(self):
        if self.control_arm_idx is None:
            return list(range(self.n))
        return [i for i in range(self.n) if i != self.control_arm_idx]

    def _sigma_hat(self, arm_idx):
        return np.sqrt(self.emp_vars[arm_idx] / (self.counts[arm_idx] - 1)) if self.counts[arm_idx] > 1 else 1.0

    def _gap(self, arm_idx):
        if self.control_arm_idx is None:
            return self.emp_means[arm_idx] - self.mu_0
        return self.emp_means[arm_idx] - self.emp_means[self.control_arm_idx]

    def _combined_phi(self, arm_idx, delta_val):
        phi_arm = self.phi(self.counts[arm_idx], delta_val, self._sigma_hat(arm_idx))
        if self.control_arm_idx is None:
            return phi_arm
        phi_control = self.phi(
            self.counts[self.control_arm_idx],
            delta_val,
            self._sigma_hat(self.control_arm_idx),
        )
        return phi_arm + phi_control

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

        # BH after init
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
            or changed_arm_idx == self.control_arm_idx
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
        test_indices = self._test_indices()
        n_tests = len(test_indices)
        if n_tests == 0:
            return self.p_values.tolist()

        p_values_with_idx = [(self.p_values[i], i) for i in test_indices]
        p_values_with_idx.sort(key=lambda x: x[0])

        current_St = set()
        for k in range(n_tests, 0, -1):
            p_val_k = p_values_with_idx[k - 1][0]
            effective_delta = self.delta * k / n_tests

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
            if self.control_arm_idx is not None:
                candidates = self._test_indices()
                if not candidates:
                    return "stop"
                return np.random.choice(candidates)
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
        # 1. Update statistics for the pulled arm (Welford)
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
        test_indices = self._test_indices()
        n_tests = len(test_indices)
        for k in range(n_tests, 0, -1):
            effective_delta = self.delta * k / n_tests
            passing_arms = []
            for i in test_indices:
                if self.control_arm_idx is None:
                    lcb = self.emp_means[i] - self._combined_phi(i, effective_delta)
                    is_passing = lcb >= self.mu_0
                else:
                    gap_lcb = self._gap(i) - self._combined_phi(i, effective_delta)
                    is_passing = gap_lcb >= 0
                if is_passing:
                    passing_arms.append(i)
            if len(passing_arms) >= k:
                k_hat = k
                current_St = set(passing_arms)
                break
        self.S_t.update(current_St)

    def get_anytime_pvalue(self, arm_idx):
        t = self.counts[arm_idx]
        emp_mean = self.emp_means[arm_idx]
        
        if self.control_arm_idx is not None and arm_idx == self.control_arm_idx:
            return 1.0

        if t == 0:
            return 1.0
        
        if self.control_arm_idx is not None and self.counts[self.control_arm_idx] == 0:
            return 1.0

        diff = self._gap(arm_idx)
        if diff <= 0:
            return 1.0

        objective_low = self._combined_phi(arm_idx, 1e-12) - diff
        if objective_low <= 0:
            return 1e-15

        objective_high = self._combined_phi(arm_idx, 0.9999) - diff
        if objective_high >= 0:
            return 1.0

        def objective(p):
            if p <= 0: return float('inf') 
            if p >= 1: return -float('inf')
            return self._combined_phi(arm_idx, p) - diff

        try:
            p_value = brentq(objective, 1e-12, 0.9999)
            return p_value
        except ValueError:
            # brentq fails when there is no sign change between bounds
            # Check which case applies:
            if objective_low <= 0:
                # combined phi at 1e-12 is below diff
                # -> evidence exceeds even the strictest bound
                # -> extremely small p-value
                return 1e-15
            else:
                # combined phi at 0.9999 is above diff (very rare)
                # -> even the loosest test does not reject
                return 1.0
            

def _should_record_history(step, horizon, history_record_every):
    return step == horizon or step % history_record_every == 0


def _run_single_simulation(algo, no_sim, all_arm_data, horizon, mode,
                            control_arm, init_nb, init_choice, variable_mu_choice,
                            n_arms, is_true_mean, true_positives,
                            history_record_every=1,
                            stop_when_all_non_control_found=False,
                            stop_control_arm=None,
                            deterministic_bootstrap_key="default"):
    """
    Runs a single simulation for a given algorithm instance.
    Common logic shared between 'adaptive' and 'uniform' modes.
    """
    p_values_list = []
    all_arm_counts = [0 for _ in range(n_arms)]
    run_pr = []
    bootstrap_start_times = {}
    history_record_every = max(1, int(history_record_every))
    stop_target_arms = set(range(n_arms))
    if stop_control_arm is not None:
        stop_target_arms.discard(int(stop_control_arm))

    def all_stop_targets_found():
        return (
            stop_when_all_non_control_found
            and stop_target_arms.issubset({int(arm) for arm in algo.S_t})
        )

    def fill_remaining(current_done):
        remaining_steps = horizon - current_done
        if remaining_steps <= 0:
            return
        if is_true_mean:
            nb_found = len(algo.S_t.intersection(true_positives))
            last_pr = nb_found / len(true_positives) if true_positives else 1.0
        else:
            last_pr = len(algo.S_t)
        run_pr.extend([last_pr] * remaining_steps)
        last_counts = algo.counts_evolution[-1] if algo.counts_evolution else algo.counts.copy()
        for step in range(current_done + 1, horizon + 1):
            if _should_record_history(step, horizon, history_record_every):
                algo.counts_evolution.append(last_counts.copy())
        last_p_values = p_values_list[-1] if p_values_list else [1.0 for _ in range(n_arms)]
        for step in range(current_done + 1, horizon + 1):
            if _should_record_history(step, horizon, history_record_every):
                p_values_list.append(list(last_p_values))

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

    discovery_times = {int(arm): 0 for arm in algo.S_t}
    if all_stop_targets_found():
        fill_remaining(0)
        return run_pr, p_values_list, discovery_times, bootstrap_start_times
    
    # --- Main loop ---
    for t in range(0, horizon):

        # Select arm
        if variable_mu_choice == True:
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
            
            fill_remaining(len(run_pr))
            break

        else:
            current_step = len(run_pr) + 1
            # Fetch the next pre-generated observation for this specific arm in this simulation
            len_arm = len(all_arm_data[no_sim][arm])
            # if we are at the end of the arm we start at zero again
            if all_arm_counts[arm] >= len_arm:
                bootstrap_start_times.setdefault(int(arm), current_step)
                observation = _deterministic_bootstrap_observation(
                    all_arm_data[no_sim][arm],
                    deterministic_bootstrap_key,
                    no_sim,
                    arm,
                    all_arm_counts[arm],
                )
            else:
                observation = all_arm_data[no_sim][arm][all_arm_counts[arm]]

            # Increment the local counter so the next pull gets the next value
            all_arm_counts[arm] += 1

            p_values_t = algo.bh_update_optimized(arm, observation)
            if _should_record_history(current_step, horizon, history_record_every):
                p_values_list.append(p_values_t)
            else:
                algo.counts_evolution.pop()

            for discovered_arm in algo.S_t:
                discovery_times.setdefault(int(discovered_arm), current_step)

            if is_true_mean:
                nb_found = len(algo.S_t.intersection(true_positives))
                current_tpr = nb_found / len(true_positives) if true_positives else 1.0
                run_pr.append(current_tpr)
            else : 
                # adding the number of arm found as positive in this turn
                nb_found = len(algo.S_t)
                run_pr.append(nb_found)  # number of positive in the simulation by draw

            if all_stop_targets_found():
                fill_remaining(current_step)
                break
    return run_pr, p_values_list, discovery_times, bootstrap_start_times


def run_experiment(arms, mu_0, delta, horizon, mode, all_arm_data, n_simulations,
                   control_arm, init_nb, init_choice, variable_mu_choice,
                   is_true_mean, return_discovery_times=False,
                   return_bootstrap_times=False, history_record_every=1,
                   stop_when_all_non_control_found=False,
                   stop_control_arm=None,
                   deterministic_bootstrap_key="default"):
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
    history_record_every = max(1, int(history_record_every))
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
    discovery_times_list = []
    bootstrap_times_list = []
    n_history_points = 1 + sum(
        1 for step in range(1, horizon + 1)
        if _should_record_history(step, horizon, history_record_every)
    )
    counts_evolution_sum = np.zeros((n_history_points, n_arms))

    # --- Algo factory ---
    algo_factory = {
        'adaptive': lambda: JamiesonJainAlgo(
            n_arms, mu_0, delta,
            control_arm_idx=control_arm if variable_mu_choice else None,
        ),
        'uniform':  lambda: UniformAlgo(
            n_arms, mu_0, delta,
            control_arm_idx=control_arm if variable_mu_choice else None,
        ),
    }

    if mode not in algo_factory:
        raise ValueError("Algorithm name not detected, choose between uniform and adaptive")

    mode_label = f"{mode.upper()} VAR" if variable_mu_choice else mode.upper()
    print(f"Simulation Mode: {mode_label} ({n_simulations} runs)")

    # --- Simulation loop ---
    for no_sim in tqdm(range(n_simulations)):

        algo = algo_factory[mode]()

        run_pr, p_values_list, discovery_times, bootstrap_start_times = _run_single_simulation(
            algo, no_sim, all_arm_data, horizon, mode,
            control_arm, init_nb, init_choice, variable_mu_choice, n_arms,
            is_true_mean, true_positives, history_record_every,
            stop_when_all_non_control_found=stop_when_all_non_control_found,
            stop_control_arm=control_arm if stop_control_arm is None else stop_control_arm,
            deterministic_bootstrap_key=deterministic_bootstrap_key,
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
        discovery_times_list.append(discovery_times)
        bootstrap_times_list.append(bootstrap_start_times)
    # --- Compute pnb_history_mean ---
    # Every simulation has exactly 'horizon' entries (guaranteed by padding in _run_single_simulation)
    pnb_history_mean = np.mean(np.array(pnb_list), axis=0)

    counts_history_mean = counts_evolution_sum / n_simulations

    # --- Compute np_p_values_mean ---
    # 1. Find the maximum number of iterations across simulations
    max_length = max(len(arr) for arr in p_values_list_by_sim)
    n_sims_total = len(p_values_list_by_sim)
    # 2. Create a large empty 3D matrix filled with NaN (Not a Number)
    padded_array = np.full((n_sims_total, max_length, n_arms), np.nan)
    # 3. Insert each simulation into this matrix
    for i, arr in enumerate(p_values_list_by_sim):
        arr_np = np.array(arr)
        padded_array[i, :arr_np.shape[0], :] = arr_np
    # 4. Compute the mean while ignoring gaps (NaN)
    np_p_values_mean = np.nanmean(padded_array, axis=0)
    np_p_values_list_by_sim = padded_array
    result = (pnb_history_mean, pnb_list, counts_history_mean, counts_list,
              np_p_values_list_by_sim, np_p_values_mean, list_positive)
    if return_discovery_times and return_bootstrap_times:
        return (*result, discovery_times_list, bootstrap_times_list)
    if return_discovery_times:
        return (*result, discovery_times_list)
    if return_bootstrap_times:
        return (*result, bootstrap_times_list)
    return result
