import numpy as np
from tqdm import tqdm

# -----------------------------------------------------------------------------
# Fused V2 for Continuous or Binary Data
# -----------------------------------------------------------------------------
# V2 BH supports NM-style confidence sequences only:
#   'normal_mixture'  — Original V_hat (sum of squared prediction errors), rigorous
#   'nm_m2'           — Uses M2 (Welford), tighter in practice, slight theoretical looseness
# Betting BH is intentionally reserved for V3.
# -----------------------------------------------------------------------------

def _phi_vector(algo, counts, delta_val, var_stats):
    counts = np.asarray(counts, dtype=float)
    var_stats = np.asarray(var_stats, dtype=float)
    phi_vals = np.full(counts.shape, np.inf, dtype=float)
    valid = counts > 0
    if not np.any(valid):
        return phi_vals

    var_plus_rho = var_stats[valid] + algo.rho
    log_term = np.log(np.sqrt(var_plus_rho / algo.rho) / delta_val)
    log_term = np.maximum(0.0, log_term)
    phi_vals[valid] = np.sqrt(2 * var_plus_rho * log_term) / counts[valid]
    return phi_vals


def _two_sample_nm_lcb_discoveries(algo, k_start, delta_denominator,
                                   phi_delta_factor=1.0):
    control_idx = algo.control_arm_idx
    treatment_idx = np.flatnonzero(np.arange(algo.n) != control_idx)
    if treatment_idx.size == 0:
        return set()

    gaps = algo.emp_means[treatment_idx] - algo.emp_means[control_idx]
    treatment_counts = algo.counts[treatment_idx]
    treatment_vars = algo.emp_vars[treatment_idx]
    control_count = algo.counts[control_idx]
    control_var = algo.emp_vars[control_idx]

    for k in range(k_start, 0, -1):
        effective_delta = algo.delta * k / delta_denominator
        phi_delta = effective_delta * phi_delta_factor
        phi_ctrl = algo.phi(control_count, phi_delta, control_var)
        scores = gaps - _phi_vector(algo, treatment_counts, phi_delta,
                                    treatment_vars) - phi_ctrl
        passing_idx = treatment_idx[scores >= 0]
        if passing_idx.size >= k:
            return set(passing_idx.tolist())
    return set()


class JamiesonJainAlgo:
    def __init__(self, n_arms, mu_0, delta, rho=0.01, cs_type='nm_m2', control_arm_idx=None):
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
            Tuning parameter for Normal Mixture CS variants (prior variance).
            Default: 0.01.
        cs_type : str
            Confidence sequence type:
            - 'normal_mixture': Original V_hat NM (Howard et al., 2021). Rigorous.
            - 'nm_m2': NM with M2 instead of V_hat. Default for fused V2.
        control_arm_idx : int or None
            If set, enables two-sample CS mode: tests mu_arm - mu_control > 0
            instead of mu_arm > mu_0. Default: None (single-sample mode).
        """
        if cs_type not in ('normal_mixture', 'nm_m2'):
            raise ValueError("V2 cs_type must be 'normal_mixture' or 'nm_m2'. Use V3 for betting BH.")

        self.n = n_arms
        self.mu_0 = mu_0
        self.delta = delta
        self.rho = max(float(rho), 1e-12)
        self.cs_type = cs_type
        self.control_arm_idx = control_arm_idx

        self.counts = np.zeros(n_arms, dtype=int)
        self.emp_means = np.zeros(n_arms, dtype=float)
        self.emp_vars = np.zeros(n_arms, dtype=float)
        self.time = 0
        self.S_t = set()

        # History for visualization
        self.counts_evolution = [np.zeros(n_arms, dtype=int)]

    # -------------------------------------------------------------------------
    # Statistics update — branching on cs_type
    # -------------------------------------------------------------------------
    def _update_stats(self, arm_idx, observation):
        """
        Updates empirical statistics for a given arm.

        - 'normal_mixture': accumulates V_hat = sum (X_i - X_bar_{i-1})^2
        - 'nm_m2': accumulates M2 = sum (X_i - X_bar_{i-1})(X_i - X_bar_i)
        """
        n = self.counts[arm_idx]
        old_mean = self.emp_means[arm_idx]

        # --- Update mean ---
        self.emp_means[arm_idx] = (old_mean * n + observation) / (n + 1)
        new_mean = self.emp_means[arm_idx]

        # --- Update variance accumulator ---
        if self.cs_type == 'normal_mixture':
            self.emp_vars[arm_idx] += (observation - old_mean) ** 2
        else:
            self.emp_vars[arm_idx] += (observation - old_mean) * (observation - new_mean)

    # -------------------------------------------------------------------------
    # Confidence radius phi
    # -------------------------------------------------------------------------
    def phi(self, t, delta_val, var_stat):
        """
        Normal Mixture confidence sequence radius (Howard et al., 2021).
        Used for arm selection (UCB) in all modes, and for p-values in NM modes.
        """
        if t == 0:
            return float('inf')

        log_term = np.log(np.sqrt((var_stat + self.rho) / self.rho) / delta_val)
        log_term = max(0.0, log_term)

        return np.sqrt(2 * (var_stat + self.rho) * log_term) / t

    # -------------------------------------------------------------------------
    # Anytime p-value
    # -------------------------------------------------------------------------
    def get_anytime_pvalue(self, arm_idx):
        """
        Computes the anytime p-value for a given arm.

        - 'normal_mixture' / 'nm_m2': Closed-form NM inversion.
        """
        t = self.counts[arm_idx]
        if t == 0:
            return 1.0

        mu0_ref = self.emp_means[self.control_arm_idx] if self.control_arm_idx is not None else self.mu_0
        diff = self.emp_means[arm_idx] - mu0_ref
        if diff <= 0:
            return 1.0

        var_stat = self.emp_vars[arm_idx]
        p_value = (np.sqrt((var_stat + self.rho) / self.rho)
                   * np.exp(-diff ** 2 * t ** 2 / (2 * (var_stat + self.rho))))
        return float(np.clip(p_value, 1e-300, 1.0))

    # -------------------------------------------------------------------------
    # Init process
    # -------------------------------------------------------------------------
    def init_process(self, data):
        """
        Initializes the algorithm with pre-collected data for each arm.
        Updates statistics observation-by-observation and runs BH afterwards.
        """
        for arm_idx, arm_data in enumerate(data):
            for obs in arm_data:
                self._update_stats(arm_idx, obs)
                self.counts[arm_idx] += 1
                self.time += 1
                self.counts_evolution.append(self.counts.copy())

        # --- Automatic rho calibration from init data ---
        var_estimates = [self.emp_vars[i] / max(self.counts[i] - 1, 1)
                         for i in range(self.n) if self.counts[i] > 1]
        if var_estimates:
            self.rho = max(float(np.median(var_estimates)), 1e-12)

        # --- BH after init ---
        p_values_with_idx = [(self.get_anytime_pvalue(i), i) for i in range(self.n)]
        p_values_with_idx.sort(key=lambda x: x[0])
        for k in range(self.n, 0, -1):
            if p_values_with_idx[k - 1][0] <= self.delta * k / self.n:
                for rank in range(k):
                    self.S_t.add(p_values_with_idx[rank][1])
                break
        # print(f"DEBUG INIT: cs_type={self.cs_type}, emp_means={self.emp_means}, "
        #       f"p_values={[pv for pv, _ in sorted(p_values_with_idx, key=lambda x: x[1])]}, "
        #       f"S_t={self.S_t}")

    # -------------------------------------------------------------------------
    # Arm selection (UCB)
    # -------------------------------------------------------------------------
    def select_arm(self):
        """
        UCB-based arm selection. Uses phi(M2 or V_hat) for all cs_types.
        """
        unsampled = [i for i in range(self.n) if self.counts[i] == 0]
        if unsampled:
            return unsampled[0]

        candidates = [i for i in range(self.n) if i not in self.S_t]
        if not candidates:
            return "stop"

        best_ucb = -float('inf')
        selected = candidates[0]

        for i in candidates:
            if self.control_arm_idx is not None:
                ucb = (self.emp_means[i] - self.emp_means[self.control_arm_idx]) \
                      + self.phi(self.counts[i], self.delta, self.emp_vars[i]) \
                      + self.phi(self.counts[self.control_arm_idx], self.delta, self.emp_vars[self.control_arm_idx])
            else:
                ucb = self.emp_means[i] + self.phi(self.counts[i], self.delta, self.emp_vars[i])
            if ucb > best_ucb:
                best_ucb = ucb
                selected = i
        return selected


    def select_arm_old(self):
        """
        Confirmation pure
        """
        unsampled = [i for i in range(self.n) if self.counts[i] == 0]
        if unsampled:
            return unsampled[0]
        
        candidates = [i for i in range(self.n) if i not in self.S_t]
        if not candidates:
            return "stop"

        best_pval = float('inf')
        selected = candidates[0]
        for i in candidates:
            pv = self.get_anytime_pvalue(i)
            if pv < best_pval:
                best_pval = pv
                selected = i
        return selected
    
    # -------------------------------------------------------------------------
    # BH update (non-optimized, LCB-based)
    # -------------------------------------------------------------------------
    def bh_update(self, arm_idx, observation):
        """
        Updates state and runs LCB-based BH procedure.
        """
        self._update_stats(arm_idx, observation)
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

    # -------------------------------------------------------------------------
    # BH update (optimized, p-value-sorted, O(n log n))
    # -------------------------------------------------------------------------
    def bh_update_optimized(self, arm_idx, observation):
        """
        Updates state and runs p-value-sorted BH procedure in O(n log n).
        """
        self._update_stats(arm_idx, observation)
        self.counts[arm_idx] += 1
        self.time += 1
        self.counts_evolution.append(self.counts.copy())

        p_values = [1.0] * self.n
        current_St = set()

        if self.control_arm_idx is not None:
            # Two-sample NM LCB
            current_St = _two_sample_nm_lcb_discoveries(
                self,
                k_start=self.n,
                delta_denominator=self.n,
            )
        else:
            # Single-sample: p-values sorted BH
            p_values_with_idx = [(self.get_anytime_pvalue(i), i) for i in range(self.n)]
            p_values = [pv for pv, _ in sorted(p_values_with_idx, key=lambda x: x[1])]
            p_values_with_idx.sort(key=lambda x: x[0])
            for k in range(self.n, 0, -1):
                if p_values_with_idx[k - 1][0] <= self.delta * k / self.n:
                    for rank in range(k):
                        current_St.add(p_values_with_idx[rank][1])
                    break

        self.S_t.update(current_St)
        return p_values


# =============================================================================
# UNIFORM ALGORITHM
# =============================================================================
class UniformAlgo:
    def __init__(self, n_arms, mu_0, delta, rho=0.01, cs_type='nm_m2', control_arm_idx=None):
        """
        Uniform (random) sampling algorithm with the same CS options.
        """
        if cs_type not in ('normal_mixture', 'nm_m2'):
            raise ValueError("V2 cs_type must be 'normal_mixture' or 'nm_m2'. Use V3 for betting BH.")

        self.n = n_arms
        self.mu_0 = mu_0
        self.delta = delta
        self.rho = max(float(rho), 1e-12)
        self.cs_type = cs_type
        self.control_arm_idx = control_arm_idx

        self.counts = np.zeros(n_arms, dtype=int)
        self.emp_means = np.zeros(n_arms, dtype=float)
        self.emp_vars = np.zeros(n_arms, dtype=float)
        self.time = 0
        self.S_t = set()

        self.counts_evolution = [np.zeros(n_arms, dtype=int)]
    def init_process(self, data):
        for arm_idx, arm_data in enumerate(data):
            for obs in arm_data:
                self._update_stats(arm_idx, obs)
                self.counts[arm_idx] += 1
                self.time += 1
                self.counts_evolution.append(self.counts.copy())

        p_values_with_idx = [(self.get_anytime_pvalue(i), i) for i in range(self.n)]
        p_values_with_idx.sort(key=lambda x: x[0])
        for k in range(self.n, 0, -1):
            if p_values_with_idx[k - 1][0] <= self.delta * k / self.n:
                for rank in range(k):
                    self.S_t.add(p_values_with_idx[rank][1])
                break
    def _update_stats(self, arm_idx, observation):
        """Same update logic as JamiesonJainAlgo."""
        n = self.counts[arm_idx]
        old_mean = self.emp_means[arm_idx]

        self.emp_means[arm_idx] = (old_mean * n + observation) / (n + 1)
        new_mean = self.emp_means[arm_idx]

        if self.cs_type == 'normal_mixture':
            self.emp_vars[arm_idx] += (observation - old_mean) ** 2
        else:
            self.emp_vars[arm_idx] += (observation - old_mean) * (observation - new_mean)

    def phi(self, t, delta_val, var_stat):
        if t == 0:
            return float('inf')
        log_term = np.log(np.sqrt((var_stat + self.rho) / self.rho) / delta_val)
        log_term = max(0.0, log_term)
        return np.sqrt(2 * (var_stat + self.rho) * log_term) / t

    def get_anytime_pvalue(self, arm_idx):
        t = self.counts[arm_idx]
        if t == 0:
            return 1.0
        mu0_ref = self.emp_means[self.control_arm_idx] if self.control_arm_idx is not None else self.mu_0
        diff = self.emp_means[arm_idx] - mu0_ref
        if diff <= 0:
            return 1.0

        var_stat = self.emp_vars[arm_idx]
        p_value = (np.sqrt((var_stat + self.rho) / self.rho)
                   * np.exp(-diff ** 2 * t ** 2 / (2 * (var_stat + self.rho))))
        return float(np.clip(p_value, 1e-300, 1.0))

    def select_arm(self):
        if self.control_arm_idx is not None:
            candidates = [i for i in range(self.n)
                          if i not in self.S_t and i != self.control_arm_idx]
            if not candidates:
                return "stop"
            return np.random.choice(candidates)
        return np.random.randint(self.n)

    def bh_update_optimized(self, arm_idx, observation):
        self._update_stats(arm_idx, observation)
        self.counts[arm_idx] += 1
        self.time += 1
        self.counts_evolution.append(self.counts.copy())

        p_values = [1.0] * self.n
        current_St = set()

        if self.control_arm_idx is not None:
            current_St = _two_sample_nm_lcb_discoveries(
                self,
                k_start=self.n,
                delta_denominator=self.n,
            )
        else:
            p_values_with_idx = [(self.get_anytime_pvalue(i), i) for i in range(self.n)]
            p_values = [pv for pv, _ in sorted(p_values_with_idx, key=lambda x: x[1])]
            p_values_with_idx.sort(key=lambda x: x[0])
            for k in range(self.n, 0, -1):
                p_val_k = p_values_with_idx[k - 1][0]
                effective_delta = self.delta * k / self.n
                if p_val_k <= effective_delta:
                    for rank in range(k):
                        current_St.add(p_values_with_idx[rank][1])
                    break

        self.S_t.update(current_St)
        return p_values

    def bh_update(self, arm_idx, observation):
        self._update_stats(arm_idx, observation)
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


# =============================================================================
# SIMULATION ENGINE
# =============================================================================
def _run_single_simulation(algo, no_sim, all_arm_data, horizon, mode,
                           control_arm, init_nb, init_choice, variable_mu_choice,
                           n_arms, is_true_mean, true_positives):
    """
    Runs a single simulation for a given algorithm instance.
    Uses sequential data access (not bootstrap) for fair comparison.
    """
    p_values_list = []
    all_arm_counts = [0 for _ in range(n_arms)]
    run_pr = []

    # --- Init ---
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
        if variable_mu_choice:
            # Two-sample mode: mu_0 update handled internally via control_arm_idx
            arm = algo.select_arm()
            # Force the draw of the control arm to reduce its uncertainty
            if all_arm_counts[control_arm] < max(all_arm_counts):
                arm = control_arm
        else:
            arm = algo.select_arm()

        if arm == "stop":
            print("stop triggered")
            remaining_steps = horizon - len(run_pr)

            last_pr = run_pr[-1] if run_pr else 0
            run_pr.extend([last_pr] * remaining_steps)

            last_counts = algo.counts_evolution[-1]
            for _ in range(remaining_steps):
                algo.counts_evolution.append(last_counts.copy())

            last_p_values = p_values_list[-1] if p_values_list else [1.0 for _ in range(n_arms)]
            for _ in range(remaining_steps):
                p_values_list.append(list(last_p_values))
            break

        else:
            # Sequential data access for fair comparison
            len_arm = len(all_arm_data[no_sim][arm])
            if all_arm_counts[arm] >= len_arm:
                observation = np.random.choice(all_arm_data[no_sim][arm]) 
            else:
                observation = all_arm_data[no_sim][arm][all_arm_counts[arm]]

            all_arm_counts[arm] += 1

            p_values_t = algo.bh_update_optimized(arm, observation)
            p_values_list.append(p_values_t)

            if is_true_mean:
                nb_found = len(algo.S_t.intersection(true_positives))
                current_tpr = nb_found / len(true_positives) if true_positives else 1.0
                run_pr.append(current_tpr)
            else:
                nb_found = len(algo.S_t)
                run_pr.append(nb_found)

    return run_pr, p_values_list


def run_experiment(arms, mu_0, delta, horizon, mode, all_arm_data, n_simulations,
                   control_arm, init_nb, init_choice, variable_mu_choice, is_true_mean,
                   rho=0.01, cs_type='nm_m2'):
    """
    Runs the bandit experiment.

    Parameters
    ----------
    arms : array-like
        The arms of the bandit (true means if known).
    mu_0 : float
        The null hypothesis mean.
    delta : float
        The significance level.
    horizon : int
        The total budget per simulation.
    mode : str
        'adaptive' or 'uniform'.
    all_arm_data : list of list of list
        Pre-generated rewards [sim][arm][pull].
    n_simulations : int
        Number of independent runs.
    control_arm : int
        Index of the control arm.
    init_nb : int
        Number of initial pulls per arm.
    init_choice : bool
        Whether to use the initialization process.
    variable_mu_choice : bool
        Whether to use variable mu_0 from control arm.
    is_true_mean : bool
        Whether to compute TPR (requires known arm means).
    rho : float
        NM tuning parameter. Default: 0.01.
    cs_type : str
        'normal_mixture' or 'nm_m2'. Default: 'nm_m2'.
    Returns
    -------
    pnb_history_mean, pnb_list, counts_history_mean, counts_list,
    np_p_values_list_by_sim, np_p_values_mean, list_positive
    """
    print(f"EXECUTION RUN EXP — cs_type={cs_type}")
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
        'adaptive': lambda: JamiesonJainAlgo(n_arms, mu_0, delta, rho=rho, cs_type=cs_type,
                                             control_arm_idx=control_arm if variable_mu_choice else None),
        'uniform': lambda: UniformAlgo(n_arms, mu_0, delta, rho=rho, cs_type=cs_type,
                                       control_arm_idx=control_arm if variable_mu_choice else None),
    }

    if mode not in algo_factory:
        raise ValueError("Algorithm name not detected, choose between uniform and adaptive")

    mode_label = f"{mode.upper()} VAR" if variable_mu_choice else mode.upper()
    print(f"Simulation Mode: {mode_label} | CS: {cs_type} | rho={rho} ({n_simulations} runs)")

    # --- Simulation loop ---
    for no_sim in tqdm(range(n_simulations)):

        algo = algo_factory[mode]()

        run_pr, p_values_list = _run_single_simulation(
            algo, no_sim, all_arm_data, horizon, mode,
            control_arm, init_nb, init_choice, variable_mu_choice,
            n_arms, is_true_mean, true_positives
        )

        list_positive.append(algo.S_t)

        pnb_i = np.array(run_pr)
        pnb_list.append(pnb_i)

        counts_arr = np.array(algo.counts_evolution)
        counts_list.append(counts_arr)
        counts_evolution_sum += counts_arr
        p_values_list_by_sim.append(p_values_list)

    # --- Aggregation ---
    pnb_history_mean = np.mean(np.array(pnb_list), axis=0)
    counts_history_mean = counts_evolution_sum / n_simulations

    max_length = max(len(arr) for arr in p_values_list_by_sim)
    n_sims_total = len(p_values_list_by_sim)
    padded_array = np.full((n_sims_total, max_length, n_arms), np.nan)
    for i, arr in enumerate(p_values_list_by_sim):
        arr_np = np.array(arr)
        padded_array[i, :arr_np.shape[0], :] = arr_np
    np_p_values_mean = np.nanmean(padded_array, axis=0)
    np_p_values_list_by_sim = padded_array

    return (pnb_history_mean, pnb_list, counts_history_mean, counts_list,
            np_p_values_list_by_sim, np_p_values_mean, list_positive)
