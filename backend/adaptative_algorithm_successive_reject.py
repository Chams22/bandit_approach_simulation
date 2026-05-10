import numpy as np
from tqdm import tqdm

# -----------------------------------------------------------------------------
# Successive Rejects (Audibert, Bubeck & Munos, COLT 2010)
# -----------------------------------------------------------------------------
# Arm selection: round-robin over active arms; at the end of each SR phase the
# arm with the lowest empirical mean (vs mu_0 or control) is eliminated.
# Inference: same NM p-values + BH as JamiesonJainAlgo / UniformAlgo so that
# the ONLY difference between the three algorithms is the sampling strategy.
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


def _two_sample_nm_lcb_discoveries(algo, k_start, delta_denominator):
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
        phi_ctrl = algo.phi(control_count, effective_delta, control_var)
        scores = (gaps
                  - _phi_vector(algo, treatment_counts, effective_delta, treatment_vars)
                  - phi_ctrl)
        passing_idx = treatment_idx[scores >= 0]
        if passing_idx.size >= k:
            return set(passing_idx.tolist())
    return set()


class SuccessiveRejectsAlgo:
    """
    Successive Rejects with NM confidence sequences and BH inference.

    Phase schedule (Audibert et al. 2010):
        log_bar_K = 1/2 + sum(1/k for k in 2..K)
        n_k = ceil((horizon - K) / (log_bar_K * (K + 1 - k)))   for phase k
    At the end of phase k every active arm has been pulled n_k times total;
    the arm with the lowest empirical mean is then eliminated.
    """

    def __init__(self, n_arms, mu_0, delta, rho=0.01, cs_type='normal_mixture',
                 control_arm_idx=None, horizon=10000):
        if cs_type != 'normal_mixture':
            raise ValueError("SuccessiveRejectsAlgo only supports 'normal_mixture'.")

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

        # Arms subject to elimination (control is never eliminated)
        self._rankable = [i for i in range(n_arms) if i != control_arm_idx]
        K = len(self._rankable)

        # SR cumulative pull targets per rankable arm at end of each phase
        log_bar_K = 0.5 + sum(1.0 / k for k in range(2, K + 1)) if K > 1 else 1.0
        self._phase_targets = [0]
        for k in range(1, K):
            nk = int(np.ceil((horizon - K) / (log_bar_K * (K + 1 - k))))
            nk = max(nk, self._phase_targets[-1] + 1)
            self._phase_targets.append(nk)

        self._active = list(self._rankable)
        self._phase = 0   # phases completed so far
        self._rr_ptr = 0  # round-robin pointer within current active set

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _update_stats(self, arm_idx, observation):
        n = self.counts[arm_idx]
        old_mean = self.emp_means[arm_idx]
        self.emp_means[arm_idx] = (old_mean * n + observation) / (n + 1)
        self.emp_vars[arm_idx] += (observation - old_mean) ** 2

    def phi(self, t, delta_val, var_stat):
        if t == 0:
            return float('inf')
        log_term = max(0.0, np.log(np.sqrt((var_stat + self.rho) / self.rho) / delta_val))
        return np.sqrt(2 * (var_stat + self.rho) * log_term) / t

    def get_anytime_pvalue(self, arm_idx):
        t = self.counts[arm_idx]
        if t == 0:
            return 1.0
        mu0_ref = (self.emp_means[self.control_arm_idx]
                   if self.control_arm_idx is not None else self.mu_0)
        diff = self.emp_means[arm_idx] - mu0_ref
        if diff <= 0:
            return 1.0
        var_stat = self.emp_vars[arm_idx]
        p = (np.sqrt((var_stat + self.rho) / self.rho)
             * np.exp(-diff ** 2 * t ** 2 / (2 * (var_stat + self.rho))))
        return float(np.clip(p, 1e-300, 1.0))

    def _eliminate_worst(self):
        if len(self._active) <= 1:
            return
        mu0_ref = (self.emp_means[self.control_arm_idx]
                   if self.control_arm_idx is not None else self.mu_0)
        worst = min(self._active, key=lambda i: self.emp_means[i] - mu0_ref)
        self._active.remove(worst)
        self._phase += 1
        self._rr_ptr = 0

    def _bh_update_St(self):
        if self.control_arm_idx is not None:
            current_St = _two_sample_nm_lcb_discoveries(
                self, k_start=self.n, delta_denominator=self.n)
        else:
            p_values_with_idx = [(self.get_anytime_pvalue(i), i) for i in range(self.n)]
            p_values_with_idx.sort(key=lambda x: x[0])
            current_St = set()
            for k in range(self.n, 0, -1):
                if p_values_with_idx[k - 1][0] <= self.delta * k / self.n:
                    for rank in range(k):
                        current_St.add(p_values_with_idx[rank][1])
                    break
        self.S_t.update(current_St)

    # ------------------------------------------------------------------
    # Public interface (matches JamiesonJainAlgo / UniformAlgo)
    # ------------------------------------------------------------------

    def init_process(self, data):
        for arm_idx, arm_data in enumerate(data):
            for obs in arm_data:
                self._update_stats(arm_idx, obs)
                self.counts[arm_idx] += 1
                self.time += 1
                self.counts_evolution.append(self.counts.copy())
        self._bh_update_St()

    def select_arm(self):
        candidates = [i for i in self._active
                      if i not in self.S_t and i != self.control_arm_idx]
        if not candidates:
            return "stop"
        idx = self._rr_ptr % len(candidates)
        self._rr_ptr += 1
        return candidates[idx]

    def bh_update_optimized(self, arm_idx, observation):
        self._update_stats(arm_idx, observation)
        self.counts[arm_idx] += 1
        self.time += 1
        self.counts_evolution.append(self.counts.copy())

        # Check phase advancement
        next_phase = self._phase + 1
        if next_phase < len(self._phase_targets) and len(self._active) > 1:
            target = self._phase_targets[next_phase]
            if all(self.counts[i] >= target for i in self._active):
                self._eliminate_worst()

        self._bh_update_St()
        return [self.get_anytime_pvalue(i) for i in range(self.n)]

    def bh_update(self, arm_idx, observation):
        return self.bh_update_optimized(arm_idx, observation)


# Alias kept so this module exposes the same adaptive class name as the other
# algorithm files.
JamiesonJainAlgo = SuccessiveRejectsAlgo


# =============================================================================
# UNIFORM ALGORITHM
# =============================================================================
class UniformAlgo:
    def __init__(self, n_arms, mu_0, delta, rho=0.01, cs_type='normal_mixture',
                 control_arm_idx=None):
        """
        Uniform random sampling with the same NM/BH inference as SuccessiveRejectsAlgo.
        """
        if cs_type != 'normal_mixture':
            raise ValueError("UniformAlgo only supports 'normal_mixture'.")

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

    def _update_stats(self, arm_idx, observation):
        n = self.counts[arm_idx]
        old_mean = self.emp_means[arm_idx]
        self.emp_means[arm_idx] = (old_mean * n + observation) / (n + 1)
        self.emp_vars[arm_idx] += (observation - old_mean) ** 2

    def phi(self, t, delta_val, var_stat):
        if t == 0:
            return float('inf')
        log_term = max(0.0, np.log(np.sqrt((var_stat + self.rho) / self.rho) / delta_val))
        return np.sqrt(2 * (var_stat + self.rho) * log_term) / t

    def get_anytime_pvalue(self, arm_idx):
        t = self.counts[arm_idx]
        if t == 0:
            return 1.0
        mu0_ref = (self.emp_means[self.control_arm_idx]
                   if self.control_arm_idx is not None else self.mu_0)
        diff = self.emp_means[arm_idx] - mu0_ref
        if diff <= 0:
            return 1.0
        var_stat = self.emp_vars[arm_idx]
        p = (np.sqrt((var_stat + self.rho) / self.rho)
             * np.exp(-diff ** 2 * t ** 2 / (2 * (var_stat + self.rho))))
        return float(np.clip(p, 1e-300, 1.0))

    def init_process(self, data):
        for arm_idx, arm_data in enumerate(data):
            for obs in arm_data:
                self._update_stats(arm_idx, obs)
                self.counts[arm_idx] += 1
                self.time += 1
                self.counts_evolution.append(self.counts.copy())
        self._bh_update_St()

    def select_arm(self):
        if self.control_arm_idx is not None:
            candidates = [i for i in range(self.n) if i != self.control_arm_idx]
            if not candidates:
                return "stop"
            return np.random.choice(candidates)
        return np.random.randint(self.n)

    def _bh_update_St(self):
        if self.control_arm_idx is not None:
            current_St = _two_sample_nm_lcb_discoveries(
                self, k_start=self.n, delta_denominator=self.n)
        else:
            p_values_with_idx = [(self.get_anytime_pvalue(i), i) for i in range(self.n)]
            p_values_with_idx.sort(key=lambda x: x[0])
            current_St = set()
            for k in range(self.n, 0, -1):
                if p_values_with_idx[k - 1][0] <= self.delta * k / self.n:
                    for rank in range(k):
                        current_St.add(p_values_with_idx[rank][1])
                    break
        self.S_t.update(current_St)

    def bh_update_optimized(self, arm_idx, observation):
        self._update_stats(arm_idx, observation)
        self.counts[arm_idx] += 1
        self.time += 1
        self.counts_evolution.append(self.counts.copy())
        self._bh_update_St()
        return [self.get_anytime_pvalue(i) for i in range(self.n)]

    def bh_update(self, arm_idx, observation):
        return self.bh_update_optimized(arm_idx, observation)


# =============================================================================
# SIMULATION ENGINE  (mirror of adaptative_algorithm_v2.run_experiment)
# =============================================================================
def _should_record_history(step, horizon, history_record_every):
    return step == horizon or step % history_record_every == 0


def _run_single_simulation(algo, no_sim, all_arm_data, horizon, mode,
                           control_arm, init_nb, init_choice, variable_mu_choice,
                           n_arms, is_true_mean, true_positives,
                           history_record_every=1):
    p_values_list = []
    all_arm_counts = [0] * n_arms
    run_pr = []
    history_record_every = max(1, int(history_record_every))

    if init_choice and init_nb > 0:
        data_init = [all_arm_data[no_sim][arm][:init_nb] for arm in range(n_arms)]
        algo.init_process(data_init)
        all_arm_counts = [init_nb] * n_arms
        algo.counts_evolution = [algo.counts.copy()]

    discovery_times = {int(arm): 0 for arm in algo.S_t}

    for t in range(horizon):
        if variable_mu_choice:
            arm = algo.select_arm()
            if all_arm_counts[control_arm] < max(all_arm_counts):
                arm = control_arm
        else:
            arm = algo.select_arm()

        if arm == "stop":
            current_done = len(run_pr)
            remaining = horizon - current_done
            if is_true_mean:
                nb_found = len(algo.S_t.intersection(true_positives))
                last_pr = nb_found / len(true_positives) if true_positives else 1.0
            else:
                last_pr = len(algo.S_t)
            run_pr.extend([last_pr] * remaining)
            last_counts = algo.counts_evolution[-1]
            for step in range(current_done + 1, horizon + 1):
                if _should_record_history(step, horizon, history_record_every):
                    algo.counts_evolution.append(last_counts.copy())
            last_pv = p_values_list[-1] if p_values_list else [1.0] * n_arms
            for step in range(current_done + 1, horizon + 1):
                if _should_record_history(step, horizon, history_record_every):
                    p_values_list.append(list(last_pv))
            break

        len_arm = len(all_arm_data[no_sim][arm])
        if all_arm_counts[arm] >= len_arm:
            observation = np.random.choice(all_arm_data[no_sim][arm])
        else:
            observation = all_arm_data[no_sim][arm][all_arm_counts[arm]]

        all_arm_counts[arm] += 1
        current_step = len(run_pr) + 1
        p_values_t = algo.bh_update_optimized(arm, observation)
        if _should_record_history(current_step, horizon, history_record_every):
            p_values_list.append(p_values_t)
        else:
            algo.counts_evolution.pop()

        for discovered_arm in algo.S_t:
            discovery_times.setdefault(int(discovered_arm), current_step)

        if is_true_mean:
            nb_found = len(algo.S_t.intersection(true_positives))
            run_pr.append(nb_found / len(true_positives) if true_positives else 1.0)
        else:
            run_pr.append(len(algo.S_t))

    return run_pr, p_values_list, discovery_times


def run_experiment(arms, mu_0, delta, horizon, mode, all_arm_data, n_simulations,
                   control_arm, init_nb, init_choice, variable_mu_choice,
                   is_true_mean, rho=0.01, cs_type='normal_mixture',
                   return_discovery_times=False, history_record_every=1):
    """
    Run the experiment with the same public interface as adaptative_algorithm_v2.

    mode:
        'adaptive' or 'successive_reject' -> SuccessiveRejectsAlgo
        'uniform'                         -> UniformAlgo
    """
    print(f"EXECUTION SUCCESSIVE REJECT MODULE — cs_type={cs_type} | rho={rho}")
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

    pnb_list, counts_list, p_values_list_by_sim, list_positive = [], [], [], []
    discovery_times_list = []
    n_history_points = 1 + sum(
        1 for step in range(1, horizon + 1)
        if _should_record_history(step, horizon, history_record_every)
    )
    counts_evolution_sum = np.zeros((n_history_points, n_arms))

    algo_factory = {
        'adaptive': lambda: SuccessiveRejectsAlgo(
            n_arms, mu_0, delta, rho=rho, cs_type=cs_type,
            control_arm_idx=control_arm if variable_mu_choice else None,
            horizon=horizon,
        ),
        'successive_reject': lambda: SuccessiveRejectsAlgo(
            n_arms, mu_0, delta, rho=rho, cs_type=cs_type,
            control_arm_idx=control_arm if variable_mu_choice else None,
            horizon=horizon,
        ),
        'successive_rejects': lambda: SuccessiveRejectsAlgo(
            n_arms, mu_0, delta, rho=rho, cs_type=cs_type,
            control_arm_idx=control_arm if variable_mu_choice else None,
            horizon=horizon,
        ),
        'sr': lambda: SuccessiveRejectsAlgo(
            n_arms, mu_0, delta, rho=rho, cs_type=cs_type,
            control_arm_idx=control_arm if variable_mu_choice else None,
            horizon=horizon,
        ),
        'uniform': lambda: UniformAlgo(
            n_arms, mu_0, delta, rho=rho, cs_type=cs_type,
            control_arm_idx=control_arm if variable_mu_choice else None,
        ),
    }

    if mode not in algo_factory:
        raise ValueError("Algorithm name not detected, choose between uniform and adaptive")

    mode_label = f"{mode.upper()} VAR" if variable_mu_choice else mode.upper()
    print(f"Simulation Mode: {mode_label} ({n_simulations} runs)")

    for no_sim in tqdm(range(n_simulations)):
        algo = algo_factory[mode]()
        run_pr, p_values_list, discovery_times = _run_single_simulation(
            algo, no_sim, all_arm_data, horizon, mode,
            control_arm, init_nb, init_choice, variable_mu_choice,
            n_arms, is_true_mean, true_positives, history_record_every,
        )
        list_positive.append(algo.S_t)
        pnb_list.append(np.array(run_pr))
        counts_arr = np.array(algo.counts_evolution)
        counts_list.append(counts_arr)
        counts_evolution_sum += counts_arr
        p_values_list_by_sim.append(p_values_list)
        discovery_times_list.append(discovery_times)

    pnb_history_mean = np.mean(np.array(pnb_list), axis=0)
    counts_history_mean = counts_evolution_sum / n_simulations

    max_length = max(len(a) for a in p_values_list_by_sim)
    padded = np.full((n_simulations, max_length, n_arms), np.nan)
    for i, arr in enumerate(p_values_list_by_sim):
        arr_np = np.array(arr)
        padded[i, :arr_np.shape[0], :] = arr_np
    np_p_values_mean = np.nanmean(padded, axis=0)
    np_p_values_list_by_sim = padded

    result = (pnb_history_mean, pnb_list, counts_history_mean, counts_list,
              np_p_values_list_by_sim, np_p_values_mean, list_positive)
    if return_discovery_times:
        return (*result, discovery_times_list)
    return result
