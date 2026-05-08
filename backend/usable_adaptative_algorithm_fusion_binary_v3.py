import numpy as np
from tqdm import tqdm

_MIN_PULLS_BEFORE_FILTER = 10
_EPSILON_RECHECK = 0.02

# -----------------------------------------------------------------------------
# Adapted for Binary (Bernoulli) Data — V3
# -----------------------------------------------------------------------------
# Two confidence sequence options via cs_type parameter:
#   'normal_mixture'  — Original V_hat NM (Howard et al., 2021). General purpose.
#   'betting'         — Bernoulli betting martingale (Shafer/Vovk/Ramdas).
#                       No tuning parameter, asymptotically optimal for binary data.
#
# V3 change vs V2: two-sample betting uses a rigorous paired e-process.
#   Instead of betting on (X_arm - mu_hat_control), we bet on the direct
#   paired difference (X_arm - X_control) where X_control is drawn simultaneously.
#   D = X_arm - X_control ∈ {-1, 0, 1}, E[D] = 0 under H0: mu_arm = mu_control.
#   This restores the martingale property under H0.
# -----------------------------------------------------------------------------

class JamiesonJainAlgo:
    def __init__(self, n_arms, mu_0, delta, rho=0.01, cs_type='betting', control_arm_idx=None):
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
            Tuning parameter for Normal Mixture CS (prior variance).
            Ignored when cs_type='betting'. Default: 0.01.
        cs_type : str
            Confidence sequence type:
            - 'normal_mixture': Original V_hat NM (Howard et al., 2021).
            - 'betting': Bernoulli betting martingale. No rho needed,
              asymptotically optimal for binary data.
        control_arm_idx : int or None
            If set, enables two-sample CS mode: tests mu_arm - mu_control > 0
            instead of mu_arm > mu_0. Default: None (single-sample mode).
        """
        self.n = n_arms
        self.mu_0 = mu_0
        self.delta = delta
        self.rho = rho
        self.cs_type = cs_type
        self.control_arm_idx = control_arm_idx

        self.counts = np.zeros(n_arms, dtype=int)
        self.emp_means = np.zeros(n_arms, dtype=float)
        self.emp_vars = np.zeros(n_arms, dtype=float)
        self.time = 0
        self.S_t = set()

        if cs_type == 'betting':
            self.log_martingale = np.zeros(n_arms, dtype=float)

        self.counts_evolution = [np.zeros(n_arms, dtype=int)]

    # -------------------------------------------------------------------------
    # Statistics update
    # -------------------------------------------------------------------------
    def _update_stats(self, arm_idx, observation, x_control=None,
                      control_mean_prev=None, control_var_prev=None, control_count_prev=None):
        """
        Updates empirical statistics for a given arm.

        x_control : float or None
            Paired control observation. Bet placed on (observation - x_control).
        control_mean_prev / control_var_prev / control_count_prev : float or None
            Previous control arm stats (before current step). Must be passed for
            predictability: lambda must be F_{t-1}-measurable, so it cannot
            depend on x_control or the updated control mean.
        """
        n = self.counts[arm_idx]
        old_mean = self.emp_means[arm_idx]

        if self.cs_type == 'betting' and n >= 1:
            if control_mean_prev is not None:
                mu0 = control_mean_prev
            elif self.control_arm_idx is not None:
                mu0 = self.emp_means[self.control_arm_idx]
            else:
                mu0 = self.mu_0
            mu0_safe = np.clip(mu0, 1e-6, 1.0 - 1e-6)

            if n >= 2:
                diff_prev = old_mean - mu0_safe
                if diff_prev > 0:
                    if x_control is not None and self.control_arm_idx is not None:
                        var_arm = max(old_mean * (1.0 - old_mean), 1e-8)
                        var_ctrl = max(mu0_safe * (1.0 - mu0_safe), 1e-8)
                        lam = diff_prev / (var_arm + var_ctrl)
                        lam = np.clip(lam, 0.0, 1.0 - 1e-6)
                        wealth_factor = 1.0 + lam * (observation - x_control)
                    else:
                        lam = diff_prev / (mu0_safe * (1.0 - mu0_safe))
                        lam = min(lam, 1.0 / mu0_safe - 1e-6)
                        lam = max(lam, 0.0)
                        wealth_factor = 1.0 + lam * (observation - mu0_safe)

                    if wealth_factor > 0:
                        self.log_martingale[arm_idx] += np.log(wealth_factor)

        self.emp_means[arm_idx] = (old_mean * n + observation) / (n + 1)
        self.emp_vars[arm_idx] += (observation - old_mean) ** 2

    # -------------------------------------------------------------------------
    # Confidence radius phi (Normal Mixture)
    # -------------------------------------------------------------------------
    def phi(self, t, delta_val, V_hat):
        """
        Normal Mixture confidence sequence radius (Howard et al., 2021).
        Used for UCB arm selection in all modes, and for p-values in NM mode.
        """
        if t == 0:
            return float('inf')

        log_term = np.log(np.sqrt((V_hat + self.rho) / self.rho) / delta_val)
        log_term = max(0.0, log_term)

        return np.sqrt(2 * (V_hat + self.rho) * log_term) / t

    # -------------------------------------------------------------------------
    # Anytime p-value
    # -------------------------------------------------------------------------
    def get_anytime_pvalue(self, arm_idx):
        """
        Computes the anytime p-value for a given arm.

        - 'normal_mixture': Closed-form NM inversion.
        - 'betting': p = exp(-log_martingale) via Ville's inequality.
        """
        t = self.counts[arm_idx]
        if t == 0:
            return 1.0

        mu0_ref = self.emp_means[self.control_arm_idx] if self.control_arm_idx is not None else self.mu_0
        diff = self.emp_means[arm_idx] - mu0_ref
        if diff <= 0:
            return 1.0

        if self.cs_type == 'betting':
            p_value = np.exp(-self.log_martingale[arm_idx])
            return float(np.clip(p_value, 1e-300, 1.0))
        else:
            V_hat = self.emp_vars[arm_idx]
            p_value = (np.sqrt((V_hat + self.rho) / self.rho)
                       * np.exp(-diff ** 2 * t ** 2 / (2 * (V_hat + self.rho))))
            return float(np.clip(p_value, 1e-300, 1.0))

    # -------------------------------------------------------------------------
    # Init process
    # -------------------------------------------------------------------------
    def init_process(self, data):
        """
        Initializes the algorithm with pre-collected data for each arm.

        Two-sample betting mode: sequential interleaved processing.
        For each index obs_idx, we first save the current control stats
        (F_{t-1}-measurable), process each test arm with the paired control
        observation using those saved stats, then update the control arm.
        This preserves predictability throughout init.

        All other modes: original sequential processing.
        """
        if self.control_arm_idx is not None and self.cs_type == 'betting':
            ctrl_data = data[self.control_arm_idx]
            for obs_idx in range(len(ctrl_data)):
                ctrl_mean_prev = self.emp_means[self.control_arm_idx]
                ctrl_var_prev = self.emp_vars[self.control_arm_idx]
                ctrl_count_prev = self.counts[self.control_arm_idx]
                x_ctrl = ctrl_data[obs_idx]
                for arm_idx, arm_data in enumerate(data):
                    if arm_idx == self.control_arm_idx:
                        continue
                    if obs_idx < len(arm_data):
                        self._update_stats(arm_idx, arm_data[obs_idx], x_control=x_ctrl,
                                           control_mean_prev=ctrl_mean_prev,
                                           control_var_prev=ctrl_var_prev,
                                           control_count_prev=ctrl_count_prev)
                        self.counts[arm_idx] += 1
                        self.time += 1
                        self.counts_evolution.append(self.counts.copy())
                self._update_stats(self.control_arm_idx, x_ctrl)
                self.counts[self.control_arm_idx] += 1
                self.time += 1
                self.counts_evolution.append(self.counts.copy())
        else:
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
            self.rho = float(np.median(var_estimates))

        # --- BH after init ---
        p_values_with_idx = [(self.get_anytime_pvalue(i), i) for i in range(self.n)]
        p_values_with_idx.sort(key=lambda x: x[0])
        for k in range(self.n, 0, -1):
            if p_values_with_idx[k - 1][0] <= self.delta * k / self.n:
                for rank in range(k):
                    self.S_t.add(p_values_with_idx[rank][1])
                break

    # -------------------------------------------------------------------------
    # Arm selection (UCB)
    # -------------------------------------------------------------------------
    def select_arm(self):
        """
        UCB-based arm selection with active arm filtering.
        Arms whose UCB stays below the detection threshold after _MIN_PULLS_BEFORE_FILTER
        pulls are classified as inactive and only revisited with probability _EPSILON_RECHECK.
        Fallback: if all arms are inactive, pick the best UCB among all treatment arms.
        """
        unsampled = [i for i in range(self.n)
                     if self.counts[i] == 0 and i != self.control_arm_idx]
        if unsampled:
            return unsampled[0]

        treatment_arms = [i for i in range(self.n)
                          if i not in self.S_t and i != self.control_arm_idx]
        if not treatment_arms:
            return "stop"

        threshold = 0.0 if self.control_arm_idx is not None else self.mu_0

        def _ucb(i):
            if self.control_arm_idx is not None:
                return ((self.emp_means[i] - self.emp_means[self.control_arm_idx])
                        + self.phi(self.counts[i], self.delta, self.emp_vars[i])
                        + self.phi(self.counts[self.control_arm_idx], self.delta,
                                   self.emp_vars[self.control_arm_idx]))
            return self.emp_means[i] + self.phi(self.counts[i], self.delta, self.emp_vars[i])

        active, inactive = [], []
        for i in treatment_arms:
            if self.counts[i] < _MIN_PULLS_BEFORE_FILTER or _ucb(i) > threshold:
                active.append(i)
            else:
                inactive.append(i)

        if not active:
            return max(treatment_arms, key=_ucb)

        if inactive and np.random.rand() < _EPSILON_RECHECK:
            return np.random.choice(inactive)

        return max(active, key=_ucb)

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
    def bh_update_optimized(self, arm_idx, observation, x_control=None):
        """
        Updates state and runs p-value-sorted BH procedure in O(n log n).

        x_control : float or None
            Paired control arm observation (two-sample betting mode only).
            If provided and arm_idx != control_arm_idx, also updates control arm stats.
        """
        # Save F_{t-1}-measurable control stats before any update
        control_mean_prev = None
        control_var_prev = None
        control_count_prev = None
        if (x_control is not None
                and self.control_arm_idx is not None
                and arm_idx != self.control_arm_idx):
            control_mean_prev = self.emp_means[self.control_arm_idx]
            control_var_prev = self.emp_vars[self.control_arm_idx]
            control_count_prev = self.counts[self.control_arm_idx]

        # Update test arm first with F_{t-1}-measurable prev control stats
        self._update_stats(arm_idx, observation, x_control=x_control,
                           control_mean_prev=control_mean_prev,
                           control_var_prev=control_var_prev,
                           control_count_prev=control_count_prev)
        self.counts[arm_idx] += 1

        # THEN update control arm
        if control_mean_prev is not None:
            self._update_stats(self.control_arm_idx, x_control)
            self.counts[self.control_arm_idx] += 1

        self.time += 1
        self.counts_evolution.append(self.counts.copy())

        p_values = [1.0] * self.n
        current_St = set()

        if self.control_arm_idx is not None:
            if self.cs_type == 'betting':
                # Two-sample betting: p-values from paired martingale
                p_values_with_idx = [(self.get_anytime_pvalue(i), i)
                                     for i in range(self.n) if i != self.control_arm_idx]
                p_values_with_idx.sort(key=lambda x: x[0])
                n_tested = self.n - 1
                for k in range(n_tested, 0, -1):
                    if p_values_with_idx[k - 1][0] <= self.delta * k / n_tested:
                        for rank in range(k):
                            current_St.add(p_values_with_idx[rank][1])
                        break
            else:
                # Two-sample NM LCB
                # Budget split by 2 for union bound: each phi uses alpha/2
                # so P(CI_arm fails OR CI_ctrl fails) <= alpha/2 + alpha/2 = alpha
                # Denominator n-1: we test n-1 hypotheses (control excluded)
                n_tested = self.n - 1
                for k in range(n_tested, 0, -1):
                    effective_delta = self.delta * k / n_tested
                    phi_ctrl = self.phi(self.counts[self.control_arm_idx], effective_delta / 2, self.emp_vars[self.control_arm_idx])
                    passing_arms = [
                        i for i in range(self.n)
                        if i != self.control_arm_idx
                        and (self.emp_means[i] - self.emp_means[self.control_arm_idx])
                            - self.phi(self.counts[i], effective_delta / 2, self.emp_vars[i])
                            - phi_ctrl >= 0
                    ]
                    if len(passing_arms) >= k:
                        current_St = set(passing_arms)
                        break
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
    def __init__(self, n_arms, mu_0, delta, rho=0.01, cs_type='betting', control_arm_idx=None):
        """
        Uniform (random) sampling algorithm with the same CS options and
        two-sample support as JamiesonJainAlgo. select_arm draws uniformly
        among test arms (control excluded in two-sample mode).
        """
        self.n = n_arms
        self.mu_0 = mu_0
        self.delta = delta
        self.rho = rho
        self.cs_type = cs_type
        self.control_arm_idx = control_arm_idx

        self.counts = np.zeros(n_arms, dtype=int)
        self.emp_means = np.zeros(n_arms, dtype=float)
        self.emp_vars = np.zeros(n_arms, dtype=float)
        self.time = 0
        self.S_t = set()

        if cs_type == 'betting':
            self.log_martingale = np.zeros(n_arms, dtype=float)

        self.counts_evolution = [np.zeros(n_arms, dtype=int)]

    def init_process(self, data):
        if self.control_arm_idx is not None and self.cs_type == 'betting':
            ctrl_data = data[self.control_arm_idx]
            for obs_idx in range(len(ctrl_data)):
                ctrl_mean_prev = self.emp_means[self.control_arm_idx]
                ctrl_var_prev = self.emp_vars[self.control_arm_idx]
                ctrl_count_prev = self.counts[self.control_arm_idx]
                x_ctrl = ctrl_data[obs_idx]
                for arm_idx, arm_data in enumerate(data):
                    if arm_idx == self.control_arm_idx:
                        continue
                    if obs_idx < len(arm_data):
                        self._update_stats(arm_idx, arm_data[obs_idx], x_control=x_ctrl,
                                           control_mean_prev=ctrl_mean_prev,
                                           control_var_prev=ctrl_var_prev,
                                           control_count_prev=ctrl_count_prev)
                        self.counts[arm_idx] += 1
                        self.time += 1
                        self.counts_evolution.append(self.counts.copy())
                self._update_stats(self.control_arm_idx, x_ctrl)
                self.counts[self.control_arm_idx] += 1
                self.time += 1
                self.counts_evolution.append(self.counts.copy())
        else:
            for arm_idx, arm_data in enumerate(data):
                for obs in arm_data:
                    self._update_stats(arm_idx, obs)
                    self.counts[arm_idx] += 1
                    self.time += 1
                    self.counts_evolution.append(self.counts.copy())

        var_estimates = [self.emp_vars[i] / max(self.counts[i] - 1, 1)
                         for i in range(self.n) if self.counts[i] > 1]
        if var_estimates:
            self.rho = float(np.median(var_estimates))

        p_values_with_idx = [(self.get_anytime_pvalue(i), i) for i in range(self.n)]
        p_values_with_idx.sort(key=lambda x: x[0])
        for k in range(self.n, 0, -1):
            if p_values_with_idx[k - 1][0] <= self.delta * k / self.n:
                for rank in range(k):
                    self.S_t.add(p_values_with_idx[rank][1])
                break

    def _update_stats(self, arm_idx, observation, x_control=None,
                      control_mean_prev=None, control_var_prev=None, control_count_prev=None):
        n = self.counts[arm_idx]
        old_mean = self.emp_means[arm_idx]

        if self.cs_type == 'betting' and n >= 1:
            if control_mean_prev is not None:
                mu0 = control_mean_prev
            elif self.control_arm_idx is not None:
                mu0 = self.emp_means[self.control_arm_idx]
            else:
                mu0 = self.mu_0
            mu0_safe = np.clip(mu0, 1e-6, 1.0 - 1e-6)
            if n >= 2:
                diff_prev = old_mean - mu0_safe
                if diff_prev > 0:
                    if x_control is not None and self.control_arm_idx is not None:
                        var_arm = max(old_mean * (1.0 - old_mean), 1e-8)
                        var_ctrl = max(mu0_safe * (1.0 - mu0_safe), 1e-8)
                        lam = diff_prev / (var_arm + var_ctrl)
                        lam = np.clip(lam, 0.0, 1.0 - 1e-6)
                        wealth_factor = 1.0 + lam * (observation - x_control)
                    else:
                        lam = diff_prev / (mu0_safe * (1.0 - mu0_safe))
                        lam = min(lam, 1.0 / mu0_safe - 1e-6)
                        lam = max(lam, 0.0)
                        wealth_factor = 1.0 + lam * (observation - mu0_safe)
                    if wealth_factor > 0:
                        self.log_martingale[arm_idx] += np.log(wealth_factor)

        self.emp_means[arm_idx] = (old_mean * n + observation) / (n + 1)
        self.emp_vars[arm_idx] += (observation - old_mean) ** 2

    def phi(self, t, delta_val, V_hat):
        if t == 0:
            return float('inf')
        log_term = np.log(np.sqrt((V_hat + self.rho) / self.rho) / delta_val)
        log_term = max(0.0, log_term)
        return np.sqrt(2 * (V_hat + self.rho) * log_term) / t

    def get_anytime_pvalue(self, arm_idx):
        t = self.counts[arm_idx]
        if t == 0:
            return 1.0
        mu0_ref = self.emp_means[self.control_arm_idx] if self.control_arm_idx is not None else self.mu_0
        diff = self.emp_means[arm_idx] - mu0_ref
        if diff <= 0:
            return 1.0

        if self.cs_type == 'betting':
            p_value = np.exp(-self.log_martingale[arm_idx])
            return float(np.clip(p_value, 1e-300, 1.0))
        else:
            V_hat = self.emp_vars[arm_idx]
            p_value = (np.sqrt((V_hat + self.rho) / self.rho)
                       * np.exp(-diff ** 2 * t ** 2 / (2 * (V_hat + self.rho))))
            return float(np.clip(p_value, 1e-300, 1.0))

    def select_arm(self):
        treatment_arms = [i for i in range(self.n)
                          if i not in self.S_t and i != self.control_arm_idx]
        if not treatment_arms:
            return "stop"

        threshold = 0.0 if self.control_arm_idx is not None else self.mu_0

        def _ucb(i):
            if self.control_arm_idx is not None:
                return ((self.emp_means[i] - self.emp_means[self.control_arm_idx])
                        + self.phi(self.counts[i], self.delta, self.emp_vars[i])
                        + self.phi(self.counts[self.control_arm_idx], self.delta,
                                   self.emp_vars[self.control_arm_idx]))
            return self.emp_means[i] + self.phi(self.counts[i], self.delta, self.emp_vars[i])

        active = [i for i in treatment_arms
                  if self.counts[i] < _MIN_PULLS_BEFORE_FILTER or _ucb(i) > threshold]
        pool = active if active else treatment_arms
        return np.random.choice(pool)

    def bh_update_optimized(self, arm_idx, observation, x_control=None):
        # Save F_{t-1}-measurable control stats before any update
        control_mean_prev = None
        control_var_prev = None
        control_count_prev = None
        if (x_control is not None
                and self.control_arm_idx is not None
                and arm_idx != self.control_arm_idx):
            control_mean_prev = self.emp_means[self.control_arm_idx]
            control_var_prev = self.emp_vars[self.control_arm_idx]
            control_count_prev = self.counts[self.control_arm_idx]

        # Update test arm first with F_{t-1}-measurable prev control stats
        self._update_stats(arm_idx, observation, x_control=x_control,
                           control_mean_prev=control_mean_prev,
                           control_var_prev=control_var_prev,
                           control_count_prev=control_count_prev)
        self.counts[arm_idx] += 1

        # THEN update control arm
        if control_mean_prev is not None:
            self._update_stats(self.control_arm_idx, x_control)
            self.counts[self.control_arm_idx] += 1

        self.time += 1
        self.counts_evolution.append(self.counts.copy())

        p_values = [1.0] * self.n
        current_St = set()

        if self.control_arm_idx is not None:
            if self.cs_type == 'betting':
                p_values_with_idx = [(self.get_anytime_pvalue(i), i)
                                     for i in range(self.n) if i != self.control_arm_idx]
                p_values_with_idx.sort(key=lambda x: x[0])
                n_tested = self.n - 1
                for k in range(n_tested, 0, -1):
                    if p_values_with_idx[k - 1][0] <= self.delta * k / n_tested:
                        for rank in range(k):
                            current_St.add(p_values_with_idx[rank][1])
                        break
            else:
                n_tested = self.n - 1
                for k in range(n_tested, 0, -1):
                    effective_delta = self.delta * k / n_tested
                    phi_ctrl = self.phi(self.counts[self.control_arm_idx], effective_delta / 2, self.emp_vars[self.control_arm_idx])
                    passing_arms = [
                        i for i in range(self.n)
                        if i != self.control_arm_idx
                        and (self.emp_means[i] - self.emp_means[self.control_arm_idx])
                            - self.phi(self.counts[i], effective_delta / 2, self.emp_vars[i])
                            - phi_ctrl >= 0
                    ]
                    if len(passing_arms) >= k:
                        current_St = set(passing_arms)
                        break
        else:
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

    def bh_update(self, arm_idx, observation):
        self._update_stats(arm_idx, observation)
        self.counts[arm_idx] += 1
        self.time += 1
        self.counts_evolution.append(self.counts.copy())

        current_St = set()
        for k in range(self.n, 0, -1):
            effective_delta = self.delta * k / self.n
            passing_arms = [i for i in range(self.n)
                            if self.emp_means[i] - self.phi(self.counts[i], effective_delta, self.emp_vars[i]) >= self.mu_0]
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

    V3 two-sample mode: each test arm pull is paired with a simultaneous
    control arm pull. The control arm is never selected directly — it is
    always updated as a side effect of test arm pulls.
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

        arm = algo.select_arm()

        if arm == "stop":
            remaining_steps = horizon - len(run_pr)
            last_pr = run_pr[-1] if run_pr else 0
            run_pr.extend([last_pr] * remaining_steps)
            last_counts = algo.counts_evolution[-1]
            for _ in range(remaining_steps):
                algo.counts_evolution.append(last_counts.copy())
            last_p_values = p_values_list[-1] if p_values_list else [1.0] * n_arms
            for _ in range(remaining_steps):
                p_values_list.append(list(last_p_values))
            break

        else:
            len_arm = len(all_arm_data[no_sim][arm])
            if all_arm_counts[arm] >= len_arm:
                observation = np.random.choice(all_arm_data[no_sim][arm])
            else:
                observation = all_arm_data[no_sim][arm][all_arm_counts[arm]]
            all_arm_counts[arm] += 1

            # Paired pull: draw a control arm observation at the same step
            x_control = None
            if variable_mu_choice and arm != control_arm:
                len_ctrl = len(all_arm_data[no_sim][control_arm])
                if all_arm_counts[control_arm] >= len_ctrl:
                    x_control = np.random.choice(all_arm_data[no_sim][control_arm])
                else:
                    x_control = all_arm_data[no_sim][control_arm][all_arm_counts[control_arm]]
                all_arm_counts[control_arm] += 1

            p_values_t = algo.bh_update_optimized(arm, observation, x_control=x_control)
            p_values_list.append(p_values_t)

            if is_true_mean:
                nb_found = len(algo.S_t.intersection(true_positives))
                current_tpr = nb_found / len(true_positives) if true_positives else 1.0
                run_pr.append(current_tpr)
            else:
                run_pr.append(len(algo.S_t))

    return run_pr, p_values_list


def run_experiment(arms, mu_0, delta, horizon, mode, all_arm_data, n_simulations,
                   control_arm, init_nb, init_choice, variable_mu_choice, is_true_mean,
                   rho=0.01, cs_type='betting'):
    """
    Runs the bandit experiment using pre-generated data.

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
        Whether to use variable mu_0 from control arm (two-sample mode).
    is_true_mean : bool
        Whether to compute TPR (requires known arm means).
    rho : float
        NM tuning parameter (ignored for betting). Default: 0.01.
    cs_type : str
        'normal_mixture' or 'betting'. Default: 'betting'.

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

    algo_factory = {
        'adaptive': lambda: JamiesonJainAlgo(n_arms, mu_0, delta, rho=rho, cs_type=cs_type,
                                             control_arm_idx=control_arm if variable_mu_choice else None),
        'uniform': lambda: UniformAlgo(n_arms, mu_0, delta, rho=rho, cs_type=cs_type,
                                       control_arm_idx=control_arm if variable_mu_choice else None),
    }

    if mode not in algo_factory:
        raise ValueError("Algorithm name not detected, choose between uniform and adaptive")

    print(f"Simulation Mode: {mode.upper()} | CS: {cs_type} | rho={rho} ({n_simulations} runs)")

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
