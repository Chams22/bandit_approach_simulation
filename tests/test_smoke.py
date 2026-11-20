"""Smoke tests: each algorithm runs end to end on a tiny simulated bandit problem."""
import importlib
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))

ALGORITHMS = ["adaptative_algorithm_jj", "adaptative_algorithm_v2"]


def _data(n_arms=6, n_pulls=4000, seed=0):
    rng = np.random.default_rng(seed)
    means = np.array([0.0, 0.0, 0.0, 0.0, 1.0, 1.0])[:n_arms]
    return means, [[list(rng.normal(m, 1.0, n_pulls)) for m in means]]


@pytest.mark.parametrize("module_name", ALGORITHMS)
@pytest.mark.parametrize("mode", ["adaptive", "uniform"])
def test_run_experiment(module_name, mode):
    module = importlib.import_module(module_name)
    means, data = _data()
    result = module.run_experiment(
        arms=means, mu_0=0.0, delta=0.05, horizon=3000, mode=mode, all_arm_data=data,
        n_simulations=1, control_arm=None, init_nb=10, init_choice=True,
        variable_mu_choice=False, is_true_mean=True,
    )
    assert result is not None


@pytest.mark.parametrize("module_name", [
    "adaptative_algorithm_continuous_v3",
    "adaptative_algorithm_binary_v3",
    "adaptative_algorithm_successive_reject",
])
def test_import(module_name):
    assert hasattr(importlib.import_module(module_name), "run_experiment")
