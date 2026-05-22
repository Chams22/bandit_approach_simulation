import pandas as pd
import numpy as np
import os
import importlib
import subprocess
import sys
import pickle
from pathlib import Path
import hashlib
import ast

import matplotlib.pyplot as plt
from statistics import mean, variance
import re


# Easy selection of implementations to run:
#   "simple" -> adaptative_algorithm_jj.py
#   "v2"     -> fused V2 module NM
#   "v3"     -> continuous_v3 / binary_v3
#   "sr"     -> successive rejects with the same interface as the others
#   "all"    -> runs simple, v2, v3, and sr in separate folders
# You can edit this list directly, or use:
#   REAL_DATA_ALGO="v2"
#   REAL_DATA_ALGOS="simple,v2"
#   REAL_DATA_ALGO="all"
DEFAULT_RUN_ALGOS = ["simple", "v2", "v3", "sr"]

# Easy selection of real datasets to process:
#   "penn", "exercise", "effort", "walmart"
#   "all" -> runs all datasets
# You can edit this list directly, or use:
#   REAL_DATA_DATASET="penn"
#   REAL_DATA_DATASETS="penn,effort"
#   REAL_DATA_DATASET="all"
DATASET_KEYS = ("penn", "exercise", "effort", "walmart")
DEFAULT_RUN_DATASETS = ["penn", "exercise", "effort", "walmart"]
COMPARISON_DATASET_ORDER = ["effort", "exercise", "penn", "walmart"]

# Cache and comparison switches:
#   REAL_DATA_USE_CACHE=1      -> reload run_experiment_cache.pkl when available
#   REAL_DATA_SAVE_CACHE=0     -> disable writing run_experiment_cache.pkl
#   REAL_DATA_COMPARE_ALGOS=0  -> disable figure_algo_compar generation
#   REAL_DATA_ONLY_COMPARISON=1 -> skip per-dataset classic plots, keep comparison plots
HISTORY_RECORD_EVERY = max(1, int(os.environ.get("REAL_DATA_HISTORY_RECORD_EVERY", "50")))
USE_EXPERIMENT_CACHE = os.environ.get("REAL_DATA_USE_CACHE", "0").lower() in {"1", "true", "yes", "load"}
SAVE_EXPERIMENT_CACHE = os.environ.get("REAL_DATA_SAVE_CACHE", "1").lower() not in {"0", "false", "no"}
GENERATE_ALGO_COMPARISON = os.environ.get("REAL_DATA_COMPARE_ALGOS", "1").lower() not in {"0", "false", "no"}
ONLY_COMPARISON_PLOTS = os.environ.get("REAL_DATA_ONLY_COMPARISON", "0").lower() in {"1", "true", "yes"}
STOP_RULE = os.environ.get("REAL_DATA_STOP_RULE", "horizon").lower()
ADAPTIVE_STOP_MAX_MULTIPLIER = max(
    1,
    int(os.environ.get("REAL_DATA_ADAPTIVE_STOP_MAX_MULTIPLIER", "5")),
)
EFFORT_BOOTSTRAP_SHORT_INIT_ARMS = os.environ.get(
    "REAL_DATA_EFFORT_BOOTSTRAP_SHORT_INIT_ARMS", "1"
).lower() not in {"0", "false", "no"}
EFFORT_INIT_BOOTSTRAP_SEED = int(os.environ.get("REAL_DATA_EFFORT_INIT_BOOTSTRAP_SEED", "12345"))
CACHE_FILENAME = "run_experiment_cache.pkl"
CACHE_VERSION = 6
SUPPORTED_CACHE_VERSIONS = {6}
ALGORITHM_CONFIGS = {
    "simple": {
        "continuous_module": "adaptative_algorithm_jj",
        "binary_module": "adaptative_algorithm_jj",
        "output_dir": "figure_real_data_simple",
    },
    "v2": {
        "continuous_module": "adaptative_algorithm_v2",
        "binary_module": "adaptative_algorithm_v2",
        "output_dir": "figure_real_data_v2",
    },
    "v3": {
        "continuous_module": "adaptative_algorithm_continuous_v3",
        "binary_module": "adaptative_algorithm_binary_v3",
        "output_dir": "figure_real_data_v3",
    },
    "sr": {
        "continuous_module": "adaptative_algorithm_successive_reject",
        "binary_module": "adaptative_algorithm_successive_reject",
        "output_dir": "figure_real_data_successive_reject",
    },
}

def parse_selection(raw_value, default_values, all_values):
    if raw_value is None or raw_value.strip() == "":
        return list(default_values)

    normalized = raw_value.lower().replace(";", ",").replace(" ", ",")
    selected = [item.strip() for item in normalized.split(",") if item.strip()]
    if selected == ["all"]:
        return list(all_values)
    return selected


def _env_truthy(value):
    return str(value).lower() in {"1", "true", "yes", "y", "load", "on"}


def _env_falsey(value):
    return str(value).lower() in {"0", "false", "no", "n", "off"}


def _format_choice(values):
    return ",".join(values)


def _prompt_text(label, default_value, valid_values=None):
    suffix = f" [{default_value}]"
    raw = input(f"{label}{suffix}: ").strip()
    value = raw if raw else default_value
    if valid_values is None:
        return value

    selected = parse_selection(value, parse_selection(default_value, [], valid_values), valid_values)
    invalid = [item for item in selected if item not in valid_values]
    while invalid:
        print(f"Invalid choice(s): {invalid}. Valid choices: {', '.join([*valid_values, 'all'])}")
        raw = input(f"{label}{suffix}: ").strip()
        value = raw if raw else default_value
        selected = parse_selection(value, parse_selection(default_value, [], valid_values), valid_values)
        invalid = [item for item in selected if item not in valid_values]
    return _format_choice(selected)


def _prompt_bool(label, default_value):
    default_label = "y" if default_value else "n"
    raw = input(f"{label} [y/n, default {default_label}]: ").strip().lower()
    if raw == "":
        return default_value
    while raw not in {"y", "yes", "1", "true", "n", "no", "0", "false"}:
        raw = input(f"{label} [y/n, default {default_label}]: ").strip().lower()
        if raw == "":
            return default_value
    return raw in {"y", "yes", "1", "true"}


def _prompt_choice(label, default_value, choices):
    choices_text = "/".join(choices)
    raw = input(f"{label} [{choices_text}, default {default_value}]: ").strip().lower()
    value = raw if raw else default_value
    while value not in choices:
        raw = input(f"{label} [{choices_text}, default {default_value}]: ").strip().lower()
        value = raw if raw else default_value
    return value


def _prompt_optional_int(label, default_value=None, min_value=0):
    default_label = "default" if default_value in {None, ""} else str(default_value)
    raw = input(f"{label} [integer or blank for default, current {default_label}]: ").strip()
    if raw == "":
        return None
    while True:
        try:
            value = int(raw)
            if value < min_value:
                raise ValueError
            return value
        except ValueError:
            raw = input(
                f"{label} [integer >= {min_value}, or blank for default]: "
            ).strip()
            if raw == "":
                return None


def configure_from_interactive_input():
    if __name__ != "__main__":
        return
    if os.environ.get("REAL_DATA_CHILD_RUN") == "1":
        return
    if _env_falsey(os.environ.get("REAL_DATA_INTERACTIVE", "1")):
        return
    if not sys.stdin.isatty():
        return

    print("\n=== Interactive real-data configuration ===")
    print("Press Enter to keep the value shown in brackets.")
    print("Available algorithms: simple, v2, v3, sr, all")
    print("Available datasets: penn, exercise, effort, walmart, all")

    default_algos = os.environ.get(
        "REAL_DATA_ALGOS",
        os.environ.get("REAL_DATA_ALGO", _format_choice(DEFAULT_RUN_ALGOS)),
    )
    default_datasets = os.environ.get(
        "REAL_DATA_DATASETS",
        os.environ.get("REAL_DATA_DATASET", _format_choice(DEFAULT_RUN_DATASETS)),
    )
    selected_algos = _prompt_text(
        "Algorithms to run",
        default_algos,
        ALGORITHM_CONFIGS.keys(),
    )
    selected_datasets = _prompt_text(
        "Datasets to run",
        default_datasets,
        DATASET_KEYS,
    )
    use_cache = _prompt_bool(
        "Use existing run_experiment cache when available",
        _env_truthy(os.environ.get("REAL_DATA_USE_CACHE", "0")),
    )
    save_cache = _prompt_bool(
        "Save run_experiment cache",
        not _env_falsey(os.environ.get("REAL_DATA_SAVE_CACHE", "1")),
    )
    generate_classic_plots = _prompt_bool(
        "Generate all per-dataset figures",
        not _env_truthy(os.environ.get("REAL_DATA_ONLY_COMPARISON", "0")),
    )
    generate_comparison = _prompt_bool(
        "Generate algorithm comparison figures",
        not _env_falsey(os.environ.get("REAL_DATA_COMPARE_ALGOS", "1")),
    )
    default_stop_rule = os.environ.get("REAL_DATA_STOP_RULE", "horizon").lower()
    if default_stop_rule in {"h", "horizon"}:
        default_stop_rule = "h"
    elif default_stop_rule in {"u", "uniform", "uniform_classic_all_non_control_arms"}:
        default_stop_rule = "u"
    else:
        default_stop_rule = "a"
    stop_rule = _prompt_choice(
        "Stopping rule",
        default_stop_rule,
        ["h", "a", "u"],
    )
    effort_init_override = _prompt_optional_int(
        "Effort initialization size override",
        os.environ.get("REAL_DATA_EFFORT_INIT_NB"),
        min_value=0,
    )

    os.environ["REAL_DATA_ALGOS"] = selected_algos
    os.environ.pop("REAL_DATA_ALGO", None)
    os.environ["REAL_DATA_DATASETS"] = selected_datasets
    os.environ.pop("REAL_DATA_DATASET", None)
    os.environ["REAL_DATA_USE_CACHE"] = "1" if use_cache else "0"
    os.environ["REAL_DATA_SAVE_CACHE"] = "1" if save_cache else "0"
    os.environ["REAL_DATA_ONLY_COMPARISON"] = "0" if generate_classic_plots else "1"
    os.environ["REAL_DATA_COMPARE_ALGOS"] = "1" if generate_comparison else "0"
    os.environ["REAL_DATA_STOP_RULE"] = {
        "h": "horizon",
        "a": "adaptive_classic_all_non_control_arms",
        "u": "uniform_classic_all_non_control_arms",
    }[stop_rule]
    if effort_init_override is None:
        os.environ.pop("REAL_DATA_EFFORT_INIT_NB", None)
    else:
        os.environ["REAL_DATA_EFFORT_INIT_NB"] = str(effort_init_override)
    print("===========================================\n")


configure_from_interactive_input()

HISTORY_RECORD_EVERY = max(1, int(os.environ.get("REAL_DATA_HISTORY_RECORD_EVERY", "50")))
USE_EXPERIMENT_CACHE = os.environ.get("REAL_DATA_USE_CACHE", "0").lower() in {"1", "true", "yes", "load"}
SAVE_EXPERIMENT_CACHE = os.environ.get("REAL_DATA_SAVE_CACHE", "1").lower() not in {"0", "false", "no"}
GENERATE_ALGO_COMPARISON = os.environ.get("REAL_DATA_COMPARE_ALGOS", "1").lower() not in {"0", "false", "no"}
ONLY_COMPARISON_PLOTS = os.environ.get("REAL_DATA_ONLY_COMPARISON", "0").lower() in {"1", "true", "yes"}
STOP_RULE = os.environ.get("REAL_DATA_STOP_RULE", "horizon").lower()
ADAPTIVE_STOP_MAX_MULTIPLIER = max(
    1,
    int(os.environ.get("REAL_DATA_ADAPTIVE_STOP_MAX_MULTIPLIER", "5")),
)
EFFORT_BOOTSTRAP_SHORT_INIT_ARMS = os.environ.get(
    "REAL_DATA_EFFORT_BOOTSTRAP_SHORT_INIT_ARMS", "1"
).lower() not in {"0", "false", "no"}
EFFORT_INIT_BOOTSTRAP_SEED = int(os.environ.get("REAL_DATA_EFFORT_INIT_BOOTSTRAP_SEED", "12345"))

RUN_ALGOS = parse_selection(
    os.environ.get("REAL_DATA_ALGOS", os.environ.get("REAL_DATA_ALGO")),
    DEFAULT_RUN_ALGOS,
    ALGORITHM_CONFIGS.keys(),
)
RUN_DATASETS = parse_selection(
    os.environ.get("REAL_DATA_DATASETS", os.environ.get("REAL_DATA_DATASET")),
    DEFAULT_RUN_DATASETS,
    DATASET_KEYS,
)

invalid_algos = [algo for algo in RUN_ALGOS if algo not in ALGORITHM_CONFIGS]
if invalid_algos:
    valid = ", ".join([*ALGORITHM_CONFIGS.keys(), "all"])
    raise ValueError(
        f"Unknown REAL_DATA_ALGO(S)={invalid_algos!r}. Choose one or more of: {valid}"
    )

invalid_datasets = [dataset for dataset in RUN_DATASETS if dataset not in DATASET_KEYS]
if invalid_datasets:
    valid = ", ".join([*DATASET_KEYS, "all"])
    raise ValueError(
        f"Unknown REAL_DATA_DATASET(S)={invalid_datasets!r}. Choose one or more of: {valid}"
    )

if STOP_RULE in {
    "a",
    "adaptive",
    "adaptive_classic",
    "adaptive_all",
    "adaptive_classic_all",
    "adaptive_classic_all_positives",
}:
    STOP_RULE = "adaptive_classic_all_non_control_arms"
if STOP_RULE in {
    "u",
    "uniform",
    "uniform_classic",
    "uniform_all",
    "uniform_classic_all",
    "uniform_classic_all_positives",
}:
    STOP_RULE = "uniform_classic_all_non_control_arms"
if STOP_RULE in {"h", "fixed"}:
    STOP_RULE = "horizon"
if STOP_RULE not in {
    "horizon",
    "adaptive_classic_all_non_control_arms",
    "uniform_classic_all_non_control_arms",
}:
    raise ValueError(
        "Unknown REAL_DATA_STOP_RULE. Choose 'horizon', "
        "'adaptive_classic_all_non_control_arms' (interactive shortcut: a), or "
        "'uniform_classic_all_non_control_arms' (interactive shortcut: u)."
    )


MODE_SPECS = [
    ("UNIF", "Uniform", "pnb_unif", "l_pos_unif", "discovery_unif", ":"),
    ("UNIF VAR", "Uniform Var", "pnb_unif_v", "l_pos_unif_v", "discovery_unif_v", "-."),
    ("ADAPT", "Adaptive", "pnb_adapt", "l_pos_adapt", "discovery_adapt", "-"),
    ("ADAPT VAR", "Adaptive Var", "pnb_adapt_v", "l_pos_adapt_v", "discovery_adapt_v", "--"),
]


def find_project_root(start_path=None):
    start = Path(start_path or __file__).resolve()
    for parent in [start.parent, *start.parents]:
        git_entry = parent / ".git"
        if git_entry.is_dir() or git_entry.is_file():
            return parent
    return Path(__file__).resolve().parents[1]


def save_experiment_cache(cache_path, payload):
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with open(cache_path, "wb") as f:
        pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)


def load_experiment_cache(cache_path):
    with open(cache_path, "rb") as f:
        payload = pickle.load(f)
    if payload.get("cache_version") not in SUPPORTED_CACHE_VERSIONS:
        raise ValueError(f"Unsupported cache version in {cache_path}")
    return payload


def _as_int_set(values):
    if values is None:
        return set()
    return {int(x) for x in values}


def _display_algo_name(algo_key):
    return "JJ" if algo_key == "simple" else algo_key.upper()


def _ordered_comparison_datasets(dataset_keys):
    selected = set(dataset_keys)
    ordered = [dataset for dataset in COMPARISON_DATASET_ORDER if dataset in selected]
    ordered.extend(dataset for dataset in dataset_keys if dataset not in ordered)
    return ordered


def _comparison_dataset_scope():
    """
    Global comparison figures should use all available cached datasets, not only
    the datasets selected for the current run.
    """
    if _env_truthy(os.environ.get("REAL_DATA_COMPARE_SELECTED_DATASETS_ONLY", "0")):
        return list(RUN_DATASETS)
    raw = os.environ.get("REAL_DATA_COMPARE_DATASETS")
    return parse_selection(raw, DEFAULT_RUN_DATASETS, DATASET_KEYS)


def _final_sets(pos_list):
    return [_as_int_set(s) for s in (pos_list or [])]


def _mean_confusion(final_sets, true_positives, n_arms, control_arm):
    true_set = _as_int_set(true_positives)
    tested_arms = set(range(n_arms))
    tested_arms.discard(int(control_arm))
    if not final_sets:
        final_sets = [set()]

    rows = []
    for detected in final_sets:
        detected = _as_int_set(detected) & tested_arms
        rows.append({
            "TP": len(detected & true_set),
            "FP": len(detected - true_set),
            "FN": len(true_set - detected),
            "TN": len(tested_arms - true_set - detected),
        })
    return {key: float(np.mean([row[key] for row in rows])) for key in ("TP", "FP", "FN", "TN")}


def _auc_score(curve, denominator):
    y = np.asarray(curve, dtype=float)
    if y.size == 0:
        return np.nan
    denom = max(float(denominator), 1.0)
    y = y / denom
    if y.size == 1:
        return float(y[0])
    area = float(np.sum((y[:-1] + y[1:]) * 0.5))
    return area / (y.size - 1)


def _curve_time_axis(curve, horizon, history_record_every):
    length = curve.shape[0] if hasattr(curve, "shape") else len(curve)
    history_steps = np.array(
        [0] + [
            step for step in range(1, int(horizon) + 1)
            if step == int(horizon) or step % max(1, int(history_record_every)) == 0
        ],
        dtype=float,
    )
    if length == len(history_steps):
        return history_steps
    if length == len(history_steps) - 1:
        return history_steps[1:]
    return np.linspace(0, horizon, length)


def _history_steps(horizon, history_record_every, include_initial=True):
    horizon = int(horizon)
    history_record_every = max(1, int(history_record_every))
    steps = [
        step for step in range(1, horizon + 1)
        if step == horizon or step % history_record_every == 0
    ]
    if include_initial:
        steps = [0] + steps
    return np.array(steps, dtype=int)


def _history_take_indices(source_horizon, target_horizon, history_record_every,
                          include_initial=True):
    source_steps = _history_steps(source_horizon, history_record_every, include_initial)
    target_steps = _history_steps(target_horizon, history_record_every, include_initial)
    indices = np.searchsorted(source_steps, target_steps, side="right") - 1
    return np.clip(indices, 0, len(source_steps) - 1)


def _truncate_history_axis(values, source_horizon, target_horizon, history_record_every,
                           axis=0, include_initial=True):
    arr = np.asarray(values)
    if arr.shape[axis] == 0:
        return arr
    indices = _history_take_indices(
        source_horizon,
        target_horizon,
        history_record_every,
        include_initial=include_initial,
    )
    indices = indices[indices < arr.shape[axis]]
    return np.take(arr, indices, axis=axis)


def _truncate_adaptive_probe_results(probe_results, source_horizon, target_horizon,
                                     history_record_every):
    (
        pnb_mean,
        pnb_list,
        counts_mean,
        counts_list,
        p_value_list,
        p_value_mean,
        _l_pos,
        discovery_times,
        bootstrap_times,
    ) = probe_results

    target_horizon = int(target_horizon)
    pnb_list = [np.asarray(arr)[:target_horizon] for arr in pnb_list]
    pnb_mean = np.mean(np.array(pnb_list), axis=0) if pnb_list else np.asarray(pnb_mean)[:target_horizon]

    counts_list = [
        _truncate_history_axis(arr, source_horizon, target_horizon, history_record_every, axis=0)
        for arr in counts_list
    ]
    counts_mean = np.mean(np.array(counts_list), axis=0) if counts_list else _truncate_history_axis(
        counts_mean, source_horizon, target_horizon, history_record_every, axis=0
    )

    p_value_list = _truncate_history_axis(
        p_value_list, source_horizon, target_horizon, history_record_every,
        axis=1, include_initial=False
    )
    p_value_mean = _truncate_history_axis(
        p_value_mean, source_horizon, target_horizon, history_record_every,
        axis=0, include_initial=False
    )

    discovery_times = [
        {int(arm): int(time) for arm, time in discovery_dict.items()
         if int(time) <= target_horizon}
        for discovery_dict in discovery_times
    ]
    l_pos = [set(discovery_dict.keys()) for discovery_dict in discovery_times]
    bootstrap_times = [
        {int(arm): int(time) for arm, time in bootstrap_dict.items()
         if int(time) <= target_horizon}
        for bootstrap_dict in bootstrap_times
    ]

    return (
        pnb_mean,
        pnb_list,
        counts_mean,
        counts_list,
        p_value_list,
        p_value_mean,
        l_pos,
        discovery_times,
        bootstrap_times,
    )


def _probe_results_as_run_results(probe_results):
    """The stopping probe is already the definitive run for its own mode."""
    return probe_results


def _initialization_cost(init_nb, n_arms, init_choice=True):
    if not init_choice:
        return 0
    return int(init_nb) * int(n_arms)


def _add_initialization_band(ax, init_cost, y_bottom, label=None):
    if init_cost <= 0:
        return
    ax.fill_between(
        [-init_cost, 0],
        [y_bottom, y_bottom],
        [0, 0],
        color="#bdbdbd",
        alpha=0.35,
        step="post",
        label=label or "_nolegend_",
        zorder=0,
    )
    ax.axvline(0, ymin=0, ymax=0.12, color="#777777",
               linestyle=":", linewidth=1.2, zorder=1)
    ax.text(
        -init_cost * 0.5,
        y_bottom * 0.55,
        f"init = {init_cost}",
        ha="center",
        va="center",
        fontsize=8,
        color="#555555",
    )


def _adaptive_classic_stop_time(discovery_list, true_positives):
    target = {int(arm) for arm in true_positives}
    if not target:
        return 1, True

    stop_times = []
    for discovery_dict in discovery_list or []:
        normalized = {int(arm): int(time) for arm, time in discovery_dict.items()}
        found = set(normalized)
        if not target.issubset(found):
            return None, False
        stop_times.append(max(normalized[arm] for arm in target))
    if not stop_times:
        return None, False
    return max(1, max(stop_times)), True


def _load_classic_positive_list(classic_stats_path):
    if not classic_stats_path.exists():
        raise FileNotFoundError(f"Missing {classic_stats_path}")
    text = classic_stats_path.read_text(encoding="utf-8")
    match = re.match(r"\s*(\[[^\]]*\])", text)
    if not match:
        raise ValueError(f"Could not find positive-arm list at start of {classic_stats_path}")
    values = ast.literal_eval(match.group(1))
    return [int(value) for value in values]


def _format_arm_set(arms):
    return "{" + ", ".join(str(int(arm)) for arm in sorted(arms)) + "}"


def _format_delta_value(value):
    if value is None or np.isnan(value):
        return "NA"
    if abs(value - round(value)) < 1e-9:
        return f"{int(round(value)):+d}"
    return f"{value:+.1f}"


def _discovery_prefixes(discovery_dict):
    if not discovery_dict:
        return {}

    cleaned_items = []
    for arm_idx, first_time in discovery_dict.items():
        try:
            arm = int(arm_idx)
            time = float(first_time)
        except (TypeError, ValueError):
            continue
        cleaned_items.append((arm, time))

    cleaned_items.sort(key=lambda item: (item[1], item[0]))
    prefixes = {}
    prefix_arms = []
    prefix_time = 0.0
    for arm, first_time in cleaned_items:
        prefix_arms.append(arm)
        prefix_time = max(prefix_time, first_time)
        prefixes[len(prefix_arms)] = (tuple(sorted(prefix_arms)), prefix_time)
    return prefixes


def _compare_same_set_discovery(method_specs, left_col="mode_a", right_col="mode_b",
                                allowed_pairs=None):
    rows = []
    if len(method_specs) < 2:
        return rows

    if allowed_pairs is not None:
        allowed_pairs = {frozenset(pair) for pair in allowed_pairs}

    prefix_cache = {}
    for method_key, method_label, discovery_list in method_specs:
        prefix_cache[method_key] = [
            _discovery_prefixes(discovery_dict)
            for discovery_dict in (discovery_list or [])
        ]

    for idx_a, (method_a, label_a, _) in enumerate(method_specs):
        for idx_b in range(idx_a + 1, len(method_specs)):
            method_b, label_b, _ = method_specs[idx_b]
            if allowed_pairs is not None:
                key_pair = frozenset((method_a, method_b))
                label_pair = frozenset((label_a, label_b))
                if key_pair not in allowed_pairs and label_pair not in allowed_pairs:
                    continue
            prefixes_a = prefix_cache[method_a]
            prefixes_b = prefix_cache[method_b]
            n_sims = max(len(prefixes_a), len(prefixes_b))
            pair_order = idx_a * len(method_specs) + idx_b

            for sim_idx in range(n_sims):
                sim_prefixes_a = prefixes_a[sim_idx] if sim_idx < len(prefixes_a) else {}
                sim_prefixes_b = prefixes_b[sim_idx] if sim_idx < len(prefixes_b) else {}
                for k in sorted(set(sim_prefixes_a) & set(sim_prefixes_b)):
                    arms_a, time_a = sim_prefixes_a[k]
                    arms_b, time_b = sim_prefixes_b[k]
                    if arms_a != arms_b:
                        continue
                    rows.append({
                        left_col: label_a,
                        right_col: label_b,
                        "simulation": sim_idx + 1,
                        "k": int(k),
                        "common_set": _format_arm_set(arms_a),
                        "time_a": time_a,
                        "time_b": time_b,
                        "delta_b_minus_a": time_b - time_a,
                        "pair_order": pair_order,
                    })
    return rows


def _prepare_same_set_heatmap(rows_df, left_col, right_col):
    if rows_df.empty:
        return None

    df = rows_df.copy()
    df["pair_label"] = df[right_col].astype(str) + " - " + df[left_col].astype(str)
    grouped = (
        df.groupby(["pair_order", "pair_label", "k"], as_index=False)
        .agg(
            mean_delta=("delta_b_minus_a", "mean"),
            match_count=("delta_b_minus_a", "size"),
        )
        .sort_values(["pair_order", "k"])
    )
    if grouped.empty:
        return None

    pair_order_df = (
        grouped[["pair_order", "pair_label"]]
        .drop_duplicates()
        .sort_values("pair_order")
    )
    pair_labels = pair_order_df["pair_label"].tolist()
    k_values = sorted(int(k) for k in grouped["k"].unique())
    matrix = np.full((len(pair_labels), len(k_values)), np.nan, dtype=float)
    counts = np.zeros_like(matrix)
    pair_index = {label: idx for idx, label in enumerate(pair_labels)}
    k_index = {k: idx for idx, k in enumerate(k_values)}

    for row in grouped.itertuples(index=False):
        i = pair_index[row.pair_label]
        j = k_index[int(row.k)]
        matrix[i, j] = float(row.mean_delta)
        counts[i, j] = int(row.match_count)

    return {
        "matrix": matrix,
        "counts": counts,
        "pair_labels": pair_labels,
        "k_values": k_values,
    }


def _draw_same_set_heatmap(ax, prepared, title, max_abs=None, show_counts=False):
    if prepared is None:
        ax.axis("off")
        ax.text(0.5, 0.5, "No identical discovered-set prefixes",
                transform=ax.transAxes, ha="center", va="center",
                color="gray", fontsize=10)
        ax.set_title(title)
        return None

    matrix = prepared["matrix"]
    pair_labels = prepared["pair_labels"]
    k_values = prepared["k_values"]
    counts = prepared["counts"]
    local_max = np.nanmax(np.abs(matrix)) if np.isfinite(matrix).any() else 1.0
    scale = max(float(max_abs or local_max), 1.0)
    cmap = plt.cm.RdBu_r.copy()
    cmap.set_bad(color="white")
    masked = np.ma.masked_invalid(matrix)
    image = ax.imshow(masked, aspect="auto", cmap=cmap, vmin=-scale, vmax=scale)

    fontsize = 8 if matrix.shape[1] <= 20 else 6
    for row_idx in range(matrix.shape[0]):
        for col_idx in range(matrix.shape[1]):
            value = matrix[row_idx, col_idx]
            if np.isnan(value):
                ax.text(col_idx, row_idx, "NA", ha="center", va="center",
                        fontsize=fontsize, color="#b0b0b0")
                continue
            text = _format_delta_value(value)
            if show_counts and counts[row_idx, col_idx] > 1:
                text = f"{text}\nn={int(counts[row_idx, col_idx])}"
            text_color = "white" if abs(value) > 0.55 * scale else "black"
            ax.text(col_idx, row_idx, text, ha="center", va="center",
                    fontsize=fontsize, color=text_color)

    ax.set_title(title)
    ax.set_xlabel("Discovered-set size k")
    ax.set_ylabel("Comparison: method B - method A")
    ax.set_xticks(np.arange(len(k_values)))
    ax.set_xticklabels(k_values)
    ax.set_yticks(np.arange(len(pair_labels)))
    ax.set_yticklabels(pair_labels)
    ax.set_facecolor("white")
    ax.set_xticks(np.arange(-0.5, len(k_values), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(pair_labels), 1), minor=True)
    ax.grid(which="minor", color="#d9d9d9", linestyle="-", linewidth=0.6)
    ax.tick_params(which="minor", bottom=False, left=False)
    return image


def _mean_discovery_sequence(discovery_list):
    rank_rows = {}
    for discovery_dict in discovery_list or []:
        ordered = sorted(
            (
                (int(arm_idx), float(first_time))
                for arm_idx, first_time in discovery_dict.items()
            ),
            key=lambda item: (item[1], item[0]),
        )
        for rank, (arm_idx, first_time) in enumerate(ordered, start=1):
            rank_rows.setdefault(rank, []).append((arm_idx, first_time))

    sequence = {}
    for rank, items in rank_rows.items():
        arm_counts = {}
        time_by_arm = {}
        for arm_idx, first_time in items:
            arm_counts[arm_idx] = arm_counts.get(arm_idx, 0) + 1
            time_by_arm.setdefault(arm_idx, []).append(first_time)
        chosen_arm = sorted(
            arm_counts,
            key=lambda arm: (-arm_counts[arm], float(np.mean(time_by_arm[arm])), arm),
        )[0]
        sequence[rank] = {
            "arm": chosen_arm,
            "mean_time": float(np.mean(time_by_arm[chosen_arm])),
            "support": int(arm_counts[chosen_arm]),
            "n_sims": len(items),
        }
    return sequence


def _draw_discovery_sequence_table(
    ax,
    method_specs,
    title="Discovery order by mode: cell = discovered arm, then first discovery time",
    control_arm=None,
    displayed_ranks=None,
):
    sequences = [
        (method_label, _mean_discovery_sequence(discovery_list))
        for _, method_label, discovery_list in method_specs
    ]
    max_rank = max((max(seq) for _, seq in sequences if seq), default=0)
    ax.axis("off")
    if max_rank == 0:
        ax.text(0.5, 0.5, "No discoveries to summarize",
                transform=ax.transAxes, ha="center", va="center",
                color="gray", fontsize=10)
        return

    if displayed_ranks is None:
        ranks_to_show = list(range(1, max_rank + 1))
    else:
        ranks_to_show = [
            int(rank) for rank in displayed_ranks
            if 1 <= int(rank) <= max_rank
        ]
    if not ranks_to_show:
        ranks_to_show = list(range(1, min(max_rank, 20) + 1))

    col_labels = [str(rank) for rank in ranks_to_show]
    row_labels = [label for label, _ in sequences]
    cell_text = []
    control_cells = set()
    for _, sequence in sequences:
        row_idx = len(cell_text) + 1
        row = []
        for col_idx, rank in enumerate(ranks_to_show):
            item = sequence.get(rank)
            if item is None:
                row.append("")
                continue
            row.append(f"a{item['arm']}\nt{_format_delta_value(item['mean_time']).lstrip('+')}")
            if control_arm is not None and int(item["arm"]) == int(control_arm):
                control_cells.add((row_idx, col_idx))
        cell_text.append(row)

    table = ax.table(
        cellText=cell_text,
        rowLabels=row_labels,
        colLabels=col_labels,
        loc="center",
        cellLoc="center",
        rowLoc="center",
    )
    table.auto_set_font_size(False)
    fontsize = 8 if len(ranks_to_show) <= 18 else 6
    table.set_fontsize(fontsize)
    table.scale(1.0, 1.45)
    for (row, col), cell in table.get_celld().items():
        cell.set_edgecolor("#d9d9d9")
        if row == 0 or col == -1:
            cell.set_text_props(weight="bold")
            cell.set_facecolor("#f2f2f2")
        elif (row, col) in control_cells:
            cell.set_facecolor("#ffb3b3")
            cell.set_edgecolor("#b30000")
            cell.set_text_props(color="#7a0000", weight="bold")
    title_suffix = " | red cell = control arm" if control_arm is not None else ""
    ax.set_title(title + title_suffix, fontsize=10, fontweight="bold", pad=12)


def write_same_set_discovery_comparison(method_specs, csv_path, figure_path, title,
                                        allowed_pairs=None, control_arm=None):
    rows = _compare_same_set_discovery(
        method_specs,
        "mode_a",
        "mode_b",
        allowed_pairs=allowed_pairs,
    )
    rows_df = pd.DataFrame(rows)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    export_columns = [
        "mode_a", "mode_b", "simulation", "k", "common_set",
        "time_a", "time_b", "delta_b_minus_a",
    ]
    if rows_df.empty:
        pd.DataFrame(columns=export_columns).to_csv(csv_path, index=False)
    else:
        rows_df[export_columns].to_csv(csv_path, index=False)

    prepared = _prepare_same_set_heatmap(rows_df, "mode_a", "mode_b")
    max_abs = None
    if prepared is not None and np.isfinite(prepared["matrix"]).any():
        max_abs = max(float(np.nanmax(np.abs(prepared["matrix"]))), 1.0)

    max_rank = 0
    for _, _, discovery_list in method_specs:
        for discovery_dict in discovery_list or []:
            max_rank = max(max_rank, len(discovery_dict or {}))
    displayed_ranks = prepared["k_values"] if prepared is not None else None
    displayed_rank_count = len(displayed_ranks) if displayed_ranks is not None else max_rank

    fig_width = 11
    fig_height = 7
    if prepared is not None:
        fig_width = max(11, 0.48 * len(prepared["k_values"]) + 5)
        fig_height = max(7.0, 0.55 * len(prepared["pair_labels"]) + 4.3)
    if max_rank:
        fig_width = max(fig_width, 0.55 * displayed_rank_count + 5.5)

    fig = plt.figure(figsize=(fig_width, fig_height))
    grid = fig.add_gridspec(2, 1, height_ratios=[1.15, 2.6], hspace=0.38)
    table_ax = fig.add_subplot(grid[0, 0])
    heatmap_ax = fig.add_subplot(grid[1, 0])
    _draw_discovery_sequence_table(
        table_ax,
        method_specs,
        control_arm=control_arm,
        displayed_ranks=displayed_ranks,
    )
    image = _draw_same_set_heatmap(
        heatmap_ax,
        prepared,
        title,
        max_abs=max_abs,
        show_counts=True,
    )
    if image is not None:
        cbar = fig.colorbar(image, ax=heatmap_ax, shrink=0.88)
        cbar.set_label("Mean time difference: method B - method A")
    fig.suptitle(
        "Same discovered-set discovery-time comparison\n"
        "Cells are filled only when both methods have found exactly the same set of arms at size k.",
        fontsize=12,
        fontweight="bold",
    )
    plt.tight_layout(rect=(0, 0, 1, 0.91))
    plt.savefig(figure_path, dpi=300, bbox_inches="tight")
    plt.close()
    return rows_df


def _collect_cached_results(project_root, algo_keys, dataset_keys):
    cached = {}
    for algo_key in algo_keys:
        config = ALGORITHM_CONFIGS[algo_key]
        for dataset_key in dataset_keys:
            cache_path = project_root / config["output_dir"] / dataset_key / CACHE_FILENAME
            if not cache_path.exists():
                print(f"[comparison] cache missing: {cache_path}")
                continue
            try:
                cached[(algo_key, dataset_key)] = load_experiment_cache(cache_path)
            except Exception as exc:
                print(f"[comparison] could not read {cache_path}: {exc}")
    return cached


def generate_algorithm_comparison_figures(project_root, algo_keys, dataset_keys):
    dataset_keys = _ordered_comparison_datasets(dataset_keys)
    cached = _collect_cached_results(project_root, algo_keys, dataset_keys)
    if not cached:
        print("[comparison] no cache available, skipping figure_algo_compar.")
        return

    output_dir = project_root / "figure_algo_compar"
    output_dir.mkdir(parents=True, exist_ok=True)

    algo_colors = {
        key: color for key, color in zip(ALGORITHM_CONFIGS.keys(), plt.cm.tab10.colors)
    }

    summary_rows = []
    for (algo_key, dataset_key), payload in cached.items():
        true_positives = _as_int_set(payload["true_positives"])
        n_true = len(true_positives)
        n_arms = int(payload["n_arms"])
        control_arm = int(payload["control_arm"])
        for mode_key, mode_label, pnb_key, pos_key, _, _ in MODE_SPECS:
            final_sets = _final_sets(payload.get(pos_key))
            final_counts = [len(s) for s in final_sets] or [0]
            confusion = _mean_confusion(final_sets, true_positives, n_arms, control_arm)
            auc = _auc_score(payload.get(pnb_key, []), n_true)
            summary_rows.append({
                "algorithm": algo_key,
                "dataset": dataset_key,
                "mode": mode_label,
                "mean_detected_positives": float(np.mean(final_counts)),
                "n_true_positives": n_true,
                "auc_discovery_curve": auc,
                **confusion,
            })

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(output_dir / "algo_comparison_summary.csv", index=False)

    _plot_positive_rate_curves(cached, algo_keys, dataset_keys, algo_colors, output_dir)
    _plot_found_positive_counts(summary_df, algo_keys, dataset_keys, output_dir)
    _plot_confusion_counts(summary_df, algo_keys, dataset_keys, output_dir)
    _plot_discovery_auc(summary_df, algo_keys, dataset_keys, output_dir)
    _plot_global_same_set_discovery_heatmaps(cached, algo_keys, dataset_keys, output_dir)
    _plot_common_adaptive_set_times(cached, algo_keys, dataset_keys, algo_colors, output_dir)
    print(f"[comparison] wrote algorithm comparison figures to {output_dir}")


def _plot_positive_rate_curves(cached, algo_keys, dataset_keys, algo_colors, output_dir):
    ncols = min(2, max(1, len(dataset_keys)))
    nrows = int(np.ceil(len(dataset_keys) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(8 * ncols, 4.8 * nrows), squeeze=False)

    flat_axes = axes.ravel()
    for ax, dataset_key in zip(flat_axes, dataset_keys):
        plotted = False
        max_x = 1.0
        max_y = 1.0
        init_costs = []
        for algo_key in algo_keys:
            payload = cached.get((algo_key, dataset_key))
            if not payload:
                continue
            init_costs.append(_initialization_cost(
                payload.get("init_nb", 0),
                payload.get("n_arms", 0),
                payload.get("init_choice", True),
            ))
            curve = np.asarray(payload.get("pnb_adapt", []), dtype=float)
            if curve.size == 0:
                continue
            x_axis = _curve_time_axis(
                curve,
                payload.get("horizon", curve.size),
                payload.get("history_record_every", 1),
            )
            y_values = curve
            ax.plot(
                x_axis,
                y_values,
                label=f"{_display_algo_name(algo_key)} - Adaptive",
                color=algo_colors.get(algo_key, "black"),
                linestyle="-",
                linewidth=2.1,
                alpha=0.86,
            )
            max_x = max(max_x, float(np.nanmax(x_axis)))
            max_y = max(max_y, float(np.nanmax(y_values)))
            plotted = True
        init_cost = max(init_costs) if init_costs else 0
        y_bottom = -0.10 * max_y
        _add_initialization_band(ax, init_cost, y_bottom, label="Initialization budget")
        n_true = None
        for algo_key in algo_keys:
            payload = cached.get((algo_key, dataset_key))
            if payload:
                n_true = len(_as_int_set(payload["true_positives"]))
                break
        if n_true:
            ax.axhline(n_true, color="black", linestyle=":", linewidth=1.0, alpha=0.5,
                       label="_nolegend_")
        ax.set_title(f"{dataset_key.upper()} - adaptive discovery trajectory")
        ax.set_ylabel("Detected positives")
        ax.set_xlabel("Rounds after initialization")
        ax.set_ylim(y_bottom * 1.15, max_y * 1.08)
        ax.set_xlim(-init_cost if init_cost > 0 else 0, max(max_x, 1))
        ax.grid(True, alpha=0.25)
        if not plotted:
            ax.text(0.5, 0.5, "No cached result", transform=ax.transAxes,
                    ha="center", va="center", color="gray")

    for ax in flat_axes[len(dataset_keys):]:
        ax.axis("off")

    handles, labels = flat_axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="lower center", ncol=min(4, len(labels)),
                   fontsize=8, bbox_to_anchor=(0.5, -0.01))
    fig.suptitle(
        "Algorithm comparison: adaptive discovery trajectories\n"
        "Only the classic Adaptive mode is shown. Y-axis is the raw number of detected positive arms.",
        fontsize=14, fontweight="bold",
    )
    plt.tight_layout(rect=(0, 0.04, 1, 0.94))
    plt.savefig(output_dir / "positive_rate_curves.png", dpi=300, bbox_inches="tight")
    plt.close()


def _plot_found_positive_counts(summary_df, algo_keys, dataset_keys, output_dir):
    if summary_df.empty:
        return
    modes = [spec[1] for spec in MODE_SPECS]
    ncols = min(2, max(1, len(dataset_keys)))
    nrows = int(np.ceil(len(dataset_keys) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(7 * ncols, 5 * nrows), squeeze=False)
    width = 0.18
    offsets = np.linspace(-1.5 * width, 1.5 * width, len(modes))

    flat_axes = axes.ravel()
    for ax, dataset_key in zip(flat_axes, dataset_keys):
        subset = summary_df[summary_df["dataset"] == dataset_key]
        x = np.arange(len(algo_keys))
        for offset, mode in zip(offsets, modes):
            values = [
                subset[(subset["algorithm"] == algo) & (subset["mode"] == mode)]["mean_detected_positives"].mean()
                for algo in algo_keys
            ]
            ax.bar(x + offset, values, width=width, label=mode)
        true_counts = subset.groupby("algorithm")["n_true_positives"].first()
        if not true_counts.empty:
            ax.axhline(float(true_counts.iloc[0]), color="black", linestyle=":",
                       linewidth=1.2, label="Classical positive arms")
        ax.set_title(dataset_key.upper())
        ax.set_xticks(x)
        ax.set_xticklabels([_display_algo_name(algo) for algo in algo_keys], rotation=35, ha="right")
        ax.set_ylabel("Mean number of detected positive arms")
        ax.grid(axis="y", alpha=0.25)

    for ax in flat_axes[len(dataset_keys):]:
        ax.axis("off")

    handles, labels = flat_axes[min(len(dataset_keys), len(flat_axes)) - 1].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=min(5, len(labels)),
               bbox_to_anchor=(0.5, -0.03))
    fig.suptitle(
        "Final discoveries by algorithm and sampling mode\n"
        "Bars show the mean final number of discovered arms; dotted line shows the classical BH-positive count.",
        fontsize=14, fontweight="bold",
    )
    plt.tight_layout(rect=(0, 0.08, 1, 0.90))
    plt.savefig(output_dir / "found_positive_counts.png", dpi=300, bbox_inches="tight")
    plt.close()


def _plot_confusion_counts(summary_df, algo_keys, dataset_keys, output_dir):
    if summary_df.empty:
        return
    dataset_keys = _ordered_comparison_datasets(dataset_keys)
    colors = {"TP": "#2ca02c", "FP": "#d62728", "FN": "#ffbf00", "TN": "#9ecae1"}
    modes = [spec[1] for spec in MODE_SPECS]
    fig, axes = plt.subplots(len(dataset_keys), 1, figsize=(16, max(5, 4.2 * len(dataset_keys))), squeeze=False)

    for row_idx, dataset_key in enumerate(dataset_keys):
        ax = axes[row_idx, 0]
        subset = summary_df[summary_df["dataset"] == dataset_key]
        labels = []
        x = []
        bottoms = []
        for algo in algo_keys:
            for mode in modes:
                row = subset[(subset["algorithm"] == algo) & (subset["mode"] == mode)]
                if row.empty:
                    continue
                labels.append(f"{_display_algo_name(algo)}\n{mode}")
                x.append(len(x))
                bottoms.append(0.0)

        x = np.array(x)
        bottoms = np.zeros(len(labels))
        for key in ("TP", "FP", "FN", "TN"):
            values = []
            for label in labels:
                algo_label, mode_label = label.split("\n", 1)
                algo_key = "simple" if algo_label == "JJ" else algo_label.lower()
                row = subset[
                    (subset["algorithm"] == algo_key) &
                    (subset["mode"] == mode_label)
                ]
                values.append(float(row[key].iloc[0]) if not row.empty else 0.0)
            ax.bar(x, values, bottom=bottoms, color=colors[key], edgecolor="white", label=key)
            bottoms += np.asarray(values)

        ax.set_title(f"{dataset_key.upper()} - confusion counts")
        ax.set_ylabel("Mean arm count")
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
        ax.grid(axis="y", alpha=0.25)

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4,
               bbox_to_anchor=(0.5, -0.02))
    fig.suptitle(
        "Detection quality against classical positives\n"
        "TP=true positive found, FP=false positive found, FN=classical positive missed, TN=correctly not found.",
        fontsize=14, fontweight="bold",
    )
    plt.tight_layout(rect=(0, 0.06, 1, 0.93))
    plt.savefig(output_dir / "confusion_counts.png", dpi=300, bbox_inches="tight")
    plt.close()


def _plot_discovery_auc(summary_df, algo_keys, dataset_keys, output_dir):
    if summary_df.empty:
        return
    modes = [spec[1] for spec in MODE_SPECS]
    ncols = min(2, max(1, len(dataset_keys)))
    nrows = int(np.ceil(len(dataset_keys) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(7 * ncols, 5 * nrows), squeeze=False)
    width = 0.18
    offsets = np.linspace(-1.5 * width, 1.5 * width, len(modes))

    flat_axes = axes.ravel()
    for ax, dataset_key in zip(flat_axes, dataset_keys):
        subset = summary_df[summary_df["dataset"] == dataset_key]
        x = np.arange(len(algo_keys))
        for offset, mode in zip(offsets, modes):
            values = [
                subset[(subset["algorithm"] == algo) & (subset["mode"] == mode)]["auc_discovery_curve"].mean()
                for algo in algo_keys
            ]
            ax.bar(x + offset, values, width=width, label=mode)
        ax.set_title(dataset_key.upper())
        ax.set_xticks(x)
        ax.set_xticklabels([_display_algo_name(algo) for algo in algo_keys], rotation=35, ha="right")
        ax.set_ylabel("Normalized area under discovery curve")
        ax.grid(axis="y", alpha=0.25)

    for ax in flat_axes[len(dataset_keys):]:
        ax.axis("off")

    handles, labels = flat_axes[min(len(dataset_keys), len(flat_axes)) - 1].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=min(4, len(labels)),
               bbox_to_anchor=(0.5, -0.03))
    fig.suptitle(
        "Discovery speed score by algorithm\n"
        "A larger area means positives were found earlier and/or more completely during the horizon.",
        fontsize=14, fontweight="bold",
    )
    plt.tight_layout(rect=(0, 0.08, 1, 0.90))
    plt.savefig(output_dir / "discovery_speed_auc.png", dpi=300, bbox_inches="tight")
    plt.close()


def _largest_common_adaptive_set_record(cached, dataset_key, algo_keys):
    required_algos = [algo for algo in ["simple", "v2", "v3", "sr"] if algo in algo_keys]
    if len(required_algos) < 4:
        return None, f"Need JJ, V2, V3, and SR caches; found {len(required_algos)}/4."

    discovery_by_algo = {}
    for algo_key in required_algos:
        payload = cached.get((algo_key, dataset_key))
        if not payload:
            return None, f"Missing cache for {_display_algo_name(algo_key)}."
        discovery_by_algo[algo_key] = payload.get("discovery_adapt", []) or []

    n_sims = min(len(discovery_by_algo[algo]) for algo in required_algos)
    if n_sims == 0:
        return None, "No adaptive discovery trajectories in cache."

    candidates = []
    for sim_idx in range(n_sims):
        prefixes = {
            algo: _discovery_prefixes(discovery_by_algo[algo][sim_idx])
            for algo in required_algos
        }
        common_k = set.intersection(
            *(set(prefixes[algo]) for algo in required_algos)
        )
        for k in common_k:
            arms = [prefixes[algo][k][0] for algo in required_algos]
            if len(set(arms)) != 1:
                continue
            times = {
                algo: float(prefixes[algo][k][1])
                for algo in required_algos
            }
            candidates.append({
                "dataset": dataset_key,
                "simulation": sim_idx + 1,
                "k": int(k),
                "common_set": _format_arm_set(arms[0]),
                "common_set_tuple": arms[0],
                "mean_time": float(np.mean(list(times.values()))),
                "times": times,
            })

    if not candidates:
        return None, "No identical adaptive discovered set shared by all four algorithms."

    candidates.sort(key=lambda row: (-row["k"], row["mean_time"], row["simulation"]))
    return candidates[0], None


def _plot_common_adaptive_set_times(cached, algo_keys, dataset_keys, algo_colors, output_dir):
    dataset_keys = _ordered_comparison_datasets(dataset_keys)
    algo_order = [algo for algo in ["simple", "v2", "v3", "sr"] if algo in algo_keys]
    display_labels = [_display_algo_name(algo) for algo in algo_order]

    rows = []
    records = {}
    messages = {}
    for dataset_key in dataset_keys:
        record, message = _largest_common_adaptive_set_record(cached, dataset_key, algo_keys)
        records[dataset_key] = record
        messages[dataset_key] = message
        if record is None:
            continue
        for algo_key in algo_order:
            rows.append({
                "dataset": dataset_key,
                "simulation": record["simulation"],
                "k": record["k"],
                "common_set": record["common_set"],
                "algorithm": _display_algo_name(algo_key),
                "discovery_time": record["times"].get(algo_key, np.nan),
            })

    pd.DataFrame(rows).to_csv(
        output_dir / "common_adaptive_set_discovery_times.csv",
        index=False,
    )

    ncols = min(2, max(1, len(dataset_keys)))
    nrows = int(np.ceil(len(dataset_keys) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(8 * ncols, 5.2 * nrows), squeeze=False)
    flat_axes = axes.ravel()

    for ax, dataset_key in zip(flat_axes, dataset_keys):
        record = records.get(dataset_key)
        if record is None:
            ax.axis("off")
            ax.text(
                0.5,
                0.5,
                messages.get(dataset_key, "No common adaptive set."),
                transform=ax.transAxes,
                ha="center",
                va="center",
                color="gray",
                fontsize=10,
                wrap=True,
            )
            ax.set_title(f"{dataset_key.upper()} - no shared adaptive set")
            continue

        values = [record["times"].get(algo, np.nan) for algo in algo_order]
        colors = [algo_colors.get(algo, "gray") for algo in algo_order]
        x = np.arange(len(algo_order))
        bars = ax.bar(x, values, color=colors, edgecolor="black", alpha=0.86)
        for bar, value in zip(bars, values):
            if np.isnan(value):
                continue
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height(),
                f"{value:.0f}",
                ha="center",
                va="bottom",
                fontsize=8,
            )

        common_tuple = record["common_set_tuple"]
        set_preview = ", ".join(str(arm) for arm in common_tuple[:10])
        if len(common_tuple) > 10:
            set_preview += ", ..."
        ax.set_title(
            f"{dataset_key.upper()} - largest common Adaptive set: k={record['k']}\n"
            f"set {{{set_preview}}}"
        )
        ax.set_xticks(x)
        ax.set_xticklabels(display_labels, rotation=0)
        ax.set_ylabel("Discovery time for the same arm set")
        ax.grid(axis="y", alpha=0.25)

    for ax in flat_axes[len(dataset_keys):]:
        ax.axis("off")

    fig.suptitle(
        "Discovery time for the largest identical Adaptive-discovered set\n"
        "Each panel compares JJ, V2, V3, and SR only at the largest k where all four found the exact same set.",
        fontsize=14,
        fontweight="bold",
    )
    plt.tight_layout(rect=(0, 0.02, 1, 0.92))
    plt.savefig(output_dir / "common_adaptive_set_discovery_times.png", dpi=300, bbox_inches="tight")
    plt.close()


def _largest_common_pair_discovery_record(discovery_a, discovery_b, label_a, label_b,
                                          dataset_key, comparison_label):
    n_sims = min(len(discovery_a or []), len(discovery_b or []))
    candidates = []
    for sim_idx in range(n_sims):
        prefixes_a = _discovery_prefixes(discovery_a[sim_idx])
        prefixes_b = _discovery_prefixes(discovery_b[sim_idx])
        common_k = set(prefixes_a).intersection(prefixes_b)
        for k in common_k:
            arms_a, time_a = prefixes_a[k]
            arms_b, time_b = prefixes_b[k]
            if arms_a != arms_b:
                continue
            candidates.append({
                "dataset": dataset_key,
                "comparison": comparison_label,
                "simulation": sim_idx + 1,
                "k": int(k),
                "common_set": _format_arm_set(arms_a),
                "common_set_tuple": arms_a,
                f"{label_a}_time": float(time_a),
                f"{label_b}_time": float(time_b),
                "mean_time": float(np.mean([time_a, time_b])),
                "delta_adaptive_minus_uniform": float(time_b - time_a),
            })

    if not candidates:
        return None
    candidates.sort(key=lambda row: (-row["k"], row["mean_time"], row["simulation"]))
    return candidates[0]


def _load_algo_root_cache_payloads(output_root, dataset_keys):
    cached = {}
    messages = {}
    for dataset_key in dataset_keys:
        cache_path = Path(output_root) / dataset_key / CACHE_FILENAME
        if not cache_path.exists():
            messages[dataset_key] = f"Missing cache: {cache_path.name}"
            continue
        try:
            cached[dataset_key] = load_experiment_cache(cache_path)
        except Exception as exc:
            messages[dataset_key] = f"Could not load cache: {exc}"
    return cached, messages


def generate_local_figure10_same_set_adapt_vs_uniform(output_root, dataset_keys=None,
                                                      payloads=None):
    output_root = Path(output_root)
    dataset_keys = _ordered_comparison_datasets(dataset_keys or DATASET_KEYS)
    cached, messages = _load_algo_root_cache_payloads(output_root, dataset_keys)
    if payloads:
        cached.update(payloads)

    rows = []
    records = {}
    for dataset_key in dataset_keys:
        payload = cached.get(dataset_key)
        if not payload:
            records[dataset_key] = {"message": messages.get(dataset_key, "Missing cache.")}
            continue

        classic_record = _largest_common_pair_discovery_record(
            payload.get("discovery_unif", []) or [],
            payload.get("discovery_adapt", []) or [],
            "uniform",
            "adaptive",
            dataset_key,
            "Classic: Adaptive vs Uniform",
        )
        var_record = _largest_common_pair_discovery_record(
            payload.get("discovery_unif_v", []) or [],
            payload.get("discovery_adapt_v", []) or [],
            "uniform",
            "adaptive",
            dataset_key,
            "Online control: Adaptive Var vs Uniform Var",
        )
        records[dataset_key] = {
            "Classic": classic_record,
            "Var": var_record,
            "message": None,
        }
        for record in [classic_record, var_record]:
            if record is None:
                continue
            rows.append({
                "dataset": record["dataset"],
                "comparison": record["comparison"],
                "simulation": record["simulation"],
                "k": record["k"],
                "common_set": record["common_set"],
                "uniform_time": record["uniform_time"],
                "adaptive_time": record["adaptive_time"],
                "delta_adaptive_minus_uniform": record["delta_adaptive_minus_uniform"],
            })

    pd.DataFrame(rows).to_csv(
        output_root / "figure10_same_set_adapt_vs_uniform.csv",
        index=False,
    )

    ncols = 2
    nrows = int(np.ceil(len(dataset_keys) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(8.2 * ncols, 5.4 * nrows), squeeze=False)
    flat_axes = axes.ravel()
    mode_colors = {
        "Uniform": "#9E9E9E",
        "Adaptive": "#2E7D32",
        "Uniform Var": "#B0B0B0",
        "Adaptive Var": "#43A047",
    }

    for ax, dataset_key in zip(flat_axes, dataset_keys):
        record_bundle = records.get(dataset_key, {})
        if record_bundle.get("message"):
            ax.axis("off")
            ax.text(0.5, 0.5, record_bundle["message"],
                    transform=ax.transAxes, ha="center", va="center",
                    color="gray", wrap=True)
            ax.set_title(f"{dataset_key.upper()} - missing data")
            continue

        specs = [
            ("Classic", "Uniform", "Adaptive"),
            ("Var", "Uniform Var", "Adaptive Var"),
        ]
        x_positions = []
        values = []
        colors = []
        labels = []
        annotations = []
        group_centers = []

        for group_idx, (record_key, uniform_label, adaptive_label) in enumerate(specs):
            record = record_bundle.get(record_key)
            center = group_idx * 3.0
            group_centers.append(center + 0.4)
            if record is None:
                x_positions.extend([center, center + 0.8])
                values.extend([np.nan, np.nan])
                colors.extend(["#dddddd", "#dddddd"])
                labels.extend([uniform_label, adaptive_label])
                annotations.extend(["NA", "NA"])
                continue

            common_tuple = record["common_set_tuple"]
            preview = ", ".join(str(arm) for arm in common_tuple[:7])
            if len(common_tuple) > 7:
                preview += ", ..."
            x_positions.extend([center, center + 0.8])
            values.extend([record["uniform_time"], record["adaptive_time"]])
            colors.extend([mode_colors[uniform_label], mode_colors[adaptive_label]])
            labels.extend([uniform_label, adaptive_label])
            annotations.extend([
                f"k={record['k']}\n{{{preview}}}",
                f"Δ={record['delta_adaptive_minus_uniform']:+.0f}",
            ])

        finite_values = [value for value in values if np.isfinite(value)]
        if not finite_values:
            ax.axis("off")
            ax.text(0.5, 0.5, "No identical discovered set between compared modes",
                    transform=ax.transAxes, ha="center", va="center", color="gray", wrap=True)
            ax.set_title(f"{dataset_key.upper()} - no common sets")
            continue

        bars = ax.bar(x_positions, np.nan_to_num(values, nan=0.0), color=colors,
                      edgecolor="black", alpha=0.88, width=0.68)
        y_max = max(finite_values)
        for bar, value, annotation in zip(bars, values, annotations):
            if not np.isfinite(value):
                ax.text(bar.get_x() + bar.get_width() / 2, y_max * 0.04, "NA",
                        ha="center", va="bottom", color="gray", fontsize=9)
                continue
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                    f"{value:.0f}", ha="center", va="bottom", fontsize=8)
            ax.text(bar.get_x() + bar.get_width() / 2, -0.12 * y_max,
                    annotation, ha="center", va="top", fontsize=7, wrap=True)

        ax.set_title(f"{dataset_key.upper()} - largest identical set per comparison")
        ax.set_xticks(x_positions)
        ax.set_xticklabels(labels, rotation=25, ha="right")
        ax.set_ylabel("Discovery time for the same arm set")
        ax.set_ylim(-0.28 * y_max, 1.18 * y_max)
        ax.grid(axis="y", alpha=0.25)
        for center in group_centers[:-1]:
            ax.axvline(center + 1.1, color="#d0d0d0", linewidth=0.8)

    for ax in flat_axes[len(dataset_keys):]:
        ax.axis("off")

    fig.suptitle(
        "Figure 10 - Adaptive vs Uniform discovery time on the largest identical discovered set\n"
        "Each dataset compares the classic pair and the online-control pair separately; lower bars are faster.",
        fontsize=14,
        fontweight="bold",
    )
    plt.tight_layout(rect=(0, 0.03, 1, 0.91))
    figure_path = output_root / "figure10_same_set_adapt_vs_uniform.png"
    plt.savefig(figure_path, dpi=300, bbox_inches="tight")
    plt.savefig(output_root / "figure10.png", dpi=300, bbox_inches="tight")
    plt.close()


def _plot_global_same_set_discovery_heatmaps(cached, algo_keys, dataset_keys, output_dir):
    all_rows = []
    dataset_keys = _ordered_comparison_datasets(dataset_keys)

    for dataset_key in dataset_keys:
        for _, mode_label, _, _, discovery_key, _ in MODE_SPECS:
            method_specs = []
            for algo_key in algo_keys:
                payload = cached.get((algo_key, dataset_key))
                if not payload:
                    continue
                method_specs.append((
                    algo_key,
                    _display_algo_name(algo_key),
                    payload.get(discovery_key, []),
                ))

            mode_rows = _compare_same_set_discovery(
                method_specs,
                left_col="algo_a",
                right_col="algo_b",
            )
            for row in mode_rows:
                row["dataset"] = dataset_key
                row["mode"] = mode_label
                all_rows.append(row)

    export_columns = [
        "dataset", "mode", "algo_a", "algo_b", "simulation", "k",
        "common_set", "time_a", "time_b", "delta_b_minus_a",
    ]
    if not all_rows:
        pd.DataFrame(columns=export_columns).to_csv(
            output_dir / "same_set_discovery_comparison.csv", index=False
        )
        return

    all_rows_df = pd.DataFrame(all_rows)
    all_rows_df[export_columns].to_csv(
        output_dir / "same_set_discovery_comparison.csv", index=False
    )

    for dataset_key in dataset_keys:
        dataset_df = all_rows_df[all_rows_df["dataset"] == dataset_key].copy()
        prepared_by_mode = []
        max_abs = 1.0
        control_arm_for_dataset = None
        for algo_key in algo_keys:
            payload = cached.get((algo_key, dataset_key))
            if payload and "control_arm" in payload:
                control_arm_for_dataset = int(payload["control_arm"])
                break
        figure_mode_specs = [
            spec for spec in MODE_SPECS
            if spec[1] in {"Adaptive", "Adaptive Var"}
        ]

        for _, mode_label, _, _, discovery_key, _ in figure_mode_specs:
            mode_df = dataset_df[dataset_df["mode"] == mode_label].copy()
            prepared = _prepare_same_set_heatmap(mode_df, "algo_a", "algo_b")
            method_specs = []
            for algo_key in algo_keys:
                payload = cached.get((algo_key, dataset_key))
                if not payload:
                    continue
                method_specs.append((
                    algo_key,
                    _display_algo_name(algo_key),
                    payload.get(discovery_key, []),
                ))
            prepared_by_mode.append((mode_label, prepared, method_specs))
            if prepared is not None and np.isfinite(prepared["matrix"]).any():
                max_abs = max(max_abs, float(np.nanmax(np.abs(prepared["matrix"]))))

        fig = plt.figure(figsize=(18, 16))
        outer_grid = fig.add_gridspec(
            len(prepared_by_mode), 1,
            left=0.05,
            right=0.88,
            bottom=0.06,
            top=0.86,
            hspace=0.36,
        )
        last_image = None
        for panel_idx, (mode_label, prepared, method_specs) in enumerate(prepared_by_mode):
            panel_grid = outer_grid[panel_idx, 0].subgridspec(
                2, 1,
                height_ratios=[0.95, 2.55],
                hspace=0.25,
            )
            table_ax = fig.add_subplot(panel_grid[0, 0])
            heatmap_ax = fig.add_subplot(panel_grid[1, 0])
            _draw_discovery_sequence_table(
                table_ax,
                method_specs,
                title=f"{mode_label}: discovery order by algorithm",
                control_arm=control_arm_for_dataset,
                displayed_ranks=prepared["k_values"] if prepared is not None else None,
            )
            image = _draw_same_set_heatmap(
                heatmap_ax,
                prepared,
                mode_label,
                max_abs=max_abs,
                show_counts=False,
            )
            if image is not None:
                last_image = image

        if last_image is not None:
            cbar_ax = fig.add_axes([0.91, 0.18, 0.015, 0.56])
            cbar = fig.colorbar(last_image, cax=cbar_ax)
            cbar.set_label("Mean time difference: algorithm B - algorithm A")

        fig.suptitle(
            f"{dataset_key.upper()} - same discovered-set timing across algorithms\n"
            "A filled cell means both algorithms found the exact same set at size k; negative means algorithm B was faster.",
            fontsize=14,
            fontweight="bold",
        )
        plt.savefig(
            output_dir / f"same_set_discovery_heatmap_{dataset_key}.png",
            dpi=300,
            bbox_inches="tight",
        )
        plt.close()


if __name__ == "__main__" and len(RUN_ALGOS) > 1:
    script_path = os.path.abspath(__file__)
    for algo_key in RUN_ALGOS:
        print(f"\n================ RUN REAL DATA WITH {algo_key.upper()} ================\n")
        env = os.environ.copy()
        env["REAL_DATA_ALGO"] = algo_key
        env["REAL_DATA_CHILD_RUN"] = "1"
        env["REAL_DATA_COMPARE_ALGOS"] = "0"
        env.pop("REAL_DATA_ALGOS", None)
        subprocess.run([sys.executable, script_path], cwd=os.path.dirname(script_path),
                       env=env, check=True)
    if GENERATE_ALGO_COMPARISON:
        generate_algorithm_comparison_figures(
            find_project_root(),
            RUN_ALGOS,
            _comparison_dataset_scope(),
        )
    sys.exit(0)

RUN_ALGO = RUN_ALGOS[0]
ACTIVE_ALGO_CONFIG = ALGORITHM_CONFIGS[RUN_ALGO]
if ACTIVE_ALGO_CONFIG["binary_module"] == ACTIVE_ALGO_CONFIG["continuous_module"]:
    usable_module = importlib.import_module(ACTIVE_ALGO_CONFIG["binary_module"])
    usable_module = importlib.reload(usable_module)
    adaptative_algorithm_binary = usable_module
    adaptative_algorithm_continuous = usable_module
else:
    adaptative_algorithm_binary = importlib.import_module(ACTIVE_ALGO_CONFIG["binary_module"])
    importlib.reload(adaptative_algorithm_binary)
    adaptative_algorithm_continuous = importlib.import_module(ACTIVE_ALGO_CONFIG["continuous_module"])
    importlib.reload(adaptative_algorithm_continuous)




# --- 1. DATA LOADING (original code, lightly cleaned) ---

# Path retrieval
current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(current_dir)

path_effort = os.path.join(root_dir, 'data', 'processed', 'effort_experiment.csv')
path_exercise = os.path.join(root_dir, 'data', 'processed', 'exercise_min.csv')
path_penn = os.path.join(root_dir, 'data', 'processed', 'penn.csv')
path_walmart = os.path.join(root_dir, 'data', 'processed', 'walmart.csv')

# Lecture des fichiers
df_effort0 = pd.read_csv(path_effort)
# Small safety check: rename if needed for effort (often 'workerId' or 'mturk_id')
if 'workerId' in df_effort0.columns: df_effort = df_effort0.rename(columns={'workerId': 'id'})
elif 'participant_id' in df_effort0.columns: df_effort = df_effort0.rename(columns={'participant_id': 'id'})

df_exercise0 = pd.read_csv(path_exercise).rename(columns={'participant_id': 'id'})
df_penn0 = pd.read_csv(path_penn).rename(columns={'participant_id': 'id'})
df_walmart0 = pd.read_csv(path_walmart).rename(columns={'participant_id': 'id'})

print("Fichiers chargés avec succès !")

# Filtrage des colonnes utiles
def sort_by_participant_id(df):
    if 'id' not in df.columns:
        return df.reset_index(drop=True)
    return df.sort_values('id', kind='mergesort').reset_index(drop=True)


def sort_by_fixed_hash_within_arm(df, seed):
    """
    Deterministic pseudo-random order for reconstructed aggregate binary data.
    This avoids the artificial 111...000... order while remaining identical
    across runs and machines.
    """
    if 'id' not in df.columns or 'arm' not in df.columns:
        return df.reset_index(drop=True)

    ordered = df.copy()

    def stable_key(row):
        raw = f"{seed}|{row['arm']}|{row['id']}".encode("utf-8")
        return int.from_bytes(hashlib.blake2b(raw, digest_size=8).digest(), "big")

    ordered["_fixed_order_key"] = ordered.apply(stable_key, axis=1)
    ordered = ordered.sort_values(
        ["arm", "_fixed_order_key"],
        kind="mergesort",
    ).drop(columns="_fixed_order_key")
    return ordered.reset_index(drop=True)


df_effort = sort_by_participant_id(df_effort0[['id', 'y', 'arm']])
df_exercise = sort_by_participant_id(df_exercise0[['id', 'y', 'arm']])
df_penn = sort_by_fixed_hash_within_arm(df_penn0[['id', 'y', 'arm']], seed="penn-fixed-v1")
df_walmart = sort_by_fixed_hash_within_arm(df_walmart0[['id', 'y', 'arm']], seed="walmart-fixed-v1")

# --- 2. NEW PREPARATION FUNCTION ---

def prepare_real_experiment(df, n_sims):
    """
    Transforme un DataFrame en structure 3D pour la simulation.
    Structure : [simulation_index][arm_index][observations_in_original_order]
    
    Returns:
        all_arm_data_by_sim: La structure de données (list of list of list)
        arm_names: La liste des noms de bras correspondant aux indices 0, 1, 2...
    """
    # 1. Group by arm and collect all Y values as lists
    # Sort arms alphabetically so index 0 is always the same
    grouped = df.groupby('arm')['y'].apply(list).sort_index()
    
    # Retrieve arm names (e.g. ['control', 'treatment_A', ...])
    arm_names = grouped.index.tolist()
    n_arms = len(arm_names)
    
    all_arm_data_by_sim = []

    # 2. Boucle sur les simulations
    for sim in range(n_sims):
        all_arm_data = []
        
        # For each arm
        for arm_name in arm_names:
            # Copy the original data while preserving its source order.
            rewards = grouped[arm_name].copy()
            
            all_arm_data.append(rewards)
            
        all_arm_data_by_sim.append(all_arm_data)
        
    return all_arm_data_by_sim, arm_names

# --- 3. RUN ON ALL DATASETS ---

def get_min_max_samples(all_arm_data):
    """
    Renvoie la taille du bras qui a le moins de données.
    Utile pour fixer l'horizon max de la simulation sans 'out of bounds'.
    """
    # Use the first simulation (index 0)
    # because the amount of data per arm is the same for all simulations
    first_simulation = all_arm_data[0]
    
    # Compute each arm length and take the minimum
    min_len = min(len(arm_data) for arm_data in first_simulation)
    max_len = max(len(arm_data) for arm_data in first_simulation)

    return min_len, max_len


def bootstrap_short_arms_for_initialization(all_arm_data, init_nb, seed):
    """
    Extend arms shorter than init_nb by sampling existing observations with
    replacement. This is meant only to make a large initialization feasible.
    """
    init_nb = int(init_nb)
    if init_nb <= 0:
        return all_arm_data, {}

    rng = np.random.default_rng(seed)
    padded = []
    report = {}

    for sim_idx, simulation in enumerate(all_arm_data):
        padded_sim = []
        for arm_idx, arm_data in enumerate(simulation):
            arm_list = list(arm_data)
            original_len = len(arm_list)
            if original_len == 0:
                raise ValueError(
                    f"Cannot bootstrap initialization for empty arm {arm_idx} "
                    f"in simulation {sim_idx}."
                )
            if original_len < init_nb:
                missing = init_nb - original_len
                extra = rng.choice(arm_list, size=missing, replace=True).tolist()
                arm_list.extend(extra)
                report[(sim_idx, arm_idx)] = {
                    "original_len": original_len,
                    "target_init_nb": init_nb,
                    "added": missing,
                }
            padded_sim.append(arm_list)
        padded.append(padded_sim)

    return padded, report

import scipy.stats as stats
from statsmodels.stats.proportion import proportion_confint

# -----------------------------------------------------------------------------
# PART 3: CONFIGURATION AND EXECUTION
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    from pathlib import Path

    def find_git_root(start: Path | None = None) -> Path:
        p = (start or Path(__file__)).resolve()
        for parent in [p, *p.parents]:
            git_entry = parent / ".git"
            if git_entry.is_dir() or git_entry.is_file():  # support worktree (.git file)
                return parent
        raise RuntimeError("Git root not found (no .git in parents)")

    git_root = find_git_root()
    output_root = git_root / ACTIVE_ALGO_CONFIG["output_dir"]
    print(f"\n=== Active usable algorithm: {RUN_ALGO.upper()} ===")
    print(f"Continuous module: {ACTIVE_ALGO_CONFIG['continuous_module']}")
    print(f"Binary module: {ACTIVE_ALGO_CONFIG['binary_module']}")
    print(f"Active datasets: {', '.join(RUN_DATASETS)}")
    print(f"Output directory: {output_root}\n")
    print(f"History record every: {HISTORY_RECORD_EVERY} step(s)\n")
    print(f"Use cache: {USE_EXPERIMENT_CACHE} | Save cache: {SAVE_EXPERIMENT_CACHE}\n")
    print(f"Only comparison plots: {ONLY_COMPARISON_PLOTS}\n")
    plt.close('all')
    
    # Scenario: 2 good arms (0, 1) and 2 bad ones (2, 3)

    n_sims = 1

    datasets = {
    "effort": (df_effort, 59),
    "exercise": (df_exercise, 53),
    "penn": (df_penn, 16),
    "walmart": (df_walmart, 3)
    }

    invert_sign_datasets = {"effort"}

    results = {}
    selected_datasets = {name: datasets[name] for name in RUN_DATASETS}
    
    print("\n--- Traitement des données ---")
    for name, df in selected_datasets.items():
        print(f"Préparation de {name}...")
        # Function call
        data_sim, arm_names = prepare_real_experiment(df[0], n_sims)
        control_arm = df[1]
        if name in invert_sign_datasets:
            data_sim = [
                [[-obs for obs in arm_data] for arm_data in simulation]
                for simulation in data_sim
            ]
            print("   -> Effort sign inverted: detecting lower original outcomes as positive effects.")
        if not 0 <= control_arm < len(arm_names):
            raise ValueError(
                f"Control arm {control_arm} is invalid for dataset {name} "
                f"with {len(arm_names)} arms."
            )
        # Stockage
        results[name] = {
            "data": data_sim,       # La liste de listes de listes
            "arm_names": arm_names,  # To know which arm index 0 corresponds to
            "control_arm": control_arm
        }
        # Quick check
        print(f"   -> {len(data_sim)} simulations générées.")
        print(f"   -> {len(arm_names)} bras trouvés.")
        print(f"   -> Exemple bras 0 ({arm_names[0]}): {len(data_sim[0][0])} observations.")

# --- 4. HOW TO USE THE DATA ---
    # Example to launch run_experiment with PENN data:
    # data_penn = results['penn']['data']
    # arm_penn = results['penn']['arm_names']
    # control_arm = 16

    # data_effort = results['effort']['data']
    # arm_effort = results['effort']['arm_names']
    # control_arm = 59, with sign-inverted observations for lower-is-better testing

    # data_walmart = results['walmart']['data']
    # arm_walmart = results['walmart']['arm_names']
    # control_arm = 3

    # data_exercise = results['exercise']['data']
    # arm_exercise = results['exercise']['arm_names']
    # control_arm = 53

    # name_data="penn"
    # name_data="effort"
    # name_data="walmart"
    # name_data="exercise"
    
    list_name=list(RUN_DATASETS)
    local_algo_payloads = {}
    num_graph=0
    for name_data in list_name:
        print("***********************name of the database treated:", name_data.upper(), "***********************")
        dataset_output_dir = output_root / name_data
        dataset_output_dir_existed = dataset_output_dir.exists()
        dataset_output_dir.mkdir(parents=True, exist_ok=True)
        run_classic_analysis = not dataset_output_dir_existed

        data_test=results[name_data]['data']
        arm_test=results[name_data]['arm_names']
        control_arm=results[name_data]['control_arm']

        # --- Utilisation ---
        min_len, max_len = get_min_max_samples(data_test)
        print("taille min =", min_len, "taille max =", max_len)

        mu_0 = 0.0
        delta = 0.05
        # horizon = min_len*10
        horizon = sum([len(arm) for arm in data_test[0]])
        n_arms = len(arm_test)
        init_nb = round(min_len*0.1)
        if name_data == "effort" and os.environ.get("REAL_DATA_EFFORT_INIT_NB"):
            init_nb = max(0, int(os.environ["REAL_DATA_EFFORT_INIT_NB"]))
            print(f"Effort init override: init_nb={init_nb}")
        init_choice = True
        mu_0_unif=mean(data_test[0][control_arm])
        print("mu_0 moyenne calcule", mu_0_unif)
        arm_test_clean = [f"{i}: {arm_test[i][:15]}" for i in range(len(arm_test))]
        classic_stats_path = dataset_output_dir / "classic_stats.txt"

        list_stat=[]
        for n in range(n_arms):
            mean_arm=round(mean(data_test[0][n]), 4)
            var_arm = round(variance(data_test[0][n]) if len(data_test[0][n]) > 1 else 0, 4)
            print("moyenne arm", n, ":", arm_test[n], "=", mean_arm, "var=", var_arm)
            list_stat.append([f"arm {n}", arm_test[n], mean_arm, var_arm, len(data_test[0][n])])

        if run_classic_analysis:
            sort_mean_desc = sorted(list_stat, key=lambda x: x[2], reverse=True)
            sort_var_desc = sorted(list_stat, key=lambda x: x[3], reverse=True)
            with open(classic_stats_path, "w", encoding="utf-8") as f:
                f.write("List of the statistics\n\n")
                for n in range(n_arms):
                    f.write(f"arm nb {n} : '{list_stat[n][1]}'\n mean = {list_stat[n][2]}\n var = {list_stat[n][3]} \n n = {list_stat[n][4]} \n")
                f.write(f"\n\n SORTING BY MEAN \n\n")
                for n in range(n_arms):
                    f.write(f"arm nb {n} : '{sort_mean_desc[n][1]}'\n mean = {sort_mean_desc[n][2]}\n var = {sort_mean_desc[n][3]} \n n = {sort_mean_desc[n][4]} \n")
                f.write(f"\n\n SORTING BY VARIANCES \n\n")
                for n in range(n_arms):
                    f.write(f"arm nb {n} : '{sort_var_desc[n][1]}'\n mean = {sort_var_desc[n][2]}\n var = {sort_var_desc[n][3]} \n n = {sort_var_desc[n][4]} \n")


        # ==========================================
        # ANALYSE STATISTIQUE
        # ==========================================
        # Choose "normal" for continuous scores (0 to 10)
        # Choose "bernouilli" for binary data (pain absent/present)
        if name_data in ["penn", "walmart"]:
            type_de_loi = "bernouilli"
        else : 
            type_de_loi = "normal"

        liste_vrai_positif=[]
        if run_classic_analysis:
            print(f"--- ANALYSE LANCÃ‰E (TYPE DE DONNÃ‰ES : {type_de_loi.upper()}) ---\n")
        else:
            liste_vrai_positif = _load_classic_positive_list(classic_stats_path)
            print(
                f"Classic statistical analysis skipped for {name_data}: "
                f"reusing {classic_stats_path}."
            )

        if run_classic_analysis and type_de_loi == "normal":
            donnees = data_test[0]
            noms_traitements = arm_test_clean[:control_arm]+arm_test_clean[control_arm+1:]
            noms_tous_groupes = arm_test_clean
            groupe_controle = donnees[control_arm]
            groupes_traitements = donnees[:control_arm]+donnees[control_arm+1:]

            # --- STATISTICAL TESTS (per-arm t-test + BH correction) ---
            from statsmodels.stats.multitest import multipletests
            indices_traitements = [i for i in range(n_arms) if i != control_arm]
            p_values_raw = []
            for groupe in groupes_traitements:
                _, p = stats.ttest_ind(groupe, groupe_controle)
                p_values_raw.append(p)

            reject, q_values, _, _ = multipletests(p_values_raw, alpha=0.05, method='fdr_bh')
            qval_dict = dict(zip(indices_traitements, q_values))
            liste_vrai_positif = [idx for idx, q in qval_dict.items() if q < 0.05]

            print("=== TESTS PAR BRAS (t-test + correction BH / q-values) ===")
            for i, (nom, q) in enumerate(zip(noms_traitements, q_values)):
                moyenne_traitement = np.mean(groupes_traitements[i])
                moyenne_controle = np.mean(groupe_controle)
                significatif = "Oui" if q < 0.05 else "Non"
                effet = "Baisse" if moyenne_traitement < moyenne_controle else "Hausse"
                print(f"Contrôle vs {nom} | q-value = {q:.4f} | Significatif : {significatif} ({effet})")

            # --- VISUALISATION ---
            means = [np.mean(d) for d in donnees]
            cis = [stats.sem(d) * 1.96 for d in donnees]
            n_obs = [len(d) for d in donnees]
            labels_courts = [nom[:25] + "…" if len(nom) > 25 else nom for nom in noms_tous_groupes]

            ordre = sorted(range(n_arms), key=lambda i: means[i])
            means_tri = [means[i] for i in ordre]
            cis_tri = [cis[i] for i in ordre]
            n_obs_tri = [n_obs[i] for i in ordre]
            labels_tri = [labels_courts[i] for i in ordre]

            # Colors based on BH q-values
            sig_flags = []
            for idx_orig in ordre:
                if idx_orig == control_arm:
                    sig_flags.append('control')
                else:
                    q = qval_dict[idx_orig]
                    sig_flags.append('sig' if q < 0.05 else 'ns')

            couleurs = []
            for flag in sig_flags:
                if flag == 'control':
                    couleurs.append('#ff6b6b')
                elif flag == 'sig':
                    couleurs.append('#8de5a1')
                else:
                    couleurs.append('#a1c9f4')

            fig, ax = plt.subplots(figsize=(10, max(6, n_arms * 0.35)))
            y_pos = range(n_arms)

            ax.barh(y_pos, means_tri, xerr=cis_tri, color=couleurs,
                    edgecolor='black', capsize=3, zorder=2, height=0.6)
            ax.axvline(x=means[control_arm], color='red', linestyle='--',
                       label=f'Moyenne contrôle ({means[control_arm]:.2f})')

            # Annotations avec q-values BH
            for idx_tri, idx_orig in enumerate(ordre):
                m = means_tri[idx_tri]
                ci = cis_tri[idx_tri]
                n = n_obs_tri[idx_tri]

                if idx_orig == control_arm:
                    label = f'{m:.2f}  (n={n})'
                else:
                    q = qval_dict[idx_orig]
                    sig = '***' if q < 0.001 else '**' if q < 0.01 else '*' if q < 0.05 else ''
                    label = f'{m:.2f}  (n={n}) {sig}'

                ax.text(m + ci + 0.01 * max(means), idx_tri, label,
                        va='center', fontsize=7)

            ax.set_yticks(y_pos)
            ax.set_yticklabels(labels_tri, fontsize=8)
            ax.set_xlabel("Moyenne ± IC 95%")
            ax.set_title(f"Comparaison des bras : {name_data}\n"
                        "t-test + correction BH (q-values) | IC 95% (moyenne ± 1.96×SEM)",
                        fontsize=14, fontweight='bold')
            ax.legend(loc='lower right')
            ax.grid(axis='x', linestyle='--', alpha=0.7, zorder=1)

            ax.text(0.99, 0.02, '* q<0.05  ** q<0.01  *** q<0.001 (BH / FDR)',
                    transform=ax.transAxes, fontsize=7, ha='right', style='italic', color='gray')

            plt.tight_layout()
            plt.savefig(dataset_output_dir / "figure0.png", dpi=300, bbox_inches="tight")
            plt.close()
        elif run_classic_analysis and type_de_loi == "bernouilli":
            # ==========================================
            # CASE 1: BINARY DATA (penn and walmart = incentive to get a vaccine)
            # ==========================================
            # --- DATA TRANSFORMATION ---
            tableau_contingence = []
            indices_valides = []
            for idx, bras in enumerate(data_test[0]):
                absents = bras.count(0)
                presents = bras.count(1)
                if absents > 0 and presents > 0:
                    tableau_contingence.append([absents, presents])
                    indices_valides.append(idx)
                else:
                    print(f"⚠️  Bras {idx} ('{arm_test_clean[idx]}') ignoré : "
                        f"données constantes ({absents} absents, {presents} présents)")

            # Recompute the control index in the filtered table
            if control_arm in indices_valides:
                control_arm_filtre = indices_valides.index(control_arm)
            else:
                print("⚠️  Le bras de contrôle a été filtré !")
                control_arm_filtre = 0

            noms_tous_groupes = [arm_test_clean[i] for i in indices_valides]
            noms_traitements = [arm_test_clean[i] for i in indices_valides if i != control_arm]

            # --- STATISTICAL TESTS (per-arm Fisher exact test + BH correction) ---
            from statsmodels.stats.multitest import multipletests
            ligne_controle = tableau_contingence[control_arm_filtre]
            lignes_traitements = (tableau_contingence[:control_arm_filtre]
                                + tableau_contingence[control_arm_filtre+1:])
            indices_traitements_filtre = [i for i in range(len(indices_valides)) if i != control_arm_filtre]

            p_values_raw = []
            for ligne_traitement in lignes_traitements:
                _, p = stats.fisher_exact([ligne_controle, ligne_traitement])
                p_values_raw.append(p)

            reject, q_values, _, _ = multipletests(p_values_raw, alpha=0.05, method='fdr_bh')
            qval_dict_bin = dict(zip(indices_traitements_filtre, q_values))

            print("=== TESTS PAR BRAS (Fisher exact + correction BH / q-values) ===")
            for i, (nom, q) in enumerate(zip(noms_traitements, q_values)):
                ligne_traitement = lignes_traitements[i]
                total_controle = sum(ligne_controle)
                total_trait = sum(ligne_traitement)
                pct_controle = (ligne_controle[1] / total_controle) * 100 if total_controle > 0 else 0
                pct_trait = (ligne_traitement[1] / total_trait) * 100 if total_trait > 0 else 0
                significatif = "Oui" if q < 0.05 else "Non"
                print(f"Contrôle ({pct_controle:.0f}%) vs {nom} ({pct_trait:.0f}%) "
                    f"| q-value = {q:.4f} | Significatif : {significatif}")
#           # --- VISUALISATION ENRICHIE ---
            proportions = [ligne[1] / sum(ligne) for ligne in tableau_contingence]
            n_obs = [sum(ligne) for ligne in tableau_contingence]
            prop_controle = proportions[control_arm_filtre]

            # IC 95% (Wilson, plus fiable que Wald pour les proportions)
            cis = []
            for p, n in zip(proportions, n_obs):
                ci = proportion_confint(round(p * n), n, alpha=0.05, method='wilson')
                cis.append((p - ci[0], ci[1] - p))  # erreur basse, erreur haute

            labels_courts = [nom[:25] + "…" if len(nom) > 25 else nom for nom in noms_tous_groupes]
            # Precompute significance for colors (BH q-values)
            sig_flags = []
            for i in range(len(proportions)):
                if i == control_arm_filtre:
                    sig_flags.append('control')
                else:
                    q = qval_dict_bin[i]
                    sig_flags.append('sig' if q < 0.05 else 'ns')
            liste_vrai_positif = [indices_valides[i] for i, flag in enumerate(sig_flags) if flag == 'sig']

            couleurs = []
            for flag in sig_flags:
                if flag == 'control':
                    couleurs.append('#ff6b6b')
                elif flag == 'sig':
                    couleurs.append('#8de5a1')
                else:
                    couleurs.append('#a1c9f4')
            fig, ax = plt.subplots(figsize=(10, max(6, len(proportions) * 0.4)))
            y_pos = range(len(proportions))

            ax.barh(y_pos, proportions,
                    xerr=list(zip(*cis)),  # asymmetric (lower, upper)
                    color=couleurs, edgecolor='black', capsize=3, zorder=2, height=0.6)

            ax.axvline(x=prop_controle, color='red', linestyle='--',
                       label=f'Contrôle ({prop_controle:.1%})')
            
            # Annotation: proportion + n + significance (BH q-values)
            for i, (p, n) in enumerate(zip(proportions, n_obs)):
                if sig_flags[i] == 'control':
                    label = f'{p:.1%}  (n={n})'
                else:
                    q = qval_dict_bin[i]
                    sig = '***' if q < 0.001 else '**' if q < 0.01 else '*' if q < 0.05 else ''
                    label = f'{p:.1%}  (n={n}) {sig}'

                ax.text(p + cis[i][1] + 0.005, i, label, va='center', fontsize=8)

            ax.set_yticks(y_pos)
            ax.set_yticklabels(labels_courts, fontsize=8)
            ax.set_xlabel("Proportion de succès ± IC 95%")
            ax.set_title(f"Proportion de succès par traitement : {name_data}\n"
             "Fisher exact + correction BH (q-values) | IC 95% Wilson",
             fontsize=14, fontweight='bold')
            ax.legend(loc='lower right')
            ax.grid(axis='x', linestyle='--', alpha=0.7, zorder=1)

            # Star legend
            ax.text(0.99, 0.02, '* q<0.05  ** q<0.01  *** q<0.001 (BH / FDR)',
                    transform=ax.transAxes, fontsize=7, ha='right', style='italic', color='gray')

            plt.tight_layout()
            plt.savefig(dataset_output_dir / "figure0.png", dpi=300, bbox_inches="tight")
            plt.close()
        elif run_classic_analysis:
            print("Erreur : La variable 'type_de_loi' doit être strictement égale à 'normal' ou 'bernouilli'.")            

        if run_classic_analysis:
            with open(classic_stats_path, "r", encoding="utf-8") as f:
                contenu_existant = f.read()
            with open(classic_stats_path, "w", encoding="utf-8") as f:
                f.write(str(liste_vrai_positif) + contenu_existant)

        init_bootstrap_report = {}
        if name_data == "effort" and init_choice and EFFORT_BOOTSTRAP_SHORT_INIT_ARMS:
            data_test, init_bootstrap_report = bootstrap_short_arms_for_initialization(
                data_test,
                init_nb,
                EFFORT_INIT_BOOTSTRAP_SEED,
            )
            if init_bootstrap_report:
                total_added = sum(row["added"] for row in init_bootstrap_report.values())
                print(
                    "Effort initialization bootstrap: "
                    f"added {total_added} synthetic initialization observation(s) "
                    f"across {len(init_bootstrap_report)} arm/simulation pair(s)."
                )
            else:
                print("Effort initialization bootstrap: no arm shorter than init_nb.")

        
        is_true_mean=False
        requested_horizon = horizon
        effective_horizon = horizon
        adaptive_stop_cap = max(1, requested_horizon * ADAPTIVE_STOP_MAX_MULTIPLIER)
        experiment_cache_path = dataset_output_dir / CACHE_FILENAME
        cached_payload = None
        adaptive_classic_probe_results = None
        uniform_classic_probe_results = None

        if USE_EXPERIMENT_CACHE and experiment_cache_path.exists():
            try:
                cached_payload = load_experiment_cache(experiment_cache_path)
                cache_checks = {
                    "algorithm": RUN_ALGO,
                    "dataset": name_data,
                    "type_de_loi": type_de_loi,
                    "control_arm": control_arm,
                    "n_arms": n_arms,
                    "requested_horizon": requested_horizon,
                    "stop_rule": STOP_RULE,
                }
                if name_data == "effort" and "effort_bootstrap_short_init_arms" in cached_payload:
                    cache_checks["effort_bootstrap_short_init_arms"] = EFFORT_BOOTSTRAP_SHORT_INIT_ARMS
                if name_data == "effort" and "effort_init_bootstrap_seed" in cached_payload:
                    cache_checks["effort_init_bootstrap_seed"] = EFFORT_INIT_BOOTSTRAP_SEED
                if "deterministic_bootstrap_key" in cached_payload:
                    cache_checks["deterministic_bootstrap_key"] = name_data
                if STOP_RULE == "horizon":
                    cache_checks["horizon"] = horizon
                mismatches = [
                    key for key, expected in cache_checks.items()
                    if cached_payload.get(key) != expected
                ]
                if mismatches:
                    print(
                        f"Cache metadata mismatch on {mismatches}; "
                        f"ignoring {experiment_cache_path}."
                    )
                    cached_payload = None
                else:
                    horizon = int(cached_payload.get("horizon", horizon))
                    effective_horizon = horizon
                    print(f"Loaded cached run_experiment variables from {experiment_cache_path}")
            except Exception as exc:
                print(f"Could not load cache {experiment_cache_path}: {exc}. Re-running experiments.")
                cached_payload = None

        adaptive_stop_target_arms = [
            int(arm_idx) for arm_idx in range(n_arms)
            if int(arm_idx) != int(control_arm)
        ]

        if STOP_RULE == "adaptive_classic_all_non_control_arms" and cached_payload is None:
            print(
                "Stopping rule: adaptive classic must find every non-control arm as positive "
                f"(cap={adaptive_stop_cap})."
            )
            runner_module = (
                adaptative_algorithm_continuous
                if type_de_loi == "normal"
                else adaptative_algorithm_binary
            )
            adaptive_classic_probe_results = runner_module.run_experiment(
                arm_test,
                mu_0_unif,
                delta,
                adaptive_stop_cap,
                'adaptive',
                data_test,
                n_sims,
                control_arm,
                init_nb,
                init_choice,
                False,
                is_true_mean,
                return_discovery_times=True,
                return_bootstrap_times=True,
                history_record_every=HISTORY_RECORD_EVERY,
                deterministic_bootstrap_key=name_data,
                stop_when_all_non_control_found=True,
                stop_control_arm=control_arm,
            )
            probe_discovery_adapt = adaptive_classic_probe_results[7]
            stop_time, reached_all = _adaptive_classic_stop_time(
                probe_discovery_adapt,
                adaptive_stop_target_arms,
            )
            if reached_all:
                effective_horizon = max(1, int(stop_time))
                print(
                    "Adaptive classic found every non-control arm as positive at "
                    f"t={effective_horizon}; using this as common horizon."
                )
            else:
                effective_horizon = adaptive_stop_cap
                print(
                    "Adaptive classic did not find every non-control arm as positive before "
                    f"the cap; using cap horizon={effective_horizon}."
                )
            horizon = effective_horizon

        if STOP_RULE == "uniform_classic_all_non_control_arms" and cached_payload is None:
            print(
                "Stopping rule: uniform classic must find every non-control arm as positive "
                f"(cap={adaptive_stop_cap})."
            )
            runner_module = (
                adaptative_algorithm_continuous
                if type_de_loi == "normal"
                else adaptative_algorithm_binary
            )
            uniform_classic_probe_results = runner_module.run_experiment(
                arm_test,
                mu_0_unif,
                delta,
                adaptive_stop_cap,
                'uniform',
                data_test,
                n_sims,
                control_arm,
                init_nb,
                init_choice,
                False,
                is_true_mean,
                return_discovery_times=True,
                return_bootstrap_times=True,
                history_record_every=HISTORY_RECORD_EVERY,
                deterministic_bootstrap_key=name_data,
                stop_when_all_non_control_found=True,
                stop_control_arm=control_arm,
            )
            probe_discovery_unif = uniform_classic_probe_results[7]
            stop_time, reached_all = _adaptive_classic_stop_time(
                probe_discovery_unif,
                adaptive_stop_target_arms,
            )
            if reached_all:
                effective_horizon = max(1, int(stop_time))
                print(
                    "Uniform classic found every non-control arm as positive at "
                    f"t={effective_horizon}; using this as common horizon."
                )
            else:
                effective_horizon = adaptive_stop_cap
                print(
                    "Uniform classic did not find every non-control arm as positive before "
                    f"the cap; using cap horizon={effective_horizon}."
                )
            horizon = effective_horizon

        if cached_payload is not None:
            pnb_unif = cached_payload["pnb_unif"]
            pnb_unif_list = cached_payload.get("pnb_unif_list")
            counts_unif_mean = cached_payload["counts_unif_mean"]
            counts_unif_list = cached_payload["counts_unif_list"]
            np_p_value_list_unif = cached_payload["np_p_value_list_unif"]
            np_p_value_mean_unif = cached_payload["np_p_value_mean_unif"]
            l_pos_unif = cached_payload["l_pos_unif"]
            discovery_unif = cached_payload["discovery_unif"]
            bootstrap_unif = cached_payload["bootstrap_unif"]

            pnb_unif_v = cached_payload["pnb_unif_v"]
            pnb_unif_v_list = cached_payload.get("pnb_unif_v_list")
            counts_unif_v_mean = cached_payload["counts_unif_v_mean"]
            counts_unif_v_list = cached_payload["counts_unif_v_list"]
            np_p_value_list_unif_v = cached_payload["np_p_value_list_unif_v"]
            np_p_value_mean_unif_v = cached_payload["np_p_value_mean_unif_v"]
            l_pos_unif_v = cached_payload["l_pos_unif_v"]
            discovery_unif_v = cached_payload["discovery_unif_v"]
            bootstrap_unif_v = cached_payload["bootstrap_unif_v"]

            pnb_adapt = cached_payload["pnb_adapt"]
            pnb_adapt_list = cached_payload.get("pnb_adapt_list")
            counts_adapt_mean = cached_payload["counts_adapt_mean"]
            counts_adapt_list = cached_payload["counts_adapt_list"]
            np_p_value_list_adapt = cached_payload["np_p_value_list_adapt"]
            np_p_value_mean_adapt = cached_payload["np_p_value_mean_adapt"]
            l_pos_adapt = cached_payload["l_pos_adapt"]
            discovery_adapt = cached_payload["discovery_adapt"]
            bootstrap_adapt = cached_payload["bootstrap_adapt"]

            pnb_adapt_v = cached_payload["pnb_adapt_v"]
            pnb_adapt_v_list = cached_payload.get("pnb_adapt_v_list")
            counts_adapt_v_mean = cached_payload["counts_adapt_v_mean"]
            counts_adapt_v_list = cached_payload["counts_adapt_v_list"]
            np_p_value_list_adapt_v = cached_payload["np_p_value_list_adapt_v"]
            np_p_value_mean_adapt_v = cached_payload["np_p_value_mean_adapt_v"]
            l_pos_adapt_v = cached_payload["l_pos_adapt_v"]
            discovery_adapt_v = cached_payload["discovery_adapt_v"]
            bootstrap_adapt_v = cached_payload["bootstrap_adapt_v"]

            cached_payload["true_positives"] = list(map(int, liste_vrai_positif))
            if SAVE_EXPERIMENT_CACHE:
                save_experiment_cache(experiment_cache_path, cached_payload)
            local_algo_payloads[name_data] = cached_payload
        else:
            # 1. Run Simulations
            if type_de_loi=="normal":
                if uniform_classic_probe_results is not None:
                    pnb_unif, pnb_unif_list, counts_unif_mean, counts_unif_list, np_p_value_list_unif, np_p_value_mean_unif, l_pos_unif, discovery_unif, bootstrap_unif = _probe_results_as_run_results(uniform_classic_probe_results)
                else:
                    pnb_unif, pnb_unif_list, counts_unif_mean, counts_unif_list,  np_p_value_list_unif, np_p_value_mean_unif, l_pos_unif, discovery_unif, bootstrap_unif = adaptative_algorithm_continuous.run_experiment(arm_test, mu_0_unif, delta, horizon, 'uniform', data_test, n_sims, control_arm, init_nb, init_choice, False, is_true_mean, return_discovery_times=True, return_bootstrap_times=True, history_record_every=HISTORY_RECORD_EVERY, deterministic_bootstrap_key=name_data)
                pnb_unif_v, pnb_unif_v_list, counts_unif_v_mean, counts_unif_v_list, np_p_value_list_unif_v, np_p_value_mean_unif_v, l_pos_unif_v, discovery_unif_v, bootstrap_unif_v = adaptative_algorithm_continuous.run_experiment(arm_test, mu_0_unif, delta, horizon, 'uniform', data_test, n_sims, control_arm, init_nb, init_choice, True, is_true_mean, return_discovery_times=True, return_bootstrap_times=True, history_record_every=HISTORY_RECORD_EVERY, deterministic_bootstrap_key=name_data)
                if adaptive_classic_probe_results is not None:
                    pnb_adapt, pnb_adapt_list, counts_adapt_mean, counts_adapt_list, np_p_value_list_adapt, np_p_value_mean_adapt, l_pos_adapt, discovery_adapt, bootstrap_adapt = _probe_results_as_run_results(adaptive_classic_probe_results)
                else:
                    pnb_adapt, pnb_adapt_list, counts_adapt_mean, counts_adapt_list, np_p_value_list_adapt, np_p_value_mean_adapt, l_pos_adapt, discovery_adapt, bootstrap_adapt = adaptative_algorithm_continuous.run_experiment(arm_test, mu_0_unif, delta, horizon, 'adaptive', data_test, n_sims, control_arm, init_nb, init_choice, False, is_true_mean, return_discovery_times=True, return_bootstrap_times=True, history_record_every=HISTORY_RECORD_EVERY, deterministic_bootstrap_key=name_data)
                pnb_adapt_v, pnb_adapt_v_list, counts_adapt_v_mean, counts_adapt_v_list, np_p_value_list_adapt_v, np_p_value_mean_adapt_v, l_pos_adapt_v, discovery_adapt_v, bootstrap_adapt_v = adaptative_algorithm_continuous.run_experiment(arm_test, mu_0_unif, delta, horizon, 'adaptive', data_test, n_sims, control_arm, init_nb, init_choice, True, is_true_mean, return_discovery_times=True, return_bootstrap_times=True, history_record_every=HISTORY_RECORD_EVERY, deterministic_bootstrap_key=name_data)
            elif type_de_loi=="bernouilli":
                if uniform_classic_probe_results is not None:
                    pnb_unif, pnb_unif_list, counts_unif_mean, counts_unif_list, np_p_value_list_unif, np_p_value_mean_unif, l_pos_unif, discovery_unif, bootstrap_unif = _probe_results_as_run_results(uniform_classic_probe_results)
                else:
                    pnb_unif, pnb_unif_list, counts_unif_mean, counts_unif_list,  np_p_value_list_unif, np_p_value_mean_unif, l_pos_unif, discovery_unif, bootstrap_unif = adaptative_algorithm_binary.run_experiment(arm_test, mu_0_unif, delta, horizon, 'uniform', data_test, n_sims, control_arm, init_nb, init_choice, False, is_true_mean, return_discovery_times=True, return_bootstrap_times=True, history_record_every=HISTORY_RECORD_EVERY, deterministic_bootstrap_key=name_data)
                pnb_unif_v, pnb_unif_v_list, counts_unif_v_mean, counts_unif_v_list, np_p_value_list_unif_v, np_p_value_mean_unif_v, l_pos_unif_v, discovery_unif_v, bootstrap_unif_v = adaptative_algorithm_binary.run_experiment(arm_test, mu_0_unif, delta, horizon, 'uniform', data_test, n_sims, control_arm, init_nb, init_choice, True, is_true_mean, return_discovery_times=True, return_bootstrap_times=True, history_record_every=HISTORY_RECORD_EVERY, deterministic_bootstrap_key=name_data)
                if adaptive_classic_probe_results is not None:
                    pnb_adapt, pnb_adapt_list, counts_adapt_mean, counts_adapt_list, np_p_value_list_adapt, np_p_value_mean_adapt, l_pos_adapt, discovery_adapt, bootstrap_adapt = _probe_results_as_run_results(adaptive_classic_probe_results)
                else:
                    pnb_adapt, pnb_adapt_list, counts_adapt_mean, counts_adapt_list, np_p_value_list_adapt, np_p_value_mean_adapt, l_pos_adapt, discovery_adapt, bootstrap_adapt = adaptative_algorithm_binary.run_experiment(arm_test, mu_0_unif, delta, horizon, 'adaptive', data_test, n_sims, control_arm, init_nb, init_choice, False, is_true_mean, return_discovery_times=True, return_bootstrap_times=True, history_record_every=HISTORY_RECORD_EVERY, deterministic_bootstrap_key=name_data)
                pnb_adapt_v, pnb_adapt_v_list, counts_adapt_v_mean, counts_adapt_v_list, np_p_value_list_adapt_v, np_p_value_mean_adapt_v, l_pos_adapt_v, discovery_adapt_v, bootstrap_adapt_v = adaptative_algorithm_binary.run_experiment(arm_test, mu_0_unif, delta, horizon, 'adaptive', data_test, n_sims, control_arm, init_nb, init_choice, True, is_true_mean, return_discovery_times=True, return_bootstrap_times=True, history_record_every=HISTORY_RECORD_EVERY, deterministic_bootstrap_key=name_data)

            cache_payload = {
                "cache_version": CACHE_VERSION,
                "algorithm": RUN_ALGO,
                "dataset": name_data,
                "type_de_loi": type_de_loi,
                "control_arm": control_arm,
                "n_arms": n_arms,
                "horizon": horizon,
                "requested_horizon": requested_horizon,
                "stop_rule": STOP_RULE,
                "adaptive_stop_max_multiplier": ADAPTIVE_STOP_MAX_MULTIPLIER,
                "effort_bootstrap_short_init_arms": (
                    EFFORT_BOOTSTRAP_SHORT_INIT_ARMS if name_data == "effort" else False
                ),
                "effort_init_bootstrap_seed": (
                    EFFORT_INIT_BOOTSTRAP_SEED if name_data == "effort" else None
                ),
                "init_bootstrap_report": init_bootstrap_report,
                "deterministic_bootstrap_key": name_data,
                "n_sims": n_sims,
                "init_nb": init_nb,
                "init_choice": init_choice,
                "history_record_every": HISTORY_RECORD_EVERY,
                "mu_0_unif": mu_0_unif,
                "true_positives": list(map(int, liste_vrai_positif)),
                "arm_names": list(arm_test),
                "pnb_unif": pnb_unif,
                "pnb_unif_list": pnb_unif_list,
                "counts_unif_mean": counts_unif_mean,
                "counts_unif_list": counts_unif_list,
                "np_p_value_list_unif": np_p_value_list_unif,
                "np_p_value_mean_unif": np_p_value_mean_unif,
                "l_pos_unif": l_pos_unif,
                "discovery_unif": discovery_unif,
                "bootstrap_unif": bootstrap_unif,
                "pnb_unif_v": pnb_unif_v,
                "pnb_unif_v_list": pnb_unif_v_list,
                "counts_unif_v_mean": counts_unif_v_mean,
                "counts_unif_v_list": counts_unif_v_list,
                "np_p_value_list_unif_v": np_p_value_list_unif_v,
                "np_p_value_mean_unif_v": np_p_value_mean_unif_v,
                "l_pos_unif_v": l_pos_unif_v,
                "discovery_unif_v": discovery_unif_v,
                "bootstrap_unif_v": bootstrap_unif_v,
                "pnb_adapt": pnb_adapt,
                "pnb_adapt_list": pnb_adapt_list,
                "counts_adapt_mean": counts_adapt_mean,
                "counts_adapt_list": counts_adapt_list,
                "np_p_value_list_adapt": np_p_value_list_adapt,
                "np_p_value_mean_adapt": np_p_value_mean_adapt,
                "l_pos_adapt": l_pos_adapt,
                "discovery_adapt": discovery_adapt,
                "bootstrap_adapt": bootstrap_adapt,
                "pnb_adapt_v": pnb_adapt_v,
                "pnb_adapt_v_list": pnb_adapt_v_list,
                "counts_adapt_v_mean": counts_adapt_v_mean,
                "counts_adapt_v_list": counts_adapt_v_list,
                "np_p_value_list_adapt_v": np_p_value_list_adapt_v,
                "np_p_value_mean_adapt_v": np_p_value_mean_adapt_v,
                "l_pos_adapt_v": l_pos_adapt_v,
                "discovery_adapt_v": discovery_adapt_v,
                "bootstrap_adapt_v": bootstrap_adapt_v,
            }
            if SAVE_EXPERIMENT_CACHE:
                save_experiment_cache(experiment_cache_path, cache_payload)
                print(f"Saved run_experiment variables to {experiment_cache_path}")
            local_algo_payloads[name_data] = cache_payload

        same_set_method_specs = [
            ("UNIF", "UNIF", discovery_unif),
            ("ADAPT", "ADAPT", discovery_adapt),
            ("UNIF VAR", "UNIF VAR", discovery_unif_v),
            ("ADAPT VAR", "ADAPT VAR", discovery_adapt_v),
        ]
        same_set_allowed_pairs = [
            ("UNIF", "ADAPT"),
            ("UNIF VAR", "ADAPT VAR"),
            ("ADAPT", "ADAPT VAR"),
        ]
        write_same_set_discovery_comparison(
            same_set_method_specs,
            dataset_output_dir / "same_set_discovery_comparison.csv",
            dataset_output_dir / "figure9_same_set_discovery_heatmap.png",
            f"{_display_algo_name(RUN_ALGO)} on {name_data}",
            allowed_pairs=same_set_allowed_pairs,
            control_arm=control_arm,
        )

        if ONLY_COMPARISON_PLOTS:
            print(f"Skipping classic per-dataset plots for {name_data}; comparison plots remain enabled.")
            num_graph += 1
            continue


        with open(dataset_output_dir / "resultats.txt", "w", encoding="utf-8") as f:
            f.write("List of the positive arm detected\n\n")
            f.write("   UNIF\n")
            for i, element in enumerate(l_pos_unif, 1):
                f.write(f"{i}. {element}\n")
            f.write("   UNIF VAR\n")
            for i, element in enumerate(l_pos_unif_v, 1):
                f.write(f"{i}. {element}\n")
            f.write("   ADAPT\n")
            for i, element in enumerate(l_pos_adapt, 1):
                f.write(f"{i}. {element}\n")
            f.write("   ADAPT VAR\n")
            for i, element in enumerate(l_pos_adapt_v, 1):
                f.write(f"{i}. {element}\n")

        with open(dataset_output_dir / "discovery_times.txt", "w", encoding="utf-8") as f:
            f.write("First discovery time by simulation and arm\n\n")
            for mode_name, discovery_list in [
                ("UNIF", discovery_unif),
                ("UNIF VAR", discovery_unif_v),
                ("ADAPT", discovery_adapt),
                ("ADAPT VAR", discovery_adapt_v),
            ]:
                f.write(f"   {mode_name}\n")
                for sim_idx, discovery_dict in enumerate(discovery_list, 1):
                    ordered = dict(sorted(discovery_dict.items()))
                    f.write(f"{sim_idx}. {ordered}\n")

        with open(dataset_output_dir / "bootstrap_times.txt", "w", encoding="utf-8") as f:
            f.write("First replacement-draw time by simulation and arm\n\n")
            f.write("A listed arm has exhausted its original observations and is now sampled with replacement.\n\n")
            for mode_name, bootstrap_list in [
                ("UNIF", bootstrap_unif),
                ("UNIF VAR", bootstrap_unif_v),
                ("ADAPT", bootstrap_adapt),
                ("ADAPT VAR", bootstrap_adapt_v),
            ]:
                f.write(f"   {mode_name}\n")
                for sim_idx, bootstrap_dict in enumerate(bootstrap_list, 1):
                    ordered = dict(sorted(bootstrap_dict.items()))
                    f.write(f"{sim_idx}. {ordered}\n")

        print("pos unif:", l_pos_unif)
        print("pos unif v:", l_pos_unif_v)
        print("pos adapt:", l_pos_adapt)
        print("pos adapt v:", l_pos_adapt_v)

        

        with open(dataset_output_dir / "resultats.txt", "r", encoding="utf-8") as f:
            contenu = f.read()

        # Regex: capture the method name and the content between {}
        pattern = r'(UNIF VAR|ADAPT VAR|UNIF|ADAPT)\s+\d+\.\s+\{([^}]*)\}'
        matches = re.findall(pattern, contenu)
        print(matches)

        resultats = {}
        print(matches)
        if matches:
            for nom, nombres in matches:
                resultats[nom] = set(int(x.strip()) for x in nombres.split(',') if x.strip())

        liste_unif = resultats.get('UNIF', set())
        liste_unif_var = resultats.get('UNIF VAR', set())
        liste_adapt = resultats.get('ADAPT', set())
        liste_adapt_var = resultats.get('ADAPT VAR', set())

        def plot_detection_comparison(vrais_positifs, detectes_list, tous_les_bras, arm_names, name_data):
            """
            vrais_positifs : liste d'indices
            detectes_list : [(set_indices, "nom_mode"), ...]
            """
            from matplotlib.patches import Patch

            n_modes = len(detectes_list)
            fig, axes = plt.subplots(1, n_modes, figsize=(6 * n_modes, max(6, len(tous_les_bras) * 0.35)),
                                     sharey=True)
            if n_modes == 1:
                axes = [axes]

            couleurs_map = {
                'TP (bien détecté)': '#8de5a1',
                'FP (faux positif)': '#ff6b6b',
                'FN (manqué)': '#ffb347',
                'TN (correct)': '#a1c9f4'
            }
            labels = [nom[:25] + "…" if len(nom) > 25 else nom for nom in arm_names]
            y_pos = range(len(tous_les_bras))

            for ax, (detectes, mode) in zip(axes, detectes_list):
                categories = []
                for i in tous_les_bras:
                    if i in vrais_positifs and i in detectes:
                        categories.append('TP (bien détecté)')
                    elif i not in vrais_positifs and i in detectes:
                        categories.append('FP (faux positif)')
                    elif i in vrais_positifs and i not in detectes:
                        categories.append('FN (manqué)')
                    else:
                        categories.append('TN (correct)')

                couleurs = [couleurs_map[c] for c in categories]
                ax.barh(y_pos, [1]*len(tous_les_bras), color=couleurs, edgecolor='black', height=0.6)

                for i, cat in enumerate(categories):
                    ax.text(0.5, i, cat, ha='center', va='center', fontsize=7, fontweight='bold')

                ax.set_xlim(0, 1)
                ax.set_xticks([])
                ax.set_title(mode.upper(), fontsize=12, fontweight='bold')

                n_tp = categories.count('TP (bien détecté)')
                n_fp = categories.count('FP (faux positif)')
                n_fn = categories.count('FN (manqué)')
                precision = f'{n_tp/(n_tp+n_fp):.0%}' if (n_tp+n_fp) > 0 else 'N/A'
                rappel = f'{n_tp/(n_tp+n_fn):.0%}' if (n_tp+n_fn) > 0 else 'N/A'
                ax.text(0.5, -0.05, f'TP={n_tp} FP={n_fp} FN={n_fn}\n'
                        f'Préc={precision} Rap={rappel}',
                        transform=ax.transAxes, fontsize=8, ha='center', style='italic', color='gray')

            axes[0].set_yticks(y_pos)
            axes[0].set_yticklabels(labels, fontsize=8)

            legend = [Patch(facecolor=c, edgecolor='black', label=l) for l, c in couleurs_map.items()]
            fig.legend(handles=legend, loc='lower center', ncol=4, fontsize=8,
                       bbox_to_anchor=(0.5, -0.02))

            fig.suptitle(f"Détection des bras significatifs : {name_data}", fontsize=14, fontweight='bold')
            plt.tight_layout()
            plt.savefig(dataset_output_dir / "figure6.png", dpi=300, bbox_inches="tight")
            plt.close()

        # Appel
        detectes_list = [(liste_unif, "unif"), (liste_unif_var, "unif var"),
                         (liste_adapt, "adapt"), (liste_adapt_var, "adapt var")]
        plot_detection_comparison(liste_vrai_positif, detectes_list, range(len(arm_test)), arm_test_clean, name_data)

        def summarize_discovery_times(discovery_list, positive_arms):
            summary = {}
            for arm_idx in positive_arms:
                found_times = [disc[arm_idx] for disc in discovery_list if arm_idx in disc]
                summary[arm_idx] = {
                    "mean_time": float(np.mean(found_times)) if found_times else np.nan,
                    "found_rate": len(found_times) / len(discovery_list) if discovery_list else 0.0,
                }
            return summary

        def plot_positive_rank_vs_discovery_time(
            positive_arms, arm_names, all_data, discovery_by_mode, horizon, output_path
        ):
            if not positive_arms:
                return

            empirical_means = np.array([float(np.mean(values)) for values in all_data])
            ranked_arms = sorted(range(len(empirical_means)),
                                 key=lambda idx: empirical_means[idx],
                                 reverse=True)
            rank_by_arm = {arm_idx: rank + 1 for rank, arm_idx in enumerate(ranked_arms)}
            positives_sorted = sorted(positive_arms, key=lambda idx: rank_by_arm[idx])
            y_means = [empirical_means[idx] for idx in positives_sorted]
            mean_span = max(y_means) - min(y_means) if y_means else 0.0
            y_jitter = max(mean_span * 0.015, 1e-4)

            summaries_by_mode = {
                mode_name: summarize_discovery_times(discovery_list, positives_sorted)
                for mode_name, discovery_list in discovery_by_mode
            }

            panels = [
                ("CLASSIQUE", [("unif", "Uniform", "#7f7f7f", -0.10),
                               ("adapt", "Adaptive", "#2ca02c", 0.10)]),
                ("VAR", [("unif var", "Uniform Var", "#7f7f7f", -0.10),
                         ("adapt var", "Adaptive Var", "#2ca02c", 0.10)]),
            ]

            all_found_times = []
            for summary in summaries_by_mode.values():
                all_found_times.extend(
                    item["mean_time"] for item in summary.values()
                    if not np.isnan(item["mean_time"])
                )

            if all_found_times:
                max_found_time = max(all_found_times)
                missed_x = max_found_time * 1.10
                x_top = max_found_time * 1.18
            else:
                missed_x = horizon
                x_top = horizon * 1.08

            x_bottom = 0
            if all_found_times:
                x_bottom = max(0, min(all_found_times) - 0.08 * max_found_time)

            rows_for_csv = []

            fig, axes = plt.subplots(
                1, len(panels),
                figsize=(8.5 * len(panels), 6.2),
                sharey=True,
                constrained_layout=True,
            )
            if len(panels) == 1:
                axes = [axes]

            for ax, (panel_title, panel_modes) in zip(axes, panels):
                label_positions = {}
                stats_lines = []

                for mode_name, pretty_name, color, y_offset in panel_modes:
                    summary = summaries_by_mode[mode_name]
                    found_count = 0
                    missed_count = 0

                    for arm_idx in positives_sorted:
                        item = summary[arm_idx]
                        is_found = not np.isnan(item["mean_time"])
                        x = item["mean_time"] if is_found else missed_x
                        y = empirical_means[arm_idx] + y_offset * y_jitter
                        size = 65 + 140 * item["found_rate"]

                        rows_for_csv.append({
                            "mode": mode_name,
                            "arm": arm_idx,
                            "arm_name": arm_names[arm_idx],
                            "empirical_rank": rank_by_arm[arm_idx],
                            "empirical_mean": empirical_means[arm_idx],
                            "mean_discovery_time": item["mean_time"],
                            "found_rate": item["found_rate"],
                        })

                        if is_found:
                            found_count += 1
                            ax.scatter(x, y, s=size, color=color, marker="o",
                                       edgecolor="black", linewidth=0.6,
                                       alpha=0.58, zorder=3,
                                       label=pretty_name if arm_idx == positives_sorted[0] else "_nolegend_")
                            if arm_idx not in label_positions:
                                label_positions[arm_idx] = (x, empirical_means[arm_idx])
                        else:
                            missed_count += 1
                            ax.scatter(x, y, s=50, color=color, marker="x",
                                       linewidth=1.4, alpha=0.72, zorder=3,
                                       label=f"{pretty_name} not found" if arm_idx == positives_sorted[0] else "_nolegend_")

                    stats_lines.append(f"{pretty_name}: {found_count}/{len(positives_sorted)}")

                for arm_idx, (x, y) in label_positions.items():
                    ax.annotate(str(arm_idx), (x, y), xytext=(4, 4),
                                textcoords="offset points", fontsize=7, color="black")

                ax.axvline(missed_x, color="red", linestyle=":", linewidth=1.3)
                ax.text(0.98, 0.98, "\n".join(stats_lines),
                        transform=ax.transAxes, ha="right", va="top",
                        fontsize=9, bbox=dict(facecolor="white", edgecolor="none", alpha=0.75))
                ax.set_title(panel_title)
                ax.set_xlabel("First discovery time (mean over simulations)")
                ax.grid(True, alpha=0.3)
                ax.set_xlim(x_bottom, x_top)
                y_padding = max(mean_span * 0.08, 4 * y_jitter)
                ax.set_ylim(min(y_means) - y_padding, max(y_means) + y_padding)

            axes[0].set_ylabel("Full-data empirical mean")
            handles = [
                plt.Line2D([0], [0], marker="o", color="w",
                           markerfacecolor="#7f7f7f", markeredgecolor="black",
                           alpha=0.58, markersize=8, label="Uniform"),
                plt.Line2D([0], [0], marker="o", color="w",
                           markerfacecolor="#1eff1e", markeredgecolor="black",
                           alpha=0.58, markersize=8, label="Adaptive"),
                plt.Line2D([0], [0], marker="x", color="black",
                           linestyle="None", markersize=8, label="not found"),
            ]
            fig.legend(handles=handles, loc="lower center", ncol=3,
                       bbox_to_anchor=(0.5, -0.04))
            fig.suptitle(f"Detected arms: discovery time vs full-data empirical mean ({name_data})",
                         fontsize=14, fontweight="bold")
            plt.savefig(output_path, dpi=300, bbox_inches="tight")
            plt.close()

            pd.DataFrame(rows_for_csv).to_csv(
                output_path.with_suffix(".csv"), index=False
            )

        discovery_by_mode = [
            ("unif", discovery_unif),
            ("unif var", discovery_unif_v),
            ("adapt", discovery_adapt),
            ("adapt var", discovery_adapt_v),
        ]
        arms_for_rank_plot = set(liste_vrai_positif)
        for _, discovery_list in discovery_by_mode:
            for discovery_dict in discovery_list:
                arms_for_rank_plot.update(discovery_dict.keys())
        arms_for_rank_plot.discard(control_arm)
        plot_positive_rank_vs_discovery_time(
            sorted(arms_for_rank_plot), arm_test_clean, data_test[0],
            discovery_by_mode, horizon, dataset_output_dir / "figure7_rank_vs_discovery.png"
        )

        history_steps = np.array(
            [0] + [
                step for step in range(1, horizon + 1)
                if step == horizon or step % HISTORY_RECORD_EVERY == 0
            ]
        )

        def time_axis_for(arr):
            length = arr.shape[0] if hasattr(arr, "shape") else len(arr)
            if length == len(history_steps):
                return history_steps
            if length == len(history_steps) - 1:
                return history_steps[1:]
            return np.linspace(0, horizon, length)

        # --- PLOT 1: pr ---
        fig, ax = plt.subplots(figsize=(10, 5))
        classic_color = '#ff7f0e'
        var_color = '#1f77b4'
        figure1_curves = [
            (pnb_adapt, "Adaptive", classic_color, "-"),
            (pnb_unif, "Uniform", classic_color, "--"),
            (pnb_adapt_v, "Adaptive_Var", var_color, "-"),
            (pnb_unif_v, "Uniform_Var", var_color, "--"),
        ]
        max_curve = max(
            [1.0] + [
                float(np.nanmax(np.asarray(curve, dtype=float)))
                for curve, _, _, _ in figure1_curves
                if len(curve)
            ]
        )
        init_cost = _initialization_cost(init_nb, n_arms, init_choice)
        y_bottom = -0.08 * max_curve
        _add_initialization_band(ax, init_cost, y_bottom, label="Initialization budget")
        for curve, label, color, linestyle in figure1_curves:
            curve_arr = np.asarray(curve, dtype=float)
            ax.plot(time_axis_for(curve_arr), curve_arr, label=label,
                    color=color, linestyle=linestyle, linewidth=2,
                    alpha=0.7)
        ax.axhline(y=1.0, color='gray', linestyle=':')
        ax.set_title("Discovery speed (pr)")
        ax.set_xlabel("Rounds after initialization")
        ax.set_ylabel("Detected positives")
        ax.set_ylim(y_bottom * 1.15, max_curve * 1.08)
        ax.set_xlim(-init_cost if init_cost > 0 else 0, max(horizon, 1))
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.savefig(dataset_output_dir / "figure1.png", dpi=300, bbox_inches="tight")
        plt.close()

        # --- PLOT 8: discovery speed with replacement-draw starts ---
        def _flatten_bootstrap_times(bootstrap_list):
            times = []
            arms = []
            for bootstrap_dict in bootstrap_list or []:
                for arm_idx, first_time in bootstrap_dict.items():
                    times.append(int(first_time))
                    arms.append(int(arm_idx))
            return np.asarray(times, dtype=float), np.asarray(arms, dtype=int)

        fig, ax = plt.subplots(figsize=(12, 6))
        figure8_specs = [
            ("Adaptive", pnb_adapt, bootstrap_adapt, classic_color, "-", 0),
            ("Uniform", pnb_unif, bootstrap_unif, classic_color, "--", 1),
            ("Adaptive Var", pnb_adapt_v, bootstrap_adapt_v, var_color, "-", 2),
            ("Uniform Var", pnb_unif_v, bootstrap_unif_v, var_color, "--", 3),
        ]
        max_curve = 1.0
        for _, curve, _, _, _, _ in figure8_specs:
            curve_arr = np.asarray(curve, dtype=float)
            if curve_arr.size:
                max_curve = max(max_curve, float(np.nanmax(curve_arr)))

        rug_step = max_curve * 0.08
        rug_levels = [-(idx + 1) * rug_step for idx in range(len(figure8_specs))]
        y_bottom = min(rug_levels) - rug_step
        _add_initialization_band(ax, init_cost, y_bottom, label="Initialization budget")

        for label, curve, bootstrap_list, color, linestyle, rug_idx in figure8_specs:
            curve_arr = np.asarray(curve, dtype=float)
            x_curve = time_axis_for(curve_arr)
            ax.plot(x_curve, curve_arr, label=label, color=color,
                    linestyle=linestyle, linewidth=2, alpha=0.78)

            times, _ = _flatten_bootstrap_times(bootstrap_list)
            if times.size:
                y_values = np.full(times.shape, rug_levels[rug_idx], dtype=float)
                ax.scatter(times, y_values, color=color, marker="|", s=180,
                           alpha=0.48, linewidths=1.4, label="_nolegend_")
                ax.text(0, rug_levels[rug_idx], f"{label} replacement starts",
                        va="center", ha="left", fontsize=8, color=color, alpha=0.9)

        ax.axhline(y=0, color="gray", linestyle=":", linewidth=1)
        ax.axhline(y=1.0, color="gray", linestyle=":", linewidth=1)
        ax.set_xlabel("Rounds after initialization")
        ax.set_ylabel("Detected positives")
        ax.set_title(
            "Discovery speed and first replacement draws\n"
            "Rug marks show when each arm first exhausts original data and starts sampling with replacement"
        )
        ax.set_ylim(y_bottom, max_curve * 1.08)
        ax.set_xlim(-init_cost if init_cost > 0 else 0, max(1, horizon))
        ax.grid(True, alpha=0.3)
        handles, labels = ax.get_legend_handles_labels()
        handles.append(plt.Line2D([0], [0], color="black", marker="|",
                                  linestyle="None", markersize=12,
                                  label="First replacement draw per arm"))
        labels.append("First replacement draw per arm")
        ax.legend(handles, labels, loc="upper left")
        plt.tight_layout()
        plt.savefig(dataset_output_dir / "figure8.png", dpi=300, bbox_inches="tight")
        plt.close()


        # --- PLOT 2: PULL EVOLUTION ---
        import numpy as np
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 4, figsize=(24, 6))

        # Find the most-pulled arm indices at the end of the adaptive algorithm
        final_pulls = counts_adapt_mean[-1, :]
        # Sort indices so the largest are at the end, then take the last 5
        top_arms_idx = np.argsort(final_pulls)[-5:] 

        # Create a distinct color palette for the top arms
        colors = plt.cm.tab10.colors 

        pull_datasets = [
            ("Uniform: Number of pulls", counts_unif_mean),
            ("Uniform VAR: Number of pulls", counts_unif_v_mean),
            ("Adaptive: Number of pulls", counts_adapt_mean),
            ("Adaptive VAR: Number of pulls", counts_adapt_v_mean),
        ]

        for subplot_idx, (title, data_mean) in enumerate(pull_datasets):
            ax = axes[subplot_idx]
            color_counter = 0
            
            for arm_idx in range(n_arms):
                is_control = (arm_test[arm_idx] == 'control')
                is_top = (arm_idx in top_arms_idx)
                
                # Logique de mise en forme
                if is_top or is_control:
                    linestyle = '--' if is_control else '-'
                    linewidth = 2.5
                    color = 'black' if is_control else colors[color_counter % len(colors)]
                    alpha = 1.0
                    label = f"Arm {arm_idx} (mu={arm_test[arm_idx][0:4]}) {'[Ctrl]' if is_control else '[Top]'}"
                    if not is_control: color_counter += 1
                else:
                    linestyle = '-'
                    linewidth = 1.0
                    color = 'grey'
                    alpha = 0.2
                    label = "_nolegend_" # Ignore this arm in the legend
                    
                ax.plot(time_axis_for(data_mean), data_mean[:, arm_idx], label=label, linewidth=linewidth, 
                        linestyle=linestyle, color=color, alpha=alpha)
            
            ax.set_xlabel("Time (t)")
            ax.grid(True, alpha=0.3)
            ax.set_title(title)

        axes[0].set_ylabel("Number of pulls ($T_i(t)$)")

        # A small clean legend with only important arms
        handles, labels = axes[-1].get_legend_handles_labels()
        fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 0.0), ncol=6)

        plt.tight_layout()
        plt.savefig(dataset_output_dir / "figure2_clean.png", dpi=300, bbox_inches="tight")
        plt.close()

        # --- PLOT 3: PULL EVOLUTION (SPAGHETTI PLOT) ---
        plt.figure(3+num_graph*10, figsize=(14, 7))
        plt.title(f"Adaptive: Number of pulls per arm ({n_sims} simulations)", fontsize=14)

        # 1. Identify arms to highlight (e.g. the 5 most-pulled at the end)
        final_pulls = counts_adapt_mean[-1, :]
        top_arms_idx = np.argsort(final_pulls)[-5:] 
        colors = plt.cm.tab10.colors
        color_counter = 0

        for arm_idx in range(n_arms):
            is_control = (arm_test[arm_idx] == 'control')
            is_top = (arm_idx in top_arms_idx)
            
            # Set style according to arm importance
            if is_top or is_control:
                base_color = 'black' if is_control else colors[color_counter % len(colors)]
                linestyle = '--' if is_control else '-'
                mean_linewidth = 2.5
                sim_alpha = 0.15 # Individual simulations remain subtle
                label = f"Arm {arm_idx} (mu={arm_test[arm_idx][0:4]}) {'[Ctrl]' if is_control else '[Top]'}"
                if not is_control: color_counter += 1
            else:
                base_color = 'gray'
                linestyle = '-'
                mean_linewidth = 1.0
                sim_alpha = 0.02 # Nearly transparent for rejected arms
                label = "_nolegend_"

            # Tracer les simulations individuelles (spaghetti)
            for sim_counts in counts_adapt_list:
                plt.plot(time_axis_for(sim_counts), sim_counts[:, arm_idx], color=base_color, alpha=sim_alpha, 
                        linewidth=0.5, linestyle=linestyle)

            # Plot the mean on top
            plt.plot(time_axis_for(counts_adapt_mean), counts_adapt_mean[:, arm_idx], label=label, color=base_color, 
                    linewidth=mean_linewidth, linestyle=linestyle)

        plt.xlabel("Time (t)", fontsize=12)
        plt.ylabel("Number of pulls ($T_i(t)$)", fontsize=12)
        plt.grid(True, alpha=0.3)

        # Simplified legend
        plt.legend(loc='upper left', fontsize=10, framealpha=0.9)

        plt.tight_layout()
        plt.savefig(dataset_output_dir / "figure3.png", dpi=300, bbox_inches="tight")
        plt.close()

        # --- PLOT 3 UNIF VAR: PULL EVOLUTION (SPAGHETTI PLOT) ---
        plt.figure(7+num_graph*10, figsize=(14, 7))
        plt.title(f"Uniform VAR: Number of pulls per arm ({n_sims} simulations)", fontsize=14)

        final_pulls_unif_v = counts_unif_v_mean[-1, :]
        top_arms_idx_unif_v = np.argsort(final_pulls_unif_v)[-5:]
        colors = plt.cm.tab10.colors
        color_counter = 0

        for arm_idx in range(n_arms):
            is_control = (arm_test[arm_idx] == 'control')
            is_top = (arm_idx in top_arms_idx_unif_v)

            if is_top or is_control:
                base_color = 'black' if is_control else colors[color_counter % len(colors)]
                linestyle = '--' if is_control else '-'
                mean_linewidth = 2.5
                sim_alpha = 0.15
                label = f"Arm {arm_idx} (mu={arm_test[arm_idx][0:4]}) {'[Ctrl]' if is_control else '[Top]'}"
                if not is_control:
                    color_counter += 1
            else:
                base_color = 'gray'
                linestyle = '-'
                mean_linewidth = 1.0
                sim_alpha = 0.02
                label = "_nolegend_"

            for sim_counts in counts_unif_v_list:
                plt.plot(time_axis_for(sim_counts), sim_counts[:, arm_idx], color=base_color, alpha=sim_alpha,
                         linewidth=0.5, linestyle=linestyle)

            plt.plot(time_axis_for(counts_unif_v_mean), counts_unif_v_mean[:, arm_idx], label=label, color=base_color,
                     linewidth=mean_linewidth, linestyle=linestyle)

        plt.xlabel("Time (t)", fontsize=12)
        plt.ylabel("Number of pulls ($T_i(t)$)", fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.legend(loc='upper left', fontsize=10, framealpha=0.9)
        plt.tight_layout()
        plt.savefig(dataset_output_dir / "figure3unifvar.png", dpi=300, bbox_inches="tight")
        plt.close()

        # --- PLOT 3 VAR: PULL EVOLUTION (SPAGHETTI PLOT) ---
        plt.figure(6+num_graph*10, figsize=(14, 7))
        plt.title(f"Adaptive VAR: Number of pulls per arm ({n_sims} simulations)", fontsize=14)

        # 1. Identify arms to highlight for the VAR variant
        # Make sure counts_adapt_v_mean is used here
        final_pulls_v = counts_adapt_v_mean[-1, :]
        top_arms_idx_v = np.argsort(final_pulls_v)[-5:] 
        colors = plt.cm.tab10.colors
        color_counter = 0

        for arm_idx in range(n_arms):
            is_control = (arm_test[arm_idx] == 'control')
            is_top = (arm_idx in top_arms_idx_v)
            
            # Set style according to arm importance
            if is_top or is_control:
                base_color = 'black' if is_control else colors[color_counter % len(colors)]
                linestyle = '--' if is_control else '-'
                mean_linewidth = 2.5
                sim_alpha = 0.15 # Transparence pour les simulations individuelles
                label = f"Arm {arm_idx} (mu={arm_test[arm_idx][0:4]}) {'[Ctrl]' if is_control else '[Top]'}"
                if not is_control: color_counter += 1
            else:
                base_color = 'gray'
                linestyle = '-'
                mean_linewidth = 1.0
                sim_alpha = 0.02 # Nearly transparent to reduce visual noise
                label = "_nolegend_"

            # Tracer les simulations individuelles (spaghetti) depuis la liste VAR
            for sim_counts in counts_adapt_v_list:
                plt.plot(time_axis_for(sim_counts), sim_counts[:, arm_idx], color=base_color, alpha=sim_alpha, 
                        linewidth=0.5, linestyle=linestyle)

            # Plot the mean on top
            plt.plot(time_axis_for(counts_adapt_v_mean), counts_adapt_v_mean[:, arm_idx], label=label, color=base_color, 
                    linewidth=mean_linewidth, linestyle=linestyle)

        plt.xlabel("Time (t)", fontsize=12)
        plt.ylabel("Number of pulls ($T_i(t)$)", fontsize=12)
        plt.grid(True, alpha=0.3)

        # Simplified legend
        plt.legend(loc='upper left', fontsize=10, framealpha=0.9)

        print("Displaying Adaptive VAR plots...")
        plt.tight_layout()
        plt.savefig(dataset_output_dir / "figure3var.png", dpi=300, bbox_inches="tight")
        plt.close()

        # --- PLOT 4: P-VALUES ---
        fig, axes = plt.subplots(1, 4, figsize=(24, 6))
        fig.suptitle("Evolution of P-values by iteration and arm", fontsize=16)

        datasets = [
            ("Uniform", np_p_value_mean_unif),
            ("Uniform VAR", np_p_value_mean_unif_v),
            ("Adaptive", np_p_value_mean_adapt),
            ("Adaptive VAR", np_p_value_mean_adapt_v)
        ]

        # Define the confidence threshold (edit this variable if needed)
        delta_threshold = 0.05 

        # Reuse top_arms to keep color consistency with Plot 3
        final_pulls = counts_adapt_mean[-1, :]
        top_arms_idx = np.argsort(final_pulls)[-5:] 
        colors = plt.cm.tab10.colors

        for idx, (title, data) in enumerate(datasets):
            ax = axes[idx]
            ax.set_title(title)
            color_counter = 0
            
            for arm_idx in range(n_arms):
                is_control = (arm_test[arm_idx] == 'control')
                is_top = (arm_idx in top_arms_idx)
                
                if is_top or is_control:
                    color = 'black' if is_control else colors[color_counter % len(colors)]
                    linestyle = '--' if is_control else '-'
                    linewidth = 2.0
                    alpha = 1.0
                    label = f"Arm {arm_idx} (mu={arm_test[arm_idx][0:4]})"
                    if not is_control: color_counter += 1
                else:
                    color = 'gray'
                    linestyle = '-'
                    linewidth = 0.8
                    alpha = 0.3
                    label = "_nolegend_"
                    
                ax.plot(time_axis_for(data), data[:, arm_idx], label=label, color=color, linewidth=linewidth, 
                        linestyle=linestyle, alpha=alpha)
            
            # THE MOST IMPORTANT CHANGE: logarithmic scale
            ax.set_yscale('log')
            # Optional: invert the Y axis so the "discovery" (dropping p-value) moves upward
            # ax.invert_yaxis() 
            
            # Horizontal threshold line
            ax.axhline(y=delta_threshold, color='red', linestyle=':', linewidth=2, 
                    label=f'Threshold ($\\delta={delta_threshold}$)')
            
            ax.set_xlabel("Time (t)")
            ax.set_ylabel("P-value (Log Scale)")
            ax.grid(True, which="both", ls="-", alpha=0.2) # Grid adapted to log scale

        # Single legend at the bottom
        handles, labels = axes[2].get_legend_handles_labels()
        # Use a dict to remove potential duplicates (such as the threshold)
        by_label = dict(zip(labels, handles))
        fig.legend(by_label.values(), by_label.keys(), loc='lower center', 
                bbox_to_anchor=(0.5, -0.15), ncol=6, fontsize='small')

        plt.tight_layout()
        fig.subplots_adjust(bottom=0.25) # Space for the legend

        plt.savefig(dataset_output_dir / "figure4.png", dpi=300, bbox_inches="tight")
        plt.close()

    # --- PLOT 5: P-VALUES (1 Colonne, 3 Trajectoires par Graphe) ---

        # Explicit color definition for each algorithm
        color_unif = 'tab:blue'
        color_unif_v = 'tab:purple'
        color_adapt = 'tab:orange'
        color_adapt_v = 'tab:green'

        # Create a grid: n_arms (rows) x 1 (column)
        # Slightly reduce width (e.g. 10) since there is only one column
        fig, axes = plt.subplots(nrows=n_arms, ncols=1, 
                                figsize=(10, 2.5 * n_arms), 
                                sharex=True)

        # Safety in case there is only one arm (axes would not be a list)
        if n_arms == 1:
            axes = [axes]

        for arm_idx in range(n_arms):
            ax = axes[arm_idx]
            arm_name = arm_test[arm_idx]
            
            # Add a title to identify which arm this row refers to
            ax.set_title(f"P-values evolution for Arm {arm_name}")

            # Plot the 3 trajectories on the SAME chart
            ax.plot(time_axis_for(np_p_value_mean_unif), np_p_value_mean_unif[:, arm_idx], label="Uniform", linewidth=2, color=color_unif)
            ax.plot(time_axis_for(np_p_value_mean_unif_v), np_p_value_mean_unif_v[:, arm_idx], label="Uniform VAR", linewidth=2, color=color_unif_v)
            ax.plot(time_axis_for(np_p_value_mean_adapt), np_p_value_mean_adapt[:, arm_idx], label="Adaptative", linewidth=2, color=color_adapt)
            ax.plot(time_axis_for(np_p_value_mean_adapt_v), np_p_value_mean_adapt_v[:, arm_idx], label="Adaptative VAR", linewidth=2, color=color_adapt_v)
            
            ax.set_ylabel("P value")
            ax.legend(loc="upper right", fontsize="small")
            ax.grid(True, alpha=0.3)

        # Add the x-axis only on the bottom-most chart
        axes[-1].set_xlabel("Time (t)")

        plt.tight_layout()
        plt.savefig(dataset_output_dir / "figure5.png", dpi=300, bbox_inches="tight")
        # plt.show()
        num_graph+=1
        plt.close()

    generate_local_figure10_same_set_adapt_vs_uniform(
        output_root,
        DATASET_KEYS,
        payloads=local_algo_payloads,
    )

    if GENERATE_ALGO_COMPARISON:
        generate_algorithm_comparison_figures(
            git_root,
            list(ALGORITHM_CONFIGS.keys()),
            _comparison_dataset_scope(),
        )
