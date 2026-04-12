import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import brentq
import pandas as pd
import time
import importlib
import sys
import os
import pickle
from datetime import datetime


# On récupère le chemin absolu du dossier parent (bandit_approach_simulation)
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, '..'))

# On l'ajoute aux chemins que Python connaît
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

# Maintenant l'import direct de backend fonctionnera
from backend import usable_adaptative_algorithm
importlib.reload(usable_adaptative_algorithm)

# Configuration de la page
st.set_page_config(
    page_title="Simulateur de Bandit Algorithm",
    page_icon="🎰",
    layout="wide"
)
# Dossier de stockage
SIMS_DIR = "saved_simulations"
if not os.path.exists(SIMS_DIR):
    os.makedirs(SIMS_DIR)

CATALOG_FILE = os.path.join(SIMS_DIR, "sim_catalog.csv")

def save_simulation(name, metadata, payload):
    # Payload contient tous les gros objets (arrays, figures, etc.)
    timestamp = int(time.time())
    filename = f"sim_{timestamp}.pkl"
    filepath = os.path.join(SIMS_DIR, filename)
    
    with open(filepath, 'wb') as f:
        pickle.dump(payload, f)
    
    # Mise à jour du catalogue CSV
    metadata['filename'] = filename
    metadata['date'] = datetime.now().strftime("%Y-%m-%d %H:%M")
    df_meta = pd.DataFrame([metadata])
    
    if not os.path.isfile(CATALOG_FILE):
        df_meta.to_csv(CATALOG_FILE, index=False)
    else:
        df_meta.to_csv(CATALOG_FILE, mode='a', header=False, index=False)
    return filename

# -----------------------------------------------------------------------------
# PART 1: LES ALGORITHMES
# -----------------------------------------------------------------------------

# class JamiesonJainAlgo:
#     def __init__(self, n_arms, mu_0, delta):
#         self.n = n_arms
#         self.mu_0 = mu_0
#         self.delta = delta

#         self.counts = np.zeros(n_arms, dtype=int)
#         self.emp_means = np.zeros(n_arms, dtype=float)
#         self.time = 0
#         self.S_t = set()
#         self.counts_evolution = [np.zeros(n_arms, dtype=int)]

#     def phi(self, t, delta_val):
#         if t == 0:
#             return float('inf')
#         num = 2 * np.log(1/delta_val) + 6 * np.log(np.log(1/delta_val) + 1e-10) + \
#               3 * np.log(np.log(np.e * t / 2) + 1e-10)
#         num = max(0.0, num)
#         return np.sqrt(num / t)

#     def select_arm(self):
#         if self.time < self.n:
#             return self.time
#         candidates = [i for i in range(self.n) if i not in self.S_t]
#         if not candidates:
#             return "stop"
#         best_ucb = -float('inf')
#         selected = candidates[0]
#         for i in candidates:
#             ucb = self.emp_means[i] + self.phi(self.counts[i], self.delta)
#             if ucb > best_ucb:
#                 best_ucb = ucb
#                 selected = i
#         return selected

#     def get_anytime_pvalue(self, arm_idx):
#         t = self.counts[arm_idx]
#         mean = self.emp_means[arm_idx]
#         if t == 0 or mean <= self.mu_0:
#             return 1.0
#         diff = mean - self.mu_0

#         def objective(p):
#             if p <= 0: return float('inf')
#             if p >= 1: return -float('inf')
#             return self.phi(t, p) - diff

#         try:
#             p_value = brentq(objective, 1e-12, 0.9999)
#             return p_value
#         except ValueError:
#             return 1.0

#     def bh_update_optimized(self, arm_idx, observation):
#         n_pulls = self.counts[arm_idx]
#         self.emp_means[arm_idx] = (self.emp_means[arm_idx] * n_pulls + observation) / (n_pulls + 1)
#         self.counts[arm_idx] += 1
#         self.time += 1
#         self.counts_evolution.append(self.counts.copy())

#         p_values_with_idx = [(self.get_anytime_pvalue(i), i) for i in range(self.n)]
#         p_values_with_idx.sort(key=lambda x: x[0])

#         current_St = set()
#         for k in range(self.n, 0, -1):
#             p_val_k = p_values_with_idx[k - 1][0]
#             effective_delta = self.delta * k / self.n
#             if p_val_k <= effective_delta:
#                 for rank in range(k):
#                     current_St.add(p_values_with_idx[rank][1])
#                 break
#         self.S_t.update(current_St)

#         # Retourner les p-values dans l'ordre des bras (0, 1, 2, ...)
#         p_values_ordered = [pv for pv, _ in sorted(p_values_with_idx, key=lambda x: x[1])]
#         return p_values_ordered


# class UniformAlgo:
#     def __init__(self, n_arms, mu_0, delta):
#         self.n = n_arms
#         self.mu_0 = mu_0
#         self.delta = delta

#         self.counts = np.zeros(n_arms, dtype=int)
#         self.emp_means = np.zeros(n_arms, dtype=float)
#         self.time = 0
#         self.S_t = set()
#         self.counts_evolution = [np.zeros(n_arms, dtype=int)]

#     def phi(self, t, delta_val):
#         if t == 0:
#             return float('inf')
#         num = 2 * np.log(1/delta_val) + 6 * np.log(np.log(1/delta_val) + 1e-10) + \
#               3 * np.log(np.log(np.e * t / 2) + 1e-10)
#         num = max(0.0, num)
#         return np.sqrt(num / t)

#     def select_arm(self):
#         return np.random.randint(self.n)

#     def get_anytime_pvalue(self, arm_idx):
#         t = self.counts[arm_idx]
#         mean = self.emp_means[arm_idx]
#         if t == 0 or mean <= self.mu_0:
#             return 1.0
#         diff = mean - self.mu_0

#         def objective(p):
#             if p <= 0: return float('inf')
#             if p >= 1: return -float('inf')
#             return self.phi(t, p) - diff

#         try:
#             p_value = brentq(objective, 1e-12, 0.9999)
#             return p_value
#         except ValueError:
#             return 1.0

#     def bh_update_optimized(self, arm_idx, observation):
#         n_pulls = self.counts[arm_idx]
#         self.emp_means[arm_idx] = (self.emp_means[arm_idx] * n_pulls + observation) / (n_pulls + 1)
#         self.counts[arm_idx] += 1
#         self.time += 1
#         self.counts_evolution.append(self.counts.copy())

#         p_values_with_idx = [(self.get_anytime_pvalue(i), i) for i in range(self.n)]
#         p_values_with_idx.sort(key=lambda x: x[0])

#         current_St = set()
#         for k in range(self.n, 0, -1):
#             p_val_k = p_values_with_idx[k - 1][0]
#             effective_delta = self.delta * k / self.n
#             if p_val_k <= effective_delta:
#                 for rank in range(k):
#                     current_St.add(p_values_with_idx[rank][1])
#                 break
#         self.S_t.update(current_St)

#         # Retourner les p-values dans l'ordre des bras (0, 1, 2, ...)
#         p_values_ordered = [pv for pv, _ in sorted(p_values_with_idx, key=lambda x: x[1])]
#         return p_values_ordered


# -----------------------------------------------------------------------------
# PART 2: SIMULATION ENGINE
# -----------------------------------------------------------------------------

def prepare_experiment(true_means, horizon, n_sims, scale, dist_type):
    n_arms = len(true_means)
    all_arm_data_by_sim = []

    progress_bar = st.progress(0)
    status_text = st.empty()

    for sim in range(n_sims):
        all_arm_data = []
        for arm in range(n_arms):
            # --- CHOIX DE LA LOI ---
            if "Normale" in dist_type:
                # Loi Normale classique
                result_arm = np.random.normal(loc=true_means[arm], scale=scale, size=horizon).tolist()
            else:
                # Loi Binomiale (n=1, p=moyenne) -> Bernoulli
                # On s'assure que p est bien entre 0 et 1 pour éviter les erreurs
                p = np.clip(true_means[arm], 0, 1)
                result_arm = np.random.binomial(n=1, p=p, size=horizon).tolist()
            
            all_arm_data.append(result_arm)
        all_arm_data_by_sim.append(all_arm_data)
        progress_bar.progress((sim + 1) / n_sims)
        status_text.text(f"Préparation ({dist_type}): {sim + 1}/{n_sims} simulations")

    status_text.empty()
    progress_bar.empty()
    return all_arm_data_by_sim


# def run_experiment(true_means, horizon, mode, all_arm_data, n_simulations=20, mu_0=0.0):
#     """
#     Runs the bandit experiment using pre-generated data for consistency.

#     Parameters
#     ----------
#     true_means : array-like
#     horizon : int
#     mode : str — 'adaptive' or 'uniform'
#     all_arm_data : list[sim][arm][pull]
#     n_simulations : int
#     mu_0 : float

#     Returns
#     -------
#     tpr_history_mean, tpr_list, counts_history_mean, counts_list,
#     np_p_values_list_by_sim, np_p_values_mean
#     """
#     n_arms = len(true_means)
#     true_positives = [i for i, m in enumerate(true_means) if m > mu_0]

#     tpr_history_sum = np.zeros(horizon)
#     tpr_list = []
#     counts_evolution_sum = np.zeros((horizon + 1, n_arms))
#     counts_list = []
#     p_values_list_by_sim = []

#     progress_bar = st.progress(0)
#     status_text = st.empty()

#     for no_sim in range(n_simulations):
#         p_values_list = []
#         all_arm_counts = [0 for _ in range(n_arms)]

#         if mode == 'adaptive':
#             algo = JamiesonJainAlgo(n_arms, mu_0, delta)
#         elif mode == 'uniform':
#             algo = UniformAlgo(n_arms, mu_0, delta)
#         else:
#             raise ValueError("Algorithm name not detected, choose between uniform and adaptive")

#         run_tpr = []

#         for t in range(horizon):
#             arm = algo.select_arm()

#             if arm == "stop":
#                 last_tpr = run_tpr[-1] if run_tpr else 1.0
#                 last_pvals = p_values_list[-1] if p_values_list else [1.0] * n_arms
#                 remaining_steps = horizon - len(run_tpr)
#                 run_tpr.extend([last_tpr] * remaining_steps)
#                 p_values_list.extend([last_pvals] * remaining_steps)
#                 last_counts = algo.counts_evolution[-1]
#                 for _ in range(remaining_steps):
#                     algo.counts_evolution.append(last_counts.copy())
#                 break

#             else:
#                 observation = all_arm_data[no_sim][arm][all_arm_counts[arm]]
#                 all_arm_counts[arm] += 1

#                 p_values_t = algo.bh_update_optimized(arm, observation)
#                 p_values_list.append(p_values_t)

#                 nb_found = len(algo.S_t.intersection(true_positives))
#                 current_tpr = nb_found / len(true_positives) if true_positives else 1.0
#                 run_tpr.append(current_tpr)

#         tpr_i = np.array(run_tpr)
#         tpr_list.append(tpr_i)
#         tpr_history_sum += tpr_i

#         counts_arr = np.array(algo.counts_evolution)[:horizon+1]
#         counts_list.append(counts_arr)
#         counts_evolution_sum += counts_arr
#         p_values_list_by_sim.append(p_values_list)

#         progress_bar.progress((no_sim + 1) / n_simulations)
#         status_text.text(f"Simulations {mode}: {no_sim + 1}/{n_simulations}")

#     tpr_history_mean = tpr_history_sum / n_simulations
#     counts_history_mean = counts_evolution_sum / n_simulations

#     np_p_values_list_by_sim = np.array(p_values_list_by_sim)    # shape: (n_sims, horizon, n_arms)
#     np_p_values_mean = np.mean(np_p_values_list_by_sim, axis=0) # shape: (horizon, n_arms)

#     status_text.text(f"Simulations {mode} terminées!")
#     time.sleep(0.5)
#     status_text.empty()
#     progress_bar.empty()

#     return tpr_history_mean, tpr_list, counts_history_mean, counts_list, np_p_values_list_by_sim, np_p_values_mean


# -----------------------------------------------------------------------------
# PART 3: FONCTIONS D'AFFICHAGE
# -----------------------------------------------------------------------------

def plot_tpr_comparison(tpr_adapt, tpr_unif, delta, sigma, n_sims):
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(tpr_adapt, label='Adaptive', color='#ff7f0e', linewidth=2)
    ax.plot(tpr_unif, label='Uniform', color='#1f77b4', linestyle='--', linewidth=2)
    ax.axhline(y=1.0, color='gray', linestyle=':', alpha=0.5)
    ax.set_xlabel("Temps (t)")
    ax.set_ylabel("Taux de Vrais Positifs (TPR)")
    ax.set_title(f"Comparaison de la vitesse de découverte\nδ={delta:.3f}, σ={sigma:.3f}, {n_sims} simulations")
    ax.legend()
    ax.grid(True, alpha=0.3)
    return fig


def plot_pulls_comparison(counts_unif_mean, counts_adapt_mean, true_means, mu_0, delta, sigma):
    n_arms = len(true_means)
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    ax1 = axes[0]
    for i in range(n_arms):
        ax1.plot(counts_unif_mean[:, i], label=f"Bras {i} (μ={true_means[i]:.2f})", linewidth=2)
    ax1.set_xlabel("Temps (t)")
    ax1.set_ylabel("Nombre de tirages")
    ax1.set_title("Uniform - Répartition des tirages")
    ax1.legend(fontsize='small')
    ax1.grid(True, alpha=0.3)

    ax2 = axes[1]
    for i in range(n_arms):
        linestyle = '-' if true_means[i] > mu_0 else '--'
        ax2.plot(counts_adapt_mean[:, i],
                label=f"Bras {i} (μ={true_means[i]:.2f})",
                linewidth=2,
                linestyle=linestyle)
    ax2.set_xlabel("Temps (t)")
    ax2.set_ylabel("Nombre de tirages")
    ax2.set_title(f"Adaptive - Répartition des tirages\nδ={delta:.3f}, σ={sigma:.3f}")
    ax2.legend(fontsize='small')
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


def plot_confidence_intervals(algo, true_means, mu_0, t):
    n_arms = len(true_means)
    arms = range(n_arms)
    means = algo.emp_means
    counts = algo.counts

    fig, ax = plt.subplots(figsize=(10, 6))

    for i in arms:
        if counts[i] > 0:
            ci_width = algo.phi(counts[i], algo.delta)
            color = 'green' if i in algo.S_t else 'blue'
            ax.errorbar(i, means[i], yerr=ci_width, fmt='o', color=color,
                       capsize=5, capthick=2, markersize=8)
        else:
            ax.plot(i, 0, 'o', color='gray', markersize=8)

    ax.axhline(y=mu_0, color='red', linestyle='--', label=f'Seuil μ₀={mu_0}')
    ax.set_xlabel("Bras")
    ax.set_ylabel("Moyenne empirique")
    ax.set_title(f"Intervalles de confiance à t={t}\n(Vert: découvert, Bleu: non découvert, Gris: non tiré)")
    ax.set_xticks(arms)
    ax.set_xticklabels([f"Bras {i}\n(μ={true_means[i]:.2f})" for i in arms])
    ax.legend()
    ax.grid(True, alpha=0.3)
    return fig


def plot_pvalues_combined(pvalues_unif_mean, pvalues_adapt_mean, true_means):
    """Plot 4 — P-values combinées : Uniform vs Adaptive côte à côte"""
    n_arms = len(true_means)
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    for arm_idx in range(n_arms):
        label = f"Arm {arm_idx} (μ={true_means[arm_idx]:.2f})"
        axes[0].plot(pvalues_unif_mean[:, arm_idx], label=label, linewidth=2)
        axes[1].plot(pvalues_adapt_mean[:, arm_idx], label=label, linewidth=2)

    for ax, title in zip(axes, ["Uniform: P-values by iteration and arm",
                                  "Adaptive: P-values by iteration and arm"]):
        ax.set_title(title)
        ax.set_xlabel("Time (t)")
        ax.set_ylabel("P-value")
        ax.legend(fontsize='small')
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


def plot_pvalues_grid(pvalues_unif_mean, pvalues_adapt_mean, true_means):
    """Plot 5 — Grille séparée par bras avec couleurs cohérentes"""
    n_arms = len(true_means)
    cmap = plt.get_cmap('tab10')
    arm_colors = [cmap(i) for i in range(n_arms)]

    fig, axes = plt.subplots(nrows=n_arms, ncols=2,
                              figsize=(12, 2.5 * n_arms),
                              sharex=True, sharey=True)

    axes[0, 0].set_title("Uniform: P-values by iteration")
    axes[0, 1].set_title("Adaptive: P-values by iteration")

    for arm_idx in range(n_arms):
        color = arm_colors[arm_idx]
        label = fr"Arm {arm_idx} ($\mu$={true_means[arm_idx]:.2f})"

        axes[arm_idx, 0].plot(pvalues_unif_mean[:, arm_idx], label=label, linewidth=2, color=color)
        axes[arm_idx, 0].set_ylabel("P-value")
        axes[arm_idx, 0].legend(loc="upper right", fontsize="small")
        axes[arm_idx, 0].grid(True, alpha=0.3)

        axes[arm_idx, 1].plot(pvalues_adapt_mean[:, arm_idx], label=label, linewidth=2, color=color)
        axes[arm_idx, 1].legend(loc="upper right", fontsize="small")
        axes[arm_idx, 1].grid(True, alpha=0.3)

    axes[-1, 0].set_xlabel("Time (t)")
    axes[-1, 1].set_xlabel("Time (t)")
    plt.tight_layout()
    return fig


def display_metrics(tpr_adapt, tpr_unif, counts_adapt_mean, true_means, mu_0):
    col1, col2, col3, col4 = st.columns(4)

    tpr_90_adapt = np.argmax(tpr_adapt >= 0.9) if np.any(tpr_adapt >= 0.9) else len(tpr_adapt)
    tpr_90_unif = np.argmax(tpr_unif >= 0.9) if np.any(tpr_unif >= 0.9) else len(tpr_unif)

    with col1:
        st.metric("Tps TPR 90% (Adaptive)", f"{tpr_90_adapt}")
    with col2:
        st.metric("Tps TPR 90% (Uniform)", f"{tpr_90_unif}")

    good_arms = [i for i, m in enumerate(true_means) if m > mu_0]
    bad_arms = [i for i, m in enumerate(true_means) if m <= mu_0]

    if good_arms and bad_arms:
        final_counts = counts_adapt_mean[-1]
        good_mean = np.mean([final_counts[i] for i in good_arms])
        bad_mean = np.mean([final_counts[i] for i in bad_arms])
        ratio = good_mean / bad_mean if bad_mean > 0 else float('inf')
        with col3:
            st.metric("Ratio tirages (bons/mauvais)", f"{ratio:.2f}")

    gain = ((tpr_90_unif - tpr_90_adapt) / tpr_90_unif * 100) if tpr_90_unif > 0 else 0
    with col4:
        st.metric("Gain d'efficacité", f"{gain:.1f}%")


# -----------------------------------------------------------------------------
# PART 4: APPLICATION STREAMLIT
# -----------------------------------------------------------------------------

st.title("🎰 Simulateur Interactif d'Algorithmes de Bandit")
st.markdown("---")

with st.sidebar:
    st.header("⚙️ Paramètres de simulation")

    st.markdown("---")
    st.header("Distribution des récompenses")
    dist_type = st.radio(
        "Type de loi",
        ["Normale", "Binomiale"],
        help="La loi normale est continue (bruit sigma), la binomiale est discrète (0 ou 1)."
    )

    if dist_type == "Binomiale":
        st.warning("⚠️ En mode Binomiale, assurez-vous que vos moyennes sont entre 0 et 1.")
    n_sims = st.slider("Nombre de simulations", 10, 500, 50, 10)
    horizon = st.slider("Horizon (T)", 100, 2000, 800, 50)
    if dist_type == "Normale":
        sigma = st.slider("Bruit (σ)", 0.1, 3.0, 1.0, 0.1)
    else:
        # On définit une valeur par défaut pour éviter que Python ne plante quand il cherchera la variable 'sigma' plus bas dans le code.
        sigma = 0.0 
    delta = st.slider("Paramètre δ (FDR)", 0.01, 0.5, 0.05, 0.01)
    mu_0 = st.number_input("Seuil μ₀", value=0.0, step=0.1)
    init_nb= st.slider("Nombre initial de tirage par bras", 0, 100, 10, 10)

    st.markdown("---")
    st.header("Configuration des bras")

    n_arms = st.number_input("Nombre de bras", min_value=2, max_value=10, value=3, step=1)

    st.subheader("Moyennes des bras")
    true_means = []
    cols = st.columns(2)
    for i in range(n_arms):
        with cols[i % 2]:
            mean = st.number_input(
                f"Bras {i}",
                value=0.5 if i < 1 else (0.35 if i < 2 else 0.0),
                step=0.05, key=f"mean_{i}", format="%.2f"
            )
            true_means.append(mean)

    true_means = np.array(true_means)

    st.markdown("---")
    run_button = st.button("🚀 Lancer la simulation", type="primary", use_container_width=True)
    st.markdown("---")
    st.caption("Basé sur l'algorithme de Jamieson & Jain (2018)")


if run_button:
    with st.spinner("Préparation des données..."):
        all_arm_data = prepare_experiment(true_means, horizon + init_nb, n_sims, sigma, dist_type)
        safe_horizon = horizon + (init_nb * n_arms) + 100 
        
        all_arm_data = prepare_experiment(true_means, safe_horizon, n_sims, sigma, dist_type)
        
    col1, col2 = st.columns(2)
    control_arm=0 # indice du bras control , donc rajouter dans le streamlit le choix du bras control
    init_choice=True
    with col1:
        st.info("Simulation Uniforme en cours...")
        # tpr_unif, tpr_list_unif, counts_unif_mean, counts_list_unif, _, pvalues_unif_mean = run_experiment(
        #     true_means, horizon, 'uniform', all_arm_data, n_sims, mu_0)
        (tpr_unif, tpr_list_unif, counts_unif_mean, counts_unif_list,  np_p_value_list_unif, pvalues_unif_mean, l_pos_unif) = usable_adaptative_algorithm.run_experiment(
            true_means, 0, delta, horizon + init_nb, 'uniform', all_arm_data, n_sims, control_arm, 0, False, False, True)
        

    with col2:
        st.info("Simulation Adaptive en cours...")
        # tpr_adapt, tpr_list_adapt, counts_adapt_mean, counts_list_adapt, _, pvalues_adapt_mean = run_experiment(
        #     true_means, horizon, 'adaptive', all_arm_data, n_sims, mu_0)
        (tpr_adapt, tpr_list_adapt, counts_adapt_mean, counts_list_adapt, np_p_value_list_adapt, pvalues_adapt_mean, l_pos_adapt) = usable_adaptative_algorithm.run_experiment(
            true_means, 0, delta, horizon, 'adaptive', all_arm_data, n_sims, control_arm, init_nb, init_choice, False, True)


    st.success("✅ Simulations terminées avec succès!")
    with st.expander("💾 Enregistrer ces résultats en local"):
        sim_name = st.text_input("Nom de la simulation", value=f"Simu {n_arms} bras - {delta} delta")
        if st.button("Confirmer la sauvegarde"):
            # On regroupe TOUT ce qui est nécessaire pour reconstruire les graphes
            payload = {
                'tpr_adapt': tpr_adapt, 'tpr_unif': tpr_unif,
                'counts_adapt': counts_adapt_mean, 'counts_unif': counts_unif_mean,
                'counts_list_adapt': counts_list_adapt, # Pour le Spaghetti plot
                'p_adapt': pvalues_adapt_mean, 'p_unif': pvalues_unif_mean,
                'true_means': true_means, 'mu_0': mu_0, 'delta': delta, 'sigma': sigma,
                'horizon': horizon, 'n_arms': n_arms
            }
            metadata = {'name': sim_name, 'n_arms': n_arms, 'delta': delta, 'horizon': horizon}
            fname = save_simulation(sim_name, metadata, payload)
            st.toast(f"Sauvegardé sous {fname}")
            
    st.markdown("---")
    st.header("📊 Résultats")
    display_metrics(tpr_adapt, tpr_unif, counts_adapt_mean, true_means, mu_0)
    st.markdown("---")

    tab1, tab2, tab3, tab4, tab5, tab6, tab7 , tab8= st.tabs([
        "TPR - Vitesse de découverte",
        "Répartition des tirages",
        "Spaghetti Plot",
        "Intervalles de confiance",
        "Données brutes",
        "P-Values (combiné)",
        "P-Values (grille)",
        "Historique Local"
    ])

    with tab1:
        st.subheader("Comparaison de la vitesse de découverte (TPR)")
        fig = plot_tpr_comparison(tpr_adapt, tpr_unif, delta, sigma, n_sims)
        st.pyplot(fig)
        plt.close(fig)
        st.info("Le TPR (True Positive Rate) mesure la proportion de bons bras correctement identifiés au cours du temps.")

    with tab2:
        st.subheader("Répartition des tirages entre les bras")
        algo_choice = st.radio("Choisir l'algorithme", ["Comparaison", "Adaptive seulement", "Uniform seulement"], horizontal=True)

        if algo_choice == "Comparaison":
            fig = plot_pulls_comparison(counts_unif_mean, counts_adapt_mean, true_means, mu_0, delta, sigma)
        elif algo_choice == "Adaptive seulement":
            fig, ax = plt.subplots(figsize=(10, 6))
            for i in range(n_arms):
                linestyle = '-' if true_means[i] > mu_0 else '--'
                ax.plot(counts_adapt_mean[:, i], label=f"Bras {i} (μ={true_means[i]:.2f})", linewidth=2, linestyle=linestyle)
            ax.set_xlabel("Temps (t)")
            ax.set_ylabel("Nombre de tirages")
            ax.set_title("Adaptive - Répartition des tirages")
            ax.legend()
            ax.grid(True, alpha=0.3)
        else:
            fig, ax = plt.subplots(figsize=(10, 6))
            for i in range(n_arms):
                ax.plot(counts_unif_mean[:, i], label=f"Bras {i} (μ={true_means[i]:.2f})", linewidth=2)
            ax.set_xlabel("Temps (t)")
            ax.set_ylabel("Nombre de tirages")
            ax.set_title("Uniform - Répartition des tirages")
            ax.legend()
            ax.grid(True, alpha=0.3)

        st.pyplot(fig)
        plt.close(fig)

    with tab3:
        st.subheader("Visualisation des simulations individuelles (Spaghetti Plot)")
        alpha = st.slider("Transparence des lignes individuelles", 0.05, 0.5, 0.1, 0.05)

        fig, ax = plt.subplots(figsize=(12, 7))
        for arm_idx in range(n_arms):
            color = f'C{arm_idx}'
            linestyle = '-' if true_means[arm_idx] > mu_0 else '--'
            for sim_counts in counts_list_adapt:
                ax.plot(sim_counts[:, arm_idx], color=color, alpha=alpha, linewidth=0.5, linestyle=linestyle)
            ax.plot(counts_adapt_mean[:, arm_idx],
                   label=f"Bras {arm_idx} (μ={true_means[arm_idx]:.2f})",
                   color=color, linewidth=2.5, linestyle=linestyle)
        ax.set_xlabel("Temps (t)")
        ax.set_ylabel("Nombre cumulé de tirages")
        ax.set_title(f"Adaptive - {n_sims} simulations individuelles")
        ax.legend(loc='upper left', bbox_to_anchor=(1, 1))
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

    with tab4:
        st.subheader("Analyse des intervalles de confiance")
        demo_algo = usable_adaptative_algorithm.UniformAlgo(n_arms, mu_0, delta)
        demo_algo.counts = counts_adapt_mean[-1].astype(int)
        demo_algo.emp_means = np.array([counts_adapt_mean[-1][i] / horizon if counts_adapt_mean[-1][i] > 0 else 0
                                        for i in range(n_arms)])
        demo_algo.S_t = set([i for i in range(n_arms) if true_means[i] > mu_0])
        fig = plot_confidence_intervals(demo_algo, true_means, mu_0, horizon)
        st.pyplot(fig)
        plt.close(fig)
        st.info("Les intervalles de confiance sont basés sur la loi du logarithme itéré (LIL).")

    with tab5:
        st.subheader("Données brutes des simulations")
        final_data = {
            'Bras': [f"Bras {i}" for i in range(n_arms)],
            'Moyenne réelle': true_means,
            '> μ₀': ['Oui' if m > mu_0 else 'Non' for m in true_means],
            'Tirages (Adaptive)': counts_adapt_mean[-1].astype(int),
            'Tirages (Uniform)': counts_unif_mean[-1].astype(int),
            'TPR final (Adaptive)': [f"{tpr_adapt[-1]:.3f}"] * n_arms,
            'TPR final (Uniform)': [f"{tpr_unif[-1]:.3f}"] * n_arms
        }
        df = pd.DataFrame(final_data)
        st.dataframe(df, use_container_width=True)

        csv = df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Télécharger les données (CSV)",
            data=csv,
            file_name=f"bandit_results_d{delta}_s{sigma}.csv",
            mime="text/csv",
        )

    with tab6:
        st.subheader("P-Values par bras et par algorithme (vue combinée)")
        fig = plot_pvalues_combined(pvalues_unif_mean, pvalues_adapt_mean, true_means)
        st.pyplot(fig)
        plt.close(fig)
        st.info("Moyenne des p-values anytime sur toutes les simulations, pour chaque bras.")

    with tab7:
        st.subheader("P-Values par bras — Grille séparée avec couleurs cohérentes")
        fig = plot_pvalues_grid(pvalues_unif_mean, pvalues_adapt_mean, true_means)
        st.pyplot(fig)
        plt.close(fig)
        st.info("Chaque ligne correspond à un bras. Les couleurs sont identiques entre Uniform et Adaptive.")

    with tab8:
        st.header("Historique des simulations")
        if os.path.exists(CATALOG_FILE):
            catalog = pd.read_csv(CATALOG_FILE)
            
            # Sélection de la simu
            selected_indices = st.selectbox(
                "Sélectionnez une simulation à recharger :", 
                options=catalog.index, 
                format_func=lambda x: f"{catalog.iloc[x]['date']} - {catalog.iloc[x]['name']}"
            )
            
            if st.button("Charger et afficher les graphiques"):
                target_file = catalog.iloc[selected_indices]['filename']
                with open(os.path.join(SIMS_DIR, target_file), 'rb') as f:
                    old = pickle.load(f)
                
                st.divider()
                st.subheader(f"Restauration de : {catalog.iloc[selected_indices]['name']}")
                
                # RE-AFFICHAGE DES GRAPHES CLÉS
                c1, c2 = st.columns(2)
                with c1:
                    st.pyplot(plot_tpr_comparison(old['tpr_adapt'], old['tpr_unif'], old['delta'], old['sigma'], 50))
                with c2:
                    st.pyplot(plot_pulls_comparison(old['counts_unif'], old['counts_adapt'], old['true_means'], old['mu_0'], old['delta'], old['sigma']))
                
                st.pyplot(plot_pvalues_grid(old['p_unif'], old['p_adapt'], old['true_means']))
        else:
            st.info("Aucun historique trouvé.")
else:
    st.markdown("""
    ## Bienvenue sur le simulateur d'algorithmes de bandit!

    Cette application vous permet d'explorer et de comparer deux algorithmes:

    * **Adaptive** (Jamieson & Jain 2018): Utilise une stratégie UCB pour allouer plus de tirages aux bras prometteurs
    * **Uniform**: Échantillonne tous les bras de manière uniforme

    ### Comment utiliser:
    1. Ajustez les paramètres dans la barre latérale
    2. Configurez les moyennes des différents bras
    3. Cliquez sur "Lancer la simulation"
    4. Explorez les résultats dans les différents onglets

    ### Les graphiques disponibles:
    * **TPR**: Vitesse à laquelle l'algorithme identifie les bons bras
    * **Répartition**: Comment les tirages sont alloués entre les bras
    * **Spaghetti**: Visualisation de la variabilité entre simulations
    * **Intervalles de confiance**: État des estimations à la fin
    * **Données brutes**: Export des résultats
    * **P-Values (combiné)**: Évolution moyenne des p-values, Uniform vs Adaptive
    * **P-Values (grille)**: Une ligne par bras, couleurs cohérentes entre les deux algos

    ---
    **Commencez par configurer vos paramètres dans le panneau de gauche**
    """)

    col1, col2, col3 = st.columns(3)
    with col1:
        st.info("**Adaptive**\nAlloue plus de ressources aux bras prometteurs")
    with col2:
        st.info("**Uniform**\nDistribution égale des tirages")
    with col3:
        st.info("**Objectif**\nIdentifier les bras avec μ > μ₀")