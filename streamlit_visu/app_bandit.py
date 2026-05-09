"""
Simulateur Interactif d'Algorithmes de Bandit
Version corrigée — tous les problèmes identifiés ont été fixés :
 1. save_test_results / RESULTS_DIR définis et unifiés avec save_simulation
 2. Boutons "Ajouter" / "Lancer tous" / "Sauvegarder" sortis du if run_button
 3. Utilisation systématique de st.session_state (plus de 'in dir()')
 4. Affichage des résultats bien scopé par test (via selectbox)
 5. Un seul système de sauvegarde (pickle + catalogue CSV)
 6. Sauvegarde persiste entre les re-runs
 7. Intervalles de confiance : calcul honnête basé sur true_means + counts
 8. control_arm exposé dans la sidebar
 9. Labels cohérents
10. init_choice exposé dans la sidebar
11. n_sims inclus dans le payload sauvegardé
12. mu_0 par défaut adapté à la loi (0.0 Normale, 0.5 Binomiale)
13. importlib.reload conditionnel (dev only)
14. safe_horizon cohérent partout (+ marge)
15. test_results réinitialisé quand besoin
"""

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import time
import importlib
import sys
import os
import pickle
from datetime import datetime

# --- Import du backend ---------------------------------------------------------
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, '..'))
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

from backend import adaptative_algorithm  # module de base (toujours chargé)

# Recharge uniquement en dev (mettre DEV_MODE=1 dans l'env pour activer)
if os.environ.get("DEV_MODE") == "1":
    importlib.reload(adaptative_algorithm)

# --- Mapping des algorithmes disponibles --------------------------------------
ALGO_OPTIONS = {
    "Simple (original)": "adaptative_algorithm",
    "Fusion V2 - NM/NM_M2": "adaptative_algorithm_v2",
    "Fusion V3 - Continue betting": "adaptative_algorithm_continuous_v3",
    "Fusion V3 - Binaire betting": "adaptative_algorithm_binary_v3",
}

LEGACY_ALGO_NAMES = {
    "Simple (original usable)": "Simple (original)",
    "Standard (original)": "Simple (original)",
    "usable_adaptative_algorithm": "Simple (original)",
    "Fusion V2 — NM/NM_M2": "Fusion V2 - NM/NM_M2",
    "usable_adaptative_algorithm_fusion_v2": "Fusion V2 - NM/NM_M2",
    "Fusion V3 — Continue betting": "Fusion V3 - Continue betting",
    "usable_adaptative_algorithm_fusion_continuous_v3": "Fusion V3 - Continue betting",
    "Fusion V3 — Binaire betting": "Fusion V3 - Binaire betting",
    "usable_adaptative_algorithm_fusion_binary_v3": "Fusion V3 - Binaire betting",
}


def load_algo_module(algo_name: str):
    """Importe dynamiquement le module backend choisi."""
    normalized_name = LEGACY_ALGO_NAMES.get(algo_name, algo_name)
    if normalized_name not in ALGO_OPTIONS:
        return adaptative_algorithm, f"Algorithme inconnu: {algo_name}"
    module_name = ALGO_OPTIONS[normalized_name]
    full_name = f"backend.{module_name}"
    try:
        mod = importlib.import_module(full_name)
        if os.environ.get("DEV_MODE") == "1":
            importlib.reload(mod)
        return mod, None
    except ImportError as e:
        return adaptative_algorithm, str(e)  # fallback sur le standard


def validate_algo_distribution(algo_name: str, dist_type: str):
    if "V3" in algo_name and "Continue" in algo_name and dist_type != "Normale":
        return "V3 Continue betting est prévu pour les récompenses normales/continues."
    if "V3" in algo_name and "Binaire" in algo_name and dist_type != "Binomiale":
        return "V3 Binaire betting est prévu pour les récompenses Bernoulli/binaires."
    return None

# --- Configuration de la page --------------------------------------------------
st.set_page_config(
    page_title="Simulateur de Bandit Algorithm",
    page_icon="🎰",
    layout="wide",
)

# --- Constantes / dossiers -----------------------------------------------------
SIMS_DIR = "saved_simulations"
RESULTS_DIR = "batch_results"
os.makedirs(SIMS_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

CATALOG_FILE = os.path.join(SIMS_DIR, "sim_catalog.csv")

SAFE_HORIZON_MARGIN = 100  # marge ajoutée partout pour éviter les IndexError

# =============================================================================
# PART 1: SAUVEGARDE / CHARGEMENT (système unifié)
# =============================================================================

def save_simulation(name, metadata, payload, base_dir=SIMS_DIR):
    """Sauvegarde une simulation en pickle + ajoute une entrée au catalogue CSV.

    Fonction unifiée utilisée à la fois pour la sauvegarde manuelle (bouton
    "Confirmer la sauvegarde") et pour la sauvegarde automatique d'un batch.
    """
    timestamp = int(time.time() * 1000)  # ms pour éviter les collisions batch
    filename = f"sim_{timestamp}.pkl"
    filepath = os.path.join(base_dir, filename)

    with open(filepath, 'wb') as f:
        pickle.dump(payload, f)

    metadata = dict(metadata)  # copie pour ne pas muter l'original
    metadata['filename'] = filename
    metadata['date'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    metadata['name'] = name

    df_meta = pd.DataFrame([metadata])
    if not os.path.isfile(CATALOG_FILE):
        df_meta.to_csv(CATALOG_FILE, index=False)
    else:
        df_meta.to_csv(CATALOG_FILE, mode='a', header=False, index=False)
    return filepath


def save_test_results(test_idx, cfg, result):
    """Wrapper pour sauvegarde automatique d'un test de la file."""
    name = f"Batch test #{test_idx} — δ={cfg['delta']}, T={cfg['horizon']}"
    metadata = {
        'n_arms': cfg['n_arms'],
        'delta': cfg['delta'],
        'horizon': cfg['horizon'],
        'n_sims': cfg['n_sims'],
        'sigma': cfg['sigma'],
        'mu_0': cfg['mu_0'],
        'algo_name': cfg.get('algo_name', 'Simple (original)'),
        'batch': True,
    }
    payload = build_payload(cfg, result)
    return save_simulation(name, metadata, payload, base_dir=RESULTS_DIR)


def build_payload(cfg, result):
    """Construit le dict sérialisable pour sauvegarde. Contient tout le
    nécessaire pour reconstruire les graphes sans relancer la simu."""
    return {
        'tpr_adapt': result['tpr_adapt'],
        'tpr_unif': result['tpr_unif'],
        'counts_adapt': result['counts_adapt_mean'],
        'counts_unif': result['counts_unif_mean'],
        'counts_list_adapt': result['counts_list_adapt'],
        'p_adapt': result['pvalues_adapt_mean'],
        'p_unif': result['pvalues_unif_mean'],
        'true_means': cfg['true_means'],
        'mu_0': cfg['mu_0'],
        'delta': cfg['delta'],
        'sigma': cfg['sigma'],
        'horizon': cfg['horizon'],
        'n_arms': cfg['n_arms'],
        'n_sims': cfg['n_sims'],
        'algo_name': cfg.get('algo_name', 'Simple (original)'),
    }

# =============================================================================
# PART 2: SIMULATION ENGINE
# =============================================================================

def prepare_experiment(true_means, horizon, n_sims, scale, dist_type):
    n_arms = len(true_means)
    all_arm_data_by_sim = []

    progress_bar = st.progress(0)
    status_text = st.empty()

    for sim in range(n_sims):
        all_arm_data = []
        for arm in range(n_arms):
            if "Normale" in dist_type:
                result_arm = np.random.normal(
                    loc=true_means[arm], scale=scale, size=horizon
                ).tolist()
            else:
                p = float(np.clip(true_means[arm], 0, 1))
                result_arm = np.random.binomial(n=1, p=p, size=horizon).tolist()
            all_arm_data.append(result_arm)
        all_arm_data_by_sim.append(all_arm_data)

        progress_bar.progress((sim + 1) / n_sims)
        status_text.text(f"Préparation ({dist_type}): {sim + 1}/{n_sims} simulations")

    status_text.empty()
    progress_bar.empty()
    return all_arm_data_by_sim


def run_single_experiment(cfg, dist_type, algo_module=None):
    """Exécute uniform + adaptive pour une config donnée. Retourne un dict
    avec tout ce qu'il faut pour l'affichage et la sauvegarde."""
    if algo_module is None:
        algo_module = adaptative_algorithm
    true_means = cfg['true_means']
    n_arms = cfg['n_arms']
    horizon = cfg['horizon']
    n_sims = cfg['n_sims']
    sigma = cfg['sigma']
    delta = cfg['delta']
    init_nb = cfg['init_nb']
    control_arm = cfg['control_arm']
    init_choice = cfg['init_choice']

    safe_horizon = horizon + init_nb * n_arms + SAFE_HORIZON_MARGIN
    all_arm_data = prepare_experiment(true_means, safe_horizon, n_sims, sigma, dist_type)

    # Uniform
    (tpr_unif, tpr_list_unif, counts_unif_mean, counts_list_unif,
     np_p_value_list_unif, pvalues_unif_mean, l_pos_unif) = \
        algo_module.run_experiment(
            true_means, 0, delta, horizon + init_nb, 'uniform',
            all_arm_data, n_sims, control_arm, 0, False, False, True,
        )

    # Adaptive
    (tpr_adapt, tpr_list_adapt, counts_adapt_mean, counts_list_adapt,
     np_p_value_list_adapt, pvalues_adapt_mean, l_pos_adapt) = \
        algo_module.run_experiment(
            true_means, 0, delta, horizon, 'adaptive',
            all_arm_data, n_sims, control_arm, init_nb, init_choice, False, True,
        )

    return {
        'tpr_adapt': tpr_adapt,
        'tpr_unif': tpr_unif,
        'tpr_list_adapt': tpr_list_adapt,
        'tpr_list_unif': tpr_list_unif,
        'counts_adapt_mean': counts_adapt_mean,
        'counts_unif_mean': counts_unif_mean,
        'counts_list_adapt': counts_list_adapt,
        'counts_list_unif': counts_list_unif,
        'pvalues_adapt_mean': pvalues_adapt_mean,
        'pvalues_unif_mean': pvalues_unif_mean,
        'l_pos_adapt': l_pos_adapt,
        'l_pos_unif':  l_pos_unif,
    }

# =============================================================================
# PART 3: FONCTIONS D'AFFICHAGE
# =============================================================================

def plot_tpr_comparison(tpr_adapt, tpr_unif, delta, sigma, n_sims, fdr_adapt=None, fdr_unif=None):
    fig, ax = plt.subplots(figsize=(10, 6))
    
    label_adapt = f"Adaptive (FDR={fdr_adapt:.3f})" if fdr_adapt is not None else "Adaptive"
    label_unif  = f"Uniform  (FDR={fdr_unif:.3f})"  if fdr_unif  is not None else "Uniform"
    
    ax.plot(tpr_adapt, label=label_adapt, color='#ff7f0e', linewidth=2)
    ax.plot(tpr_unif,  label=label_unif,  color='#1f77b4', linestyle='--', linewidth=2)
    ax.axhline(y=1.0, color='gray', linestyle=':', alpha=0.5)
    ax.set_xlabel("Temps (t)")
    ax.set_ylabel("Taux de Vrais Positifs (TPR)")
    ax.set_title(
        f"Comparaison de la vitesse de découverte\n"
        f"δ={delta:.3f}, σ={sigma:.3f}, {n_sims} simulations"
    )
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
        ax2.plot(
            counts_adapt_mean[:, i],
            label=f"Bras {i} (μ={true_means[i]:.2f})",
            linewidth=2, linestyle=linestyle,
        )
    ax2.set_xlabel("Temps (t)")
    ax2.set_ylabel("Nombre de tirages")
    ax2.set_title(f"Adaptive - Répartition des tirages\nδ={delta:.3f}, σ={sigma:.3f}")
    ax2.legend(fontsize='small')
    ax2.grid(True, alpha=0.3)
    plt.tight_layout()
    return fig


def plot_confidence_intervals_schematic(true_means, final_counts, mu_0, delta, horizon):
    """Affiche des CI *schématiques* centrés sur true_means avec largeur phi(counts, delta).

    Note: c'est une représentation idéalisée (centrée sur la vérité terrain)
    utilisée à titre pédagogique — les vraies moyennes empiriques ne sont pas
    exposées par le backend actuellement.
    """
    n_arms = len(true_means)
    # UniformAlgo peut ne pas exister dans tous les modules — fallback sur le standard
    _base = adaptative_algorithm
    tmp = _base.UniformAlgo(n_arms, mu_0, delta)

    fig, ax = plt.subplots(figsize=(10, 6))
    for i in range(n_arms):
        c = int(final_counts[i])
        if c > 0:
            ci_width = tmp.phi(c, delta)
            color = 'green' if true_means[i] > mu_0 else 'blue'
            ax.errorbar(i, true_means[i], yerr=ci_width, fmt='o', color=color,
                        capsize=5, capthick=2, markersize=8)
        else:
            ax.plot(i, 0, 'o', color='gray', markersize=8)

    ax.axhline(y=mu_0, color='red', linestyle='--', label=f'Seuil μ₀={mu_0}')
    ax.set_xlabel("Bras")
    ax.set_ylabel("Moyenne (idéalisée à la vérité terrain)")
    ax.set_title(
        f"Intervalles de confiance schématiques à t={horizon}\n"
        "(Vert: μ > μ₀, Bleu: μ ≤ μ₀, Gris: non tiré)"
    )
    ax.set_xticks(range(n_arms))
    ax.set_xticklabels([f"Bras {i}\n(μ={true_means[i]:.2f}, n={int(final_counts[i])})"
                        for i in range(n_arms)])
    ax.legend()
    ax.grid(True, alpha=0.3)
    return fig


def plot_pvalues_combined(pvalues_unif_mean, pvalues_adapt_mean, true_means):
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
    n_arms = len(true_means)
    cmap = plt.get_cmap('tab10')
    arm_colors = [cmap(i) for i in range(n_arms)]

    fig, axes = plt.subplots(nrows=n_arms, ncols=2,
                             figsize=(12, 2.5 * n_arms),
                             sharex=True, sharey=True,
                             squeeze=False)

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


def plot_spaghetti(counts_list_adapt, counts_adapt_mean, true_means, mu_0, n_sims, alpha):
    n_arms = len(true_means)
    fig, ax = plt.subplots(figsize=(12, 7))
    for arm_idx in range(n_arms):
        color = f'C{arm_idx}'
        linestyle = '-' if true_means[arm_idx] > mu_0 else '--'
        for sim_counts in counts_list_adapt:
            ax.plot(sim_counts[:, arm_idx], color=color, alpha=alpha,
                    linewidth=0.5, linestyle=linestyle)
        ax.plot(counts_adapt_mean[:, arm_idx],
                label=f"Bras {arm_idx} (μ={true_means[arm_idx]:.2f})",
                color=color, linewidth=2.5, linestyle=linestyle)
    ax.set_xlabel("Temps (t)")
    ax.set_ylabel("Nombre cumulé de tirages")
    ax.set_title(f"Adaptive - {n_sims} simulations individuelles")
    ax.legend(loc='upper left', bbox_to_anchor=(1, 1))
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    return fig


def display_metrics(tpr_adapt, tpr_unif, counts_adapt_mean, true_means, mu_0):
    col1, col2, col3, col4 = st.columns(4)

    tpr_90_adapt = int(np.argmax(tpr_adapt >= 0.9)) if np.any(tpr_adapt >= 0.9) else len(tpr_adapt)
    tpr_90_unif = int(np.argmax(tpr_unif >= 0.9)) if np.any(tpr_unif >= 0.9) else len(tpr_unif)

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

def compute_fdr_metrics(list_positive, true_means, mu_0):
    """FDR/TPR réels — uniquement en simulation car H1 connu."""
    H1 = {int(i) for i in np.where(np.array(true_means) > mu_0)[0]}
    H0 = set(range(len(true_means))) - H1

    fdrs, tprs = [], []
    for St in list_positive:
        tp = len(St & H1)
        fp = len(St & H0)
        fdrs.append(fp / max(len(St), 1))
        tprs.append(tp / max(len(H1), 1))

    return {
        "FDR_mean": np.mean(fdrs),
        "TPR_mean": np.mean(tprs),
        "H1": H1,
        "H0": H0,
        "last_St": list_positive[-1] if list_positive else set(),
    }

def render_result_tabs(result, cfg, tab_prefix=""):
    """Affiche tous les onglets de résultat pour un test donné.

    Le tab_prefix sert de clé unique pour les widgets quand plusieurs résultats
    coexistent (sinon Streamlit râle sur les duplicate keys).
    """
    true_means = cfg['true_means']
    n_arms = cfg['n_arms']
    mu_0 = cfg['mu_0']
    delta = cfg['delta']
    sigma = cfg['sigma']
    horizon = cfg['horizon']
    n_sims = cfg['n_sims']

    tpr_adapt = result['tpr_adapt']
    tpr_unif = result['tpr_unif']
    counts_adapt_mean = result['counts_adapt_mean']
    counts_unif_mean = result['counts_unif_mean']
    counts_list_adapt = result['counts_list_adapt']
    pvalues_adapt_mean = result['pvalues_adapt_mean']
    pvalues_unif_mean = result['pvalues_unif_mean']
    m_adapt = compute_fdr_metrics(result['l_pos_adapt'], true_means, mu_0)
    m_unif  = compute_fdr_metrics(result['l_pos_unif'],  true_means, mu_0)


    display_metrics(tpr_adapt, tpr_unif, counts_adapt_mean, true_means, mu_0)
    st.markdown("---")

    tabs = st.tabs([
        "TPR - Vitesse de découverte",
        "Répartition des tirages",
        "Spaghetti Plot",
        "Intervalles de confiance",
        "Données brutes",
        "P-Values (combiné)",
        "P-Values (grille)",
        "FDR — Découvertes",
    ])

    with tabs[0]:
        st.subheader("Comparaison de la vitesse de découverte (TPR)")
        fig = plot_tpr_comparison(
            tpr_adapt, tpr_unif, delta, sigma, n_sims,
            fdr_adapt=m_adapt['FDR_mean'],
            fdr_unif=m_unif['FDR_mean'],
        )
        st.pyplot(fig)
        plt.close(fig)
        st.info("Le TPR (True Positive Rate) mesure la proportion de bons bras correctement identifiés au cours du temps.")

    with tabs[1]:
        st.subheader("Répartition des tirages entre les bras")
        algo_choice = st.radio(
            "Choisir l'algorithme",
            ["Comparaison", "Adaptive seulement", "Uniform seulement"],
            horizontal=True,
            key=f"{tab_prefix}_algo_choice",
        )
        if algo_choice == "Comparaison":
            fig = plot_pulls_comparison(counts_unif_mean, counts_adapt_mean, true_means, mu_0, delta, sigma)
        elif algo_choice == "Adaptive seulement":
            fig, ax = plt.subplots(figsize=(10, 6))
            for i in range(n_arms):
                ls = '-' if true_means[i] > mu_0 else '--'
                ax.plot(counts_adapt_mean[:, i], label=f"Bras {i} (μ={true_means[i]:.2f})", linewidth=2, linestyle=ls)
            ax.set_xlabel("Temps (t)"); ax.set_ylabel("Nombre de tirages")
            ax.set_title("Adaptive - Répartition des tirages")
            ax.legend(); ax.grid(True, alpha=0.3)
        else:
            fig, ax = plt.subplots(figsize=(10, 6))
            for i in range(n_arms):
                ax.plot(counts_unif_mean[:, i], label=f"Bras {i} (μ={true_means[i]:.2f})", linewidth=2)
            ax.set_xlabel("Temps (t)"); ax.set_ylabel("Nombre de tirages")
            ax.set_title("Uniform - Répartition des tirages")
            ax.legend(); ax.grid(True, alpha=0.3)
        st.pyplot(fig)
        plt.close(fig)

    with tabs[2]:
        st.subheader("Visualisation des simulations individuelles (Spaghetti Plot)")
        alpha = st.slider("Transparence des lignes individuelles", 0.05, 0.5, 0.1, 0.05,
                          key=f"{tab_prefix}_alpha")
        fig = plot_spaghetti(counts_list_adapt, counts_adapt_mean, true_means, mu_0, n_sims, alpha)
        st.pyplot(fig)
        plt.close(fig)

    with tabs[3]:
        st.subheader("Analyse des intervalles de confiance (schématique)")
        fig = plot_confidence_intervals_schematic(
            true_means, counts_adapt_mean[-1], mu_0, delta, horizon
        )
        st.pyplot(fig)
        plt.close(fig)
        st.info(
            "Les largeurs des IC sont basées sur la loi du logarithme itéré (LIL), "
            "calculées à partir du nombre de tirages final. Les centres sont "
            "placés sur les vraies moyennes (idéalisation pédagogique)."
        )

    with tabs[4]:
        st.subheader("Données brutes des simulations")
        final_data = {
            'Bras': [f"Bras {i}" for i in range(n_arms)],
            'Moyenne réelle': true_means,
            '> μ₀': ['Oui' if m > mu_0 else 'Non' for m in true_means],
            'Tirages (Adaptive)': counts_adapt_mean[-1].astype(int),
            'Tirages (Uniform)': counts_unif_mean[-1].astype(int),
            'TPR final (Adaptive)': [f"{tpr_adapt[-1]:.3f}"] * n_arms,
            'TPR final (Uniform)': [f"{tpr_unif[-1]:.3f}"] * n_arms,
        }
        df = pd.DataFrame(final_data)
        st.dataframe(df, use_container_width=True)
        csv = df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Télécharger les données (CSV)",
            data=csv,
            file_name=f"bandit_results_d{delta}_s{sigma}.csv",
            mime="text/csv",
            key=f"{tab_prefix}_csv",
        )

    with tabs[5]:
        st.subheader("P-Values par bras et par algorithme (vue combinée)")
        fig = plot_pvalues_combined(pvalues_unif_mean, pvalues_adapt_mean, true_means)
        st.pyplot(fig)
        plt.close(fig)
        st.info("Moyenne des p-values anytime sur toutes les simulations, pour chaque bras.")

    with tabs[6]:
        st.subheader("P-Values par bras — Grille séparée avec couleurs cohérentes")
        fig = plot_pvalues_grid(pvalues_unif_mean, pvalues_adapt_mean, true_means)
        st.pyplot(fig)
        plt.close(fig)
        st.info("Chaque ligne correspond à un bras. Les couleurs sont identiques entre Uniform et Adaptive.")
    
    with tabs[7]:
        st.subheader("Analyse des découvertes (FDR contrôlé)")
        for label, l_pos in [("Adaptive", result['l_pos_adapt']),
                          ("Uniform",  result['l_pos_unif'])]:
            m = compute_fdr_metrics(l_pos, true_means, mu_0)
            st.markdown(f"**{label}**")
            c1, c2, c3 = st.columns(3)
            c1.metric("FDR moyen", f"{m['FDR_mean']:.3f}")
            c2.metric("TPR moyen", f"{m['TPR_mean']:.3f}")
            c3.metric("Vrais positifs H1", str(sorted(m['H1'])))
            st.write(f"Dernière simulation — S_t : `{m['last_St']}`")

            # --- Interprétation ---
            n_H1 = len(m['H1'])
            fdr_pct = m['FDR_mean'] * 100
            tpr_pct = m['TPR_mean'] * 100
            missed = n_H1 - round(m['TPR_mean'] * n_H1)

            fdr_ok = m['FDR_mean'] <= delta
            st.info(
                f"Sur {n_sims} simulations, l'algorithme **{label}** a identifié en moyenne "
                f"**{tpr_pct:.1f}%** des {n_H1} vrais bras positifs (H1 = {sorted(m['H1'])}), "
                f"soit environ **{missed} bras manqué(s)** en moyenne. "
                f"Le taux de fausses découvertes moyen est de **{fdr_pct:.1f}%**, "
                f"{'ce qui respecte' if fdr_ok else '⚠️ ce qui dépasse'} le seuil fixé δ = {delta:.2f}. "
                f"{' Le contrôle FDR est garanti.' if fdr_ok else '❌ Le contrôle FDR est violé.'}"
            )
            st.markdown("---")
# =============================================================================
# PART 4: INITIALISATION DU SESSION STATE
# =============================================================================

def init_state():
    defaults = {
        "test_configs": [],      # file d'attente (list[cfg])
        "test_results": [],      # résultats des tests batch (list[dict])
        "single_result": None,   # dernier résultat de simu simple (dict)
        "single_cfg": None,      # config du dernier résultat simple
        "run_batch_requested": False,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


init_state()

# =============================================================================
# PART 5: SIDEBAR — PARAMÈTRES
# =============================================================================

st.title("🎰 Simulateur Interactif d'Algorithmes de Bandit")
st.markdown("---")

with st.sidebar:
    st.header("⚙️ Paramètres de simulation")

    st.subheader("Algorithme backend")
    algo_choice = st.selectbox(
        "Version de l'algorithme",
        options=list(ALGO_OPTIONS.keys()),
        help=(
            "Simple : algorithme original.\n"
            "Fusion V2 : module unique NM/NM_M2, utilisable en Normale et Binomiale.\n"
            "Fusion V3 : betting, choisir Continue pour Normale et Binaire pour Bernoulli."
        ),
    )
    st.markdown("---")

    st.subheader("Distribution des récompenses")
    dist_type = st.radio(
        "Type de loi", ["Normale", "Binomiale"],
        help="La loi normale est continue (bruit σ), la binomiale est discrète (0 ou 1).",
    )
    if dist_type == "Binomiale":
        st.warning("⚠️ En mode Binomiale, les moyennes doivent être entre 0 et 1.")

    n_sims = st.slider("Nombre de simulations", 10, 500, 50, 10)
    horizon = st.slider("Horizon (T)", 100, 2000, 800, 50)

    if dist_type == "Normale":
        sigma = st.slider("Bruit (σ)", 0.1, 3.0, 0.5, 0.1)
    else:
        sigma = 0.0  # pas de bruit paramétrique en Bernoulli

    delta = st.slider("Paramètre δ (FDR)", 0.01, 0.5, 0.05, 0.01)

    default_mu_0 = 0.5 if dist_type == "Binomiale" else 0.0
    mu_0 = st.number_input("Seuil μ₀", value=default_mu_0, step=0.1)

    init_nb = st.slider("Nombre initial de tirages par bras", 1, 100, 1, 1)
    init_choice = st.checkbox(
        "Activer l'initialisation (init_choice)", value=True,
        help="Si coché, l'algorithme Adaptive commence par tirer chaque bras init_nb fois.",
    )

    st.markdown("---")
    st.subheader("Configuration des bras")
    n_arms = st.number_input("Nombre de bras", min_value=2, max_value=10, value=3, step=1)

    control_arm = st.number_input(
        "Indice du bras de contrôle", min_value=0, max_value=int(n_arms) - 1,
        value=0, step=1,
    )

    st.caption("Moyennes des bras")
    true_means = []
    cols = st.columns(2)
    for i in range(n_arms):
        with cols[i % 2]:
            default_val = 0.5 if i == 0 else (0.35 if i == 1 else 0.0)
            if dist_type == "Binomiale":
                default_val = max(0.0, min(1.0, default_val + 0.3 if default_val == 0.0 else default_val))
            mean = st.number_input(
                f"Bras {i}", value=default_val, step=0.05,
                key=f"mean_{i}", format="%.2f",
            )
            true_means.append(mean)
    true_means = np.array(true_means)

    st.markdown("---")
    run_button = st.button("🚀 Lancer la simulation", type="primary", use_container_width=True)
    add_to_queue_button = st.button("➕ Ajouter cette config à la file", use_container_width=True)
    st.markdown("---")
    st.caption("Basé sur l'algorithme de Jamieson & Jain (2018)")


def current_cfg():
    """Construit la config à partir des widgets de la sidebar."""
    return {
        "n_sims": n_sims,
        "horizon": horizon,
        "sigma": sigma,
        "delta": delta,
        "mu_0": mu_0,
        "init_nb": init_nb,
        "init_choice": init_choice,
        "control_arm": int(control_arm),
        "n_arms": int(n_arms),
        "true_means": np.array(true_means),
        "dist_type": dist_type,
        "algo_name": algo_choice,   # ← nom de l'algo sélectionné
    }

# =============================================================================
# PART 6: ACTIONS — simu simple / ajout à la file
# =============================================================================

if run_button:
    cfg = current_cfg()
    validation_msg = validate_algo_distribution(cfg["algo_name"], dist_type)
    if validation_msg:
        st.warning(f"⚠️ {validation_msg}")
    algo_module, import_err = load_algo_module(cfg["algo_name"])
    if import_err:
        st.warning(f"⚠️ Module '{cfg['algo_name']}' introuvable — fallback sur Standard.\n\n`{import_err}`")
    else:
        st.caption(f"Algorithme chargé : **{cfg['algo_name']}**")
    with st.spinner("Simulation en cours..."):
        result = run_single_experiment(cfg, dist_type, algo_module)
    st.session_state.single_result = result
    st.session_state.single_cfg = cfg
    st.success("✅ Simulation terminée avec succès !")

if add_to_queue_button:
    cfg = current_cfg()
    validation_msg = validate_algo_distribution(cfg["algo_name"], dist_type)
    if validation_msg:
        st.warning(f"⚠️ {validation_msg}")
    st.session_state.test_configs.append(cfg)
    st.success(f"Test #{len(st.session_state.test_configs)} ajouté à la file.")

# =============================================================================
# PART 7: BLOC — SAUVEGARDE DU DERNIER RÉSULTAT SIMPLE
# =============================================================================

if st.session_state.single_result is not None:
    st.markdown("---")
    with st.expander("💾 Enregistrer le dernier résultat en local"):
        cfg = st.session_state.single_cfg
        default_name = f"Simu {cfg['n_arms']} bras - δ={cfg['delta']}"
        sim_name = st.text_input("Nom de la simulation", value=default_name, key="save_name")
        if st.button("Confirmer la sauvegarde", key="save_confirm"):
            metadata = {
                'n_arms': cfg['n_arms'], 'delta': cfg['delta'],
                'horizon': cfg['horizon'], 'n_sims': cfg['n_sims'],
                'sigma': cfg['sigma'], 'mu_0': cfg['mu_0'],
                'algo_name': cfg.get('algo_name', 'Simple (original)'),
                'batch': False,
            }
            payload = build_payload(cfg, st.session_state.single_result)
            path = save_simulation(sim_name, metadata, payload)
            st.toast(f"Sauvegardé : {os.path.basename(path)}")

# =============================================================================
# PART 8: BLOC — FILE D'ATTENTE
# =============================================================================

st.markdown("---")
st.subheader(f"🗂️ File d'attente ({len(st.session_state.test_configs)} tests)")

if st.session_state.test_configs:
    for i, cfg in enumerate(st.session_state.test_configs):
        with st.expander(f"Test #{i+1} — δ={cfg['delta']}, T={cfg['horizon']}, σ={cfg['sigma']}"):
            st.write(f"**Algo:** {cfg.get('algo_name', 'Simple (original)')}")
            st.write(f"**Bras:** {cfg['n_arms']}, **Sims:** {cfg['n_sims']}, **Loi:** {cfg['dist_type']}")
            st.write(f"**Moyennes:** {cfg['true_means']}")
            st.write(f"**μ₀:** {cfg['mu_0']}, **Init pulls:** {cfg['init_nb']}, **Init choice:** {cfg['init_choice']}")
            st.write(f"**Bras contrôle:** {cfg['control_arm']}")

    col_a, col_b = st.columns(2)
    with col_a:
        run_all_btn = st.button("▶️ Lancer tous les tests de la file",
                                type="primary", use_container_width=True)
    with col_b:
        clear_queue_btn = st.button("🗑️ Vider la file", use_container_width=True)

    if clear_queue_btn:
        st.session_state.test_configs = []
        st.rerun()

    if run_all_btn:
        st.session_state.run_batch_requested = True
else:
    st.info("Ajoute des tests avec le bouton « ➕ Ajouter cette config à la file » dans la sidebar.")

# =============================================================================
# PART 9: EXÉCUTION DU BATCH
# =============================================================================

if st.session_state.run_batch_requested and st.session_state.test_configs:
    configs_to_run = list(st.session_state.test_configs)
    st.session_state.test_results = []

    global_progress = st.progress(0)
    global_status = st.empty()

    for test_idx, cfg in enumerate(configs_to_run):
        global_status.info(f"⏳ Test {test_idx + 1}/{len(configs_to_run)} en cours...")
        global_progress.progress(test_idx / len(configs_to_run))

        result = run_single_experiment(
            cfg, cfg['dist_type'],
            load_algo_module(cfg.get("algo_name", "Simple (original)"))[0]
        )

        # Sauvegarde auto
        saved_path = save_test_results(test_idx + 1, cfg, result)
        result['saved_path'] = saved_path
        result['config'] = cfg
        st.session_state.test_results.append(result)

    global_progress.progress(1.0)
    global_status.success(
        f"✅ {len(configs_to_run)} tests terminés ! "
        f"Résultats sauvegardés dans '{RESULTS_DIR}/'"
    )
    time.sleep(1)
    global_progress.empty()

    st.session_state.test_configs = []
    st.session_state.run_batch_requested = False
    st.rerun()

# =============================================================================
# PART 10: AFFICHAGE DES RÉSULTATS
# =============================================================================

st.markdown("---")

# A) Résultat de la dernière simu simple
if st.session_state.single_result is not None:
    st.header("📊 Dernière simulation")
    render_result_tabs(
        st.session_state.single_result,
        st.session_state.single_cfg,
        tab_prefix="single",
    )

# B) Résultats de batch
if st.session_state.test_results:
    st.markdown("---")
    st.header(f"📦 Résultats du batch ({len(st.session_state.test_results)} tests)")

    labels = []
    for i, res in enumerate(st.session_state.test_results):
        cfg = res['config']
        labels.append(f"Test #{i+1} — δ={cfg['delta']}, T={cfg['horizon']}, σ={cfg['sigma']}")

    sel = st.selectbox("Sélectionner un test à explorer :",
                       options=list(range(len(labels))),
                       format_func=lambda x: labels[x],
                       key="batch_select")

    chosen = st.session_state.test_results[sel]
    st.caption(f"💾 {chosen.get('saved_path', '')}")
    render_result_tabs(chosen, chosen['config'], tab_prefix=f"batch_{sel}")

    if st.button("Effacer les résultats du batch"):
        st.session_state.test_results = []
        st.rerun()

# =============================================================================
# PART 11: HISTORIQUE LOCAL
# =============================================================================

st.markdown("---")
with st.expander("🗄️ Historique local des simulations"):
    if os.path.exists(CATALOG_FILE):
        catalog = pd.read_csv(CATALOG_FILE)
        if len(catalog) == 0:
            st.info("Catalogue vide.")
        else:
            selected_idx = st.selectbox(
                "Sélectionnez une simulation à recharger :",
                options=catalog.index,
                format_func=lambda x: f"{catalog.iloc[x]['date']} - {catalog.iloc[x]['name']}",
                key="history_select",
            )
            if st.button("Charger et afficher les graphiques", key="history_load"):
                target_file = catalog.iloc[selected_idx]['filename']
                # On cherche dans les deux dossiers possibles (manuel + batch)
                candidate_paths = [
                    os.path.join(SIMS_DIR, target_file),
                    os.path.join(RESULTS_DIR, target_file),
                ]
                fpath = next((p for p in candidate_paths if os.path.isfile(p)), None)
                if fpath is None:
                    st.error(f"Fichier introuvable : {target_file}")
                else:
                    with open(fpath, 'rb') as f:
                        old = pickle.load(f)
                    st.subheader(f"Restauration : {catalog.iloc[selected_idx]['name']}")
                    n_sims_old = old.get('n_sims', 'N/A')

                    c1, c2 = st.columns(2)
                    with c1:
                        fig = plot_tpr_comparison(
                            old['tpr_adapt'], old['tpr_unif'],
                            old['delta'], old['sigma'],
                            n_sims_old if isinstance(n_sims_old, int) else 0,
                        )
                        st.pyplot(fig); plt.close(fig)
                    with c2:
                        fig = plot_pulls_comparison(
                            old['counts_unif'], old['counts_adapt'],
                            old['true_means'], old['mu_0'],
                            old['delta'], old['sigma'],
                        )
                        st.pyplot(fig); plt.close(fig)

                    fig = plot_pvalues_grid(old['p_unif'], old['p_adapt'], old['true_means'])
                    st.pyplot(fig); plt.close(fig)
    else:
        st.info("Aucun historique trouvé.")

# =============================================================================
# PART 12: ÉCRAN D'ACCUEIL (si aucun résultat)
# =============================================================================

if st.session_state.single_result is None and not st.session_state.test_results:
    st.markdown("""
## Bienvenue sur le simulateur d'algorithmes de bandit !

Cette application permet de comparer les stratégies d'échantillonnage pour plusieurs backends :

- **Adaptive** (Jamieson & Jain 2018) : stratégie UCB, alloue plus de tirages aux bras prometteurs.
- **Uniform** : échantillonne tous les bras de manière uniforme.
- **Backends disponibles** : Simple, Fusion V2 NM/NM_M2, Fusion V3 betting continu/binaire.

### Comment utiliser :
1. Ajustez les paramètres dans la barre latérale.
2. Configurez les moyennes des différents bras.
3. Cliquez sur **🚀 Lancer la simulation** (test unique) ou **➕ Ajouter cette config à la file** puis **▶️ Lancer tous les tests** (batch).
4. Explorez les résultats dans les différents onglets.

### Graphiques disponibles :
- **TPR** : vitesse à laquelle l'algorithme identifie les bons bras.
- **Répartition** : allocation des tirages entre les bras.
- **Spaghetti** : variabilité entre simulations.
- **Intervalles de confiance** : état schématique des estimations.
- **Données brutes** : export CSV.
- **P-Values (combiné / grille)** : évolution des p-values anytime.
""")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.info("**Adaptive**\nAlloue plus de ressources aux bras prometteurs")
    with col2:
        st.info("**Uniform**\nDistribution égale des tirages")
    with col3:
        st.info("**Objectif**\nIdentifier les bras avec μ > μ₀")
