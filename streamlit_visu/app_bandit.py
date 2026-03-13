import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import brentq
from tqdm import tqdm
import pandas as pd
from pathlib import Path
import time

# Configuration de la page
st.set_page_config(
    page_title="Simulateur de Bandit Algorithm",
    page_icon="🎰",
    layout="wide"
)

# -----------------------------------------------------------------------------
# PART 1: LES ALGORITHMES (tels que définis dans votre code)
# -----------------------------------------------------------------------------

class JamiesonJainAlgo:
    def __init__(self, n_arms, mu_0, delta):
        self.n = n_arms
        self.mu_0 = mu_0
        self.delta = delta
        
        self.counts = np.zeros(n_arms, dtype=int)
        self.emp_means = np.zeros(n_arms, dtype=float)
        self.time = 0
        self.S_t = set()
        self.counts_evolution = [np.zeros(n_arms, dtype=int)]

    def phi(self, t, delta_val):
        if t == 0: 
            return float('inf')
        
        num = 2 * np.log(1/delta_val) + 6 * np.log(np.log(1/delta_val) + 1e-10) + \
              3 * np.log(np.log(np.e * t / 2) + 1e-10)
        num = max(0.0, num)
        return np.sqrt(num / t)

    def select_arm(self):
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

    def get_anytime_pvalue(self, arm_idx):
        t = self.counts[arm_idx]
        mean = self.emp_means[arm_idx]
        
        if t == 0 or mean <= self.mu_0:
            return 1.0

        diff = mean - self.mu_0
        
        def objective(p):
            if p <= 0: return float('inf') 
            if p >= 1: return -float('inf')
            return self.phi(t, p) - diff

        try:
            p_value = brentq(objective, 1e-12, 0.9999)
            return p_value
        except ValueError:
            return 1.0

    def bh_update_optimized(self, arm_idx, observation):
        n_pulls = self.counts[arm_idx]
        self.emp_means[arm_idx] = (self.emp_means[arm_idx] * n_pulls + observation) / (n_pulls + 1)
        self.counts[arm_idx] += 1
        self.time += 1
        self.counts_evolution.append(self.counts.copy()) 
        
        p_values_with_idx = [(self.get_anytime_pvalue(i), i) for i in range(self.n)]
        p_values_with_idx.sort(key=lambda x: x[0])
        
        current_St = set()
        
        for k in range(self.n, 0, -1):
            p_val_k = p_values_with_idx[k - 1][0]
            effective_delta = self.delta * k / self.n
            
            if p_val_k <= effective_delta:
                for rank in range(k):
                    discovered_arm_idx = p_values_with_idx[rank][1]
                    current_St.add(discovered_arm_idx)
                break
                
        self.S_t.update(current_St)

class UniformAlgo:
    def __init__(self, n_arms, mu_0, delta):
        self.n = n_arms
        self.mu_0 = mu_0
        self.delta = delta
        
        self.counts = np.zeros(n_arms, dtype=int)
        self.emp_means = np.zeros(n_arms, dtype=float)
        self.time = 0
        self.S_t = set()
        self.counts_evolution = [np.zeros(n_arms, dtype=int)]

    def phi(self, t, delta_val):
        if t == 0: 
            return float('inf')
        
        num = 2 * np.log(1/delta_val) + 6 * np.log(np.log(1/delta_val) + 1e-10) + \
              3 * np.log(np.log(np.e * t / 2) + 1e-10)
        num = max(0.0, num)
        return np.sqrt(num / t)

    def select_arm(self):
        return np.random.randint(self.n)
    
    def get_anytime_pvalue(self, arm_idx):
        t = self.counts[arm_idx]
        mean = self.emp_means[arm_idx]
        
        if t == 0 or mean <= self.mu_0:
            return 1.0

        diff = mean - self.mu_0
        
        def objective(p):
            if p <= 0: return float('inf') 
            if p >= 1: return -float('inf')
            return self.phi(t, p) - diff

        try:
            p_value = brentq(objective, 1e-12, 0.9999)
            return p_value
        except ValueError:
            return 1.0

    def bh_update_optimized(self, arm_idx, observation):
        n_pulls = self.counts[arm_idx]
        self.emp_means[arm_idx] = (self.emp_means[arm_idx] * n_pulls + observation) / (n_pulls + 1)
        self.counts[arm_idx] += 1
        self.time += 1
        self.counts_evolution.append(self.counts.copy()) 
        
        p_values_with_idx = [(self.get_anytime_pvalue(i), i) for i in range(self.n)]
        p_values_with_idx.sort(key=lambda x: x[0])
        
        current_St = set()
        
        for k in range(self.n, 0, -1):
            p_val_k = p_values_with_idx[k - 1][0]
            effective_delta = self.delta * k / self.n
            
            if p_val_k <= effective_delta:
                for rank in range(k):
                    discovered_arm_idx = p_values_with_idx[rank][1]
                    current_St.add(discovered_arm_idx)
                break
                
        self.S_t.update(current_St)

# -----------------------------------------------------------------------------
# PART 2: SIMULATION ENGINE
# -----------------------------------------------------------------------------

def prepare_experiment(true_means, horizon, n_sims, scale):
    n_arms = len(true_means)
    all_arm_data_by_sim = []
    
    # Barre de progression pour la préparation des données
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for sim in range(n_sims):
        all_arm_data = []   
        for arm in range(n_arms):
            result_arm = []
            for t in range(horizon):
                observation = np.random.normal(loc=true_means[arm], scale=scale)
                result_arm.append(observation)
            all_arm_data.append(result_arm)
        all_arm_data_by_sim.append(all_arm_data)
        
        # Mise à jour de la progression
        progress = (sim + 1) / n_sims
        progress_bar.progress(progress)
        status_text.text(f"Préparation des données: {sim + 1}/{n_sims} simulations")
    
    status_text.text("Préparation terminée!")
    time.sleep(0.5)
    status_text.empty()
    progress_bar.empty()
    
    return all_arm_data_by_sim

def run_experiment(true_means, horizon, mode, all_arm_data, n_simulations=20, mu_0=0.0):
    n_arms = len(true_means)
    true_positives = [i for i, m in enumerate(true_means) if m > mu_0]
    
    tpr_history_sum = np.zeros(horizon)
    tpr_list = []
    
    counts_evolution_sum = np.zeros((horizon + 1, n_arms))
    counts_list = []

    # Barre de progression pour les simulations
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for no_sim in range(n_simulations):
        all_arm_counts = [0 for _ in range(n_arms)]

        if mode == 'adaptive':
            algo = JamiesonJainAlgo(n_arms, mu_0, delta)
        elif mode == 'uniform':
            algo = UniformAlgo(n_arms, mu_0, delta)
        else:
            raise ValueError("Algorithm name not detected")
        
        run_tpr = []
        
        for t in range(horizon):
            arm = algo.select_arm()
            
            if arm == "stop":
                last_tpr = run_tpr[-1] if run_tpr else 1.0
                remaining_steps = horizon - len(run_tpr)
                run_tpr.extend([last_tpr] * remaining_steps)
                
                last_counts = algo.counts_evolution[-1]
                for _ in range(remaining_steps):
                    algo.counts_evolution.append(last_counts.copy())
                break
            
            else:
                observation = all_arm_data[no_sim][arm][all_arm_counts[arm]]
                all_arm_counts[arm] += 1
                
                algo.bh_update_optimized(arm, observation)
                
                nb_found = len(algo.S_t.intersection(true_positives))
                current_tpr = nb_found / len(true_positives) if true_positives else 1.0
                run_tpr.append(current_tpr)

        tpr_i = np.array(run_tpr)
        tpr_list.append(tpr_i)
        tpr_history_sum += tpr_i
        
        counts_arr = np.array(algo.counts_evolution)[:horizon+1]
        counts_list.append(counts_arr)
        counts_evolution_sum += counts_arr
        
        # Mise à jour de la progression
        progress = (no_sim + 1) / n_simulations
        progress_bar.progress(progress)
        status_text.text(f"Simulations {mode}: {no_sim + 1}/{n_simulations}")

    tpr_history_mean = tpr_history_sum / n_simulations
    counts_history_mean = counts_evolution_sum / n_simulations
    
    status_text.text(f"Simulations {mode} terminées!")
    time.sleep(0.5)
    status_text.empty()
    progress_bar.empty()

    return tpr_history_mean, tpr_list, counts_history_mean, counts_list

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
    
    # Uniform
    ax1 = axes[0]
    for i in range(n_arms):
        ax1.plot(counts_unif_mean[:, i], label=f"Bras {i} (μ={true_means[i]:.2f})", linewidth=2)
    ax1.set_xlabel("Temps (t)")
    ax1.set_ylabel("Nombre de tirages")
    ax1.set_title("Uniform - Répartition des tirages")
    ax1.legend(fontsize='small')
    ax1.grid(True, alpha=0.3)
    
    # Adaptive
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

def plot_spaghetti(counts_adapt_list, counts_adapt_mean, true_means, mu_0, delta, sigma):
    n_arms = len(true_means)
    fig, ax = plt.subplots(figsize=(12, 7))
    
    for arm_idx in range(n_arms):
        color = f'C{arm_idx}'
        linestyle = '-' if true_means[arm_idx] > mu_0 else '--'
        
        # Simulations individuelles
        for sim_counts in counts_adapt_list:
            ax.plot(sim_counts[:, arm_idx], 
                   color=color, 
                   alpha=0.1, 
                   linewidth=0.5, 
                   linestyle=linestyle)
        
        # Moyenne
        ax.plot(counts_adapt_mean[:, arm_idx], 
               label=f"Bras {arm_idx} (μ={true_means[arm_idx]:.2f})", 
               color=color, 
               linewidth=2.5, 
               linestyle=linestyle)
    
    ax.set_xlabel("Temps (t)")
    ax.set_ylabel("Nombre cumulé de tirages")
    ax.set_title(f"Adaptive - Simulations individuelles\nδ={delta:.3f}, σ={sigma:.3f}")
    ax.legend(loc='upper left', bbox_to_anchor=(1, 1))
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    return fig

def plot_confidence_intervals(algo, true_means, mu_0, t):
    """Affiche les intervalles de confiance à un instant donné"""
    n_arms = len(true_means)
    arms = range(n_arms)
    means = algo.emp_means
    counts = algo.counts
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    for i in arms:
        if counts[i] > 0:
            ci_width = algo.phi(counts[i], algo.delta)
            ci_lower = means[i] - ci_width
            ci_upper = means[i] + ci_width
            
            # Couleur selon si découvert ou non
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

def plot_pvalues_evolution(algo_history, n_arms, horizon):
    """Simule l'évolution des p-values (version simplifiée)"""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Génération de p-values simulées pour la démonstration
    t = np.arange(horizon)
    for i in range(n_arms):
        # Simulation d'une décroissance des p-values
        p_values = 1.0 / (1 + np.exp((t - 200*i) / 50))
        ax.plot(t, p_values, label=f'Bras {i}')
    
    ax.set_xlabel("Temps")
    ax.set_ylabel("p-value")
    ax.set_title("Évolution des p-values (simulation conceptuelle)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    return fig

def display_metrics(tpr_adapt, tpr_unif, counts_adapt_mean, true_means):
    """Affiche des métriques de performance"""
    col1, col2, col3, col4 = st.columns(4)
    
    # Temps pour atteindre 90% de TPR
    tpr_90_adapt = np.argmax(tpr_adapt >= 0.9) if np.any(tpr_adapt >= 0.9) else len(tpr_adapt)
    tpr_90_unif = np.argmax(tpr_unif >= 0.9) if np.any(tpr_unif >= 0.9) else len(tpr_unif)
    
    with col1:
        st.metric("Tps TPR 90% (Adaptive)", f"{tpr_90_adapt}")
    with col2:
        st.metric("Tps TPR 90% (Uniform)", f"{tpr_90_unif}")
    
    # Ratio de tirages entre bons et mauvais bras (Adaptive)
    good_arms = [i for i, m in enumerate(true_means) if m > mu_0]
    bad_arms = [i for i, m in enumerate(true_means) if m <= mu_0]
    
    if good_arms and bad_arms:
        final_counts = counts_adapt_mean[-1]
        good_mean = np.mean([final_counts[i] for i in good_arms])
        bad_mean = np.mean([final_counts[i] for i in bad_arms])
        ratio = good_mean / bad_mean if bad_mean > 0 else float('inf')
        
        with col3:
            st.metric("Ratio tirages (bons/mauvais)", f"{ratio:.2f}")
    
    # Gain en efficacité
    gain = ((tpr_90_unif - tpr_90_adapt) / tpr_90_unif * 100) if tpr_90_unif > 0 else 0
    with col4:
        st.metric("Gain d'efficacité", f"{gain:.1f}%")

# -----------------------------------------------------------------------------
# PART 4: APPLICATION STREAMLIT
# -----------------------------------------------------------------------------

# Titre principal
st.title("🎰 Simulateur Interactif d'Algorithmes de Bandit")
st.markdown("---")

# Barre latérale - Paramètres
with st.sidebar:
    st.header("⚙️ Paramètres de simulation")
    
    # Paramètres principaux
    n_sims = st.slider("Nombre de simulations", 10, 500, 100, 10)
    horizon = st.slider("Horizon (T)", 100, 2000, 800, 50)
    sigma = st.slider("Bruit (σ)", 0.1, 3.0, 1.0, 0.1)
    delta = st.slider("Paramètre δ (FDR)", 0.01, 0.5, 0.05, 0.01)
    mu_0 = st.number_input("Seuil μ₀", value=0.0, step=0.1)
    
    st.markdown("---")
    st.header("Configuration des bras")
    
    # Configuration des bras
    n_arms = st.number_input("Nombre de bras", min_value=2, max_value=10, value=6, step=1)
    
    # Création des entrées pour les moyennes
    st.subheader("Moyennes des bras")
    true_means = []
    cols = st.columns(2)
    for i in range(n_arms):
        with cols[i % 2]:
            mean = st.number_input(f"Bras {i}", value=0.5 if i < 2 else (0.35 if i < 4 else 0.0), 
                                   step=0.05, key=f"mean_{i}", format="%.2f")
            true_means.append(mean)
    
    true_means = np.array(true_means)
    
    st.markdown("---")
    
    # Bouton de lancement
    run_button = st.button("🚀 Lancer la simulation", type="primary", use_container_width=True)
    
    st.markdown("---")
    st.caption("Basé sur l'algorithme de Jamieson & Jain (2018)")

# Contenu principal avec onglets
if run_button:
    # Préparation des données
    with st.spinner("Préparation des données..."):
        all_arm_data = prepare_experiment(true_means, horizon, n_sims, sigma)
    
    # Exécution des simulations
    col1, col2 = st.columns(2)
    with col1:
        st.info("Simulation Uniforme en cours...")
        tpr_unif, tpr_list_unif, counts_unif_mean, counts_list_unif = run_experiment(
            true_means, horizon, 'uniform', all_arm_data, n_sims, mu_0)
    
    with col2:
        st.info("Simulation Adaptive en cours...")
        tpr_adapt, tpr_list_adapt, counts_adapt_mean, counts_list_adapt = run_experiment(
            true_means, horizon, 'adaptive', all_arm_data, n_sims, mu_0)
    
    st.success("✅ Simulations terminées avec succès!")
    
    # Métriques de performance
    st.markdown("---")
    st.header("📊 Résultats")
    display_metrics(tpr_adapt, tpr_unif, counts_adapt_mean, true_means)
    st.markdown("---")
    
    # Création des onglets
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "TPR - Vitesse de découverte", 
        "Répartition des tirages", 
        "Spaghetti Plot",
        "Intervalles de confiance",
        "Données brutes"
    ])
    
    with tab1:
        st.subheader("Comparaison de la vitesse de découverte (TPR)")
        fig = plot_tpr_comparison(tpr_adapt, tpr_unif, delta, sigma, n_sims)
        st.pyplot(fig)
        plt.close(fig)
        
        st.info("Le TPR (True Positive Rate) mesure la proportion de bons bras correctement identifiés au cours du temps.")
    
    with tab2:
        st.subheader("Répartition des tirages entre les bras")
        
        # Option pour choisir l'algorithme à afficher
        algo_choice = st.radio("Choisir l'algorithme", ["Comparaison", "Adaptive seulement", "Uniform seulement"], horizontal=True)
        
        if algo_choice == "Comparaison":
            fig = plot_pulls_comparison(counts_unif_mean, counts_adapt_mean, true_means, mu_0, delta, sigma)
        elif algo_choice == "Adaptive seulement":
            fig, ax = plt.subplots(figsize=(10, 6))
            for i in range(n_arms):
                linestyle = '-' if true_means[i] > mu_0 else '--'
                ax.plot(counts_adapt_mean[:, i], 
                       label=f"Bras {i} (μ={true_means[i]:.2f})", 
                       linewidth=2, 
                       linestyle=linestyle)
            ax.set_xlabel("Temps (t)")
            ax.set_ylabel("Nombre de tirages")
            ax.set_title(f"Adaptive - Répartition des tirages")
            ax.legend()
            ax.grid(True, alpha=0.3)
        else:  # Uniform seulement
            fig, ax = plt.subplots(figsize=(10, 6))
            for i in range(n_arms):
                ax.plot(counts_unif_mean[:, i], 
                       label=f"Bras {i} (μ={true_means[i]:.2f})", 
                       linewidth=2)
            ax.set_xlabel("Temps (t)")
            ax.set_ylabel("Nombre de tirages")
            ax.set_title(f"Uniform - Répartition des tirages")
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        st.pyplot(fig)
        plt.close(fig)
    
    with tab3:
        st.subheader("Visualisation des simulations individuelles (Spaghetti Plot)")
        
        # Option pour choisir la transparence
        alpha = st.slider("Transparence des lignes individuelles", 0.05, 0.5, 0.1, 0.05)
        
        fig, ax = plt.subplots(figsize=(12, 7))
        
        for arm_idx in range(n_arms):
            color = f'C{arm_idx}'
            linestyle = '-' if true_means[arm_idx] > mu_0 else '--'
            
            # Simulations individuelles
            for sim_counts in counts_list_adapt:
                ax.plot(sim_counts[:, arm_idx], 
                       color=color, 
                       alpha=alpha, 
                       linewidth=0.5, 
                       linestyle=linestyle)
            
            # Moyenne
            ax.plot(counts_adapt_mean[:, arm_idx], 
                   label=f"Bras {arm_idx} (μ={true_means[arm_idx]:.2f})", 
                   color=color, 
                   linewidth=2.5, 
                   linestyle=linestyle)
        
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
        
        # Créer un algo factice pour la démonstration à la fin de la simulation
        demo_algo = UniformAlgo(n_arms, mu_0, delta)
        demo_algo.counts = counts_adapt_mean[-1].astype(int)
        demo_algo.emp_means = np.array([counts_adapt_mean[-1][i] / horizon if counts_adapt_mean[-1][i] > 0 else 0 
                                        for i in range(n_arms)])
        demo_algo.S_t = set([i for i in range(n_arms) if true_means[i] > mu_0])  # Simulation
        
        fig = plot_confidence_intervals(demo_algo, true_means, mu_0, horizon)
        st.pyplot(fig)
        plt.close(fig)
        
        st.info("Les intervalles de confiance sont basés sur la loi du logarithme itéré (LIL).")
    
    with tab5:
        st.subheader("Données brutes des simulations")
        
        # Création d'un DataFrame pour les données finales
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
        
        # Option de téléchargement
        csv = df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Télécharger les données (CSV)",
            data=csv,
            file_name=f"bandit_results_d{delta}_s{sigma}.csv",
            mime="text/csv",
        )

else:
    # Message d'accueil
    st.markdown("""
    ## 👋 Bienvenue sur le simulateur d'algorithmes de bandit!
    
    Cette application vous permet d'explorer et de comparer deux algorithmes:
    
    * **Adaptive** (Jamieson & Jain 2018): Utilise une stratégie UCB pour allouer plus de tirages aux bras prometteurs
    * **Uniform**: Échantillonne tous les bras de manière uniforme
    
    ### 🎮 Comment utiliser:
    1. Ajustez les paramètres dans la barre latérale
    2. Configurez les moyennes des différents bras
    3. Cliquez sur "Lancer la simulation"
    4. Explorez les résultats dans les différents onglets
    
    ### 📊 Les graphiques disponibles:
    * **TPR**: Vitesse à laquelle l'algorithme identifie les bons bras
    * **Répartition**: Comment les tirages sont alloués entre les bras
    * **Spaghetti**: Visualisation de la variabilité entre simulations
    * **Intervalles de confiance**: État des estimations à la fin
    * **Données brutes**: Export des résultats
    
    ---
    **Commencez par configurer vos paramètres dans le panneau de gauche ➡️**
    """)
    
    # Image ou information supplémentaire
    col1, col2, col3 = st.columns(3)
    with col1:
        st.info("📈 **Adaptive**\nAlloue plus de ressources aux bras prometteurs")
    with col2:
        st.info("📊 **Uniform**\nDistribution égale des tirages")
    with col3:
        st.info("🎯 **Objectif**\nIdentifier les bras avec μ > μ₀")