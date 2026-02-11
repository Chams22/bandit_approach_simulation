import pandas as pd
import numpy as np
import os
import usable_adaptative_algorithm
import matplotlib.pyplot as plt


# --- 1. CHARGEMENT DES DONNÉES (Votre code, légèrement nettoyé) ---

# Récupération des chemins
current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(current_dir)

path_effort = os.path.join(root_dir, 'data', 'processed', 'effort_experiment.csv')
path_exercise = os.path.join(root_dir, 'data', 'processed', 'exercise_min.csv')
path_penn = os.path.join(root_dir, 'data', 'processed', 'penn.csv')
path_walmart = os.path.join(root_dir, 'data', 'processed', 'walmart.csv')

# Lecture des fichiers
df_effort0 = pd.read_csv(path_effort)
# Petite sécurité : renommage si nécessaire pour effort (souvent 'workerId' ou 'mturk_id')
if 'workerId' in df_effort0.columns: df_effort = df_effort0.rename(columns={'workerId': 'id'})
elif 'participant_id' in df_effort0.columns: df_effort = df_effort0.rename(columns={'participant_id': 'id'})

df_exercise0 = pd.read_csv(path_exercise).rename(columns={'participant_id': 'id'})
df_penn0 = pd.read_csv(path_penn).rename(columns={'participant_id': 'id'})
df_walmart0 = pd.read_csv(path_walmart).rename(columns={'participant_id': 'id'})

print("Fichiers chargés avec succès !")

# Filtrage des colonnes utiles
df_effort = df_effort0[['id', 'y', 'arm']]
df_exercise = df_exercise0[['id', 'y', 'arm']]
df_penn = df_penn0[['id', 'y', 'arm']]
df_walmart = df_walmart0[['id', 'y', 'arm']]

# --- 2. NOUVELLE FONCTION DE PRÉPARATION ---

def prepare_real_experiment(df, n_sims=20):
    """
    Transforme un DataFrame en structure 3D pour la simulation.
    Structure : [simulation_index][arm_index][shuffled_observations]
    
    Returns:
        all_arm_data_by_sim: La structure de données (list of list of list)
        arm_names: La liste des noms de bras correspondant aux indices 0, 1, 2...
    """
    # 1. On groupe par bras et on récupère tous les Y sous forme de liste
    # On trie les bras par ordre alphabétique pour que l'index 0 soit toujours le même
    grouped = df.groupby('arm')['y'].apply(list).sort_index()
    
    # On récupère les noms des bras (ex: ['control', 'treatment_A', ...])
    arm_names = grouped.index.tolist()
    n_arms = len(arm_names)
    
    all_arm_data_by_sim = []

    # 2. Boucle sur les simulations
    for sim in range(n_sims):
        all_arm_data = []
        
        # Pour chaque bras
        for arm_name in arm_names:
            # On copie les données originales
            rewards = grouped[arm_name].copy()
            
            # SHUFFLE : On mélange aléatoirement l'ordre des récompenses
            # Cela simule un ordre d'arrivée différent des patients/participants à chaque simu
            np.random.shuffle(rewards)
            
            all_arm_data.append(rewards)
            
        all_arm_data_by_sim.append(all_arm_data)
        
    return all_arm_data_by_sim, arm_names

# --- 3. EXÉCUTION SUR TOUS LES DATASETS ---

datasets = {
    "effort": df_effort,
    "exercise": df_exercise,
    "penn": df_penn,
    "walmart": df_walmart
}

results = {}

print("\n--- Traitement des données ---")
for name, df in datasets.items():
    print(f"Préparation de {name}...")
    
    # Appel de la fonction
    data_sim, arm_names = prepare_real_experiment(df, n_sims=20)
    
    # Stockage
    results[name] = {
        "data": data_sim,       # La liste de listes de listes
        "arm_names": arm_names  # Pour savoir que l'index 0 correspond à tel bras
    }
    
    # Vérification rapide
    print(f"   -> {len(data_sim)} simulations générées.")
    print(f"   -> {len(arm_names)} bras trouvés.")
    print(f"   -> Exemple bras 0 ({arm_names[0]}): {len(data_sim[0][0])} observations.")


# --- 4. COMMENT UTILISER LES DONNÉES ---
# Exemple pour lancer votre run_experiment avec les données de PENN :
data_penn = results['penn']['data']
arm_penn = results['penn']['arm_names']
def get_min_max_samples(all_arm_data):
    """
    Renvoie la taille du bras qui a le moins de données.
    Utile pour fixer l'horizon max de la simulation sans 'out of bounds'.
    """
    # On prend la première simulation (index 0)
    # car la quantité de données par bras est la même pour toutes les sims
    first_simulation = all_arm_data[0]
    
    # On calcule la longueur de chaque bras et on prend le minimum
    min_len = min(len(arm_data) for arm_data in first_simulation)
    max_len = max(len(arm_data) for arm_data in first_simulation)

    return min_len, max_len

# --- Utilisation ---
min_len, max_len = get_min_max_samples(data_penn)
print("taille min =", min_len, "taille max =", max_len)


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
    plt.close('all')
    
    # Scenario: 2 good arms (0, 1) and 2 bad ones (2, 3)
    mu_0 = 0.0
    delta = 0.05
    horizon = min_len
    n_sims = 20
    n_arms = len(arm_penn)
        
    # 1. Run Simulations
    pnb_unif, _, counts_unif_mean, counts_unif_list = usable_adaptative_algorithm.run_experiment(arm_penn, mu_0, delta, horizon, mode='uniform', all_arm_data=data_penn, n_simulations=n_sims)
    pnb_adapt, _, counts_adapt_mean, counts_adapt_list = usable_adaptative_algorithm.run_experiment(arm_penn, mu_0, delta, horizon, mode='adaptive', all_arm_data=data_penn, n_simulations=n_sims)

    
    # --- PLOT 1: pr ---
    plt.figure(1, figsize=(10, 5))
    plt.plot(pnb_adapt, label='Adaptive', color='#ff7f0e', linewidth=2)
    plt.plot(pnb_unif, label='Uniform', color='#1f77b4', linestyle='--')
    plt.axhline(y=1.0, color='gray', linestyle=':')
    plt.title("Discovery speed (pr)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(git_root / "figure_real_data/figure1.png", dpi=300, bbox_inches="tight")


    # --- PLOT 2: PULL EVOLUTION ---
    plt.figure(2, figsize=(12, 6))
    
    # Subplot 1: Uniform
    plt.subplot(1, 2, 1)
    plt.title("Uniform: Number of pulls per arm")
    for arm_idx in range(n_arms):
        label = f"Arm {arm_idx} ($mu$={arm_penn[arm_idx][0:4]})"
        plt.plot(counts_unif_mean[:, arm_idx], label=label, linewidth=2)
    plt.xlabel("Time (t)")
    plt.ylabel("Number of pulls ($T_i(t)$)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Subplot 2: Adaptive
    plt.subplot(1, 2, 2)
    plt.title("Adaptive: Number of pulls per arm")
    for arm_idx in range(n_arms):
        linestyle = '-' if arm_penn[arm_idx]!='control' else '--'
        label = f"Arm {arm_idx} ($mu$={arm_penn[arm_idx][0:4]})"
        plt.plot(counts_adapt_mean[:, arm_idx], label=label, linewidth=2, linestyle=linestyle)
        
    plt.xlabel("Time (t)")
    plt.ylabel("Number of pulls ($T_i(t)$)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(git_root / "figure_real_data/figure2.png", dpi=300, bbox_inches="tight")


    # --- PLOT 3: PULL EVOLUTION (SPAGHETTI PLOT) ---
    plt.figure(3, figsize=(14, 6))
    plt.title(f"Adaptive: Number of pulls per arm ({n_sims} simulations)")
    
    for arm_idx in range(n_arms):
        color = f'C{arm_idx}' 
        linestyle = '-' if arm_penn[arm_idx]!='control' else '--'
        label = f"Arm {arm_idx} ($mu$={arm_penn[arm_idx][0:4]})"
        
        for sim_counts in counts_adapt_list:
            plt.plot(sim_counts[:, arm_idx], 
                     color=color, 
                     alpha=0.15,
                     linewidth=0.8,
                     linestyle=linestyle)

        plt.plot(counts_adapt_mean[:, arm_idx], 
                 label=label, 
                 color=color, 
                 linewidth=2.5,
                 linestyle=linestyle)
        
    plt.xlabel("Time (t)")
    plt.ylabel("Number of pulls ($T_i(t)$)")
    plt.legend(loc='upper left')
    plt.grid(True, alpha=0.3)
    
    print("Displaying plots...")
    plt.tight_layout()
    plt.savefig(git_root / "figure_real_data/figure3.png", dpi=300, bbox_inches="tight")


# attention changer le controle !!!!!