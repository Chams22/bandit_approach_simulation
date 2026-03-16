import pandas as pd
import numpy as np
import os
import importlib
import usable_adaptative_algorithm
importlib.reload(usable_adaptative_algorithm)
import matplotlib.pyplot as plt
from statistics import mean



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

def prepare_real_experiment(df, n_sims):
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

    n_sims = 5

    datasets = {
    "effort": (df_effort, 59),
    "exercise": (df_exercise, 53),
    "penn": (df_penn, 16),
    "walmart": (df_walmart, 3)
    }

    results = {}
    
    print("\n--- Traitement des données ---")
    for name, df in datasets.items():
        print(f"Préparation de {name}...")
        # Appel de la fonction
        data_sim, arm_names = prepare_real_experiment(df[0], n_sims)
        # Stockage
        results[name] = {
            "data": data_sim,       # La liste de listes de listes
            "arm_names": arm_names,  # Pour savoir que l'index 0 correspond à tel bras
            "control_arm": df[1]
        }
        # Vérification rapide
        print(f"   -> {len(data_sim)} simulations générées.")
        print(f"   -> {len(arm_names)} bras trouvés.")
        print(f"   -> Exemple bras 0 ({arm_names[0]}): {len(data_sim[0][0])} observations.")

    # --- 4. COMMENT UTILISER LES DONNÉES ---
    # Exemple pour lancer votre run_experiment avec les données de PENN :
    # data_penn = results['penn']['data']
    # arm_penn = results['penn']['arm_names']
    # control_arm = 16

    # data_effort = results['effort']['data']
    # arm_effort = results['effort']['arm_names']
    # control_arm = 59

    # data_walmart = results['walmart']['data']
    # arm_walmart = results['walmart']['arm_names']
    # control_arm = 3

    # data_exercise = results['exercise']['data']
    # arm_exercise = results['exercise']['arm_names']
    # control_arm = 53

    name_data="penn"
    # name_data="effort"
    # name_data="walmart"
    # name_data="exercise"
    list_name=["penn", "exercise", "effort", "walmart"]
    num_graph=0
    for name_data in list_name:
        print("nom de la base étudiée", name_data)

        data_test=results[name_data]['data']
        arm_test=results[name_data]['arm_names']
        control_arm=results[name_data]['control_arm']

        # --- Utilisation ---
        min_len, max_len = get_min_max_samples(data_test)
        print("taille min =", min_len, "taille max =", max_len)

        mu_0 = 0.0
        delta = 0.05
        horizon = min_len
        n_arms = len(arm_test)
        init_nb = round(min_len*0.01)
        init_choice = True
        mu_0_unif=mean(data_test[0][control_arm])
        print("mu_0 moyenne calcule", mu_0_unif)
        for n in range(n_arms):
            mean_arm=mean(data_test[0][n])
            print("moyenne arm", n, ":", arm_test[n], "=", round(mean_arm, 4))

            
        # 1. Run Simulations
        pnb_unif, _, counts_unif_mean, counts_unif_list,  np_p_value_list_unif, np_p_value_mean_unif = usable_adaptative_algorithm.run_experiment(arm_test, mu_0_unif, delta, horizon, 'uniform', data_test, n_sims, control_arm, 0, False, False)
        pnb_adapt, _, counts_adapt_mean, counts_adapt_list, np_p_value_list_adapt, np_p_value_mean_adapt = usable_adaptative_algorithm.run_experiment(arm_test, mu_0_unif, delta, horizon, 'adaptive', data_test, n_sims, control_arm, init_nb, init_choice, False)
        pnb_adapt_v, _, counts_adapt_v_mean, counts_adapt_v_list, np_p_value_list_adapt_v, np_p_value_mean_adapt_v = usable_adaptative_algorithm.run_experiment(arm_test, mu_0_unif, delta, horizon, 'adaptive', data_test, n_sims, control_arm, init_nb, init_choice, True)


        # --- PLOT 1: pr ---
        plt.figure(1+num_graph*10, figsize=(10, 5))
        plt.plot(pnb_adapt, label='Adaptive', color='#ff7f0e', linewidth=2)
        plt.plot(pnb_adapt_v, label='Adaptive_Var', color="#59e244", linewidth=2)    
        plt.plot(pnb_unif, label='Uniform', color='#1f77b4', linestyle='--')
        plt.axhline(y=1.0, color='gray', linestyle=':')
        plt.title("Discovery speed (pr)")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(git_root / f"figure_real_data/{name_data}/figure1.png", dpi=300, bbox_inches="tight")


        # --- PLOT 2: PULL EVOLUTION ---
        plt.figure(2+num_graph*10, figsize=(12, 6))
        
        # Subplot 1: Uniform
        plt.subplot(1, 3, 1)
        plt.title("Uniform: Number of pulls per arm")
        for arm_idx in range(n_arms):
            label = f"Arm {arm_idx} ($mu$={arm_test[arm_idx][0:4]})"
            plt.plot(counts_unif_mean[:, arm_idx], label=label, linewidth=2)
        plt.xlabel("Time (t)")
        plt.ylabel("Number of pulls ($T_i(t)$)")
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Subplot 2: Adaptive
        plt.subplot(1, 3, 2)
        plt.title("Adaptive: Number of pulls per arm")
        for arm_idx in range(n_arms):
            linestyle = '-' if arm_test[arm_idx]!='control' else '--'
            label = f"Arm {arm_idx} ($mu$={arm_test[arm_idx][0:4]})"
            plt.plot(counts_adapt_mean[:, arm_idx], label=label, linewidth=2, linestyle=linestyle)
        
        # Subplot 3: Adaptive VAR
        plt.subplot(1, 3, 3)
        plt.title("Adaptive VAR: Number of pulls per arm")
        for arm_idx in range(n_arms):
            linestyle = '-' if arm_test[arm_idx]!='control' else '--'
            label = f"Arm {arm_idx} ($mu$={arm_test[arm_idx][0:4]})"
            plt.plot(counts_adapt_v_mean[:, arm_idx], label=label, linewidth=2, linestyle=linestyle)
            
        plt.xlabel("Time (t)")
        plt.ylabel("Number of pulls ($T_i(t)$)")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(git_root / f"figure_real_data/{name_data}/figure2.png", dpi=300, bbox_inches="tight")


        # --- PLOT 3: PULL EVOLUTION (SPAGHETTI PLOT) ---
        plt.figure(3+num_graph*10, figsize=(14, 6))
        plt.title(f"Adaptive: Number of pulls per arm ({n_sims} simulations)")
        
        for arm_idx in range(n_arms):
            color = f'C{arm_idx}' 
            linestyle = '-' if arm_test[arm_idx]!='control' else '--'
            label = f"Arm {arm_idx} ($mu$={arm_test[arm_idx][0:4]})"
            
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
        plt.savefig(git_root / f"figure_real_data/{name_data}/figure3.png", dpi=300, bbox_inches="tight")

        # --- PLOT 3 VAR: PULL EVOLUTION (SPAGHETTI PLOT) ---
        plt.figure(6+num_graph*10, figsize=(14, 6))
        plt.title(f"Adaptive: Number of pulls per arm ({n_sims} simulations)")
        
        for arm_idx in range(n_arms):
            color = f'C{arm_idx}' 
            linestyle = '-' if arm_test[arm_idx]!='control' else '--'
            label = f"Arm {arm_idx} ($mu$={arm_test[arm_idx][0:4]})"
            
            for sim_counts in counts_adapt_v_list:
                plt.plot(sim_counts[:, arm_idx], 
                        color=color, 
                        alpha=0.15,
                        linewidth=0.8,
                        linestyle=linestyle)

            plt.plot(counts_adapt_v_mean[:, arm_idx], 
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
        plt.savefig(git_root / f"figure_real_data/{name_data}/figure3var.png", dpi=300, bbox_inches="tight")

        # --- PLOT 4: P-VALUES ---
        plt.figure(4+num_graph*10, figsize=(12, 6))
        # Subplot 1: Uniform
        plt.subplot(1, 3, 1)
        plt.title("Uniform: P values by iteration and arm")
        for arm_idx in range(n_arms):
            print("bc1")
            label = f"Arm {arm_test[arm_idx][0:4]}"
            plt.plot(np_p_value_mean_unif[:, arm_idx], label=label, linewidth=2)
        plt.xlabel("Time (t)")
        plt.ylabel("P values")
        plt.legend()
        plt.grid(True, alpha=0.3)
        # Subplot 2: Adaptative
        plt.subplot(1, 3, 2)
        plt.title("Adaptative: P values by iteration and arm")
        for arm_idx in range(n_arms):
            print("bc2")
            label = f"Arm {arm_test[arm_idx][0:4]}"
            plt.plot(np_p_value_mean_adapt[:, arm_idx], label=label, linewidth=2)
        plt.xlabel("Time (t)")
        plt.ylabel("P values")
        plt.legend()
        plt.grid(True, alpha=0.3)
        # Subplot 3: Adaptative VAR
        plt.subplot(1, 3, 3)
        plt.title("Adaptative VAR: P values by iteration and arm")
        for arm_idx in range(n_arms):
            print("bc3")
            label = f"Arm {arm_test[arm_idx][0:4]}"
            plt.plot(np_p_value_mean_adapt_v[:, arm_idx], label=label, linewidth=2)
        plt.xlabel("Time (t)")
        plt.ylabel("P values")
        plt.legend()
        plt.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(git_root / f"figure_real_data/{name_data}/figure4.png", dpi=300, bbox_inches="tight")

    # --- PLOT 5: P-VALUES (1 Colonne, 3 Trajectoires par Graphe) ---

        # Définition explicite des couleurs pour chaque algorithme
        color_unif = 'tab:blue'
        color_adapt = 'tab:orange'
        color_adapt_v = 'tab:green'

        # Création d'une grille : n_arms (lignes) x 1 (colonne)
        # On réduit un peu la largeur (ex: 10) vu qu'il n'y a plus qu'une seule colonne
        fig, axes = plt.subplots(nrows=n_arms, ncols=1, 
                                figsize=(10, 2.5 * n_arms), 
                                sharex=True)

        # Sécurité au cas où il n'y aurait qu'un seul bras (axes ne serait pas une liste)
        if n_arms == 1:
            axes = [axes]

        for arm_idx in range(n_arms):
            ax = axes[arm_idx]
            arm_name = arm_test[arm_idx]
            
            # Ajout du titre pour identifier de quel bras on parle sur cette ligne
            ax.set_title(f"P-values evolution for Arm {arm_name}")

            # Tracé des 3 trajectoires sur le MÊME graphique
            ax.plot(np_p_value_mean_unif[:, arm_idx], label="Uniform", linewidth=2, color=color_unif)
            ax.plot(np_p_value_mean_adapt[:, arm_idx], label="Adaptative", linewidth=2, color=color_adapt)
            ax.plot(np_p_value_mean_adapt_v[:, arm_idx], label="Adaptative VAR", linewidth=2, color=color_adapt_v)
            
            ax.set_ylabel("P value")
            ax.legend(loc="upper right", fontsize="small")
            ax.grid(True, alpha=0.3)

        # Ajout de l'axe des abscisses uniquement sur le tout dernier graphique du bas
        axes[-1].set_xlabel("Time (t)")

        plt.tight_layout()
        plt.savefig(git_root / f"figure_real_data/{name_data}/figure5.png", dpi=300, bbox_inches="tight")
        # plt.show()
        num_graph+=1