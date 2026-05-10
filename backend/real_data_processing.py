import pandas as pd
import numpy as np
import os
import importlib
import subprocess
import sys

import matplotlib.pyplot as plt
from statistics import mean, variance
import re


# Easy selection of implementations to run:
#   "simple" -> adaptative_algorithm.py
#   "v2"     -> fused V2 module NM/NM_M2
#   "v3"     -> continuous_v3 / binary_v3
#   "sr"     -> successive rejects with the same interface as the others
#   "all"    -> runs simple, v2, v3, and sr in separate folders
# You can edit this list directly, or use:
#   REAL_DATA_ALGO="v2"
#   REAL_DATA_ALGOS="simple,v2"
#   REAL_DATA_ALGO="all"
DEFAULT_RUN_ALGOS = ["v2", "v3", "sr"]
HISTORY_RECORD_EVERY = max(1, int(os.environ.get("REAL_DATA_HISTORY_RECORD_EVERY", "50")))

ALGORITHM_CONFIGS = {
    "simple": {
        "continuous_module": "adaptative_algorithm",
        "binary_module": "adaptative_algorithm",
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

def parse_run_algos(raw_value):
    if raw_value is None or raw_value.strip() == "":
        return list(DEFAULT_RUN_ALGOS)

    normalized = raw_value.lower().replace(";", ",").replace(" ", ",")
    selected = [item.strip() for item in normalized.split(",") if item.strip()]
    if selected == ["all"]:
        return list(ALGORITHM_CONFIGS.keys())
    return selected


RUN_ALGOS = parse_run_algos(
    os.environ.get("REAL_DATA_ALGOS", os.environ.get("REAL_DATA_ALGO"))
)

invalid_algos = [algo for algo in RUN_ALGOS if algo not in ALGORITHM_CONFIGS]
if invalid_algos:
    valid = ", ".join([*ALGORITHM_CONFIGS.keys(), "all"])
    raise ValueError(
        f"Unknown REAL_DATA_ALGO(S)={invalid_algos!r}. Choose one or more of: {valid}"
    )

if __name__ == "__main__" and len(RUN_ALGOS) > 1:
    script_path = os.path.abspath(__file__)
    for algo_key in RUN_ALGOS:
        print(f"\n================ RUN REAL DATA WITH {algo_key.upper()} ================\n")
        env = os.environ.copy()
        env["REAL_DATA_ALGO"] = algo_key
        env.pop("REAL_DATA_ALGOS", None)
        subprocess.run([sys.executable, script_path], cwd=os.path.dirname(script_path),
                       env=env, check=True)
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
df_effort = df_effort0[['id', 'y', 'arm']]
df_exercise = df_exercise0[['id', 'y', 'arm']]
df_penn = df_penn0[['id', 'y', 'arm']]
df_walmart = df_walmart0[['id', 'y', 'arm']]

# --- 2. NEW PREPARATION FUNCTION ---

def prepare_real_experiment(df, n_sims):
    """
    Transforme un DataFrame en structure 3D pour la simulation.
    Structure : [simulation_index][arm_index][shuffled_observations]
    
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
            # Copy the original data
            rewards = grouped[arm_name].copy()
            
            # SHUFFLE: randomly permute reward order
            # This simulates a different arrival order for patients/participants in each simulation
            np.random.shuffle(rewards)
            
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
    print(f"Output directory: {output_root}\n")
    print(f"History record every: {HISTORY_RECORD_EVERY} step(s)\n")
    plt.close('all')
    
    # Scenario: 2 good arms (0, 1) and 2 bad ones (2, 3)

    n_sims = 1

    datasets = {
    "effort": (df_effort, "min_mean"),
    "exercise": (df_exercise, 53),
    "penn": (df_penn, 16),
    "walmart": (df_walmart, 3)
    }

    results = {}
    
    print("\n--- Traitement des données ---")
    for name, df in datasets.items():
        print(f"Préparation de {name}...")
        # Function call
        data_sim, arm_names = prepare_real_experiment(df[0], n_sims)
        control_arm = df[1]
        if control_arm == "min_mean":
            arm_means = [mean(arm_data) for arm_data in data_sim[0]]
            control_arm = int(np.argmin(arm_means))
            print(f"   -> Contrôle auto min mean: arm {control_arm} ({arm_names[control_arm]}) "
                  f"mean={arm_means[control_arm]:.4f}")
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
    # control_arm = min mean arm

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
    
    list_name=["penn", "exercise", "effort", "walmart"]
    num_graph=0
    for name_data in list_name:
        print("***********************name of the database treated:", name_data.upper(), "***********************")
        dataset_output_dir = output_root / name_data
        dataset_output_dir.mkdir(parents=True, exist_ok=True)

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
        init_choice = True
        mu_0_unif=mean(data_test[0][control_arm])
        print("mu_0 moyenne calcule", mu_0_unif)

        list_stat=[]
        for n in range(n_arms):
            mean_arm=round(mean(data_test[0][n]), 4)
            var_arm = round(variance(data_test[0][n]) if len(data_test[0][n]) > 1 else 0, 4)
            print("moyenne arm", n, ":", arm_test[n], "=", mean_arm, "var=", var_arm)
            list_stat.append([f"arm {n}", arm_test[n], mean_arm, var_arm, len(data_test[0][n])])

        
        sort_mean_desc = sorted(list_stat, key=lambda x: x[2], reverse=True)
        sort_var_desc = sorted(list_stat, key=lambda x: x[3], reverse=True)
        with open(dataset_output_dir / "classic_stats.txt", "w", encoding="utf-8") as f:
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

        print(f"--- ANALYSE LANCÉE (TYPE DE DONNÉES : {type_de_loi.upper()}) ---\n")

        arm_test_clean = [f"{i}: {arm_test[i][:15]}" for i in range(len(arm_test))]
        liste_vrai_positif=[]

        if type_de_loi == "normal":
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
        elif type_de_loi == "bernouilli":
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
        else:
            print("Erreur : La variable 'type_de_loi' doit être strictement égale à 'normal' ou 'bernouilli'.")            

        with open(dataset_output_dir / "classic_stats.txt", "r", encoding="utf-8") as f:
            contenu_existant = f.read()
        with open(dataset_output_dir / "classic_stats.txt", "w", encoding="utf-8") as f:
            f.write(str(liste_vrai_positif) + contenu_existant)

        
        is_true_mean=False
        # 1. Run Simulations
        if type_de_loi=="normal":
            pnb_unif, _, counts_unif_mean, counts_unif_list,  np_p_value_list_unif, np_p_value_mean_unif, l_pos_unif, discovery_unif = adaptative_algorithm_continuous.run_experiment(arm_test, mu_0_unif, delta, horizon, 'uniform', data_test, n_sims, control_arm, init_nb, init_choice, False, is_true_mean, return_discovery_times=True, history_record_every=HISTORY_RECORD_EVERY)
            pnb_unif_v, _, counts_unif_v_mean, counts_unif_v_list, np_p_value_list_unif_v, np_p_value_mean_unif_v, l_pos_unif_v, discovery_unif_v = adaptative_algorithm_continuous.run_experiment(arm_test, mu_0_unif, delta, horizon, 'uniform', data_test, n_sims, control_arm, init_nb, init_choice, True, is_true_mean, return_discovery_times=True, history_record_every=HISTORY_RECORD_EVERY)
            pnb_adapt, _, counts_adapt_mean, counts_adapt_list, np_p_value_list_adapt, np_p_value_mean_adapt, l_pos_adapt, discovery_adapt = adaptative_algorithm_continuous.run_experiment(arm_test, mu_0_unif, delta, horizon, 'adaptive', data_test, n_sims, control_arm, init_nb, init_choice, False, is_true_mean, return_discovery_times=True, history_record_every=HISTORY_RECORD_EVERY)
            pnb_adapt_v, _, counts_adapt_v_mean, counts_adapt_v_list, np_p_value_list_adapt_v, np_p_value_mean_adapt_v, l_pos_adapt_v, discovery_adapt_v = adaptative_algorithm_continuous.run_experiment(arm_test, mu_0_unif, delta, horizon, 'adaptive', data_test, n_sims, control_arm, init_nb, init_choice, True, is_true_mean, return_discovery_times=True, history_record_every=HISTORY_RECORD_EVERY)
        elif type_de_loi=="bernouilli":
            pnb_unif, _, counts_unif_mean, counts_unif_list,  np_p_value_list_unif, np_p_value_mean_unif, l_pos_unif, discovery_unif = adaptative_algorithm_binary.run_experiment(arm_test, mu_0_unif, delta, horizon, 'uniform', data_test, n_sims, control_arm, init_nb, init_choice, False, is_true_mean, return_discovery_times=True, history_record_every=HISTORY_RECORD_EVERY)
            pnb_unif_v, _, counts_unif_v_mean, counts_unif_v_list, np_p_value_list_unif_v, np_p_value_mean_unif_v, l_pos_unif_v, discovery_unif_v = adaptative_algorithm_binary.run_experiment(arm_test, mu_0_unif, delta, horizon, 'uniform', data_test, n_sims, control_arm, init_nb, init_choice, True, is_true_mean, return_discovery_times=True, history_record_every=HISTORY_RECORD_EVERY)
            pnb_adapt, _, counts_adapt_mean, counts_adapt_list, np_p_value_list_adapt, np_p_value_mean_adapt, l_pos_adapt, discovery_adapt = adaptative_algorithm_binary.run_experiment(arm_test, mu_0_unif, delta, horizon, 'adaptive', data_test, n_sims, control_arm, init_nb, init_choice, False, is_true_mean, return_discovery_times=True, history_record_every=HISTORY_RECORD_EVERY)
            pnb_adapt_v, _, counts_adapt_v_mean, counts_adapt_v_list, np_p_value_list_adapt_v, np_p_value_mean_adapt_v, l_pos_adapt_v, discovery_adapt_v = adaptative_algorithm_binary.run_experiment(arm_test, mu_0_unif, delta, horizon, 'adaptive', data_test, n_sims, control_arm, init_nb, init_choice, True, is_true_mean, return_discovery_times=True, history_record_every=HISTORY_RECORD_EVERY)
        

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
            y_ranks = [rank_by_arm[idx] for idx in positives_sorted]

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
                        y = rank_by_arm[arm_idx] + y_offset
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
                                label_positions[arm_idx] = (x, rank_by_arm[arm_idx])
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
                ax.set_ylim(max(y_ranks) + 0.5, min(y_ranks) - 0.5)

            axes[0].set_ylabel("Empirical mean rank (1 = highest)")
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
            fig.suptitle(f"Detected arms: discovery time vs empirical rank ({name_data})",
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
        plt.figure(1+num_graph*10, figsize=(10, 5))
        classic_color = '#ff7f0e'
        var_color = '#1f77b4'
        plt.plot(pnb_adapt, label='Adaptive', color=classic_color, linewidth=2, alpha=0.7)
        plt.plot(pnb_unif, label='Uniform', color=classic_color, linestyle='--', alpha=0.7)
        plt.plot(pnb_adapt_v, label='Adaptive_Var', color=var_color, linewidth=2, alpha=0.7)
        plt.plot(pnb_unif_v, label='Uniform_Var', color=var_color, linestyle='--', alpha=0.7)
        plt.axhline(y=1.0, color='gray', linestyle=':')
        plt.title("Discovery speed (pr)")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(dataset_output_dir / "figure1.png", dpi=300, bbox_inches="tight")
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
