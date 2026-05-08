import pandas as pd
import numpy as np
import os
import importlib
import usable_adaptative_algorithm_fusion_binary_v2 as usable_adaptative_algorithm_fusion_binary
importlib.reload(usable_adaptative_algorithm_fusion_binary)
import usable_adaptative_algorithm_fusion_continuous_v2 as usable_adaptative_algorithm_fusion_continuous
importlib.reload(usable_adaptative_algorithm_fusion_continuous)
import matplotlib.pyplot as plt
from statistics import mean, variance
import re




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
    plt.close('all')
    
    # Scenario: 2 good arms (0, 1) and 2 bad ones (2, 3)

    n_sims = 1

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

    # name_data="penn"
    # name_data="effort"
    # name_data="walmart"
    # name_data="exercise"
    
    list_name=["penn", "exercise", "effort", "walmart"]
    num_graph=0
    for name_data in list_name:
        print("***********************name of the database treated:", name_data.upper(), "***********************")

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
        with open(git_root / f"figure_real_data/{name_data}/classic_stats.txt", "w", encoding="utf-8") as f:
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
        # Choisissez "normal" pour des scores continus (0 à 10)
        # Choisissez "bernouilli" pour du binaire (Douleur absente/présente)
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

            # --- TESTS STATISTIQUES ---
            stat_f, p_value_anova = stats.f_oneway(*donnees)
            print("=== ÉTAPE 1 : TEST GLOBAL (ANOVA) ===")
            print(f"P-value de l'ANOVA : {p_value_anova:.5f}")

            # Dunnett toujours calculé (utilisé aussi pour le graphe)
            res_dunnett = stats.dunnett(*groupes_traitements, control=groupe_controle)

            # Mapping : indice original du bras → p-value Dunnett
            indices_traitements = [i for i in range(n_arms) if i != control_arm]
            dunnett_pvals = dict(zip(indices_traitements, res_dunnett.pvalue))
            liste_vrai_positif = [idx for idx, p_val in dunnett_pvals.items() if p_val < 0.05]


            if p_value_anova < 0.05:
                print("-> Résultat significatif.\n")
                print("=== ÉTAPE 2 : TESTS POST-HOC (Test de Dunnett) ===")
                for i, p_val in enumerate(res_dunnett.pvalue):
                    nom = noms_traitements[i]
                    moyenne_traitement = np.mean(groupes_traitements[i])
                    moyenne_controle = np.mean(groupe_controle)
                    significatif = "Oui" if p_val < 0.05 else "Non"
                    effet = "Baisse" if moyenne_traitement < moyenne_controle else "Hausse"
                    print(f"Contrôle vs {nom} | P-value = {p_val:.4f} | Significatif : {significatif} ({effet})")
            else:
                print("-> Résultat non significatif.")

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

            # Couleurs basées sur Dunnett
            sig_flags = []
            for idx_orig in ordre:
                if idx_orig == control_arm:
                    sig_flags.append('control')
                else:
                    p_val = dunnett_pvals[idx_orig]
                    sig_flags.append('sig' if p_val < 0.05 else 'ns')

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

            # Annotations avec Dunnett
            for idx_tri, idx_orig in enumerate(ordre):
                m = means_tri[idx_tri]
                ci = cis_tri[idx_tri]
                n = n_obs_tri[idx_tri]

                if idx_orig == control_arm:
                    label = f'{m:.2f}  (n={n})'
                else:
                    p_val = dunnett_pvals[idx_orig]
                    sig = '***' if p_val < 0.001 else '**' if p_val < 0.01 else '*' if p_val < 0.05 else ''
                    label = f'{m:.2f}  (n={n}) {sig}'

                ax.text(m + ci + 0.01 * max(means), idx_tri, label,
                        va='center', fontsize=7)

            ax.set_yticks(y_pos)
            ax.set_yticklabels(labels_tri, fontsize=8)
            ax.set_xlabel("Moyenne ± IC 95%")
            ax.set_title(f"Comparaison des bras : {name_data}\n"
                        "ANOVA + post-hoc Dunnett | IC 95% (moyenne ± 1.96×SEM)",
                        fontsize=14, fontweight='bold')
            ax.legend(loc='lower right')
            ax.grid(axis='x', linestyle='--', alpha=0.7, zorder=1)

            ax.text(0.99, 0.02, '* p<0.05  ** p<0.01  *** p<0.001 (Dunnett)',
                    transform=ax.transAxes, fontsize=7, ha='right', style='italic', color='gray')

            plt.tight_layout()
            plt.savefig(git_root / f"figure_real_data/{name_data}/figure0.png", dpi=300, bbox_inches="tight")
            plt.close()
        elif type_de_loi == "bernouilli":
            # ==========================================
            # CAS 1 : DONNÉES BINAIRES (penn et walmart = incitation à prendre un vaccin)
            # ==========================================
            # --- TRANSFORMATION DES DONNÉES ---
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

            # Recalculer l'index du contrôle dans le tableau filtré
            if control_arm in indices_valides:
                control_arm_filtre = indices_valides.index(control_arm)
            else:
                print("⚠️  Le bras de contrôle a été filtré !")
                control_arm_filtre = 0

            noms_tous_groupes = [arm_test_clean[i] for i in indices_valides]
            noms_traitements = [arm_test_clean[i] for i in indices_valides if i != control_arm]

            # --- TESTS STATISTIQUES ---
            stat_chi2, p_val_globale, dof, expected = stats.chi2_contingency(tableau_contingence)
            print("=== ÉTAPE 1 : TEST GLOBAL (Chi-deux) ===")
            print(f"P-value globale : {p_val_globale:.5f}")

            if p_val_globale < 0.05:
                print("-> Résultat significatif.\n")
                print("=== ÉTAPE 2 : TESTS POST-HOC (Fisher exact + Bonferroni) ===")

                ligne_controle = tableau_contingence[control_arm_filtre]
                lignes_traitements = (tableau_contingence[:control_arm_filtre]
                                    + tableau_contingence[control_arm_filtre+1:])
                nombre_de_comparaisons = len(lignes_traitements)

                for i, ligne_traitement in enumerate(lignes_traitements):
                    nom = noms_traitements[i]
                    sous_tableau = [ligne_controle, ligne_traitement]

                    stat_odds, p_val_brute = stats.fisher_exact(sous_tableau)
                    p_val_corrigee = min(p_val_brute * nombre_de_comparaisons, 1.0)

                    total_controle = sum(ligne_controle)
                    total_trait = sum(ligne_traitement)
                    pct_controle = (ligne_controle[0] / total_controle) * 100 if total_controle > 0 else 0
                    pct_trait = (ligne_traitement[0] / total_trait) * 100 if total_trait > 0 else 0

                    significatif = "Oui" if p_val_corrigee < 0.05 else "Non"
                    print(f"Contrôle ({pct_controle:.0f}%) vs {nom} ({pct_trait:.0f}%) "
                        f"| P-val = {p_val_corrigee:.4f} | Significatif : {significatif}")
            else:
                print("-> Résultat non significatif.")
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
            # Pré-calcul de la significativité pour les couleurs
            sig_flags = []
            for i in range(len(proportions)):
                if i == control_arm_filtre:
                    sig_flags.append('control')
                else:
                    sous_tableau = [tableau_contingence[control_arm_filtre],
                                   tableau_contingence[i]]
                    _, p_val = stats.fisher_exact(sous_tableau)
                    p_val_corr = min(p_val * (len(proportions) - 1), 1.0)
                    sig_flags.append('sig' if p_val_corr < 0.05 else 'ns')
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
                    xerr=list(zip(*cis)),  # asymétrique (bas, haut)
                    color=couleurs, edgecolor='black', capsize=3, zorder=2, height=0.6)

            ax.axvline(x=prop_controle, color='red', linestyle='--',
                       label=f'Contrôle ({prop_controle:.1%})')
            
            # Annotation : proportion + n + significativité
            for i, (p, n) in enumerate(zip(proportions, n_obs)):
                if sig_flags[i] == 'control':
                    label = f'{p:.1%}  (n={n})'
                else:
                    sous_tableau = [tableau_contingence[control_arm_filtre],
                                   tableau_contingence[i]]
                    _, p_val = stats.fisher_exact(sous_tableau)
                    p_val_corr = min(p_val * (len(proportions) - 1), 1.0)
                    sig = '***' if p_val_corr < 0.001 else '**' if p_val_corr < 0.01 else '*' if p_val_corr < 0.05 else ''
                    label = f'{p:.1%}  (n={n}) {sig}'

                ax.text(p + cis[i][1] + 0.005, i, label, va='center', fontsize=8)

            ax.set_yticks(y_pos)
            ax.set_yticklabels(labels_courts, fontsize=8)
            ax.set_xlabel("Proportion de succès ± IC 95%")
            ax.set_title(f"Proportion de succès par traitement : {name_data}\n"
             "Chi-deux + post-hoc Fisher exact (Bonferroni) | IC 95% Wilson",
             fontsize=14, fontweight='bold')    
            ax.legend(loc='lower right')
            ax.grid(axis='x', linestyle='--', alpha=0.7, zorder=1)

            # Légende des étoiles
            ax.text(0.99, 0.02, '* p<0.05  ** p<0.01  *** p<0.001 (Fisher + Bonferroni)',
                    transform=ax.transAxes, fontsize=7, ha='right', style='italic', color='gray')

            plt.tight_layout()
            plt.savefig(git_root / f"figure_real_data/{name_data}/figure0.png", dpi=300, bbox_inches="tight")
            plt.close()
        else:
            print("Erreur : La variable 'type_de_loi' doit être strictement égale à 'normal' ou 'bernouilli'.")            

        with open(git_root / f"figure_real_data/{name_data}/classic_stats.txt", "r", encoding="utf-8") as f:
            contenu_existant = f.read()
        with open(git_root / f"figure_real_data/{name_data}/classic_stats.txt", "w", encoding="utf-8") as f:
            f.write(str(liste_vrai_positif) + contenu_existant)

        
        is_true_mean=False
        # 1. Run Simulations
        if type_de_loi=="normal":
            pnb_unif, _, counts_unif_mean, counts_unif_list,  np_p_value_list_unif, np_p_value_mean_unif, l_pos_unif = usable_adaptative_algorithm_fusion_continuous.run_experiment(arm_test, mu_0_unif, delta, horizon, 'uniform', data_test, n_sims, control_arm, init_nb, init_choice, False, is_true_mean)
            pnb_adapt, _, counts_adapt_mean, counts_adapt_list, np_p_value_list_adapt, np_p_value_mean_adapt, l_pos_adapt = usable_adaptative_algorithm_fusion_continuous.run_experiment(arm_test, mu_0_unif, delta, horizon, 'adaptive', data_test, n_sims, control_arm, init_nb, init_choice, False, is_true_mean)
            pnb_adapt_v, _, counts_adapt_v_mean, counts_adapt_v_list, np_p_value_list_adapt_v, np_p_value_mean_adapt_v, l_pos_adapt_v = usable_adaptative_algorithm_fusion_continuous.run_experiment(arm_test, mu_0_unif, delta, horizon, 'adaptive', data_test, n_sims, control_arm, init_nb, init_choice, True, is_true_mean)
        elif type_de_loi=="bernouilli":
            pnb_unif, _, counts_unif_mean, counts_unif_list,  np_p_value_list_unif, np_p_value_mean_unif, l_pos_unif = usable_adaptative_algorithm_fusion_binary.run_experiment(arm_test, mu_0_unif, delta, horizon, 'uniform', data_test, n_sims, control_arm, init_nb, init_choice, False, is_true_mean)
            pnb_adapt, _, counts_adapt_mean, counts_adapt_list, np_p_value_list_adapt, np_p_value_mean_adapt, l_pos_adapt = usable_adaptative_algorithm_fusion_binary.run_experiment(arm_test, mu_0_unif, delta, horizon, 'adaptive', data_test, n_sims, control_arm, init_nb, init_choice, False, is_true_mean)
            pnb_adapt_v, _, counts_adapt_v_mean, counts_adapt_v_list, np_p_value_list_adapt_v, np_p_value_mean_adapt_v, l_pos_adapt_v = usable_adaptative_algorithm_fusion_binary.run_experiment(arm_test, mu_0_unif, delta, horizon, 'adaptive', data_test, n_sims, control_arm, init_nb, init_choice, True, is_true_mean)
        

        with open(git_root / f"figure_real_data/{name_data}/resultats.txt", "w", encoding="utf-8") as f:
            f.write("List of the positive arm detected\n\n")
            f.write("   UNIF\n")
            for i, element in enumerate(l_pos_unif, 1):
                f.write(f"{i}. {element}\n")
            f.write("   ADAPT\n")
            for i, element in enumerate(l_pos_adapt, 1):
                f.write(f"{i}. {element}\n")
            f.write("   ADAPT VAR\n")
            for i, element in enumerate(l_pos_adapt_v, 1):
                f.write(f"{i}. {element}\n")

        print("pos unif:", l_pos_unif)
        print("pos adapt:", l_pos_adapt)
        print("pos adapt v:", l_pos_adapt_v)

        

        with open(git_root / f"figure_real_data/{name_data}/resultats.txt", "r", encoding="utf-8") as f:
            contenu = f.read()

        # Regex : capture le nom de la méthode et le contenu entre {}
        pattern = r'(UNIF|ADAPT VAR|ADAPT)\s+\d+\.\s+\{([^}]+)\}'
        matches = re.findall(pattern, contenu)
        print(matches)

        resultats = {}
        print(matches)
        if matches:
            for nom, nombres in matches:
                resultats[nom] = set(int(x.strip()) for x in nombres.split(','))

        liste_unif = resultats.get('UNIF', set())
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
            plt.savefig(git_root / f"figure_real_data/{name_data}/figure6.png", dpi=300, bbox_inches="tight")
            plt.close()

        # Appel
        detectes_list = [(liste_unif, "unif"), (liste_adapt, "adapt"), (liste_adapt_var, "adapt var")]
        plot_detection_comparison(liste_vrai_positif, detectes_list, range(len(arm_test)), arm_test_clean, name_data)

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
        plt.close()


        # --- PLOT 2: PULL EVOLUTION ---
        import numpy as np
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 3, figsize=(18, 6))

        # Trouver les index des bras les plus tirés à la fin dans l'algo adaptatif
        final_pulls = counts_adapt_mean[-1, :]
        # Trie les index pour avoir les plus grands à la fin, on prend les 5 derniers
        top_arms_idx = np.argsort(final_pulls)[-5:] 

        # On crée une palette de couleurs distinctes pour les top bras
        colors = plt.cm.tab10.colors 

        for subplot_idx, data_mean in enumerate([counts_unif_mean, counts_adapt_mean, counts_adapt_v_mean]):
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
                    label = "_nolegend_" # Ignore ce bras dans la légende
                    
                ax.plot(data_mean[:, arm_idx], label=label, linewidth=linewidth, 
                        linestyle=linestyle, color=color, alpha=alpha)
            
            ax.set_xlabel("Time (t)")
            ax.grid(True, alpha=0.3)

        axes[0].set_title("Uniform: Number of pulls")
        axes[0].set_ylabel("Number of pulls ($T_i(t)$)")
        axes[1].set_title("Adaptive: Number of pulls")
        axes[2].set_title("Adaptive VAR: Number of pulls")

        # Une petite légende propre avec uniquement les bras importants
        handles, labels = axes[2].get_legend_handles_labels()
        fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 0.0), ncol=6)

        plt.tight_layout()
        plt.savefig(git_root / f"figure_real_data/{name_data}/figure2_clean.png", dpi=300, bbox_inches="tight")
        plt.close()

        # --- PLOT 3: PULL EVOLUTION (SPAGHETTI PLOT) ---
        plt.figure(3+num_graph*10, figsize=(14, 7))
        plt.title(f"Adaptive: Number of pulls per arm ({n_sims} simulations)", fontsize=14)

        # 1. Identifier les bras à mettre en valeur (ex: les 5 plus tirés à la fin)
        final_pulls = counts_adapt_mean[-1, :]
        top_arms_idx = np.argsort(final_pulls)[-5:] 
        colors = plt.cm.tab10.colors
        color_counter = 0

        for arm_idx in range(n_arms):
            is_control = (arm_test[arm_idx] == 'control')
            is_top = (arm_idx in top_arms_idx)
            
            # Définir le style selon l'importance du bras
            if is_top or is_control:
                base_color = 'black' if is_control else colors[color_counter % len(colors)]
                linestyle = '--' if is_control else '-'
                mean_linewidth = 2.5
                sim_alpha = 0.15 # Les simulations individuelles restent discrètes
                label = f"Arm {arm_idx} (mu={arm_test[arm_idx][0:4]}) {'[Ctrl]' if is_control else '[Top]'}"
                if not is_control: color_counter += 1
            else:
                base_color = 'gray'
                linestyle = '-'
                mean_linewidth = 1.0
                sim_alpha = 0.02 # Quasi-transparent pour les bras rejetés
                label = "_nolegend_"

            # Tracer les simulations individuelles (spaghetti)
            for sim_counts in counts_adapt_list:
                plt.plot(sim_counts[:, arm_idx], color=base_color, alpha=sim_alpha, 
                        linewidth=0.5, linestyle=linestyle)

            # Tracer la moyenne par-dessus
            plt.plot(counts_adapt_mean[:, arm_idx], label=label, color=base_color, 
                    linewidth=mean_linewidth, linestyle=linestyle)

        plt.xlabel("Time (t)", fontsize=12)
        plt.ylabel("Number of pulls ($T_i(t)$)", fontsize=12)
        plt.grid(True, alpha=0.3)

        # Légende épurée
        plt.legend(loc='upper left', fontsize=10, framealpha=0.9)

        plt.tight_layout()
        plt.savefig(git_root / f"figure_real_data/{name_data}/figure3.png", dpi=300, bbox_inches="tight")
        plt.close()

        # --- PLOT 3 VAR: PULL EVOLUTION (SPAGHETTI PLOT) ---
        plt.figure(6+num_graph*10, figsize=(14, 7))
        plt.title(f"Adaptive VAR: Number of pulls per arm ({n_sims} simulations)", fontsize=14)

        # 1. Identifier les bras à mettre en valeur pour la variante VAR
        # On utilise bien counts_adapt_v_mean ici
        final_pulls_v = counts_adapt_v_mean[-1, :]
        top_arms_idx_v = np.argsort(final_pulls_v)[-5:] 
        colors = plt.cm.tab10.colors
        color_counter = 0

        for arm_idx in range(n_arms):
            is_control = (arm_test[arm_idx] == 'control')
            is_top = (arm_idx in top_arms_idx_v)
            
            # Définir le style selon l'importance du bras
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
                sim_alpha = 0.02 # Quasi-transparent pour éviter le bruit visuel
                label = "_nolegend_"

            # Tracer les simulations individuelles (spaghetti) depuis la liste VAR
            for sim_counts in counts_adapt_v_list:
                plt.plot(sim_counts[:, arm_idx], color=base_color, alpha=sim_alpha, 
                        linewidth=0.5, linestyle=linestyle)

            # Tracer la moyenne par-dessus
            plt.plot(counts_adapt_v_mean[:, arm_idx], label=label, color=base_color, 
                    linewidth=mean_linewidth, linestyle=linestyle)

        plt.xlabel("Time (t)", fontsize=12)
        plt.ylabel("Number of pulls ($T_i(t)$)", fontsize=12)
        plt.grid(True, alpha=0.3)

        # Légende épurée
        plt.legend(loc='upper left', fontsize=10, framealpha=0.9)

        print("Displaying Adaptive VAR plots...")
        plt.tight_layout()
        plt.savefig(git_root / f"figure_real_data/{name_data}/figure3var.png", dpi=300, bbox_inches="tight")
        plt.close()

        # --- PLOT 4: P-VALUES ---
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        fig.suptitle("Evolution of P-values by iteration and arm", fontsize=16)

        datasets = [
            ("Uniform", np_p_value_mean_unif),
            ("Adaptive", np_p_value_mean_adapt),
            ("Adaptive VAR", np_p_value_mean_adapt_v)
        ]

        # Définir ton seuil de confiance (modifie cette variable si besoin)
        delta_threshold = 0.05 

        # On réutilise les top_arms pour garder une cohérence de couleurs avec le Plot 3
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
                    
                ax.plot(data[:, arm_idx], label=label, color=color, linewidth=linewidth, 
                        linestyle=linestyle, alpha=alpha)
            
            # LE CHANGEMENT LE PLUS IMPORTANT : Échelle logarithmique
            ax.set_yscale('log')
            # Optionnel : inverser l'axe Y pour que la "découverte" (p-value qui chute) aille vers le haut
            # ax.invert_yaxis() 
            
            # Ligne horizontale pour le seuil
            ax.axhline(y=delta_threshold, color='red', linestyle=':', linewidth=2, 
                    label=f'Threshold ($\\delta={delta_threshold}$)')
            
            ax.set_xlabel("Time (t)")
            ax.set_ylabel("P-value (Log Scale)")
            ax.grid(True, which="both", ls="-", alpha=0.2) # Grille adaptée au log

        # Légende unique en bas
        handles, labels = axes[1].get_legend_handles_labels()
        # On utilise un dict pour enlever les doublons potentiels (comme le seuil)
        by_label = dict(zip(labels, handles))
        fig.legend(by_label.values(), by_label.keys(), loc='lower center', 
                bbox_to_anchor=(0.5, -0.15), ncol=6, fontsize='small')

        plt.tight_layout()
        fig.subplots_adjust(bottom=0.25) # Place pour la légende

        plt.savefig(git_root / f"figure_real_data/{name_data}/figure4.png", dpi=300, bbox_inches="tight")
        plt.close()

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
        plt.close()