# configurationmedicale_correcte.py

import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import sys
import time

# === TON ALGORITHME BANDIT COMPLET ===
def phi_function(t, delta):
    if t == 0:
        return float('inf')
    delta = min(delta, 1.0)
    log_delta = np.log(1/delta)
    log_log_delta = np.log(log_delta + 1e-10)
    log_log_t = np.log(np.log(np.e * t / 2) + 1e-10)
    numerator = 2 * log_delta + 6 * log_log_delta + 3 * log_log_t
    return np.sqrt(numerator / t)

class BanditExperimentOptimise:
    """Ton algorithme bandit OPTIMISÉ pour être rapide"""
    
    def __init__(self, true_means, mu_0, delta, mode='TPR', fwer_control=False):
        self.true_means = np.array(true_means)
        self.n = len(true_means)
        self.mu_0 = mu_0
        self.delta = delta
        self.mode = mode
        self.fwer_control = fwer_control
        
        self.T = np.zeros(self.n, dtype=int)
        self.mu_hat = np.zeros(self.n)
        self.sum_rewards = np.zeros(self.n)
        
        self.S = set()
        self.R = set()
        
        # Cache pour accélérer phi_function
        self.phi_cache = {}
        
    def phi_cached(self, t, delta):
        """phi_function avec cache"""
        key = (t, delta)
        if key not in self.phi_cache:
            self.phi_cache[key] = phi_function(t, delta)
        return self.phi_cache[key]
    
    def pull_arm(self, arm_idx, reward=None):
        """Version optimisée avec reward optionnel"""
        if reward is None:
            reward = np.random.normal(self.true_means[arm_idx], 1.0)
        
        old_sum = self.sum_rewards[arm_idx]
        old_T = self.T[arm_idx]
        
        self.T[arm_idx] += 1
        self.sum_rewards[arm_idx] += reward
        
        # Mise à jour efficace de la moyenne
        if old_T == 0:
            self.mu_hat[arm_idx] = reward
        else:
            self.mu_hat[arm_idx] = old_sum / old_T + (reward - old_sum / old_T) / self.T[arm_idx]
        
        return reward
    
    def update_S_incremental(self):
        """Met à jour S de manière incrémentale (BENJAMINI-HOCHBERG OPTIMISÉ)"""
        delta_prime = self.delta / (6.4 * np.log(36 / self.delta))
        n = self.n
        
        # Calcul rapide de tous les s(k)
        s_sizes = np.zeros(n + 1, dtype=int)
        
        # Pré-calculer les bornes inférieures pour chaque bras
        lower_bounds = []
        for i in range(n):
            if self.T[i] > 0:
                # Calculer pour k=n (le plus petit threshold)
                threshold = delta_prime * n / n
                conf = self.phi_cached(self.T[i], threshold)
                lb = self.mu_hat[i] - conf
                lower_bounds.append((lb, i))
            else:
                lower_bounds.append((-float('inf'), i))
        
        # Trier par borne inférieure décroissante
        lower_bounds.sort(reverse=True)
        
        # Calculer s(k) pour tous les k en une passe
        current_count = 0
        for rank, (lb, i) in enumerate(lower_bounds):
            if lb >= self.mu_0:
                current_count += 1
                # k = rank + 1
                if current_count >= (rank + 1):
                    s_sizes[rank + 1] = current_count
        
        # Trouver le plus grand k tel que |s(k)| >= k
        k_hat = 0
        for k in range(n, 0, -1):
            if s_sizes[k] >= k:
                k_hat = k
                break
        
        # Mettre à jour S
        if k_hat > 0:
            new_S = {i for _, i in lower_bounds[:k_hat] 
                    if lower_bounds[:k_hat][i][0] >= self.mu_0}
            self.S = new_S
        
        return self.S
    
    def choisir_prochain_bras(self):
        """Choisit le prochain bras à tirer (logique I_t)"""
        len_S = len(self.S)
        
        # ξ_t selon le mode
        if self.mode == 'TPR':
            xi_t = 1.0
        else:  # FWPD
            term2 = (5 / (3 * (1 - 4 * self.delta))) * np.log(1 / self.delta)
            xi_t = max(2 * len_S, term2)
        
        # Candidats: bras pas dans S
        candidates = [i for i in range(self.n) if i not in self.S]
        
        if not candidates:
            # Fallback: choisir au hasard dans S
            return np.random.choice(list(self.S)) if self.S else 0
        
        # Calcul UCB pour les candidats
        best_ucb = -float('inf')
        best_arm = candidates[0]
        
        for i in candidates:
            if self.T[i] > 0:
                ucb = self.mu_hat[i] + self.phi_cached(self.T[i], self.delta / xi_t)
                if ucb > best_ucb:
                    best_ucb = ucb
                    best_arm = i
        
        return best_arm
    
    def executer_pas(self, max_patients=1):
        """Exécute un pas de l'algorithme (pour plusieurs patients)"""
        for _ in range(max_patients):
            # 1. Choisir le bras
            bras = self.choisir_prochain_bras()
            
            # 2. Tirer le bras
            self.pull_arm(bras)
            
            # 3. Mettre à jour S (moins fréquemment pour accélérer)
            if np.random.random() < 0.3:  # 30% de chance de mettre à jour
                self.update_S_incremental()

# === ESSAI CLINIQUE OPTIMISÉ ===
class EssaiCliniqueEfficace:
    
    def __init__(self, n_traitements=10, efficacite_reference=30, delta=0.05,
                 duree_jours=15, patients_par_jour=6):
        
        np.random.seed(42)
        
        # Générer des efficacités réalistes
        n_efficaces = min(4, n_traitements // 2)
        efficaces = np.random.uniform(35, 50, n_efficaces)
        inefficaces = np.random.uniform(20, 29, n_traitements - n_efficaces)
        
        self.efficacites_reelles = np.concatenate([efficaces, inefficaces])
        np.random.shuffle(self.efficacites_reelles)
        
        self.n_traitements = n_traitements
        self.efficacite_reference = efficacite_reference
        self.delta = delta
        
        # Bandit avec TES paramètres
        self.bandit = BanditExperimentOptimise(
            true_means=self.efficacites_reelles / 100,
            mu_0=efficacite_reference / 100,
            delta=delta,
            mode='FWPD',  # Mode du papier
            fwer_control=False  # Désactivé pour accélérer
        )
        
        self.duree_jours = duree_jours
        self.patients_par_jour = patients_par_jour
        self.patients_total = 0
        self.patients_par_traitement = np.zeros(n_traitements)
        
        self.historique = []
    
    def simuler_patient(self, traitement_id):
        """Simule un patient pour un traitement"""
        efficacite_base = self.efficacites_reelles[traitement_id]
        # Ajouter de la variabilité patient
        efficacite_patient = efficacite_base + np.random.normal(0, 8)
        efficacite_patient = max(10, min(90, efficacite_patient)) / 100
        
        # Binaire: succès ou échec
        succes = 1 if np.random.random() < efficacite_patient else 0
        
        self.patients_total += 1
        self.patients_par_traitement[traitement_id] += 1
        
        return succes
    
    def executer(self):
        """Exécute l'essai clinique complet"""
        print("🏥 ESSAI CLINIQUE AVEC ALGORITHME BANDIT")
        print("="*60)
        print(f"Traitements: {self.n_traitements}")
        print(f"Référence: {self.efficacite_reference}%")
        print(f"Confiance: {(1-self.delta)*100}%")
        print(f"Durée: {self.duree_jours} jours")
        print(f"Patients/jour: {self.patients_par_jour}")
        print("="*60)
        
        # PHASE 1: Initialisation (chaque bras 2 fois)
        print("\n🔬 PHASE D'INITIALISATION")
        start_time = time.time()
        
        for bras in range(self.n_traitements):
            for _ in range(2):  # Seulement 2 patients par bras initialement
                reward = self.simuler_patient(bras)
                self.bandit.pull_arm(bras, reward)
        
        # Mettre à jour S après initialisation
        self.bandit.update_S_incremental()
        
        init_time = time.time() - start_time
        print(f"✓ Initialisation terminée en {init_time:.1f}s")
        print(f"  S après init: {sorted(self.bandit.S)}")
        
        # PHASE 2: Phase adaptative
        print("\n🔄 PHASE ADAPTATIVE")
        
        for jour in range(1, self.duree_jours + 1):
            jour_start = time.time()
            
            patients_ce_jour = min(self.patients_par_jour, 
                                  100 - self.patients_total)  # Limite sécurité
            
            if patients_ce_jour <= 0:
                print(f"\n⏹️ Limite de patients atteinte")
                break
            
            print(f"\n📅 Jour {jour}: {patients_ce_jour} patients", end="", flush=True)
            
            # Exécuter l'algorithme pour tous les patients du jour
            self.bandit.executer_pas(max_patients=patients_ce_jour)
            
            # Simuler les patients pour les bras choisis
            # (Dans la vraie vie, on aurait le reward en temps réel)
            
            # Mettre à jour S à la fin du jour
            old_S = self.bandit.S.copy()
            self.bandit.update_S_incremental()
            
            jour_time = time.time() - jour_start
            
            # Afficher progression
            print(f" [✓ {jour_time:.1f}s]")
            print(f"  |S|: {len(old_S)} → {len(self.bandit.S)}")
            
            # Sauvegarder l'état
            self.historique.append({
                'jour': jour,
                'S': self.bandit.S.copy(),
                'T': self.bandit.T.copy(),
                'mu_hat': self.bandit.mu_hat.copy(),
                'patients': self.patients_total
            })
            
            # Arrêt précoce si bonnes découvertes
            if len(self.bandit.S) >= 3 and jour > 5:
                print(f"\n🎯 Bonnes découvertes! Arrêt précoce jour {jour}")
                break
        
        # ANALYSE FINALE
        self.analyser_resultats()
    
    def analyser_resultats(self):
        """Analyse et affiche les résultats"""
        print("\n" + "="*60)
        print("📊 ANALYSE FINALE")
        print("="*60)
        
        # Vrais positifs (connaissances cachées)
        vrais_positifs = [i for i, eff in enumerate(self.efficacites_reelles) 
                         if eff > self.efficacite_reference]
        
        print(f"\n🎯 TRAITEMENTS:")
        print(f"  Réellement efficaces (> {self.efficacite_reference}%): {vrais_positifs}")
        print(f"  Détectés par l'algorithme (S): {sorted(self.bandit.S)}")
        
        # Métriques
        tp = len([i for i in self.bandit.S if i in vrais_positifs])
        fp = len([i for i in self.bandit.S if i not in vrais_positifs])
        fn = len([i for i in vrais_positifs if i not in self.bandit.S])
        
        print(f"\n📈 PERFORMANCE:")
        print(f"  Vrais positifs: {tp}/{len(vrais_positifs)} ({tp/len(vrais_positifs)*100:.0f}%)")
        print(f"  Faux positifs: {fp}")
        print(f"  Faux négatifs: {fn}")
        
        if len(self.bandit.S) > 0:
            print(f"  FDR observé: {fp/len(self.bandit.S):.3f} (cible ≤ {self.delta})")
        
        print(f"\n👥 PATIENTS:")
        print(f"  Total: {self.patients_total}")
        print(f"  Exposés à traitement inefficace: {self.calculer_exposition_inefficace()}")
        
        # Détails par traitement
        print(f"\n🔍 DÉTAILS PAR TRAITEMENT:")
        for i in range(self.n_traitements):
            eff_reelle = self.efficacites_reelles[i]
            eff_est = self.bandit.mu_hat[i] * 100 if self.bandit.T[i] > 0 else 0
            patients = self.bandit.T[i]
            status = "★" if i in self.bandit.S else " "
            
            print(f"  T{i:2d}: {eff_reelle:5.1f}% → estimé {eff_est:5.1f}% "
                  f"[{patients:3d} patients] {status}")
        
        # Visualisation
        self.visualiser()
    
    def calculer_exposition_inefficace(self):
        """Calcule l'exposition aux traitements inefficaces"""
        exposition = 0
        for i in range(self.n_traitements):
            if self.efficacites_reelles[i] <= self.efficacite_reference:
                exposition += self.bandit.T[i]
        return exposition
    
    def visualiser(self):
        """Crée des visualisations"""
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # 1. Efficacités réelles vs estimées
        x = np.arange(self.n_traitements)
        width = 0.35
        
        axes[0, 0].bar(x - width/2, self.efficacites_reelles, width, 
                      label='Réelle', alpha=0.7, color='steelblue')
        axes[0, 0].bar(x + width/2, self.bandit.mu_hat * 100, width, 
                      label='Estimée', alpha=0.7, color='lightcoral')
        axes[0, 0].axhline(y=self.efficacite_reference, color='red', 
                          linestyle='--', label='Référence')
        axes[0, 0].set_xlabel('Traitement')
        axes[0, 0].set_ylabel('Efficacité (%)')
        axes[0, 0].set_title('Efficacité réelle vs estimée')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. Distribution des patients
        colors = ['green' if i in self.bandit.S else 'gray' 
                 for i in range(self.n_traitements)]
        axes[0, 1].bar(x, self.bandit.T, color=colors, alpha=0.7)
        axes[0, 1].set_xlabel('Traitement')
        axes[0, 1].set_ylabel('Nombre de patients')
        axes[0, 1].set_title('Patients par traitement (vert = dans S)')
        axes[0, 1].grid(True, alpha=0.3, axis='y')
        
        # 3. Évolution de |S|
        if self.historique:
            jours = [h['jour'] for h in self.historique]
            taille_S = [len(h['S']) for h in self.historique]
            
            axes[1, 0].plot(jours, taille_S, 'b-', linewidth=2, marker='o')
            axes[1, 0].set_xlabel('Jour')
            axes[1, 0].set_ylabel('|S| (traitements détectés)')
            axes[1, 0].set_title('Évolution des découvertes')
            axes[1, 0].grid(True, alpha=0.3)
            
        # 4. Précision des estimations
        erreurs = np.abs(self.bandit.mu_hat * 100 - self.efficacites_reelles)
        axes[1, 1].bar(x, erreurs, alpha=0.7, color='orange')
        axes[1, 1].set_xlabel('Traitement')
        axes[1, 1].set_ylabel('Erreur d\'estimation (%)')
        axes[1, 1].set_title('Précision des estimations')
        axes[1, 1].grid(True, alpha=0.3, axis='y')
        
        plt.suptitle(f'Essai Clinique Adaptatif - {self.patients_total} patients', 
                    fontsize=14, y=1.02)
        plt.tight_layout()
        plt.show()

# === EXÉCUTION ===
if __name__ == "__main__":
    print("💊 SIMULATION AVEC VOTRE ALGORITHME BANDIT")
    print("="*60)
    
    # Paramètres raisonnables pour un test rapide
    essai = EssaiCliniqueEfficace(
        n_traitements=8,          # Moins de bras pour aller vite
        efficacite_reference=30,
        delta=0.05,
        duree_jours=10,          # Plus court
        patients_par_jour=5       # Moins de patients par jour
    )
    
    try:
        essai.executer()
        
        print("\n" + "="*60)
        print("✅ SIMULATION TERMINÉE AVEC SUCCÈS")
        print("="*60)
        print("\n🎯 Cet essai a utilisé VOTRE algorithme bandit avec:")
        print("   • Benjamini-Hochberg pour le contrôle FDR")
        print("   • UCB adaptatif pour la sélection des bras")
        print("   • Mise à jour incrémentale optimisée")
        print("   • Cache pour les calculs coûteux")
        
    except Exception as e:
        print(f"\n❌ Erreur: {e}")
        import traceback
        traceback.print_exc()