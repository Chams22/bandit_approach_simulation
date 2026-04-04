import numpy as np
import math

def phi_function(t, delta):
    """
    Fonction d'intervalle de confiance "Anytime".
    Basée sur l'équation citée dans la section 'Experiments' du papier (Source 246).
    """
    if t == 0:
        return float('inf')
    
    # Évite les erreurs log(<=0)
    delta = min(delta, 1.0)
    
    # Terme numérateur complexe de la borne lil'UCB
    log_delta = np.log(1/delta)
    log_log_delta = np.log(log_delta + 1e-10) # petit epsilon pour stabilité
    log_log_t = np.log(np.log(np.e * t / 2) + 1e-10)
    
    numerator = 2 * log_delta + 6 * log_log_delta + 3 * log_log_t
    return np.sqrt(numerator / t)

class BanditExperiment:
    def __init__(self, true_means, mu_0, delta, mode='TPR', fwer_control=False):
        """
        true_means: Liste des vraies moyennes (inconnues de l'algo, pour la simulation)
        mu_0: Le seuil de référence (baseline)
        delta: Niveau de confiance
        mode: 'TPR' ou 'FWPD' (définit xi_t et nu_t)
        fwer_control: Booléen, activer ou non le contrôle FWER (Set R_t)
        """
        self.true_means = np.array(true_means)
        self.n = len(true_means)
        self.mu_0 = mu_0
        self.delta = delta
        self.mode = mode
        self.fwer_control = fwer_control
        
        # Stats internes
        self.T = np.zeros(self.n)      # Nombre de tirages par bras
        self.mu_hat = np.zeros(self.n) # Moyenne empirique
        self.sum_rewards = np.zeros(self.n)
        
        # Ensembles S (FDR) et R (FWER)
        self.S = set()
        self.R = set()
        
        # Historique pour analyse
        self.history = []

    def pull_arm(self, arm_idx):
        """Simule un tirage (Reward gaussien N(mu, 1))"""
        reward = np.random.normal(self.true_means[arm_idx], 1.0)
        
        self.T[arm_idx] += 1
        self.sum_rewards[arm_idx] += reward
        self.mu_hat[arm_idx] = self.sum_rewards[arm_idx] / self.T[arm_idx]
        return reward

    def run(self, max_steps=5000):
        # --- INITIALIZATION ---
        # "Pull each arm i in [n] once"
        for i in range(self.n):
            self.pull_arm(i)
        
        # Boucle principale (commence à n+1)
        for t in range(self.n + 1, max_steps + 1):
            
            # --- DEFINITION DE xi_t et nu_t ---
            len_S = len(self.S)
            
            if self.mode == 'TPR':
                xi_t = 1.0
                nu_t = 1.0
            elif self.mode == 'FWPD':
                # Formule complexe de l'image pour FWPD
                term2 = (5 / (3 * (1 - 4 * self.delta))) * np.log(1 / self.delta)
                xi_t = max(2 * len_S, term2)
                nu_t = max(len_S, 1)
            else:
                raise ValueError("Mode must be TPR or FWPD")

            # --- CHOIX DU BRAS I_t (Pour S_t) ---
            # I_t = argmax des bras PAS dans S_t
            candidates_S = [i for i in range(self.n) if i not in self.S]
            
            if not candidates_S:
                # Tous les bras sont dans S, on s'arrête ou on continue l'exploitation pure
                # Pour éviter crash, on prend un bras au hasard dans S si S est plein
                I_t = np.random.choice(list(self.S)) if self.S else 0
            else:
                ucb_values = []
                for i in candidates_S:
                    val = self.mu_hat[i] + phi_function(self.T[i], self.delta / xi_t)
                    ucb_values.append(val)
                I_t = candidates_S[np.argmax(ucb_values)]
            
            # Action: Tirer le bras I_t
            self.pull_arm(I_t)

            # --- MISE A JOUR DE S_t (Benjamini-Hochberg) ---
            # delta' calculation
            delta_prime = self.delta / (6.4 * np.log(36 / self.delta))
            
            # Calcul de k_hat
            k_hat = 0
            
            # On cherche le max k tel que |s(k)| >= k
            # Note: C'est l'étape coûteuse en calcul, optimisable mais faite ici "by the book"
            for k in range(self.n, 0, -1):
                # Calcul de l'ensemble s(k)
                threshold_sk = delta_prime * k / self.n
                s_k_size = 0
                temp_s_k = []
                
                for i in range(self.n):
                    conf = phi_function(self.T[i], threshold_sk)
                    if (self.mu_hat[i] - conf) >= self.mu_0:
                        s_k_size += 1
                        temp_s_k.append(i)
                
                if s_k_size >= k:
                    k_hat = k
                    self.S = set(temp_s_k) # Update S_{t+1}
                    break
            
            # --- FWER LOGIC (Optional) ---
            if self.fwer_control and len(self.S) > 0:
                # Candidats pour J_t: dans S mais pas dans R
                candidates_R = [i for i in self.S if i not in self.R]
                
                if candidates_R:
                    # Choix du bras J_t
                    ucb_R_values = []
                    for i in candidates_R:
                        val = self.mu_hat[i] + phi_function(self.T[i], self.delta / nu_t)
                        ucb_R_values.append(val)
                    J_t = candidates_R[np.argmax(ucb_R_values)]
                    
                    # Action: Tirer le bras J_t (Sampling supplémentaire)
                    self.pull_arm(J_t)
                    
                    # Mise à jour de R_t (Bonferroni-like)
                    # Calcul de Chi_t
                    term_log = np.log(5 * np.log2(self.n / delta_prime) / delta_prime)
                    term_complex = (4 * (1 + 4 * delta_prime) / 3) * term_log
                    term_S = (1 - 2 * delta_prime * (1 + 4 * delta_prime)) * len(self.S)
                    
                    chi_t = self.n - term_S + term_complex
                                        
                    # Update R_{t+1}
                    new_discoveries = []
                    for i in self.S:
                        conf = phi_function(self.T[i], self.delta / chi_t)
                        if (self.mu_hat[i] - conf) >= self.mu_0:
                            new_discoveries.append(i)
                    
                    self.R.update(new_discoveries)

            # Logging
            self.history.append({
                'step': t,
                'S_size': len(self.S),
                'R_size': len(self.R) if self.fwer_control else 0
            })

        return self.S, self.R

# --- EXEMPLE D'UTILISATION ---

# 1. Configuration de l'expérience
n_arms = 20
# On crée des moyennes: les 5 premiers sont "bons" (0.5), les autres nuls (0)
true_means = [0.5] * 5 + [0.0] * 15 
threshold = 0.1 # mu_0
delta = 0.05

print(f"Lancement de la simulation avec {n_arms} bras...")
print(f"Vrais positifs attendus (indices): 0, 1, 2, 3, 4")

# 2. Initialisation
algo = BanditExperiment(
    true_means=true_means,
    mu_0=threshold,
    delta=delta,
    mode='TPR', 
    fwer_control=True
)

# 3. Exécution
S_final, R_final = algo.run(max_steps=3000)

print("-" * 30)
print(f"Résultats après 3000 tirages:")
print(f"Découvertes FDR (Set S): {sorted(list(S_final))}")
print(f"Découvertes FWER (Set R): {sorted(list(R_final))}")