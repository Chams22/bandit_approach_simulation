# Configuration médicale
n_traitements = 20

# Efficacité réelle des traitements (inconnue en pratique)
# Les 5 premiers sont vraiment efficaces, les autres non
efficacites_reelles = [45, 42, 38, 35, 33] + [25, 28, 22, 27, 24, 
                                             26, 23, 29, 21, 25,
                                             27, 24, 26, 22, 28]  # en %

# Traitement de référence (standard de soins)
efficacite_reference = 30  # %

# Niveau de confiance requis pour publication
niveau_confiance = 0.05  # 95% de confiance

# Traduction directe :
# µ₀ = efficacité du traitement de référence
# true_means = efficacités réelles des nouveaux traitements
# δ = 1 - niveau de confiance

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime, timedelta

class EssaiCliniqueBandit:
    """
    Simulation d'un essai clinique utilisant l'algorithme bandit adaptatif
    """
    
    def __init__(self, n_traitements=20, efficacite_reference=30, delta=0.05,
                 duree_max_jours=180, patients_par_jour=10):
        """
        n_traitements: nombre de nouveaux traitements à tester
        efficacite_reference: efficacité du traitement standard (%)
        delta: niveau de signification (1 - confiance)
        duree_max_jours: durée maximale de l'étude
        patients_par_jour: nombre de patients pouvant être recrutés par jour
        """
        # Générer aléatoirement les efficacités réelles
        np.random.seed(42)
        n_efficaces = 5  # 5 traitements vraiment efficaces
        efficaces = np.random.uniform(32, 50, n_efficaces)  # 32-50%
        inefficaces = np.random.uniform(20, 29, n_traitements - n_efficaces)  # 20-29%
        
        self.efficacites_reelles = np.concatenate([efficaces, inefficaces])
        np.random.shuffle(self.efficacites_reelles)  # Mélanger
        
        self.n_traitements = n_traitements
        self.efficacite_reference = efficacite_reference
        self.delta = delta
        self.duree_max_jours = duree_max_jours
        self.patients_par_jour = patients_par_jour
        
        # Initialiser l'algorithme bandit
        self.bandit = BanditExperiment(
            true_means=self.efficacites_reelles / 100,  # Convertir en proportion
            mu_0=efficacite_reference / 100,
            delta=delta,
            mode='FWPD',  # Mode conservateur pour essai clinique
            fwer_control=True  # Contrôle strict des faux positifs
        )
        
        # Statistiques médicales
        self.patients_total = 0
        self.patients_par_traitement = np.zeros(n_traitements)
        self.efficacite_estimee = np.zeros(n_traitements)
        self.date_debut = datetime.now()
        self.historique_patients = []
        self.resultats_intermediaires = []
        
        # Coûts (en milliers d'euros)
        self.cout_par_patient = 50  # Coût par patient inclus
        self.cout_par_traitement = 200  # Coût pour administrer un traitement
        
    def recruter_patient(self, traitement_id, jour):
        """Simule le recrutement et le suivi d'un patient"""
        # Effet aléatoire du traitement sur ce patient
        efficacite_base = self.efficacites_reelles[traitement_id]
        variabilite = np.random.normal(0, 10)  # Variabilité inter-patient
        efficacite_patient = max(0, min(100, efficacite_base + variabilite)) / 100
        
        # Symptômes réduits (0-1)
        reduction_symptomes = np.random.binomial(1, efficacite_patient)
        
        # Effets secondaires (probabilité basse)
        effets_secondaires = np.random.random() < 0.05  # 5% de chance
        
        patient = {
            'id_patient': self.patients_total + 1,
            'traitement': traitement_id,
            'jour_inclusion': jour,
            'reduction_symptomes': reduction_symptomes,
            'effets_secondaires': effets_secondaires,
            'efficacite_reelle': efficacite_patient,
            'cout': self.cout_par_patient + (self.cout_par_traitement if traitement_id < 5 else 0)
        }
        
        self.historique_patients.append(patient)
        self.patients_total += 1
        self.patients_par_traitement[traitement_id] += 1
        
        return reduction_symptomes
    
    def executer_etude(self):
        """Exécute l'étude clinique adaptative"""
        print("="*70)
        print("🚀 DÉBUT DE L'ESSAI CLINIQUE ADAPTATIF")
        print("="*70)
        print(f"Date de début: {self.date_debut.strftime('%d/%m/%Y')}")
        print(f"Durée maximale: {self.duree_max_jours} jours")
        print(f"Patients par jour: {self.patients_par_jour}")
        print(f"Traitement de référence: {self.efficacite_reference}% d'efficacité")
        print(f"Niveau de confiance: {(1-self.delta)*100}%")
        print("\n" + "="*70)
        
        # Phase d'initialisation (chaque traitement testé sur 10 patients)
        print("\n📋 PHASE D'INITIALISATION")
        print("Chaque traitement testé sur 10 patients...")
        
        for traitement in range(self.n_traitements):
            for _ in range(10):
                reward = self.recruter_patient(traitement, jour=0)
                self.bandit.pull_arm(traitement)  # Utilise la reward
        
        # Phase adaptative
        print("\n🔄 PHASE ADAPTATIVE")
        print("L'algorithme ajuste les allocations en temps réel...")
        
        for jour in range(1, self.duree_max_jours + 1):
            # Nombre de patients à recruter aujourd'hui
            patients_aujourdhui = min(self.patients_par_jour, 
                                     3000 - self.patients_total)  # Limite totale
            
            if patients_aujourdhui <= 0:
                print(f"\n⏹️ Arrêt de l'étude: limite de patients atteinte")
                break
            
            print(f"\n📅 Jour {jour}: recrutement de {patients_aujourdhui} patient(s)")
            
            for _ in range(patients_aujourdhui):
                # L'algorithme bandit choisit le traitement
                S, R = self.bandit.run(max_steps=self.bandit.T.sum() + 1)
                
                # Choisir quel bras tirer (dans la pratique, on utiliserait la logique de l'algorithme)
                # Pour simplifier, on prend le bras avec le plus haut UCB parmi ceux pas dans S
                candidates = [i for i in range(self.n_traitements) if i not in self.bandit.S]
                
                if candidates:
                    # Calculer UCB pour chaque candidat
                    ucb_values = []
                    for i in candidates:
                        xi_t = max(2 * len(self.bandit.S), 
                                  (5 / (3 * (1 - 4 * self.delta))) * np.log(1 / self.delta))
                        ucb = (self.bandit.mu_hat[i] + 
                              phi_function(self.bandit.T[i], self.delta / xi_t))
                        ucb_values.append(ucb)
                    
                    traitement_choisi = candidates[np.argmax(ucb_values)]
                else:
                    # Tous les traitements sont dans S, choisir au hasard parmi les prometteurs
                    traitement_choisi = np.random.choice(list(self.bandit.S))
                
                # Recruter le patient
                reward = self.recruter_patient(traitement_choisi, jour)
                self.bandit.pull_arm(traitement_choisi)
            
            # Analyse intermédiaire tous les 30 jours
            if jour % 30 == 0:
                self.analyse_intermediaire(jour)
                
                # Critère d'arrêt précoce pour efficacité
                if self.critere_arret_precoce():
                    print(f"\n🎉 ARRÊT PRÉCOCE: Traitement(s) supérieur(s) identifié(s)")
                    break
        
        # Analyse finale
        self.analyse_finale()
        
        return self.bandit.S, self.bandit.R
    
    def analyse_intermediaire(self, jour):
        """Analyse intermédiaire de l'étude"""
        print(f"\n📊 ANALYSE INTERMÉDIAIRE - Jour {jour}")
        print("-"*50)
        
        # Traitements dans S (FDR controlé)
        traitements_S = sorted(self.bandit.S)
        traitements_R = sorted(self.bandit.R)
        
        print(f"Traitements potentiellement supérieurs (FDR controlé): {traitements_S}")
        print(f"Traitements très prometteurs (FWER controlé): {traitements_R}")
        
        # Efficacités estimées
        print("\nEfficacités estimées (%):")
        for i in range(self.n_traitements):
            if i in traitements_S:
                marque = "★" if i in traitements_R else "☆"
                status = f"{marque} Supérieur"
            elif self.bandit.T[i] > 0:
                status = "En test"
            else:
                status = "Non testé"
            
            eff_est = self.bandit.mu_hat[i] * 100 if self.bandit.T[i] > 0 else 0
            eff_reelle = self.efficacites_reelles[i]
            
            print(f"  Traitement {i:2d}: {eff_est:5.1f}% (réel: {eff_reelle:4.1f}%) "
                  f"[{self.bandit.T[i]:3d} patients] - {status}")
        
        # Statistiques patients
        print(f"\n📈 STATISTIQUES:")
        print(f"  Patients totaux: {self.patients_total}")
        print(f"  Patients exposés à traitement inefficace: "
              f"{self.calculer_patients_inefficaces():d}")
        print(f"  Coût total estimé: {self.calculer_cout_total():,.0f} k€")
        
        self.resultats_intermediaires.append({
            'jour': jour,
            'patients': self.patients_total,
            'S': traitements_S.copy(),
            'R': traitements_R.copy(),
            'cout': self.calculer_cout_total()
        })
    
    def analyse_finale(self):
        """Analyse finale complète de l'étude"""
        print("\n" + "="*70)
        print("📋 RAPPORT FINAL DE L'ESSAI CLINIQUE")
        print("="*70)
        
        date_fin = self.date_debut + timedelta(days=min(self.duree_max_jours, 
                                                       len(self.resultats_intermediaires)*30))
        duree_reelle = (date_fin - self.date_debut).days
        
        print(f"\n📅 DURÉE DE L'ÉTUDE: {duree_reelle} jours")
        print(f"   Du {self.date_debut.strftime('%d/%m/%Y')} au {date_fin.strftime('%d/%m/%Y')}")
        
        print(f"\n👥 PATIENTS:")
        print(f"   Total recrutés: {self.patients_total}")
        print(f"   Par traitement (moyenne): {self.patients_par_traitement.mean():.1f}")
        print(f"   Patients exposés à traitement inefficace: "
              f"{self.calculer_patients_inefficaces():d} "
              f"({self.calculer_patients_inefficaces()/self.patients_total*100:.1f}%)")
        
        print(f"\n💰 COÛTS:")
        cout_total = self.calculer_cout_total()
        print(f"   Coût total: {cout_total:,.0f} k€")
        print(f"   Coût par patient: {cout_total/self.patients_total:,.0f} k€")
        
        print(f"\n🎯 RÉSULTATS:")
        
        # Vrais traitements efficaces (inconnu en pratique)
        vrais_positifs = [i for i, eff in enumerate(self.efficacites_reelles) 
                         if eff > self.efficacite_reference]
        
        print(f"   Traitements réellement efficaces (> {self.efficacite_reference}%): {vrais_positifs}")
        print(f"   Traitements détectés par l'algorithme (S): {sorted(self.bandit.S)}")
        print(f"   Traitements hautement prometteurs (R): {sorted(self.bandit.R)}")
        
        # Métriques
        tp = len([i for i in self.bandit.S if i in vrais_positifs])
        fp = len([i for i in self.bandit.S if i not in vrais_positifs])
        fn = len([i for i in vrais_positifs if i not in self.bandit.S])
        
        print(f"\n📊 MÉTRIQUES DE PERFORMANCE:")
        print(f"   Vrais positifs: {tp}/{len(vrais_positifs)} "
              f"({tp/len(vrais_positifs)*100:.1f}%)")
        print(f"   Faux positifs: {fp}")
        print(f"   Faux négatifs: {fn}")
        
        if len(self.bandit.S) > 0:
            print(f"   FDR observé: {fp/len(self.bandit.S):.3f} "
                  f"(cible: ≤ {self.delta})")
        
        # Recommandations
        print(f"\n💡 RECOMMANDATIONS:")
        if self.bandit.R:
            print(f"   1. Poursuivre en phase III: traitements {sorted(self.bandit.R)}")
        if self.bandit.S - self.bandit.R:
            print(f"   2. Études complémentaires: traitements {sorted(self.bandit.S - self.bandit.R)}")
        if not self.bandit.S:
            print(f"   3. Aucun traitement n'a montré de supériorité significative")
        
        # Visualisation
        self.visualiser_resultats()
    
    def calculer_patients_inefficaces(self):
        """Calcule le nombre de patients exposés à des traitements inefficaces"""
        count = 0
        for i in range(self.n_traitements):
            if self.efficacites_reelles[i] <= self.efficacite_reference:
                count += self.patients_par_traitement[i]
        return count
    
    def calculer_cout_total(self):
        """Calcule le coût total de l'étude"""
        cout_patients = self.patients_total * self.cout_par_patient
        cout_traitements = sum(self.patients_par_traitement[:5]) * self.cout_par_traitement
        return cout_patients + cout_traitements
    
    def critere_arret_precoce(self):
        """Critères d'arrêt précoce pour efficacité"""
        # Arrêt si un traitement a une forte probabilité d'être supérieur
        for i in self.bandit.R:
            if self.bandit.T[i] >= 100:  # Suffisamment de patients
                eff_est = self.bandit.mu_hat[i] * 100
                if eff_est > self.efficacite_reference + 15:  # Supériorité claire
                    return True
        return False
    
    def visualiser_resultats(self):
        """Visualise les résultats de l'étude"""
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # Graphique 1: Efficacités réelles vs estimées
        x = np.arange(self.n_traitements)
        width = 0.35
        
        axes[0, 0].bar(x - width/2, self.efficacites_reelles, width, 
                      label='Efficacité réelle', alpha=0.7)
        axes[0, 0].bar(x + width/2, self.bandit.mu_hat * 100, width, 
                      label='Estimation', alpha=0.7)
        axes[0, 0].axhline(y=self.efficacite_reference, color='r', 
                          linestyle='--', label='Référence')
        axes[0, 0].set_xlabel('Traitement')
        axes[0, 0].set_ylabel('Efficacité (%)')
        axes[0, 0].set_title('Efficacité réelle vs estimée')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Graphique 2: Distribution des patients
        colors = ['green' if eff > self.efficacite_reference else 'red' 
                 for eff in self.efficacites_reelles]
        axes[0, 1].bar(x, self.patients_par_traitement, color=colors, alpha=0.7)
        axes[0, 1].set_xlabel('Traitement')
        axes[0, 1].set_ylabel('Nombre de patients')
        axes[0, 1].set_title('Distribution des patients par traitement')
        axes[0, 1].grid(True, alpha=0.3, axis='y')
        
        # Graphique 3: Évolution des découvertes
        if self.resultats_intermediaires:
            jours = [r['jour'] for r in self.resultats_intermediaires]
            taille_S = [len(r['S']) for r in self.resultats_intermediaires]
            taille_R = [len(r['R']) for r in self.resultats_intermediaires]
            
            axes[1, 0].plot(jours, taille_S, 'b-', label='|S| (FDR)', linewidth=2)
            axes[1, 0].plot(jours, taille_R, 'r-', label='|R| (FWER)', linewidth=2)
            axes[1, 0].set_xlabel('Jour')
            axes[1, 0].set_ylabel('Nombre de traitements')
            axes[1, 0].set_title('Évolution des découvertes')
            axes[1, 0].legend()
            axes[1, 0].grid(True, alpha=0.3)
        
        # Graphique 4: Coût cumulatif
        if self.resultats_intermediaires:
            cout_cumul = [r['cout'] for r in self.resultats_intermediaires]
            axes[1, 1].plot(jours, cout_cumul, 'g-', linewidth=2)
            axes[1, 1].set_xlabel('Jour')
            axes[1, 1].set_ylabel('Coût cumulé (k€)')
            axes[1, 1].set_title('Coût de l\'étude au cours du temps')
            axes[1, 1].grid(True, alpha=0.3)
        
        plt.suptitle('Essai Clinique Adaptatif - Résultats Finaux', fontsize=16)
        plt.tight_layout()
        plt.show()

# ==================== EXÉCUTION DE L'ESSAI CLINIQUE ====================

if __name__ == "__main__":
    print("🏥 SIMULATION D'ESSAI CLINIQUE ADAPTATIF")
    print("="*70)
    
    # Créer et exécuter l'essai
    essai = EssaiCliniqueBandit(
        n_traitements=20,
        efficacite_reference=30,
        delta=0.05,
        duree_max_jours=180,
        patients_par_jour=10
    )
    
    # Exécuter l'étude
    S, R = essai.executer_etude()
    
    print("\n" + "="*70)
    print("✅ SIMULATION TERMINÉE")
    print("="*70)
    print("\nAvantages de cette approche:")
    print("1. ✅ Moins de patients exposés à des traitements inefficaces")
    print("2. ✅ Identification plus rapide des traitements prometteurs")
    print("3. ✅ Contrôle statistique rigoureux des erreurs")
    print("4. ✅ Réduction des coûts de développement")
    print("5. ✅ Décisions adaptatives basées sur les données accumulées")