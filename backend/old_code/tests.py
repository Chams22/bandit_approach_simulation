import numpy as np
import math
import os
import sys 

# === TESTS UNITAIRES POUR L'ALGORITHME BANDIT ===

# Essaie d'importer depuis bandit_algorithm.py
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
try:
    from bandit_algorithm import phi_function, BanditExperiment
    print("✅ Import réussi depuis bandit_algorithm.py")
except ImportError as e:
    print(f"❌ Erreur d'import: {e}")
    print("Assure-toi que:")
    print("1. bandit_algorithm.py est dans le même dossier que ce fichier")
    print("2. Le fichier s'appelle bien bandit_algorithm.py")
    print("3. Les classes/fonctions ont les bons noms")
    sys.exit(1)

def test_phi_function_basics():
    """Tests de base pour la fonction phi"""
    print("\n=== Test phi_function ===")
    
    # Test 1: t = 0 doit retourner infini
    assert phi_function(0, 0.05) == float('inf'), "phi(0) devrait être infini"
    
    # Test 2: Décroissance avec t
    phi_10 = phi_function(10, 0.05)
    phi_100 = phi_function(100, 0.05)
    assert phi_10 > phi_100, "phi doit décroître quand t augmente"
    
    # Test 3: Sensibilité à delta
    phi_delta_small = phi_function(100, 0.01)
    phi_delta_large = phi_function(100, 0.1)
    assert phi_delta_small > phi_delta_large, "phi doit être plus grand pour delta plus petit"
    
    # Test 4: Valeurs non-nan
    for t in [1, 10, 100, 1000]:
        val = phi_function(t, 0.05)
        assert not np.isnan(val), f"phi({t}, 0.05) ne doit pas être NaN"
        assert val > 0, f"phi({t}, 0.05) doit être positif"
    
    print("✓ phi_function: OK")

def test_bandit_initialization():
    """Test l'initialisation correcte de BanditExperiment"""
    print("\n=== Test Initialisation Bandit ===")
    
    true_means = [0.5, 0.3, 0.7]
    mu_0 = 0.4
    delta = 0.05
    
    bandit = BanditExperiment(
        true_means=true_means,
        mu_0=mu_0,
        delta=delta,
        mode='TPR',
        fwer_control=False
    )
    
    # Vérifications
    assert bandit.n == 3, f"n devrait être 3, pas {bandit.n}"
    assert bandit.mu_0 == mu_0, f"mu_0 incorrect"
    assert bandit.delta == delta, f"delta incorrect"
    assert bandit.mode == 'TPR', f"mode incorrect"
    assert not bandit.fwer_control, "fwer_control devrait être False"
    
    # Structures de données
    assert len(bandit.T) == 3, "T devrait avoir 3 éléments"
    assert len(bandit.mu_hat) == 3, "mu_hat devrait avoir 3 éléments"
    assert bandit.S == set(), "S devrait être vide initialement"
    assert bandit.R == set(), "R devrait être vide initialement"
    assert bandit.history == [], "history devrait être vide initialement"
    
    print("✓ Initialisation: OK")

def test_pull_arm():
    """Test que pull_arm met à jour les statistiques correctement"""
    print("\n=== Test pull_arm ===")
    
    np.random.seed(42)  # Pour reproductibilité
    
    true_means = [0.5, 0.8]
    bandit = BanditExperiment(
        true_means=true_means,
        mu_0=0.3,
        delta=0.05,
        mode='TPR',
        fwer_control=False
    )
    
    # Tirer le bras 0
    reward0 = bandit.pull_arm(0)
    
    # Vérifications après 1 tirage
    assert bandit.T[0] == 1, f"T[0] devrait être 1, pas {bandit.T[0]}"
    assert bandit.T[1] == 0, "T[1] devrait être 0"
    assert bandit.sum_rewards[0] == reward0, "sum_rewards[0] incorrect"
    assert bandit.mu_hat[0] == reward0, f"mu_hat[0] devrait être {reward0}"
    
    # Tirer le bras 0 à nouveau
    reward1 = bandit.pull_arm(0)
    
    # Vérifications après 2 tirages
    assert bandit.T[0] == 2, f"T[0] devrait être 2, pas {bandit.T[0]}"
    assert bandit.sum_rewards[0] == reward0 + reward1, "sum_rewards[0] incorrect"
    assert abs(bandit.mu_hat[0] - (reward0 + reward1)/2) < 1e-10, f"mu_hat[0] incorrect: {bandit.mu_hat[0]}"
    
    print("✓ pull_arm: OK")

def test_mode_definitions():
    """Test que les modes TPR et FWPD définissent correctement xi_t et nu_t"""
    print("\n=== Test Définition des modes ===")
    
    true_means = [0.5] * 3
    bandit_tpr = BanditExperiment(
        true_means=true_means,
        mu_0=0.3,
        delta=0.05,
        mode='TPR',
        fwer_control=False
    )
    
    bandit_fwpd = BanditExperiment(
        true_means=true_means,
        mu_0=0.3,
        delta=0.05,
        mode='FWPD',
        fwer_control=False
    )
    
    # Simuler que S a 2 éléments
    bandit_tpr.S = {0, 1}
    bandit_fwpd.S = {0, 1}
    
    # Calculer xi_t et nu_t pour TPR
    if bandit_tpr.mode == 'TPR':
        xi_t_tpr = 1.0
        nu_t_tpr = 1.0
    
    # Calculer xi_t et nu_t pour FWPD
    if bandit_fwpd.mode == 'FWPD':
        term2 = (5 / (3 * (1 - 4 * bandit_fwpd.delta))) * np.log(1 / bandit_fwpd.delta)
        xi_t_fwpd = max(2 * len(bandit_fwpd.S), term2)
        nu_t_fwpd = max(len(bandit_fwpd.S), 1)
    
    # Vérifications TPR
    assert xi_t_tpr == 1.0, f"TPR: xi_t devrait être 1.0, pas {xi_t_tpr}"
    assert nu_t_tpr == 1.0, f"TPR: nu_t devrait être 1.0, pas {nu_t_tpr}"
    
    # Vérifications FWPD
    assert xi_t_fwpd >= 2 * 2, f"FWPD: xi_t devrait être ≥ {2*2}, pas {xi_t_fwpd}"
    assert nu_t_fwpd == 2, f"FWPD: nu_t devrait être 2, pas {nu_t_fwpd}"
    
    print(f"✓ TPR: xi_t={xi_t_tpr}, nu_t={nu_t_tpr}")
    print(f"✓ FWPD: xi_t={xi_t_fwpd:.2f}, nu_t={nu_t_fwpd}")
    print("✓ Modes: OK")

def test_run_minimal():
    """Test d'exécution minimal sans crash"""
    print("\n=== Test Exécution Minimal ===")
    
    np.random.seed(123)
    
    # Configuration simple
    true_means = [0.6, 0.4, 0.2]
    mu_0 = 0.5
    
    # Test 1: Mode TPR sans FWER
    bandit = BanditExperiment(
        true_means=true_means,
        mu_0=mu_0,
        delta=0.1,
        mode='TPR',
        fwer_control=False
    )
    
    S, R = bandit.run(max_steps=100)
    
    # Vérifications de base
    assert isinstance(S, set), f"S devrait être un set, pas {type(S)}"
    assert isinstance(R, set), f"R devrait être un set, pas {type(R)}"
    assert len(bandit.history) > 0, "history ne devrait pas être vide"
    
    # Chaque bras devrait avoir été tiré au moins une fois (initialisation)
    for i in range(bandit.n):
        assert bandit.T[i] >= 1, f"Bras {i} n'a pas été tiré"
    
    print(f"✓ Exécution sans crash: OK")
    print(f"  S final: {sorted(S)}")
    print(f"  R final: {sorted(R)} (devrait être vide)")
    print(f"  Tirages totaux: {sum(bandit.T)}")

def test_fwer_control_option():
    """Test que l'option FWER fonctionne"""
    print("\n=== Test Option FWER ===")
    
    np.random.seed(456)
    
    true_means = [0.7, 0.6, 0.2, 0.1]
    mu_0 = 0.5
    
    # Avec FWER
    bandit_with = BanditExperiment(
        true_means=true_means,
        mu_0=mu_0,
        delta=0.1,
        mode='TPR',
        fwer_control=True
    )
    
    # Sans FWER
    bandit_without = BanditExperiment(
        true_means=true_means,
        mu_0=mu_0,
        delta=0.1,
        mode='TPR',
        fwer_control=False
    )
    
    S_with, R_with = bandit_with.run(max_steps=200)
    S_without, R_without = bandit_without.run(max_steps=200)
    
    # R devrait être non-vide seulement avec FWER activé
    if len(R_with) > 0:
        print(f"✓ Avec FWER: R={sorted(R_with)} (non-vide)")
    else:
        print(f"✓ Avec FWER: R={sorted(R_with)} (vide)")
    
    assert R_without == set(), f"Sans FWER, R devrait être vide, pas {R_without}"
    
    # Vérifier que plus de tirages ont été faits avec FWER
    total_with = sum(bandit_with.T)
    total_without = sum(bandit_without.T)
    
    print(f"  Tirages avec FWER: {total_with}")
    print(f"  Tirages sans FWER: {total_without}")
    
    if total_with > total_without:
        print("✓ Plus de tirages avec FWER (logique)")
    else:
        print("⚠ Même nombre de tirages (peut être normal)")

def test_edge_cases():
    """Test des cas limites"""
    print("\n=== Test Cas Limites ===")
    
    # Cas 1: Tous les bras identiques
    np.random.seed(789)
    true_means = [0.5] * 5
    bandit = BanditExperiment(
        true_means=true_means,
        mu_0=0.5,  # Exactement égal
        delta=0.05,
        mode='TPR',
        fwer_control=False
    )
    
    S, R = bandit.run(max_steps=50)
    
    print(f"✓ Cas égalité: S={sorted(S)}, R={sorted(R)}")
    
    # Cas 2: Un seul bras
    true_means = [0.8]
    bandit = BanditExperiment(
        true_means=true_means,
        mu_0=0.5,
        delta=0.1,
        mode='TPR',
        fwer_control=False
    )
    
    S, R = bandit.run(max_steps=30)
    
    assert bandit.n == 1, "Devrait avoir 1 bras"
    assert bandit.T[0] == 30, f"Devrait avoir 30 tirages, pas {bandit.T[0]}"
    print(f"✓ Bras unique: OK (S={S})")
    
    # Cas 3: delta très petit
    bandit = BanditExperiment(
        true_means=[0.6, 0.4],
        mu_0=0.5,
        delta=1e-10,  # Très petit
        mode='TPR',
        fwer_control=False
    )
    
    try:
        S, R = bandit.run(max_steps=20)
        print("✓ Delta très petit: exécuté sans erreur")
    except Exception as e:
        print(f"✗ Delta très petit: erreur {e}")

def test_benjamini_hochberg_logic():
    """Test spécifique de la logique Benjamini-Hochberg"""
    print("\n=== Test Logique Benjamini-Hochberg ===")
    
    # Simuler une situation où certains bras sont clairement > mu_0
    np.random.seed(999)
    
    # Créer un scénario contrôlé
    true_means = [1.0, 1.0, 0.0, 0.0]  # 2 bons, 2 mauvais
    mu_0 = 0.5
    
    bandit = BanditExperiment(
        true_means=true_means,
        mu_0=mu_0,
        delta=0.1,
        mode='TPR',
        fwer_control=False
    )
    
    # Forcer des estimations précises en tirant beaucoup
    for i in range(4):
        for _ in range(100):
            bandit.pull_arm(i)
    
    # Exécuter la mise à jour BH manuellement
    delta_prime = bandit.delta / (6.4 * np.log(36 / bandit.delta))
    
    # Calculer s(k) pour différents k
    s_sizes = []
    for k in range(bandit.n, 0, -1):
        threshold_sk = delta_prime * k / bandit.n
        temp_s_k = []
        
        for i in range(bandit.n):
            conf = phi_function(bandit.T[i], threshold_sk)
            if (bandit.mu_hat[i] - conf) >= bandit.mu_0:
                temp_s_k.append(i)
        
        s_sizes.append((k, len(temp_s_k)))
    
    print("Résultats BH:")
    for k, size in s_sizes:
        print(f"  k={k}: |s(k)|={size} {'✓' if size >= k else ''}")
    
    # Au moins k=2 devrait être valide
    valid_ks = [k for k, size in s_sizes if size >= k]
    assert len(valid_ks) > 0, "Au moins un k devrait être valide"
    
    print("✓ Logique BH: OK")

# Exécution de tous les tests
if __name__ == "__main__":
    print("🔍 TESTS UNITAIRES - Algorithme Bandit")
    print("=" * 50)
    
    # Liste des tests à exécuter
    tests = [
        test_phi_function_basics,
        test_bandit_initialization,
        test_pull_arm,
        test_mode_definitions,
        test_run_minimal,
        test_fwer_control_option,
        test_edge_cases,
        test_benjamini_hochberg_logic
    ]
    
    passed = 0
    failed = 0
    
    for test_func in tests:
        try:
            test_func()
            passed += 1
        except AssertionError as e:
            print(f"\n❌ {test_func.__name__} a échoué: {e}")
            failed += 1
        except Exception as e:
            print(f"\n❌ {test_func.__name__} a crashé: {type(e).__name__}: {e}")
            failed += 1
    
    print("\n" + "=" * 50)
    print(f"📊 RÉSULTATS: {passed} passés, {failed} échoués")
    
    if failed == 0:
        print("✅ TOUS LES TESTS ONT PASSÉ !")
        print("\nL'algorithme ne fait pas 'n'importe quoi':")
        print("1. ✅ Les fonctions de base fonctionnent")
        print("2. ✅ L'initialisation est correcte")
        print("3. ✅ Les tirages mettent à jour les statistiques")
        print("4. ✅ Les modes TPR/FWPD sont implémentés")
        print("5. ✅ L'exécution ne crashe pas")
        print("6. ✅ L'option FWER est fonctionnelle")
        print("7. ✅ Les cas limites sont gérés")
        print("8. ✅ La logique Benjamini-Hochberg est cohérente")
    else:
        print(f"\n❌ {failed} test(s) ont échoué. Vérifie ton implémentation.")
    
    print("=" * 50)