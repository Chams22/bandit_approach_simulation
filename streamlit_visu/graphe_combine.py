import pickle
import matplotlib.pyplot as plt
import numpy as np
import os

# ============================================================
# PARAMETERS - adapt paths to your pickle files
# ============================================================

# Put the path to your saved files here
# (ils sont dans batch_results/ ou saved_simulations/)
RESULTS_DIR = "batch_results"  # ou "saved_simulations"

# Map each proportion to its pickle file
# Remplace les noms par les vrais noms de tes fichiers
SIMULATIONS = {
    "1 / 9" : "sim_1777028900697.pkl",   # remplace par le vrai nom
    "3 / 7" : "sim_1778246856902.pkl",
    "5 / 5" : "sim_1778246905765.pkl",
    "7 / 3" : "sim_1778246962658.pkl",
    "9 / 1" : "sim_1778246962658.pkl",
}

# Couleurs pour chaque proportion
COLORS = ["#e41a1c", "#ff7f00", "#4daf4a", "#377eb8", "#984ea3"]

# ============================================================
# CHART GENERATION
# ============================================================

fig, ax = plt.subplots(figsize=(9, 5))

for (label, filename), color in zip(SIMULATIONS.items(), COLORS):
    filepath = os.path.join(RESULTS_DIR, filename)
    
    with open(filepath, "rb") as f:
        data = pickle.load(f)
    
    tpr_adapt = data["tpr_adapt"]
    ax.plot(tpr_adapt, label=f"Adaptatif — {label}",
            color=color, linewidth=2)
    ax.plot(data["tpr_unif"], label=f"Uniforme — {label}",
            color=color, linewidth=1.2, linestyle="--", alpha=0.5)

ax.axhline(y=0.9, color="gray", linestyle=":", linewidth=1, label="Seuil 90%")
ax.set_xlabel("Temps (t)", fontsize=12)
ax.set_ylabel("Taux de Vrais Positifs (TPR)", fontsize=12)
ax.set_title("Impact de la proportion de vrais positifs\nsur la vitesse de convergence (Positif/Négatif)", fontsize=13)
ax.legend(fontsize=8, ncol=2, loc="lower right")
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("exp6_toutes_proportions.png", dpi=150, bbox_inches="tight")
plt.show()
print("Graphe sauvegardé : exp6_toutes_proportions.png")
