# Bandit Approach Simulation

Simulation and analysis of adaptive bandit algorithms for sequential multiple testing with FDR control, based on the NeurIPS 2018 paper *"A Bandit Approach to Sequential Experimental Design with False Discovery Control"* (Jamieson & Jain).

The goal is to **maximize the True Positive Rate (TPR)** while controlling the **False Discovery Rate (FDR)** at level δ, using as few samples as possible — on both simulated and real megastudy data.

---

## Getting started

**Install dependencies:**
```bash
pip install streamlit numpy pandas matplotlib scipy tqdm
```

**Launch the Streamlit app:**
```bash
streamlit run streamlit_visu/app_bandit.py
```

---

## Repository structure

```
├── backend/                        # Algorithm implementations
│   ├── adaptative_algorithm_jj.py          # JJ — LIL-based anytime p-values (Jamieson & Jain)
│   ├── adaptative_algorithm_v2.py          # V2 — Normal Mixture closed-form p-values
│   ├── adaptative_algorithm_continuous_v3.py   # V3 — Betting martingale (continuous data)
│   ├── adaptative_algorithm_binary_v3.py       # V3 — Betting martingale (binary/Bernoulli)
│   ├── adaptative_algorithm_successive_reject.py  # SR — Successive Rejects (best-arm identification)
│   └── real_data_processing.py             # Pipeline for real megastudy data
│
├── streamlit_visu/
│   └── app_bandit.py               # Interactive Streamlit app (simulation + visualisation)
│
├── data/                           # Real megastudy datasets (raw, processed, estimates)
├── figure_*/                       # Saved experiment figures
└── saved_results/                  # Saved simulation outputs
```

---

## Algorithms

| Name | Module | P-value method | Notes |
|------|--------|---------------|-------|
| **JJ** | `adaptative_algorithm_jj.py` | LIL confidence sequences (brentq inversion) | Original Jamieson & Jain 2018 |
| **V2 / NM** | `adaptative_algorithm_v2.py` | Normal Mixture closed-form | More powerful than LIL bounds |
| **V3** | `adaptative_algorithm_continuous_v3.py` | Betting martingale (Ville's inequality) | Strongest anytime guarantee |
| **SR** | `adaptative_algorithm_successive_reject.py` | None (elimination-based) | Fixed-budget best-arm identification |

Each algorithm runs in two modes:
- **Adaptive** — UCB-based arm selection, samples more from promising arms
- **Uniform** — equal allocation baseline

FDR control is achieved via the **Benjamini-Hochberg (BH)** procedure at each time step.

---

## Streamlit app features

- **Single simulation** — run one algorithm (JJ / V2 / V3 / SR) and compare adaptive vs uniform
- **Algorithm comparison** — run all algorithms on the same shared data for fair comparison
- **Batch testing** — queue and run multiple configurations
- **Visualisations** — TPR curves, pull distribution, spaghetti plots, p-value grids, bootstrap CIs, detection order, exact discovery sequences (Kendall τ heatmap)
- **Metrics** — TPR@90%, Gain d'efficacité, ΔAUC, Cohen's D (δ/σ), FDR

---

## Reference

Jamieson, K., & Jain, L. (2018). *A Bandit Approach to Sequential Experimental Design with False Discovery Control*. NeurIPS 2018.
