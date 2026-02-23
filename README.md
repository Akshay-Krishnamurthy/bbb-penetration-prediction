# 🧠 Blood-Brain Barrier Penetration Prediction Pipeline

[![Python 3.8+](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![RDKit](https://img.shields.io/badge/RDKit-Cheminformatics-green.svg)](https://www.rdkit.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-orange.svg)](https://scikit-learn.org/)
[![XGBoost](https://img.shields.io/badge/XGBoost-Boosting-red.svg)](https://xgboost.readthedocs.io/)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Akshay-Krishnamurthy/bbb-penetration-prediction/blob/main/Blood_Brain_Barrier_Penetration_Prediction.ipynb)

---

## 📌 Overview

A **full end-to-end computational pipeline** for predicting Blood-Brain Barrier (BBB) permeability of drug-like compounds, combining:

- **60+ molecular descriptors** (physicochemical, topological, electrostatic, binary flags)
- **6 ML classifiers** with 10-fold stratified cross-validation
- **SHAP explainability** for mechanistic insight
- **Mechanistic PK decomposition** — P-gp efflux, fup, fubrain, Kp,brain, Kp,uu,brain
- **2-compartment PBPK simulation** with P-gp inhibition DDI scenarios
- **Automated Excel reporting** (9 worksheets, colour-coded)

> Designed for CNS drug discovery: upload a CSV of SMILES and get a full tiered decision report.

---

## 🧪 Scientific Background

The Blood-Brain Barrier (BBB) is a selective semipermeable membrane that restricts entry of most molecules into the brain. Predicting BBB permeability is a critical early-stage filter in CNS drug discovery.

This pipeline integrates three tiers of analysis:

| Tier | Analysis | Key Output |
|------|----------|-----------|
| **Level 1** | ML classification + EDA | BBB+/BBB− probability |
| **Level 2** | Mechanistic PK decomposition | Kp,uu,brain, P-gp NER, fup, fubrain |
| **Level 3/4** | PBPK ODE simulation | Brain AUC, DDI risk ratio |

---

## 🔬 Descriptor Classes

Four descriptor classes are computed from SMILES strings using RDKit:

| Class | Examples | Count |
|-------|---------|-------|
| **A — Physicochemical** | MW, LogP, TPSA, HBD, HBA, QED, CNS MPO, BBB Score | ~14 |
| **B — Topological/Structural** | Chi indices, Kappa indices, BertzCT, ring counts | ~25 |
| **C — Electrostatic** | Partial charges, VSA contributions, LabuteASA | ~13 |
| **D — Binary/Categorical** | Lipinski flags, ionization class (one-hot), functional groups | ~12 |

### Composite Drug-likeness Scores
- **BBB Score** (Gupta et al. 2019): 0–6 scale; ≥4 = BBB-penetrant
- **CNS MPO** (Wager et al. 2010, Pfizer): 0–6 scale; ≥4 = CNS-favourable

---

## 🤖 Machine Learning Models

All models trained on the **BBBP benchmark dataset** (DeepChem, ~2000 compounds) with **10-fold stratified cross-validation**:

| Model | Notes |
|-------|-------|
| Random Forest (n=300) | Class-balanced, sqrt features |
| **XGBoost** (n=300) | Scale-pos-weight for imbalance |
| LightGBM (n=300) | num_leaves=63, class-balanced |
| Extra Trees (n=300) | Class-balanced |
| Gradient Boosting (n=200) | Subsample=0.8 |
| Logistic Regression | Pipeline with StandardScaler |

### Feature Selection Pipeline (4 steps)
1. **Variance threshold** (removes near-zero variance features)
2. **Correlation filter** (removes |r| > 0.90 pairs)
3. **Mutual information** (top 40 features with MI > 0.005)
4. **VIF check** (variance inflation factor — multicollinearity audit)

---

## 📊 Pipeline Outputs

### Plots Generated (13 total)
| Plot | Description |
|------|------------|
| `plot_01_distributions.png` | KDE distributions of key descriptors (BBB+ vs BBB−) |
| `plot_02_correlations.png` | Top-30 point-biserial correlations with BBB label |
| `plot_03_corr_heatmap.png` | Inter-descriptor correlation heatmap (top 15 features) |
| `plot_04_roc_pr.png` | ROC + Precision-Recall curves for all models (OOF) |
| `plot_05_model_comparison.png` | Grouped bar chart — 6 metrics × 6 models |
| `plot_06_confusion.png` | Confusion matrices for top-3 models |
| `plot_07_cv_boxplots.png` | 10-fold CV score distributions (boxplots) |
| `plot_08_shap_*.png` | SHAP summary plots (beeswarm) per model |
| `plot_09_feature_importance.png` | Gini importance: RF vs XGBoost |
| `plot_10_pbpk_curves.png` | Brain vs plasma PK curves — normal vs P-gp inhibited |
| `plot_11_decision_dashboard.png` | Unified CNS decision dashboard (7 panels) |
| `plot_12_train_vs_pred.png` | KDE comparison: training set vs query compounds |
| `plot_13_radar.png` | Normalised radar chart — descriptor profiles |

### Excel Report (9 worksheets)
| Sheet | Contents |
|-------|---------|
| `1_Predictions` | Full predictions with colour-coded BBB class, all PK metrics |
| `2_Model_Statistics` | CV stats + per-fold AUC table with Excel formulas |
| `3_Descriptor_Stats` | Mann-Whitney U + point-biserial per descriptor |
| `4_Feature_Selection` | MI scores + VIF table |
| `5_SHAP_Summary` | Mean \|SHAP\| values + direction per model |
| `6_Mechanistic_PK` | P-gp class, fup, fubrain, Kp,brain, Kp,uu,brain |
| `7_PBPK_Results` | Brain AUC normal vs inhibited, DDI ratio |
| `8_Training_Data` | Full BBBP benchmark with descriptors |
| `9_Thresholds_Reference` | Complete BBB decision rules from literature |

---

## 🚀 Quick Start

### Option 1: Google Colab (Recommended — no setup needed)

Click the badge above or open directly:

```
https://colab.research.google.com/github/Akshay-Krishnamurthy/bbb-penetration-prediction/blob/main/Blood_Brain_Barrier_Penetration_Prediction.ipynb
```

Run all cells top to bottom. Cell 1 installs all dependencies automatically (~3–4 min on first run).

### Option 2: Local Setup

```bash
git clone https://github.com/Akshay-Krishnamurthy/bbb-penetration-prediction.git
cd bbb-penetration-prediction
pip install -r requirements.txt
jupyter notebook Blood_Brain_Barrier_Penetration_Prediction.ipynb
```

---

## 📁 Repository Structure

```
bbb-penetration-prediction/
│
├── Blood_Brain_Barrier_Penetration_Prediction.ipynb   ← Main pipeline notebook
├── requirements.txt                                   ← All Python dependencies
├── README.md                                          ← This file
├── .gitignore                                         ← Git ignore rules
│
├── sample_data/
│   └── demo_compounds.csv                             ← 14 demo SMILES with known BBB status
│
└── results/                                           ← Example outputs (gitignored by default)
    ├── CNS_BBB_Complete_Results.xlsx
    └── plots/
```

---

## 📋 Input Format

Upload a CSV with a column named `SMILES` (or `smiles`). Any additional columns (Name, ID, CAS, etc.) are preserved in output.

```csv
SMILES,Name,Reference
CN1CCC[C@H]1c2cccnc2,Nicotine,Known BBB+
CC(=O)Oc1ccccc1C(=O)O,Aspirin,Known BBB-
COc1ccc2[nH]cc(CC(N)C(=O)O)c2c1,5-Methoxytryptophan,Test
```

---

## 🧬 Mechanistic PK Framework

Based on the **J. Med. Chem. 2021 tiered framework**:

```
BBB+ Probability (ML)
        ↓
P-gp Efflux Class (Low/Medium/High) → NER value
        ↓
fup (plasma unbound fraction)
fubrain (brain unbound fraction)
        ↓
Kp,brain (brain-to-plasma partition, Rodgers & Rowland 2006)
        ↓
Kp,uu,brain = (Kp,brain / NER) × (fup / fubrain)
```

### Kp,uu,brain Interpretation

| Value | Interpretation |
|-------|---------------|
| > 1.0 | Net brain accumulation |
| 0.3–1.0 | Good CNS exposure |
| 0.1–0.3 | Efflux-limited (moderate) |
| < 0.1 | Poor CNS exposure |

---

## ⚡ PBPK Model

Two-compartment ODE (plasma ↔ brain) with IV bolus assumption:

```
dC_plasma/dt = -(CL_systemic + CL_passive)/Vp × Cp + (CL_passive + CL_efflux)/Vp × Cb
dC_brain/dt  =  (CL_passive/Vb) × Cp - (CL_passive + CL_efflux)/Vb × Cb
```

**Scenarios simulated:**
- Normal conditions
- P-gp inhibited (90% efflux inhibition — DDI scenario)

**DDI Risk Classification:**
- 🔴 High: AUC ratio > 5×
- 🟡 Moderate: 2–5×
- 🟢 Low: < 2×

---

## 🔑 Key BBB Thresholds

| Descriptor | BBB+ Favoured | BBB− Risk |
|-----------|--------------|-----------|
| TPSA | < 90 Å² | > 120 Å² |
| LogP | 1.0–5.0 | < 0 or > 5 |
| MW | < 450 Da | > 500 Da |
| HBD | ≤ 3 | > 5 |
| HBA | ≤ 8 | > 10 |
| BBB Score | ≥ 4 | < 3 |
| CNS MPO | ≥ 4 | < 3 |
| Kp,uu,brain | > 0.3 | < 0.1 |

---

## 📚 References

1. **Gupta, M. et al.** (2019). *BBB Score*. J. Med. Chem., 62(19), 9134–9141.
2. **Wager, T.T. et al.** (2010). *CNS MPO Score (Pfizer)*. ACS Chem. Neurosci., 1(6), 435–449.
3. **Rodgers, T. & Rowland, M.** (2006). *Mechanistic approaches to volume of distribution predictions*. J. Pharm. Sci., 95(6), 1238–1257.
4. **Fridén, M. et al.** (2010). *Prediction of brain unbound fraction using in vitro measurement*. Drug Metab. Dispos., 38(6), 1090–1099.
5. **BBBP Dataset** — MoleculeNet benchmark (DeepChem).
6. **Martins, I.F. et al.** (2012). *A Bayesian Approach to in Silico Blood-Brain Barrier Penetration Modeling.* J. Chem. Inf. Model., 52(6), 1686–1697.
   — **Primary source of the BBBP benchmark dataset used for training.**
7. **Wu, Z. et al.** (2018). *MoleculeNet: A Benchmark for Molecular Machine Learning.* Chemical Science, 9(2), 513–530.
   — **MoleculeNet benchmark suite (BBBP is part of this collection).**
8. **DeepChem** (https://deepchem.io) — Open-source platform hosting the BBBP dataset used in this pipeline.

---

## 👤 Author

**Akshay Krishnamurthy Hegde**
- Field: Computational Drug Discovery / Cheminformatics
- Tools: RDKit, scikit-learn, XGBoost, SHAP, PBPK modelling

---

## 📄 License

MIT License — free to use and adapt with attribution.
