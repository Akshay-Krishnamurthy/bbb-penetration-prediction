# 🧠 Structure-Brain Link (SBL)
### Blood-Brain Barrier Penetration Prediction Pipeline

[![Python 3.8+](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![RDKit](https://img.shields.io/badge/RDKit-Cheminformatics-green.svg)](https://www.rdkit.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-orange.svg)](https://scikit-learn.org/)
[![XGBoost](https://img.shields.io/badge/XGBoost-Boosting-red.svg)](https://xgboost.readthedocs.io/)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Akshay-Krishnamurthy/structure-brain-link/blob/main/Blood_Brain_Barrier_Penetration_Prediction.ipynb)

---
## 🎯 Summary
Most BBB predictors give a binary label. This pipeline goes further:
it tells you HOW MUCH free drug reaches the brain target (Kp,uu,brain),
whether P-gp is actively pumping your compound out (NER classification),
what happens when a co-drug inhibits that efflux (DDI simulation),
and which exact structural features are causing BBB failure (SHAP).
The result is not a prediction — it's a full CNS drug profile.


## 📈 Benchmark Results (Actual Run)

| Model | AUC-ROC | AUC-PR | F1 | MCC | Balanced Acc |
|-------|---------|--------|-----|-----|-------------|
| **Extra Trees** 🏆 | **0.9540 ± 0.0059** | **0.9725** | **0.9054** | **0.7441** | **0.8712** |
| Random Forest | 0.9524 ± 0.0063 | 0.9714 | 0.9038 | 0.7354 | 0.8638 |
| XGBoost | 0.9496 ± 0.0071 | 0.9680 | 0.9054 | 0.7454 | 0.8726 |
| LightGBM | 0.9490 ± 0.0068 | 0.9671 | 0.9054 | 0.7462 | 0.8735 |
| Gradient Boosting | 0.9402 ± 0.0074 | 0.9607 | 0.8982 | 0.7130 | 0.8466 |
| Logistic Regression | 0.8614 ± 0.0098 | 0.8926 | 0.8468 | 0.5784 | 0.7862 |

> Trained on 8,115 compounds (BBBP + B3DB merged). 10-fold stratified CV. Best model: **Extra Trees** (AUC-ROC 0.9540).

### Top-5 BBB-Predictive Descriptors (Point-Biserial r, Mann-Whitney p < 0.001)

| Rank | Descriptor | r | Direction |
|------|-----------|---|-----------|
| 1 | CNS_MPO | +0.525 | ↑ promotes BBB+ |
| 2 | TPSA | −0.517 | ↓ inhibits BBB+ |
| 3 | NOCount | −0.504 | ↓ inhibits BBB+ |
| 4 | BBB_Score | +0.495 | ↑ promotes BBB+ |
| 5 | NumHeteroatoms | −0.492 | ↓ inhibits BBB+ |

## 📌 Overview
📐 [**View Full Pipeline Diagram →**](https://akshay-krishnamurthy.github.io/structure-brain-link/BBB_Pipeline_Diagram.html)

A **full end-to-end computational pipeline** for predicting Blood-Brain Barrier (BBB) permeability of drug-like compounds, combining:

- **66 molecular descriptors** (physicochemical, topological, electrostatic, binary flags)
- **6 ML classifiers** with 10-fold stratified cross-validation
- **SHAP explainability** (XGBoost; TreeExplainer)
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

All models trained on the **BBBP + B3DB merged benchmark** (8,115 compounds after deduplication (1,975 BBBP + 6,140 B3DB)) with **10-fold stratified cross-validation**:

| Model | Notes |
|-------|-------|
| Random Forest (n=300) | Class-balanced, sqrt features |
| XGBoost (n=300) | Scale-pos-weight for imbalance |
| LightGBM (n=300) | num_leaves=63, class-balanced |
| **Extra Trees** (n=300) | Class-balanced — 🏆 Best Model |
| Gradient Boosting (n=200) | Subsample=0.8 |
| Logistic Regression | Pipeline with StandardScaler |

### Feature Selection Pipeline (4 steps)
1. **Variance threshold** (removes near-zero variance features)
2. **Correlation filter** (removes |r| > 0.90 pairs)
3. **Mutual information** (top 39 features with MI > 0.005)
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
| `8_Training_Data` | BBBP + B3DB merged benchmark with descriptors + source column |
| `9_Thresholds_Reference` | Complete BBB decision rules from literature |

---

## 🚀 Quick Start

### Option 1: Google Colab (Recommended — no setup needed)

Click the badge above or open directly:

```
https://colab.research.google.com/github/Akshay-Krishnamurthy/structure-brain-link/blob/main/Blood_Brain_Barrier_Penetration_Prediction.ipynb
```

Run all cells top to bottom. Cell 1 installs all dependencies automatically (~3–4 min on first run).

### Option 2: Local Setup

```bash
git clone https://github.com/Akshay-Krishnamurthy/structure-brain-link.git
cd structure-brain-link
pip install -r requirements.txt
jupyter notebook Blood_Brain_Barrier_Penetration_Prediction.ipynb
```

---

## 📁 Repository Structure

```
structure-brain-link/
│
├── Blood_Brain_Barrier_Penetration_Prediction.ipynb   ← Main pipeline notebook
├── BBB_Pipeline_Diagram.html                          ← Interactive pipeline diagram
├── requirements.txt                                   ← All Python dependencies
├── README.md                                          ← This file
├── .gitignore                                         ← Git ignore rules
│
├── sample_data/
│   └── demo_compounds.csv                             ← 14 demo SMILES with known BBB status
│
└── results/                                           ← Example outputs (gitignored by default)
    ├── SBL_Complete_Results.xlsx
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

1. **Martins, I.F. et al.** (2012). *A Bayesian Approach to in Silico Blood-Brain Barrier Penetration Modeling.* J. Chem. Inf. Model., 52(6), 1686–1697.
   — **Primary source of the BBBP benchmark dataset used for training.**

2. **Meng, J. et al.** (2021). *B3DB: A Curated Blood-Brain Barrier Database.* Scientific Data, 8, 289.
   — **B3DB dataset merged with BBBP for training (~7,800+ compounds total).**

3. **Wu, Z. et al.** (2018). *MoleculeNet: A Benchmark for Molecular Machine Learning.* Chemical Science, 9(2), 513–530.
   — **MoleculeNet benchmark suite (BBBP is part of this collection).**

4. **DeepChem** (https://deepchem.io) — Open-source platform hosting the BBBP dataset.

5. **Gupta, M. et al.** (2019). *BBB Score — A Composite Score for Predicting Blood-Brain Barrier Permeation.* J. Med. Chem., 62(19), 9134–9141.

6. **Wager, T.T. et al.** (2010). *Moving beyond Rules: The Development of a Central Nervous System Multiparameter Optimization (CNS MPO) Approach To Enable Alignment of Druglike Properties.* ACS Chem. Neurosci., 1(6), 435–449.

7. **Rodgers, T. & Rowland, M.** (2006). *Mechanistic approaches to volume of distribution predictions: understanding the processes.* J. Pharm. Sci., 95(6), 1238–1257.

8. **Fridén, M. et al.** (2010). *Prediction of drug brain concentrations using an in vivo steady state brain slice model.* Drug Metab. Dispos., 38(6), 1087–1093.

9. **Lobell, M. & Sivarajah, V.** (2003). *In silico prediction of aqueous solubility, human plasma protein binding and the volume of distribution of compounds from calculated pKa and AlogP98 values.* Mol. Divers., 7(1), 69–87.
   — **fup (plasma unbound fraction) approximation method.**

---

## 👤 Author

**Akshay Krishnamurthy Hegde**
- Field: Computational Drug Discovery / Machine Learning / Cheminformatics
- Tools: RDKit, scikit-learn, XGBoost, SHAP, PBPK modelling

---

## 📄 License

MIT License — free to use and adapt with attribution.
