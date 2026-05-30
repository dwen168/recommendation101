# 🍫 Chocolate Shop Intelligent Recommendation Pipeline

An industrial-grade, high-performance recommendation ecosystem benchmarked on a real retail setting (low SKU, high-traffic, sparse transactions, high repeat purchases).

This repository integrates and compares **five** state-of-the-art recommendation strategies: **Item-based Collaborative Filtering (ItemCF)**, **Market Basket Analysis (MBA)**, **XGBoost**, **LightGBM**, and **Deep Neural Collaborative Filtering (NCF)** in PyTorch. It features a context-aware **Two-Stage Hybrid Pipeline** serving live predictions through a responsive, glassmorphic sandboxed web interface.

---

## 🎯 Project Overview & Scenario
We simulate a premium physical chocolate shop (200 unique SKUs, infinite customer flow, and sparse buying habits but high average customer loyalty/repeat rates). The goal is to maximize cross-selling conversion rate ($\text{pCTR}$ / $\text{pCVR}$) and order size under strict inference latency boundaries.

### ⚡ Key Architectural Innovation: The Two-Stage Pipeline
* **The Dilemma**: Running complex GBDT or Deep Learning models with 20+ features over all 200+ candidates for every request results in high serving latency (~2.39ms+).
* **The Solution**: An industrial **Recall ➔ Ranking** cascade:
  1. **Stage 1 (Recall)**: Use a lightweight **ItemCF** or rule-based generator to filter down the 200+ candidates to **10 highly relevant items** based on co-occurrence in just **0.12ms**.
  2. **Stage 2 (Ranking)**: Use a trained **LightGBM/XGBoost Classifier** or **Neural NCF Model** to rank only these 10 candidates using complex user demographics, store types, and calendar time contexts in **0.32ms**.
* **Result**: Total latency drops by **~85% (to ~0.45ms)** while maintaining peak personalization and context accuracy!

---

## 📂 Repository Structure
```
recommendation101/
├── rawdata/                      # Raw transaction database tables (CSV)
│   ├── sales.csv                 # ~900K+ transaction records (Jan 2023 – Dec 2024)
│   ├── customers.csv             # Customer demographics (age, gender, loyalty)
│   ├── products.csv              # 200 SKUs (brand, category, cocoa%, weight)
│   ├── stores.csv                # Store metadata (type, region)
│   └── calendar.csv             # Calendar/holiday features
├── itemcf/
│   ├── itemcf_recommender.py    # Cosine co-occurrence ItemCF Candidate Generator
│   └── README.md                # Detailed ItemCF & Two-Stage hybrid explanation
├── mba/
│   ├── market_basket_analysis.py # Unsupervised Apriori Association Rules Mining
│   ├── mba_results.md           # Top mined cross-selling association rules
│   └── README.md                # Mathematical metrics (Support, Confidence, Lift)
├── xg_boost/
│   └── xgboost_recommender.py   # Feature engineering & XGBClassifier ranker
├── lightgbm/
│   ├── lightgbm_recommender.py  # High-speed LightGBMClassifier ranker
│   └── README.md                # Training, negative sampling & feature injection flow
├── neural/
│   ├── neural_recommender.py    # PyTorch Deep Learning NCF model with aux features
│   └── README.md                # Backpropagation, vector embeddings & PyTorch DataLoader
├── simulation/
│   ├── train_and_save.py        # ⚠️ Offline training script — must run before server
│   ├── server.py                # Micro-webserver loading pre-trained models & serving APIs
│   ├── models/                  # Serialised .pkl model artefacts (git-ignored)
│   ├── index.html               # Interactive Chinese Sandbox UI (Glassmorphic dark theme)
│   └── index_en.html            # Interactive English Sandbox UI (dataset attribute alignment)
├── calculate_auc.py             # AUC/ROC curve validation evaluator
├── eda.py                       # Exploratory Data Analysis & customer profiling
├── evaluate.py                  # Chronological Split (Jan–Oct 2024 Train / Nov–Dec 2024 Test) & Metric Evaluator
├── run_comparison.py            # One-click command to train and compare all offline models
├── model_comparison_results.md  # Auto-generated benchmark report comparing all models
└── README.md                    # System master guide (This file)
```

---

## ⚙️ Algorithm Classification: Negative Sampling Strategy

A core design distinction divides the recommendation engines in this repository:

| Model Category | Representative Models | Requires Negative Sampling? | Mathematical Core & Loss Function |
| :--- | :--- | :---: | :--- |
| **Supervised Classification & Ranking** | `LightGBM`, `XGBoost`, `Neural (NCF)` | **🟢 YES** | Models recommendation as a binary purchase prediction ($P(\text{Buy} = 1.0)$). Synthesizes non-purchase samples (`0.0` labels) to build the decision boundary via `LogLoss` or Binary Cross-Entropy (`BCELoss`). |
| **Unsupervised Heuristics & Statistical Rules** | `ItemCF`, `Market Basket Analysis (Apriori)` | **🔴 NO** | Learns patterns strictly from positive transactional records. Computes co-occurrence frequencies, cosine similarities, or rule metrics (Support, Confidence, Lift). |

---

## ❄️ Real-time Cold-Start Guest Resolution
For unidentified guest users with no historical database record, the sandbox implements a highly decoupled **In-memory Feature Injection Wrapper**:
1. **Dynamic Guest Registry**: When a request without a history arrives, `server.py` creates a temporary `GUEST_XXXX` session ID.
2. **Context-Aware Inference**: The backend injects the guest's stated attributes (e.g. `Gender=Female`, `Age=25`) and active shopping channel (e.g. `Store_Type=Airport`) into the loaded GBDT/Deep Learning recommender profiles in RAM on-the-fly.
3. **Graceful Fallback**:
   * **ItemCF / MBA** falls back to regional/store-specific popularity recall.
   * **LightGBM / XGBoost / Neural** natively splits on the newly-injected demographics and active LBS/store context, yielding tailored recommendations (e.g. Godiva truffles for female travelers at the airport) in microseconds without any database writes.

---

## 📊 Offline Performance Comparison
*All models evaluated using a strict temporal split (Train: Jan 2023 – Oct 2024 | Test: Nov – Dec 2024) based on `model_comparison_results.md` benchmarks. MBA is an unsupervised rule-miner and is evaluated live via lift-ranked recommendations in the simulation.*

| Recommendation Strategy | Precision@5 | Recall@5 | F1@5 | Hit Rate@5 | Inference Latency | Strengths & Best Use Cases |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **Item-based CF** | `1.04%` | `2.87%` | `1.53%` | `5.20%` | **`0.19 ms`** | Lightning-fast; excellent for strong product co-occurrences. Ideal cold-start popularity fallback & Stage 1 recall layer. |
| **Market Basket Analysis** | — | — | — | — | `< 0.5 ms` | Fully interpretable lift-based rules; no training labels required. Best for basket-level cross-sell explainability. |
| **XGBoost Ranking** | `1.07%` | `2.51%` | `1.50%` | `5.20%` | `2.96 ms` | Strong feature combination logic; good interpretability via SHAP. Higher latency at full-catalogue ranking. |
| **LightGBM Ranking** | **`1.15%`** | **`2.65%`** | **`1.60%`** | **`5.70%`** | `3.09 ms` | Peak precision; extremely fast training; handles large datasets efficiently. |
| **Neural NCF** | `0.84%` | `1.99%` | `1.18%` | `4.15%` | `0.50 ms` | Latent embedding representations; captures dense hidden user–item interactions. |
| **Two-Stage Hybrid (CF + LGBM)** | `1.10%` | `2.59%` | — | `5.45%` | **`0.45 ms`** | **Optimal industrial compromise**: ~85% latency reduction vs. standalone LGBM while retaining ~95% of peak precision. |

---

## 🚀 Getting Started

### 1. Prerequisites & Virtual Environment Setup
Ensure you have **Python 3.8+** installed. All dependencies (`lightgbm`, `xgboost`, `pandas`, `numpy`, `scikit-learn`, `torch`, `mlxtend`) are pre-configured in the virtual environment.

```bash
# Activate virtual environment
source .venv/bin/activate
```

### 2. Run the Offline Benchmark
Train all offline recommenders (ItemCF, XGBoost, LightGBM, Neural NCF) on 900K+ historical records, evaluate on the chronologically split test set, and write a detailed comparison to `model_comparison_results.md`:

```bash
python run_comparison.py
```

### 3. Pre-train & Serialise Models for the Simulation Server
The simulation server loads **pre-trained pickled models** rather than training at startup. Run this once (or after any code changes) to train all five models — including MBA — and save them to `simulation/models/`:

```bash
python simulation/train_and_save.py
```

> ⚠️ **This step is required before launching the server.** The server will exit with an error if the model `.pkl` files are missing from `simulation/models/`.

### 4. Launch the Online Simulation Server
Start the lightweight HTTP recommendation server to serve the frontend sandboxes and `/api/*` endpoints:

```bash
python simulation/server.py
```

Upon startup, the server reads all serialised model artefacts from `simulation/models/` and begins listening on port `5001`.

### 5. Open the Interactive Sandboxes
Open your browser and navigate to:
* 🇨🇳 **Chinese Version**: [http://localhost:5001/](http://localhost:5001/)
* 🇬🇧 **English Version**: [http://localhost:5001/en](http://localhost:5001/en) *(Recommended for direct comparison against dataset parameter attributes)*

---

## 🛡️ License
This project is proprietary and built for high-performance retail algorithmic recommendation simulations.
