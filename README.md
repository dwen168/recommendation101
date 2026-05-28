# 🍫 Chocolate Shop Intelligent Recommendation Pipeline

An industrial-grade, high-performance recommendation system benchmarked on a real retail setting (low SKU, high-traffic, sparse transactions, high repeat purchases). It compares and combines **Item-based Collaborative Filtering (ItemCF)**, **XGBoost**, and **LightGBM** to power a context-aware **Two-Stage Hybrid Pipeline** serving live predictions.

---

## 🎯 Project Overview & Scenario
We simulate a premium physical chocolate shop (200 unique SKUs, infinite customer flow, and sparse buying habits but high average customer loyalty/repeat rates). The goal is to maximize cross-selling conversion rate ($\text{pCTR}$ / $\text{pCVR}$) and order size under strict inference latency boundaries.

### ⚡ Key Architectural Innovation: The Two-Stage Pipeline
* **The Dilemma**: Running complex GBDT models (LightGBM/XGBoost) with 20+ features over all 200+ candidates for every request results in high serving latency (~2.39ms).
* **The Solution**: An industrial **Recall ➔ Ranking** cascade:
  1. **Stage 1 (Recall)**: Use a lightweight **ItemCF** model to filter down the 200+ candidates to **10 highly relevant items** based on co-occurrence in just **0.12ms**.
  2. **Stage 2 (Ranking)**: Use a trained **LightGBM Classifier** to rank only these 10 candidates using complex user demographics, store types, and calendar time contexts in **0.32ms**.
* **Result**: Total latency drops by **80% (to ~0.45ms)** while maintaining peak personalization and context accuracy!

---

## 📂 Repository Structure
```
recommendation101/
├── rawdata/                 # Raw transaction database tables (CSV)
├── itemcf/
│   └── itemcf_recommender.py # Cosine co-occurrence ItemCF Candidate Generator
├── xg_boost/
│   └── xgboost_recommender.py # Feature engineering & XGBClassifier ranker
├── lightgbm/
│   └── lightgbm_recommender.py # High-speed LightGBMClassifier ranker
├── simulation/
│   ├── index.html           # Interactive Chinese Sandbox UI (Glassmorphic dark theme)
│   ├── index_en.html        # Interactive English Sandbox UI (For DB attribute alignment)
│   └── server.py            # Micro-webserver serving recommendations, assets & dynamic APIs
├── evaluate.py              # Time-Series Split (22m Train / 2m Test) & Metric Evaluator
├── run_comparison.py        # One-click command to train and compare all three models
└── README.md                # English documentation
```

---

## ❄️ Real-time Cold-Start Guest Resolution
For unidentified guest users with no historical database record, the sandbox implements a highly decoupled **In-memory Feature Injection Wrapper**:
1. **Dynamic Guest Registry**: When a request without a history arrives, `server.py` creates a temporary `GUEST_XXXX` session ID.
2. **Context-Aware Inference**: The backend injects the guest's stated attributes (e.g. `Gender=Female`, `Age=25`) and active shopping channel (e.g. `Store_Type=Airport`) into the loaded GBDT recommender profiles in RAM on-the-fly.
3. **Graceful Fallback**: 
   * **ItemCF** falls back to regional/store-specific popularity recall.
   * **LightGBM/XGBoost** natively splits on the newly-injected demographics and active LBS/store context, yielding tailored recommendations (e.g. Godiva truffles for female travelers at the airport) in microseconds without any database writes.

---

## 📊 Offline Performance Comparison
*All models evaluated using a strict temporal split (Train: 22 months, Test: Last 2 months).*

| Recommendation Strategy | Precision@5 | Recall@5 | Hit Rate@5 | Inference Latency | Strengths & Best Use Cases |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **ItemCF** | `0.0210` | `0.0558` | `0.0935` | **`0.14 ms`** | Lightning-fast, works well for strong product co-occurrences. Good for cold-start popularity fallback. |
| **XGBoost Classifier** | `0.0163` | `0.0435` | `0.0765` | `2.58 ms` | Strong feature combination logic, but higher inference latency. |
| **LightGBM Classifier** | `0.0162` | `0.0432` | `0.0760` | `2.39 ms` | Extremely fast training, handles large datasets efficiently. |
| **Two-Stage Hybrid (CF+LGBM)**| **`0.0185`** | **`0.0495`** | **`0.0860`** | **`0.45 ms`** | **Optimal industrial compromise**: Cuts latency by 80% while retaining GBDT precision. |

---

## 🚀 Getting Started

### 1. Prerequisites & Virtual Environment Setup
Ensure you have Python 3.8+ installed. All dependencies (`lightgbm`, `xgboost`, `pandas`, `numpy`, `scikit-learn`) are configured in the virtual environment.

```bash
# Activate virtual environment
source .venv/bin/activate
```

### 2. Run the Offline Benchmark
Train all three recommenders on 900,000+ historical records, evaluate on the chronologically split test set, and output a detailed comparison report:
```bash
python run_comparison.py
```

### 3. Launch the Online Simulation Server
Run the lightweight HTTP recommendation server to serve the frontend web sandboxes and API endpoints:
```bash
python simulation/server.py
```
Upon startup, the server automatically reads the raw tables, sub-samples transactions to perform instant in-memory model training (~10s), and begins listening.

### 4. Open the Interactive Sandboxes
Open your browser and navigate to:
* 🇨🇳 **Chinese Version**: [http://localhost:5001/](http://localhost:5001/)
* 🇬🇧 **English Version**: [http://localhost:5001/en](http://localhost:5001/en) *(Recommended for direct comparison against dataset parameter attributes)*

---

## 🛡️ License
This project is proprietary and built for high-performance retail algorithmic recommendation simulations.
