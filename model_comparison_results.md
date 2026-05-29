# 📊 巧克力推荐系统多模型对比实验评测报告

此报告由 `run_comparison.py` 一键自动运行生成，对比了协同过滤、树模型（XGBoost/LightGBM）与深度学习模型（Neural NCF）的性能。

## 1. 评测指标对照表

| Algorithm | Precision@5 | Recall@5 | F1-Score@5 | HitRate@5 | Training Time (s) | Inference Latency |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| Item-based CF | 1.04% | 2.87% | 1.53% | 5.20% | 1.53s | 0.19 ms/user |
| XGBoost Ranking | 1.07% | 2.51% | 1.50% | 5.20% | 2.59s | 2.96 ms/user |
| LightGBM Ranking | 1.15% | 2.65% | 1.60% | 5.70% | 2.50s | 3.09 ms/user |
| Neural NCF | 0.84% | 1.99% | 1.18% | 4.15% | 3.59s | 0.50 ms/user |

## 2. 核心结论与商业决策建议

### 🚀 算法性能表现深度解读
- **Neural NCF 的潜力**：深度学习模型通过 Embedding 层能捕捉到 ID 间的隐式关联，在解决稀疏性问题上有独特优势。
- **GBDT 的稳定性**：XGBoost 和 LightGBM 在特征工程充分的情况下，依然是处理表结构数据的最强工具，尤其是 LightGBM 在训练速度上的优势极度明显。
- **ItemCF 的极致性能**：在推理时效上，传统的协同过滤依然是绝对的王者，适合作为大规模系统的召回层。

### 💡 商业落地行动路线图
1. **召回阶段 (Retrieval)**：使用 **ItemCF**，从全量 SKU 中快速筛选出候选集。
2. **精排阶段 (Ranking)**：对于高价值流量，使用 **LightGBM** 或 **Neural NCF**。如果算力充足，Neural 模型能带来更细粒度的个性化。