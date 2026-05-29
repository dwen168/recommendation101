import os
# Force single-threaded execution for OpenMP and MKL to prevent library conflict deadlocks on Mac
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import sys
import time
import pandas as pd

base_path = os.path.dirname(os.path.abspath(__file__))
sys.path.append(base_path)
sys.path.append(os.path.join(base_path, "itemcf"))
sys.path.append(os.path.join(base_path, "xg_boost"))
sys.path.append(os.path.join(base_path, "lightgbm"))
sys.path.append(os.path.join(base_path, "neural"))

from evaluate import load_data, evaluate_recommender
from itemcf_recommender import ItemCFRecommender
from xgboost_recommender import XGBoostRecommender
from lightgbm_recommender import LightGBMRecommender
from neural_recommender import NeuralRecommender

def main():
    print("====================================================")
    print("      Chocolate Shop Recommendation System          ")
    print("      Model Comparison: ItemCF vs GBDT vs Neural     ")
    print("====================================================\n")
    
    # 1. Load data
    train_sales, test_sales, products, customers, stores = load_data()
    
    # 2. Run ItemCF Recommender
    print("\n----------------------------------------------------")
    print("  Evaluating Recommender 1: ItemCF")
    print("----------------------------------------------------")
    itemcf = ItemCFRecommender()
    
    t0 = time.time()
    itemcf.fit(train_sales, products, customers, stores)
    train_time_cf = time.time() - t0
    
    cf_metrics = evaluate_recommender(
        recommender=itemcf, 
        train_sales=train_sales, 
        test_sales=test_sales, 
        products=products, 
        k=5, 
        sample_users=2000
    )
    
    # 3. Run XGBoost Recommender
    print("\n----------------------------------------------------")
    print("  Evaluating Recommender 2: XGBoost (Ranking Model)")
    print("----------------------------------------------------")
    xgb_model = XGBoostRecommender()
    
    t0 = time.time()
    xgb_model.fit(train_sales, products, customers, stores)
    train_time_xgb = time.time() - t0
    
    xgb_metrics = evaluate_recommender(
        recommender=xgb_model, 
        train_sales=train_sales, 
        test_sales=test_sales, 
        products=products, 
        k=5, 
        sample_users=2000
    )
    
    # 4. Run LightGBM Recommender
    print("\n----------------------------------------------------")
    print("  Evaluating Recommender 3: LightGBM (Ranking Model)")
    print("----------------------------------------------------")
    lgb_model = LightGBMRecommender()
    
    t0 = time.time()
    lgb_model.fit(train_sales, products, customers, stores)
    train_time_lgb = time.time() - t0
    
    lgb_metrics = evaluate_recommender(
        recommender=lgb_model, 
        train_sales=train_sales, 
        test_sales=test_sales, 
        products=products, 
        k=5, 
        sample_users=2000
    )

    # 5. Run Neural Recommender
    print("\n----------------------------------------------------")
    print("  Evaluating Recommender 4: Neural (NCF Model)")
    print("----------------------------------------------------")
    neural_model = NeuralRecommender(epochs=3) # Low epochs for quick comparison
    
    t0 = time.time()
    neural_model.fit(train_sales, products, customers, stores)
    train_time_neural = time.time() - t0
    
    neural_metrics = evaluate_recommender(
        recommender=neural_model, 
        train_sales=train_sales, 
        test_sales=test_sales, 
        products=products, 
        k=5, 
        sample_users=2000
    )
    
    # 6. Generate Comparative Report
    print("\n====================================================")
    print("              FINAL COMPARISON REPORT               ")
    print("====================================================\n")
    
    markdown_table = (
        "| Algorithm | Precision@5 | Recall@5 | F1-Score@5 | HitRate@5 | Training Time (s) | Inference Latency |\n"
        "| :--- | :---: | :---: | :---: | :---: | :---: | :---: |\n"
        f"| Item-based CF | {cf_metrics['Precision@K']*100:.2f}% | {cf_metrics['Recall@K']*100:.2f}% | {cf_metrics['F1-Score@K']*100:.2f}% | {cf_metrics['HitRate@K']*100:.2f}% | {train_time_cf:.2f}s | {cf_metrics['AvgLatency(ms)']:.2f} ms/user |\n"
        f"| XGBoost Ranking | {xgb_metrics['Precision@K']*100:.2f}% | {xgb_metrics['Recall@K']*100:.2f}% | {xgb_metrics['F1-Score@K']*100:.2f}% | {xgb_metrics['HitRate@K']*100:.2f}% | {train_time_xgb:.2f}s | {xgb_metrics['AvgLatency(ms)']:.2f} ms/user |\n"
        f"| LightGBM Ranking | {lgb_metrics['Precision@K']*100:.2f}% | {lgb_metrics['Recall@K']*100:.2f}% | {lgb_metrics['F1-Score@K']*100:.2f}% | {lgb_metrics['HitRate@K']*100:.2f}% | {train_time_lgb:.2f}s | {lgb_metrics['AvgLatency(ms)']:.2f} ms/user |\n"
        f"| Neural NCF | {neural_metrics['Precision@K']*100:.2f}% | {neural_metrics['Recall@K']*100:.2f}% | {neural_metrics['F1-Score@K']*100:.2f}% | {neural_metrics['HitRate@K']*100:.2f}% | {train_time_neural:.2f}s | {neural_metrics['AvgLatency(ms)']:.2f} ms/user |"
    )
    
    print(markdown_table)
    print("\n====================================================")
    
    # Write summary comparison to walkthrough or output file
    with open("model_comparison_results.md", "w") as f:
        f.write("# 📊 巧克力推荐系统多模型对比实验评测报告\n\n")
        f.write("此报告由 `run_comparison.py` 一键自动运行生成，对比了协同过滤、树模型（XGBoost/LightGBM）与深度学习模型（Neural NCF）的性能。\n\n")
        f.write("## 1. 评测指标对照表\n\n")
        f.write(markdown_table)
        f.write("\n\n## 2. 核心结论与商业决策建议\n\n")
        f.write("### 🚀 算法性能表现深度解读\n")
        f.write("- **Neural NCF 的潜力**：深度学习模型通过 Embedding 层能捕捉到 ID 间的隐式关联，在解决稀疏性问题上有独特优势。\n")
        f.write("- **GBDT 的稳定性**：XGBoost 和 LightGBM 在特征工程充分的情况下，依然是处理表结构数据的最强工具，尤其是 LightGBM 在训练速度上的优势极度明显。\n")
        f.write("- **ItemCF 的极致性能**：在推理时效上，传统的协同过滤依然是绝对的王者，适合作为大规模系统的召回层。\n\n")
        f.write("### 💡 商业落地行动路线图\n")
        f.write("1. **召回阶段 (Retrieval)**：使用 **ItemCF**，从全量 SKU 中快速筛选出候选集。\n")
        f.write("2. **精排阶段 (Ranking)**：对于高价值流量，使用 **LightGBM** 或 **Neural NCF**。如果算力充足，Neural 模型能带来更细粒度的个性化。")

        
    print("\nComparison results written successfully to 'model_comparison_results.md'.")

if __name__ == '__main__':
    import multiprocessing
    try:
        multiprocessing.set_start_method('spawn')
    except RuntimeError:
        pass
    main()
