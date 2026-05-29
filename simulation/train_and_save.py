import os
# Force single-threaded execution for OpenMP and MKL to prevent library conflict deadlocks on Mac
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import sys
import pickle
import time
import pandas as pd
import numpy as np
from collections import defaultdict, Counter

# Add parent directory to path to import models
base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(base_path)
sys.path.append(os.path.join(base_path, "itemcf"))
sys.path.append(os.path.join(base_path, "xg_boost"))
sys.path.append(os.path.join(base_path, "lightgbm"))
sys.path.append(os.path.join(base_path, "neural"))

from evaluate import load_data, BaseRecommender
from itemcf_recommender import ItemCFRecommender
from xgboost_recommender import XGBoostRecommender
from lightgbm_recommender import LightGBMRecommender
from neural_recommender import NeuralRecommender

class MBARecommender(BaseRecommender):
    """Association Rules Recommender based on Market Basket Analysis."""
    def __init__(self):
        self.rules = {}
        self.popular_items = []
        self.user_items = {}
        
    def fit(self, train_sales, products, customers, stores):
        self.popular_items = [item for item, _ in Counter(train_sales['product_id']).most_common(50)]
        self.user_items = train_sales.groupby('customer_id')['product_id'].apply(set).to_dict()
        
        co_occurrence = defaultdict(lambda: defaultdict(int))
        item_counts = train_sales['product_id'].value_counts().to_dict()
        num_users = len(self.user_items)
        
        for user, items in self.user_items.items():
            item_list = list(items)
            n = len(item_list)
            for i in range(n):
                for j in range(i + 1, n):
                    co_occurrence[item_list[i]][item_list[j]] += 1
                    co_occurrence[item_list[j]][item_list[i]] += 1
                    
        self.rules = defaultdict(list)
        for item_i, related_items in co_occurrence.items():
            for item_j, count in related_items.items():
                support_j = item_counts[item_j] / num_users
                confidence = count / item_counts[item_i]
                lift = confidence / support_j
                
                if lift > 1.0:
                    self.rules[item_i].append((item_j, lift))
                    
        # Sort rules by lift descending
        for item in self.rules:
            self.rules[item] = sorted(self.rules[item], key=lambda x: x[1], reverse=True)
            
        self.rules = dict(self.rules)

    def recommend(self, user_id, k=5, store_id=None, order_date=None):
        history = self.user_items.get(user_id, set())
        candidates = defaultdict(float)
        for hist_item in history:
            for sim_item, lift in self.rules.get(hist_item, []):
                if sim_item not in history:
                    candidates[sim_item] += lift
                    
        rec_ids = [item for item, score in sorted(candidates.items(), key=lambda x: x[1], reverse=True)[:k]]
        
        if len(rec_ids) < k:
            for item in self.popular_items:
                if item not in history and item not in rec_ids:
                    rec_ids.append(item)
                    if len(rec_ids) == k:
                        break
        return rec_ids[:k]

def main():
    print("====================================================")
    print("      Chocolate Recommender Offline Training        ")
    print("====================================================\n")
    
    # 1. Load Data
    train_sales, test_sales, products, customers, stores = load_data()
    
    models_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
    os.makedirs(models_dir, exist_ok=True)
    
    # We will train on the full training set (189k or 916k rows depending on the dataset loaded)
    print(f"\n[Training] Total training sales records: {len(train_sales)}")
    
    # 1. ItemCF Recommender
    print("\n--- Training Item-based CF Model ---")
    t0 = time.time()
    itemcf = ItemCFRecommender()
    itemcf.fit(train_sales, products, customers, stores)
    with open(os.path.join(models_dir, "itemcf.pkl"), "wb") as f:
        pickle.dump(itemcf, f)
    print(f"ItemCF saved successfully in {time.time() - t0:.2f}s")
    
    # 2. XGBoost Recommender
    print("\n--- Training XGBoost Model ---")
    t0 = time.time()
    xgb_model = XGBoostRecommender()
    xgb_model.fit(train_sales, products, customers, stores)
    with open(os.path.join(models_dir, "xgboost.pkl"), "wb") as f:
        pickle.dump(xgb_model, f)
    print(f"XGBoost saved successfully in {time.time() - t0:.2f}s")
    
    # 3. LightGBM Recommender
    print("\n--- Training LightGBM Model ---")
    t0 = time.time()
    lgb_model = LightGBMRecommender()
    lgb_model.fit(train_sales, products, customers, stores)
    with open(os.path.join(models_dir, "lightgbm.pkl"), "wb") as f:
        pickle.dump(lgb_model, f)
    print(f"LightGBM saved successfully in {time.time() - t0:.2f}s")
    
    # 4. Neural NCF Recommender
    print("\n--- Training Neural NCF Model ---")
    t0 = time.time()
    neural_model = NeuralRecommender(epochs=3) # High performance NCF
    neural_model.fit(train_sales, products, customers, stores)
    with open(os.path.join(models_dir, "neural.pkl"), "wb") as f:
        pickle.dump(neural_model, f)
    print(f"Neural NCF saved successfully in {time.time() - t0:.2f}s")
    
    # 5. Market Basket Analysis (MBA) Recommender
    print("\n--- Training Market Basket Analysis (MBA) Model ---")
    t0 = time.time()
    mba_model = MBARecommender()
    mba_model.fit(train_sales, products, customers, stores)
    with open(os.path.join(models_dir, "mba.pkl"), "wb") as f:
        pickle.dump(mba_model, f)
    print(f"MBA saved successfully in {time.time() - t0:.2f}s")
    
    # 6. Save Metadata & Caches
    print("\n--- Saving Metadata Caches ---")
    products_dict = products.set_index('product_id').to_dict(orient='index')
    customers_dict = customers.set_index('customer_id').to_dict(orient='index')
    stores_dict = stores.set_index('store_id').to_dict(orient='index')
    
    # Select 100 sample customers for UI selection
    np.random.seed(42)
    sample_cids = np.random.choice(customers['customer_id'].tolist(), size=100, replace=False)
    sample_customers = [{**{'customer_id': cid}, **customers_dict[cid]} for cid in sample_cids]
    sample_stores = [{**{'store_id': sid}, **stores_dict[sid]} for sid in stores['store_id'].tolist()]
    
    metadata = {
        'products': products_dict,
        'customers': customers_dict,
        'stores': stores_dict,
        'sample_customers': sample_customers,
        'sample_stores': sample_stores
    }
    with open(os.path.join(models_dir, "metadata.pkl"), "wb") as f:
        pickle.dump(metadata, f)
    print("Metadata caches saved successfully!")
    
    print("\n====================================================")
    print("      Offline Training Complete! All Models Saved!  ")
    print("====================================================")

if __name__ == "__main__":
    main()
