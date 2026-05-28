import pandas as pd
import numpy as np
import time
import os
from collections import defaultdict

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "rawdata")

def load_data():
    """Loads all dataset files and performs chronological train/test split."""
    print("Loading datasets...")
    sales = pd.read_csv(os.path.join(DATA_DIR, "sales.csv"))
    products = pd.read_csv(os.path.join(DATA_DIR, "products.csv"))
    customers = pd.read_csv(os.path.join(DATA_DIR, "customers.csv"))
    stores = pd.read_csv(os.path.join(DATA_DIR, "stores.csv"))
    
    # Chronological Split
    # Train: 2023-01-01 to 2024-10-31
    # Test: 2024-11-01 to 2024-12-31
    print("Splitting data chronologically...")
    sales['order_date'] = pd.to_datetime(sales['order_date'])
    split_date = pd.to_datetime('2024-11-01')
    
    train_sales = sales[sales['order_date'] < split_date].copy()
    test_sales = sales[sales['order_date'] >= split_date].copy()
    
    print(f"Train sales shape: {train_sales.shape}")
    print(f"Test sales shape: {test_sales.shape}")
    
    return train_sales, test_sales, products, customers, stores

class BaseRecommender:
    def fit(self, train_sales, products, customers, stores):
        """Train the recommender model."""
        pass
        
    def recommend(self, user_id, k=5, store_id=None, order_date=None):
        """Return a list of top-k recommended product_ids."""
        return []

def evaluate_recommender(recommender, train_sales, test_sales, products, k=5, sample_users=2000):
    """
    Evaluates a recommender model using offline metrics.
    We evaluate on a random sample of users present in the test set to ensure fast execution.
    """
    print(f"Evaluating recommender on {sample_users} test users (Top-{k} recommendations)...")
    
    # Group test purchases by user
    test_purchases = test_sales.groupby('customer_id')['product_id'].apply(set).to_dict()
    
    # We also need store context for context-aware recommenders (like XGBoost)
    # Let's map each user to their most common store in the test set
    user_stores = test_sales.groupby('customer_id')['store_id'].agg(lambda x: x.mode()[0]).to_dict()
    # Most common order date for context
    user_dates = test_sales.groupby('customer_id')['order_date'].agg(lambda x: x.mode()[0]).to_dict()
    
    test_users = list(test_purchases.keys())
    
    # Seed for deterministic evaluation
    np.random.seed(42)
    eval_users = np.random.choice(test_users, size=min(sample_users, len(test_users)), replace=False)
    
    precisions = []
    recalls = []
    hit_rates = []
    
    start_time = time.time()
    
    for i, user_id in enumerate(eval_users):
        actual_items = test_purchases[user_id]
        store_id = user_stores.get(user_id)
        order_date = user_dates.get(user_id)
        
        # Get recommendations
        rec_items = recommender.recommend(user_id, k=k, store_id=store_id, order_date=order_date)
        
        # Ensure we have a list/array of items
        rec_items = list(rec_items)[:k]
        
        # Compute metrics
        intersection = actual_items.intersection(set(rec_items))
        
        # Precision@K = (actual items recommended) / K
        precision = len(intersection) / k
        precisions.append(precision)
        
        # Recall@K = (actual items recommended) / (total actual items)
        recall = len(intersection) / len(actual_items) if len(actual_items) > 0 else 0.0
        recalls.append(recall)
        
        # HitRate@K = 1 if at least one actual item is recommended, else 0
        hit = 1 if len(intersection) > 0 else 0
        hit_rates.append(hit)
        
    end_time = time.time()
    total_latency_ms = (end_time - start_time) * 1000
    avg_latency_ms = total_latency_ms / len(eval_users)
    
    mean_precision = np.mean(precisions)
    mean_recall = np.mean(recalls)
    mean_hit_rate = np.mean(hit_rates)
    
    f1 = 2 * (mean_precision * mean_recall) / (mean_precision + mean_recall) if (mean_precision + mean_recall) > 0 else 0.0
    
    metrics = {
        'Precision@K': mean_precision,
        'Recall@K': mean_recall,
        'F1-Score@K': f1,
        'HitRate@K': mean_hit_rate,
        'AvgLatency(ms)': avg_latency_ms
    }
    
    print("\nEvaluation Results:")
    for metric, val in metrics.items():
        if metric != 'AvgLatency(ms)':
            print(f"  {metric}: {val*100:.4f}%")
        else:
            print(f"  {metric}: {val:.4f} ms")
            
    return metrics
