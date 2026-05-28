import math
import os
import pandas as pd
from collections import defaultdict, Counter
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from evaluate import BaseRecommender

class ItemCFRecommender(BaseRecommender):
    def __init__(self):
        self.user_items = {}       # customer_id -> set of purchased product_ids
        self.item_counts = {}      # product_id -> total purchases
        self.similarity = {}       # product_id -> {other_product_id -> sim_score}
        self.popular_items = []    # List of top popular products (fallback)
        
    def fit(self, train_sales, products, customers, stores):
        print("Training ItemCF Model...")
        
        # 1. Group train purchases by user
        print("  Grouping transactions by user...")
        # To save memory and process fast, extract values directly
        user_items_dict = defaultdict(set)
        for row in train_sales[['customer_id', 'product_id']].itertuples(index=False):
            user_items_dict[row.customer_id].add(row.product_id)
        self.user_items = dict(user_items_dict)
        
        # 2. Count occurrences of each item
        print("  Counting item purchases...")
        self.item_counts = train_sales['product_id'].value_counts().to_dict()
        self.popular_items = [item for item, _ in Counter(train_sales['product_id']).most_common(50)]
        
        # 3. Calculate Co-occurrence matrix
        print("  Building co-occurrence matrix...")
        co_occurrence = defaultdict(lambda: defaultdict(int))
        for user, items in self.user_items.items():
            item_list = list(items)
            n = len(item_list)
            for i in range(n):
                for j in range(i + 1, n):
                    item_i = item_list[i]
                    item_j = item_list[j]
                    co_occurrence[item_i][item_j] += 1
                    co_occurrence[item_j][item_i] += 1
                    
        # 4. Calculate Cosine Similarity Matrix
        print("  Calculating Cosine Similarity matrix...")
        self.similarity = defaultdict(dict)
        for item_i, related_items in co_occurrence.items():
            for item_j, count in related_items.items():
                denom = math.sqrt(self.item_counts[item_i] * self.item_counts[item_j])
                if denom > 0:
                    self.similarity[item_i][item_j] = count / denom
                    
        self.similarity = dict(self.similarity)
        print(f"  Training complete. Indexed {len(self.similarity)} items with non-zero similarities.")
        
    def recommend(self, user_id, k=5, store_id=None, order_date=None):
        # Retrieve user history
        history = self.user_items.get(user_id, set())
        
        # Score candidates
        scores = defaultdict(float)
        for hist_item in history:
            sim_items = self.similarity.get(hist_item, {})
            for sim_item, sim_score in sim_items.items():
                if sim_item in history:
                    continue
                scores[sim_item] += sim_score
                
        # Sort and recommend
        recommended = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        rec_ids = [item for item, score in recommended[:k]]
        
        # Fallback to popular items if we don't have enough recommendations (cold start or low history)
        if len(rec_ids) < k:
            for item in self.popular_items:
                if item not in history and item not in rec_ids:
                    rec_ids.append(item)
                    if len(rec_ids) == k:
                        break
                        
        return rec_ids[:k]
