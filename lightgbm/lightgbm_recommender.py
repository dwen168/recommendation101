import pandas as pd
import numpy as np
try:
    import lightgbm as lgb
    HAS_LGB = True
except ImportError:
    HAS_LGB = False
import os
from collections import defaultdict, Counter
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from evaluate import BaseRecommender

class LightGBMRecommender(BaseRecommender):
    def __init__(self):
        self.model = None
        self.popular_items = []
        self.products_df = None
        
        # Encoders
        self.gender_map = {'Male': 0, 'Female': 1, 'Unknown': 2}
        self.brand_map = {'Ferrero': 0, 'Cadbury': 1, 'Lindt': 2, 'Mars': 3, 'Godiva': 4, 'Hershey': 5, 'Unknown': 6}
        self.category_map = {'Praline': 0, 'White': 1, 'Dark': 2, 'Truffle': 3, 'Milk': 4, 'Unknown': 5}
        self.store_type_map = {'Airport': 0, 'Mall': 1, 'Online': 2, 'Retail': 3, 'Unknown': 4}
        
        # Aggregated stats
        self.user_stats = {}      # user_id -> dict of stats
        self.item_stats = {}      # product_id -> dict of stats
        self.user_item_counts = defaultdict(int)      # (user_id, product_id) -> count
        self.user_brand_counts = defaultdict(int)     # (user_id, brand) -> count
        self.user_category_counts = defaultdict(int)  # (user_id, category) -> count
        
        self.all_product_ids = []
        
    def _encode_series(self, series, mapping, default_val):
        return series.map(mapping).fillna(default_val).astype(int)

    def fit(self, train_sales, products, customers, stores):
        print("Training LightGBM Model...")
        self.products_df = products.copy()
        self.all_product_ids = products['product_id'].tolist()
        self.popular_items = [item for item, _ in Counter(train_sales['product_id']).most_common(50)]
        
        # Initialize extra lookup dicts
        self.fit_extra(train_sales, products, customers, stores)
        
        # 1. Compute Aggregated Statistics from Training Set
        print("  Computing user and item statistical profiles...")
        # User stats
        user_grp = train_sales.groupby('customer_id')
        user_counts = user_grp.size().to_dict()
        user_avg_rev = user_grp['revenue'].mean().to_dict()
        user_avg_disc = user_grp['discount'].mean().to_dict()
        
        for cid in customers['customer_id']:
            self.user_stats[cid] = {
                'user_total_purchases': user_counts.get(cid, 0),
                'user_avg_revenue': user_avg_rev.get(cid, 0.0),
                'user_avg_discount': user_avg_disc.get(cid, 0.0)
            }
            
        # Item stats
        item_grp = train_sales.groupby('product_id')
        item_counts = item_grp.size().to_dict()
        item_avg_disc = item_grp['discount'].mean().to_dict()
        
        for pid in products['product_id']:
            self.item_stats[pid] = {
                'item_total_sales': item_counts.get(pid, 0),
                'item_avg_discount': item_avg_disc.get(pid, 0.0)
            }
            
        # Interaction stats
        # Interaction stats
        print("  Computing user-item interaction frequencies...")
        prod_brand_map = products.set_index('product_id')['brand'].to_dict()
        prod_cat_map = products.set_index('product_id')['category'].to_dict()
        
        user_item_freq = train_sales.groupby(['customer_id', 'product_id']).size()
        for (cid, pid), count in user_item_freq.items():
            self.user_item_counts[(cid, pid)] = count
            
        train_sales_merged = train_sales[['customer_id', 'product_id']].copy()
        train_sales_merged['brand'] = train_sales_merged['product_id'].map(prod_brand_map)
        train_sales_merged['category'] = train_sales_merged['product_id'].map(prod_cat_map)
        
        user_brand_freq = train_sales_merged.groupby(['customer_id', 'brand']).size()
        for (cid, brand), count in user_brand_freq.items():
            self.user_brand_counts[(cid, brand)] = count
            
        user_cat_freq = train_sales_merged.groupby(['customer_id', 'category']).size()
        for (cid, cat), count in user_cat_freq.items():
            self.user_category_counts[(cid, cat)] = count
                
        # 2. Build Supervised Dataset (Sampling Positive and Negative Labels)
        print("  Constructing supervised training samples (Negative Sampling)...")
        # Subsample positive cases to train efficiently
        np.random.seed(42)
        sample_size = min(50000, len(train_sales))
        pos_samples = train_sales.sample(n=sample_size, random_state=42).copy()
        
        pos_samples['label'] = 1
        
        # Negative sampling (4 negatives per positive)
        cids = []
        pids = []
        dates = []
        store_ids = []
        quantities = []
        
        all_prods = np.array(self.all_product_ids)
        for row in pos_samples.itertuples():
            negs = np.random.choice(all_prods, size=4, replace=False)
            while row.product_id in negs:
                negs = np.random.choice(all_prods, size=4, replace=False)
            
            cids.extend([row.customer_id] * 4)
            pids.extend(negs)
            dates.extend([row.order_date] * 4)
            store_ids.extend([row.store_id] * 4)
            quantities.extend([row.quantity] * 4)
            
        neg_samples = pd.DataFrame({
            'customer_id': cids,
            'product_id': pids,
            'order_date': dates,
            'store_id': store_ids,
            'quantity': quantities,
            'discount': 0.0,
            'label': 0
        })
        
        train_df = pd.concat([pos_samples[['customer_id', 'product_id', 'order_date', 'store_id', 'quantity', 'discount', 'label']], neg_samples], ignore_index=True)
        
        # 3. Merge Features
        print("  Merging user, product, store, and context features...")
        train_df = train_df.merge(products, on='product_id', how='left')
        train_df = train_df.merge(customers, on='customer_id', how='left')
        train_df = train_df.merge(stores, on='store_id', how='left')
        
        # Add calendar features
        train_df['order_date'] = pd.to_datetime(train_df['order_date'])
        train_df['day_of_week'] = train_df['order_date'].dt.dayofweek
        train_df['month'] = train_df['order_date'].dt.month
        
        # 4. Feature Extraction & Encodings
        X = pd.DataFrame()
        
        # Categorical Categories definitions
        self.gender_categories = [0, 1, 2]
        self.loyalty_categories = [0, 1]
        self.category_categories = [0, 1, 2, 3, 4, 5]
        self.brand_categories = [0, 1, 2, 3, 4, 5, 6]
        self.store_type_categories = [0, 1, 2, 3, 4]
        self.day_of_week_categories = list(range(7))
        self.month_categories = list(range(1, 13))
        
        # Encode Categoricals
        X['user_age'] = train_df['age'].fillna(35)
        X['user_gender'] = pd.Categorical(self._encode_series(train_df['gender'], self.gender_map, 2), categories=self.gender_categories)
        X['user_loyalty'] = pd.Categorical(train_df['loyalty_member'].fillna(0).astype(int), categories=self.loyalty_categories)
        
        X['item_cocoa'] = train_df['cocoa_percent'].fillna(0.0).astype(float)
        X['item_weight'] = train_df['weight_g'].fillna(0.0).astype(float)
        X['item_category'] = pd.Categorical(self._encode_series(train_df['category'], self.category_map, 5), categories=self.category_categories)
        X['item_brand'] = pd.Categorical(self._encode_series(train_df['brand'], self.brand_map, 6), categories=self.brand_categories)
        
        X['store_type'] = pd.Categorical(self._encode_series(train_df['store_type'], self.store_type_map, 4), categories=self.store_type_categories)
        X['day_of_week'] = pd.Categorical(train_df['day_of_week'].fillna(0).astype(int), categories=self.day_of_week_categories)
        X['month'] = pd.Categorical(train_df['month'].fillna(1).astype(int), categories=self.month_categories)
        
        # Merge stats features
        print("  Injecting aggregated statistics and interaction features...")
        user_total_purchases_map = {cid: stats['user_total_purchases'] for cid, stats in self.user_stats.items()}
        user_avg_revenue_map = {cid: stats['user_avg_revenue'] for cid, stats in self.user_stats.items()}
        user_avg_discount_map = {cid: stats['user_avg_discount'] for cid, stats in self.user_stats.items()}
        
        X['user_total_purchases'] = train_df['customer_id'].map(user_total_purchases_map).fillna(0).astype(int)
        X['user_avg_revenue'] = train_df['customer_id'].map(user_avg_revenue_map).fillna(0.0).astype(float)
        X['user_avg_discount'] = train_df['customer_id'].map(user_avg_discount_map).fillna(0.0).astype(float)
        
        item_total_sales_map = {pid: stats['item_total_sales'] for pid, stats in self.item_stats.items()}
        item_avg_discount_map = {pid: stats['item_avg_discount'] for pid, stats in self.item_stats.items()}
        
        X['item_total_sales'] = train_df['product_id'].map(item_total_sales_map).fillna(0).astype(int)
        X['item_avg_discount'] = train_df['product_id'].map(item_avg_discount_map).fillna(0.0).astype(float)
        
        # User-Item interaction history
        X['user_item_purchase_count'] = [self.user_item_counts.get((cid, pid), 0) for cid, pid in zip(train_df['customer_id'], train_df['product_id'])]
        X['user_brand_purchase_count'] = [self.user_brand_counts.get((cid, brand), 0) for cid, brand in zip(train_df['customer_id'], train_df['brand'])]
        X['user_category_purchase_count'] = [self.user_category_counts.get((cid, cat), 0) for cid, cat in zip(train_df['customer_id'], train_df['category'])]
        
        y = train_df['label']
        
        # 5. Fit LightGBM Classifier
        print("  Fitting LGBMClassifier...")
        self.model = lgb.LGBMClassifier(
            n_estimators=120,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            n_jobs=-1,
            verbosity=-1
        )
        self.model.fit(X, y)
        print("  LightGBM training complete.")
 
    def predict_numpy(self, rows):
        if not hasattr(self, 'np_trees') or self.np_trees is None:
            if hasattr(self.model, 'booster_'):
                d = self.model.booster_.dump_model()
                self.np_trees = [t['tree_structure'] for t in d['tree_info']]
            else:
                return np.zeros(len(rows))
        
        def eval_node(node, row):
            if 'leaf_value' in node:
                return node['leaf_value']
            feat_idx = node['split_feature']
            val = row[feat_idx]
            thresh = node['threshold']
            decision_type = node.get('decision_type', '<=')
            
            if decision_type == '==':
                cat_values = node.get('threshold', '')
                if isinstance(cat_values, str):
                    cat_list = [int(x) for x in cat_values.split('||')] if cat_values else []
                elif isinstance(cat_values, (int, float)):
                    cat_list = [int(cat_values)]
                else:
                    cat_list = []
                go_left = int(val) in cat_list
            else:
                go_left = val <= thresh
                
            if go_left:
                return eval_node(node['left_child'], row)
            else:
                return eval_node(node['right_child'], row)

        probs = []
        for row in rows:
            raw_score = sum(eval_node(tree, row) for tree in self.np_trees)
            prob = 1.0 / (1.0 + np.exp(-raw_score))
            probs.append(prob)
        return np.array(probs)

    def recommend(self, user_id, k=5, store_id=None, order_date=None):
        candidates = self.products_df.copy()
        num_candidates = len(candidates)
        
        u_stats = self.user_stats.get(user_id, {'user_total_purchases':0, 'user_avg_revenue':0.0, 'user_avg_discount':0.0})
        profile = self.customer_profiles.get(user_id, {'age': 35, 'gender': 'Unknown', 'loyalty_member': 0})
        stype = self.store_types.get(store_id, 'Unknown') if store_id else 'Unknown'
        odate = pd.to_datetime(order_date) if order_date else pd.to_datetime('2024-11-15')
        
        rows = []
        for _, cand in candidates.iterrows():
            row = [
                profile['age'],
                self.gender_map.get(profile['gender'], 2),
                profile['loyalty_member'],
                cand['cocoa_percent'] if not pd.isna(cand['cocoa_percent']) else 0.0,
                cand['weight_g'] if not pd.isna(cand['weight_g']) else 0.0,
                self.category_map.get(cand['category'], 5),
                self.brand_map.get(cand['brand'], 6),
                self.store_type_map.get(stype, 4),
                odate.dayofweek,
                odate.month,
                u_stats['user_total_purchases'],
                u_stats['user_avg_revenue'],
                u_stats['user_avg_discount'],
                self.item_stats.get(cand['product_id'], {}).get('item_total_sales', 0),
                self.item_stats.get(cand['product_id'], {}).get('item_avg_discount', 0.0),
                self.user_item_counts.get((user_id, cand['product_id']), 0),
                self.user_brand_counts.get((user_id, cand['brand']), 0),
                self.user_category_counts.get((user_id, cand['category']), 0)
            ]
            rows.append(row)
            
        probs = self.predict_numpy(rows)
        candidates['pred_prob'] = probs
        ranked = candidates.sort_values(by='pred_prob', ascending=False)
        return ranked['product_id'].head(k).tolist()

    def fit_extra(self, train_sales, products, customers, stores):
        self.customer_profiles = {}
        for row in customers.itertuples(index=False):
            self.customer_profiles[row.customer_id] = {
                'age': row.age if not pd.isna(row.age) else 35,
                'gender': row.gender if not pd.isna(row.gender) else 'Unknown',
                'loyalty_member': row.loyalty_member if not pd.isna(row.loyalty_member) else 0
            }
        self.store_types = stores.set_index('store_id')['store_type'].to_dict()
