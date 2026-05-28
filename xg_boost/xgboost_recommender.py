import pandas as pd
import numpy as np
import xgboost as xgb
import os
from collections import defaultdict, Counter
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from evaluate import BaseRecommender

class XGBoostRecommender(BaseRecommender):
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
        print("Training XGBoost Model...")
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
        print("  Computing user-item interaction frequencies...")
        for row in train_sales.itertuples(index=False):
            cid, pid = row.customer_id, row.product_id
            self.user_item_counts[(cid, pid)] += 1
            
            p_info = products[products['product_id'] == pid]
            if not p_info.empty:
                brand = p_info.iloc[0]['brand']
                category = p_info.iloc[0]['category']
                self.user_brand_counts[(cid, brand)] += 1
                self.user_category_counts[(cid, category)] += 1
                
        # 2. Build Supervised Dataset (Sampling Positive and Negative Labels)
        print("  Constructing supervised training samples (Negative Sampling)...")
        # Subsample positive cases to train efficiently
        np.random.seed(42)
        sample_size = min(50000, len(train_sales))
        pos_samples = train_sales.sample(n=sample_size, random_state=42).copy()
        
        pos_samples['label'] = 1
        
        # Negative sampling (4 negatives per positive)
        neg_data = []
        for row in pos_samples.itertuples():
            cid = row.customer_id
            order_date = row.order_date
            store_id = row.store_id
            qty = row.quantity
            
            # Select 4 random products the user hasn't bought or doesn't buy constantly
            potential_negs = [p for p in self.all_product_ids if p != row.product_id]
            negs = np.random.choice(potential_negs, size=4, replace=False)
            
            for neg_pid in negs:
                neg_data.append({
                    'customer_id': cid,
                    'product_id': neg_pid,
                    'order_date': order_date,
                    'store_id': store_id,
                    'quantity': qty,
                    'discount': 0.0,
                    'label': 0
                })
        neg_samples = pd.DataFrame(neg_data)
        
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
        
        # Encode Categoricals
        X['user_age'] = train_df['age'].fillna(35)
        X['user_gender'] = self._encode_series(train_df['gender'], self.gender_map, 2)
        X['user_loyalty'] = train_df['loyalty_member'].fillna(0).astype(int)
        
        X['item_cocoa'] = train_df['cocoa_percent'].fillna(0.0).astype(float)
        X['item_weight'] = train_df['weight_g'].fillna(0.0).astype(float)
        X['item_category'] = self._encode_series(train_df['category'], self.category_map, 5)
        X['item_brand'] = self._encode_series(train_df['brand'], self.brand_map, 6)
        
        X['store_type'] = self._encode_series(train_df['store_type'], self.store_type_map, 4)
        X['day_of_week'] = train_df['day_of_week'].fillna(0).astype(int)
        X['month'] = train_df['month'].fillna(1).astype(int)
        
        # Merge stats features
        print("  Injecting aggregated statistics and interaction features...")
        u_p = [self.user_stats.get(cid, {'user_total_purchases':0, 'user_avg_revenue':0.0, 'user_avg_discount':0.0}) for cid in train_df['customer_id']]
        i_s = [self.item_stats.get(pid, {'item_total_sales':0, 'item_avg_discount':0.0}) for pid in train_df['product_id']]
        
        X['user_total_purchases'] = [item['user_total_purchases'] for item in u_p]
        X['user_avg_revenue'] = [item['user_avg_revenue'] for item in u_p]
        X['user_avg_discount'] = [item['user_avg_discount'] for item in u_p]
        
        X['item_total_sales'] = [item['item_total_sales'] for item in i_s]
        X['item_avg_discount'] = [item['item_avg_discount'] for item in i_s]
        
        # User-Item interaction history
        X['user_item_purchase_count'] = [self.user_item_counts[(cid, pid)] for cid, pid in zip(train_df['customer_id'], train_df['product_id'])]
        X['user_brand_purchase_count'] = [self.user_brand_counts[(cid, brand)] for cid, brand in zip(train_df['customer_id'], train_df['brand'])]
        X['user_category_purchase_count'] = [self.user_category_counts[(cid, cat)] for cid, cat in zip(train_df['customer_id'], train_df['category'])]
        
        y = train_df['label']
        
        # 5. Fit XGBoost Classifier
        print("  Fitting XGBClassifier...")
        self.model = xgb.XGBClassifier(
            n_estimators=120,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            n_jobs=-1
        )
        self.model.fit(X, y)
        print("  XGBoost training complete.")

    def recommend(self, user_id, k=5, store_id=None, order_date=None):
        # We need to rank all 202 product candidates for this user
        candidates = self.products_df.copy()
        
        # Build features for candidates
        num_candidates = len(candidates)
        
        X_cand = pd.DataFrame()
        
        # 1. User properties
        u_stats = self.user_stats.get(user_id, {'user_total_purchases':0, 'user_avg_revenue':0.0, 'user_avg_discount':0.0})
        
        # We don't have the customers dataframe in recommend, but we can infer user profile or store it
        # Wait, let's pre-store customer profiles during fit!
        # Let's save a dictionary self.customer_profiles during fit
        profile = self.customer_profiles.get(user_id, {'age': 35, 'gender': 'Unknown', 'loyalty_member': 0})
        
        X_cand['user_age'] = [profile['age']] * num_candidates
        X_cand['user_gender'] = [self.gender_map.get(profile['gender'], 2)] * num_candidates
        X_cand['user_loyalty'] = [profile['loyalty_member']] * num_candidates
        
        # 2. Product properties
        X_cand['item_cocoa'] = candidates['cocoa_percent'].fillna(0.0).values
        X_cand['item_weight'] = candidates['weight_g'].fillna(0.0).values
        X_cand['item_category'] = self._encode_series(candidates['category'], self.category_map, 5).values
        X_cand['item_brand'] = self._encode_series(candidates['brand'], self.brand_map, 6).values
        
        # 3. Context properties
        # In evaluate.py, store_id and order_date are passed!
        # If store_id or order_date is missing, fall back
        stype = self.store_types.get(store_id, 'Unknown') if store_id else 'Unknown'
        X_cand['store_type'] = [self.store_type_map.get(stype, 4)] * num_candidates
        
        odate = pd.to_datetime(order_date) if order_date else pd.to_datetime('2024-11-15')
        X_cand['day_of_week'] = [odate.dayofweek] * num_candidates
        X_cand['month'] = [odate.month] * num_candidates
        
        # 4. Stats features
        X_cand['user_total_purchases'] = [u_stats['user_total_purchases']] * num_candidates
        X_cand['user_avg_revenue'] = [u_stats['user_avg_revenue']] * num_candidates
        X_cand['user_avg_discount'] = [u_stats['user_avg_discount']] * num_candidates
        
        i_sales = []
        i_avg_discs = []
        ui_purchases = []
        ub_purchases = []
        uc_purchases = []
        
        for row in candidates.itertuples(index=False):
            pid = row.product_id
            brand = row.brand
            cat = row.category
            
            i_stats = self.item_stats.get(pid, {'item_total_sales': 0, 'item_avg_discount': 0.0})
            i_sales.append(i_stats['item_total_sales'])
            i_avg_discs.append(i_stats['item_avg_discount'])
            
            ui_purchases.append(self.user_item_counts[(user_id, pid)])
            ub_purchases.append(self.user_brand_counts[(user_id, brand)])
            uc_purchases.append(self.user_category_counts[(user_id, cat)])
            
        X_cand['item_total_sales'] = i_sales
        X_cand['item_avg_discount'] = i_avg_discs
        
        X_cand['user_item_purchase_count'] = ui_purchases
        X_cand['user_brand_purchase_count'] = ub_purchases
        X_cand['user_category_purchase_count'] = uc_purchases
        
        # Ensure column ordering is exactly the same as during training!
        # XGBoost requires columns to be in identical order.
        cols_order = [
            'user_age', 'user_gender', 'user_loyalty',
            'item_cocoa', 'item_weight', 'item_category', 'item_brand',
            'store_type', 'day_of_week', 'month',
            'user_total_purchases', 'user_avg_revenue', 'user_avg_discount',
            'item_total_sales', 'item_avg_discount',
            'user_item_purchase_count', 'user_brand_purchase_count', 'user_category_purchase_count'
        ]
        X_cand = X_cand[cols_order]
        
        # Predict probability of purchase
        probs = self.model.predict_proba(X_cand)[:, 1]
        
        # Rank candidates
        candidates['pred_prob'] = probs
        ranked = candidates.sort_values(by='pred_prob', ascending=False)
        
        return ranked['product_id'].head(k).tolist()

    # Auxiliary variables to store profiles during fit
    def fit_extra(self, train_sales, products, customers, stores):
        # Store customer profiles for fast recommend lookup
        print("  Caching profile maps...")
        self.customer_profiles = {}
        for row in customers.itertuples(index=False):
            self.customer_profiles[row.customer_id] = {
                'age': row.age if not pd.isna(row.age) else 35,
                'gender': row.gender if not pd.isna(row.gender) else 'Unknown',
                'loyalty_member': row.loyalty_member if not pd.isna(row.loyalty_member) else 0
            }
            
        # Store store type map for context lookup
        self.store_types = stores.set_index('store_id')['store_type'].to_dict()
