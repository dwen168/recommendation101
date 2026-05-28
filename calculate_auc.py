import pandas as pd
import numpy as np
import xgboost as xgb
import os
import sys
from collections import defaultdict, Counter
from sklearn.metrics import roc_auc_score

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "rawdata")

def calculate_model_auc():
    print("Loading datasets for AUC calculation...")
    sales = pd.read_csv(os.path.join(DATA_DIR, "sales.csv"))
    products = pd.read_csv(os.path.join(DATA_DIR, "products.csv"))
    customers = pd.read_csv(os.path.join(DATA_DIR, "customers.csv"))
    stores = pd.read_csv(os.path.join(DATA_DIR, "stores.csv"))
    
    # Chronological Split
    sales['order_date'] = pd.to_datetime(sales['order_date'])
    split_date = pd.to_datetime('2024-11-01')
    
    train_sales = sales[sales['order_date'] < split_date].copy()
    test_sales = sales[sales['order_date'] >= split_date].copy()
    
    # Maps
    gender_map = {'Male': 0, 'Female': 1, 'Unknown': 2}
    brand_map = {'Ferrero': 0, 'Cadbury': 1, 'Lindt': 2, 'Mars': 3, 'Godiva': 4, 'Hershey': 5, 'Unknown': 6}
    category_map = {'Praline': 0, 'White': 1, 'Dark': 2, 'Truffle': 3, 'Milk': 4, 'Unknown': 5}
    store_type_map = {'Airport': 0, 'Mall': 1, 'Online': 2, 'Retail': 3, 'Unknown': 4}
    
    # 1. Compute stats on train
    print("Computing train statistics...")
    user_grp = train_sales.groupby('customer_id')
    user_counts = user_grp.size().to_dict()
    user_avg_rev = user_grp['revenue'].mean().to_dict()
    user_avg_disc = user_grp['discount'].mean().to_dict()
    
    user_stats = {}
    for cid in customers['customer_id']:
        user_stats[cid] = {
            'user_total_purchases': user_counts.get(cid, 0),
            'user_avg_revenue': user_avg_rev.get(cid, 0.0),
            'user_avg_discount': user_avg_disc.get(cid, 0.0)
        }
        
    item_grp = train_sales.groupby('product_id')
    item_counts = item_grp.size().to_dict()
    item_avg_disc = item_grp['discount'].mean().to_dict()
    
    item_stats = {}
    for pid in products['product_id']:
        item_stats[pid] = {
            'item_total_sales': item_counts.get(pid, 0),
            'item_avg_discount': item_avg_disc.get(pid, 0.0)
        }
        
    user_item_counts = defaultdict(int)
    user_brand_counts = defaultdict(int)
    user_category_counts = defaultdict(int)
    for row in train_sales.itertuples(index=False):
        cid, pid = row.customer_id, row.product_id
        user_item_counts[(cid, pid)] += 1
        p_info = products[products['product_id'] == pid]
        if not p_info.empty:
            brand = p_info.iloc[0]['brand']
            category = p_info.iloc[0]['category']
            user_brand_counts[(cid, brand)] += 1
            user_category_counts[(cid, category)] += 1
            
    all_product_ids = products['product_id'].tolist()
    
    # 2. Build Train Dataset
    print("Building train samples...")
    np.random.seed(42)
    pos_train = train_sales.sample(n=50000, random_state=42).copy()
    pos_train['label'] = 1
    
    neg_train_data = []
    for row in pos_train.itertuples():
        potential_negs = [p for p in all_product_ids if p != row.product_id]
        negs = np.random.choice(potential_negs, size=4, replace=False)
        for neg_pid in negs:
            neg_train_data.append({
                'customer_id': row.customer_id, 'product_id': neg_pid, 'order_date': row.order_date,
                'store_id': row.store_id, 'quantity': row.quantity, 'discount': 0.0, 'label': 0
            })
    neg_train = pd.DataFrame(neg_train_data)
    train_df = pd.concat([pos_train[['customer_id', 'product_id', 'order_date', 'store_id', 'quantity', 'discount', 'label']], neg_train], ignore_index=True)
    
    # 3. Build Test Dataset (for classification evaluation)
    print("Building test samples...")
    pos_test = test_sales.sample(n=10000, random_state=42).copy()
    pos_test['label'] = 1
    
    neg_test_data = []
    for row in pos_test.itertuples():
        potential_negs = [p for p in all_product_ids if p != row.product_id]
        negs = np.random.choice(potential_negs, size=4, replace=False)
        for neg_pid in negs:
            neg_test_data.append({
                'customer_id': row.customer_id, 'product_id': neg_pid, 'order_date': row.order_date,
                'store_id': row.store_id, 'quantity': row.quantity, 'discount': 0.0, 'label': 0
            })
    neg_test = pd.DataFrame(neg_test_data)
    test_df = pd.concat([pos_test[['customer_id', 'product_id', 'order_date', 'store_id', 'quantity', 'discount', 'label']], neg_test], ignore_index=True)
    
    def process_features(df):
        m_df = df.merge(products, on='product_id', how='left')
        m_df = m_df.merge(customers, on='customer_id', how='left')
        m_df = m_df.merge(stores, on='store_id', how='left')
        
        m_df['order_date'] = pd.to_datetime(m_df['order_date'])
        m_df['day_of_week'] = m_df['order_date'].dt.dayofweek
        m_df['month'] = m_df['order_date'].dt.month
        
        X = pd.DataFrame()
        X['user_age'] = m_df['age'].fillna(35)
        X['user_gender'] = m_df['gender'].map(gender_map).fillna(2).astype(int)
        X['user_loyalty'] = m_df['loyalty_member'].fillna(0).astype(int)
        
        X['item_cocoa'] = m_df['cocoa_percent'].fillna(0.0).astype(float)
        X['item_weight'] = m_df['weight_g'].fillna(0.0).astype(float)
        X['item_category'] = m_df['category'].map(category_map).fillna(5).astype(int)
        X['item_brand'] = m_df['brand'].map(brand_map).fillna(6).astype(int)
        
        X['store_type'] = m_df['store_type'].map(store_type_map).fillna(4).astype(int)
        X['day_of_week'] = m_df['day_of_week'].fillna(0).astype(int)
        X['month'] = m_df['month'].fillna(1).astype(int)
        
        u_p = [user_stats.get(cid, {'user_total_purchases':0, 'user_avg_revenue':0.0, 'user_avg_discount':0.0}) for cid in m_df['customer_id']]
        i_s = [item_stats.get(pid, {'item_total_sales':0, 'item_avg_discount':0.0}) for pid in m_df['product_id']]
        
        X['user_total_purchases'] = [item['user_total_purchases'] for item in u_p]
        X['user_avg_revenue'] = [item['user_avg_revenue'] for item in u_p]
        X['user_avg_discount'] = [item['user_avg_discount'] for item in u_p]
        
        X['item_total_sales'] = [item['item_total_sales'] for item in i_s]
        X['item_avg_discount'] = [item['item_avg_discount'] for item in i_s]
        
        X['user_item_purchase_count'] = [user_item_counts[(cid, pid)] for cid, pid in zip(m_df['customer_id'], m_df['product_id'])]
        X['user_brand_purchase_count'] = [user_brand_counts[(cid, brand)] for cid, brand in zip(m_df['customer_id'], m_df['brand'])]
        X['user_category_purchase_count'] = [user_category_counts[(cid, cat)] for cid, cat in zip(m_df['customer_id'], m_df['category'])]
        
        return X, m_df['label']

    print("Extracting train features...")
    X_train, y_train = process_features(train_df)
    
    print("Extracting test features...")
    X_test, y_test = process_features(test_df)
    
    print("Fitting XGBClassifier...")
    model = xgb.XGBClassifier(
        n_estimators=120,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train, y_train)
    
    print("Predicting purchase probabilities on test set...")
    probs = model.predict_proba(X_test)[:, 1]
    
    auc = roc_auc_score(y_test, probs)
    print(f"\n>>> Model Test AUC: {auc:.6f} <<<")

if __name__ == '__main__':
    calculate_model_auc()
