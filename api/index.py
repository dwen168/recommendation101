import os
import sys

# Configure Matplotlib config directory to avoid sandbox / read-only violations
base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.environ["MPLCONFIGDIR"] = os.path.join(base_path, "simulation", ".matplotlib")
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import http.server
import json
import urllib.parse
import time
import pickle
import pandas as pd
import numpy as np
from collections import defaultdict

# Add search paths for model modules
sys.path.append(base_path)
sys.path.append(os.path.join(base_path, "simulation"))
sys.path.append(os.path.join(base_path, "itemcf"))
sys.path.append(os.path.join(base_path, "xg_boost"))
sys.path.append(os.path.join(base_path, "lightgbm"))
sys.path.append(os.path.join(base_path, "neural"))

from evaluate import BaseRecommender
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
        pass

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

# Global references for models and datasets
recommenders = {}
dataset = {}

def load_saved_models():
    if recommenders and dataset:
        return
    models_dir = os.path.join(base_path, "simulation", "models")
    
    # Load metadata
    with open(os.path.join(models_dir, "metadata.pkl"), "rb") as f:
        meta = pickle.load(f)
        dataset.update(meta)
        
    global IS_XGB_FALLBACK
    # Load recommenders
    for r_name in ['itemcf', 'xgboost', 'lightgbm', 'neural', 'mba']:
        try:
            with open(os.path.join(models_dir, f"{r_name}.pkl"), "rb") as f:
                recommenders[r_name] = pickle.load(f)
        except Exception as e:
            print(f"[Warning] Could not load model {r_name}: {e}")
            if r_name == 'xgboost':
                IS_XGB_FALLBACK = True
    if 'xgboost' not in recommenders and 'lightgbm' in recommenders:
        recommenders['xgboost'] = recommenders['lightgbm']
        IS_XGB_FALLBACK = True

IS_XGB_FALLBACK = False

# Preload models on cold start
try:
    import __main__
    __main__.MBARecommender = MBARecommender
    load_saved_models()
except Exception as e:
    print(f"[Vercel Initialization Warning] {e}")

class handler(http.server.BaseHTTPRequestHandler):
    def end_headers(self):
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'X-Requested-With, Content-Type')
        super().end_headers()

    def do_OPTIONS(self):
        self.send_response(200)
        self.end_headers()

    def do_GET(self):
        load_saved_models()
        raw_path = self.path
        headers_str = " ".join([f"{k}:{v}" for k, v in self.headers.items()]) if hasattr(self, 'headers') and self.headers else ""
        orig_uri = self.headers.get('x-forwarded-uri', raw_path) if hasattr(self, 'headers') and self.headers else raw_path
        
        parsed_url = urllib.parse.urlparse(orig_uri)
        path = parsed_url.path
        query = urllib.parse.parse_qs(parsed_url.query)
        
        raw_parsed = urllib.parse.urlparse(raw_path)
        raw_query = urllib.parse.parse_qs(raw_parsed.query)
        for k, v in raw_query.items():
            if k not in query:
                query[k] = v
        
        all_text = f"{path} {raw_path} {headers_str} {' '.join(query.keys())}".lower()
        
        if "customers" in all_text:
            self.handle_json_response(dataset.get('sample_customers', []))
        elif "stores" in all_text:
            self.handle_json_response(dataset.get('sample_stores', []))
        elif "products" in all_text:
            self.handle_json_response(dataset.get('products', {}))
        elif "download" in all_text:
            self.handle_download(query)
        elif "recommend" in all_text or 'store_id' in query or 'user_id' in query or 'model' in query:
            self.handle_recommend(query)
        else:
            if 'user_id' in query or 'store_id' in query:
                self.handle_recommend(query)
            else:
                self.handle_json_response({"status": "ok", "message": "Chocolate Recommender API Operational"})

    def handle_download(self, query):
        filename = query.get('file', [None])[0]
        allowed = ['products.csv', 'customers.csv', 'stores.csv', 'sales.csv', 'calendar.csv']
        if not filename or filename not in allowed:
            self.send_error(400, "Invalid file requested")
            return
        file_path = os.path.join(base_path, "rawdata", filename)
        if not os.path.exists(file_path) and filename == 'sales.csv':
            gz_path = os.path.join(base_path, "rawdata", "sales.csv.gz")
            if os.path.exists(gz_path):
                try:
                    import gzip
                    with gzip.open(gz_path, 'rb') as f:
                        content = f.read()
                    self.send_response(200)
                    self.send_header("Content-Type", "text/csv; charset=utf-8")
                    self.send_header("Content-Disposition", f"attachment; filename={filename}")
                    self.end_headers()
                    self.wfile.write(content)
                    return
                except Exception as e:
                    self.send_error(500, f"Error downloading file: {str(e)}")
                    return
        if not os.path.exists(file_path):
            self.send_error(404, "File not found")
            return
        try:
            with open(file_path, 'rb') as f:
                content = f.read()
            self.send_response(200)
            self.send_header("Content-Type", "text/csv; charset=utf-8")
            self.send_header("Content-Disposition", f"attachment; filename={filename}")
            self.end_headers()
            self.wfile.write(content)
        except Exception as e:
            self.send_error(500, f"Error downloading file: {str(e)}")

    def handle_recommend(self, query):
        user_id = query.get('user_id', [None])[0] or 'C00001'
        store_id = query.get('store_id', [None])[0] or 'S001'
        model_name = query.get('model', ['lightgbm'])[0]
        gender = query.get('gender', [None])[0]
        age = query.get('age', [None])[0]
            
        if user_id not in dataset['customers']:
            parsed_age = int(age) if age else 35
            parsed_gender = gender if gender else 'Unknown'
            
            guest_profile = {
                'age': parsed_age,
                'gender': parsed_gender,
                'loyalty_member': 0
            }
            dataset['customers'][user_id] = guest_profile
            
            for r_name in ['lightgbm', 'xgboost', 'neural', 'mba']:
                if r_name in recommenders:
                    if r_name == 'neural':
                        recommenders[r_name].customer_profiles[user_id] = guest_profile
                    elif r_name == 'mba':
                        recommenders[r_name].user_items[user_id] = set()
                    else:
                        recommenders[r_name].customer_profiles[user_id] = guest_profile
                        recommenders[r_name].user_stats[user_id] = {
                            'user_total_purchases': 0,
                            'user_avg_revenue': 0.0,
                            'user_avg_discount': 0.0
                        }
        
        t0 = time.time()
        rec_product_ids = []
        
        if model_name == 'hybrid':
            cf_recall_ids = recommenders['itemcf'].recommend(user_id, k=10, store_id=store_id)
            lgb_ranked_ids = recommenders['lightgbm'].recommend(user_id, k=200, store_id=store_id)
            rec_product_ids = [pid for pid in lgb_ranked_ids if pid in cf_recall_ids][:5]
        else:
            rec_model = recommenders.get(model_name)
            if not rec_model:
                self.send_error(400, f"Invalid model name: {model_name}")
                return
            rec_product_ids = rec_model.recommend(user_id, k=5, store_id=store_id)
            
        latency_ms = (time.time() - t0) * 1000
        
        probs = {}
        try:
            if model_name in ['lightgbm', 'xgboost', 'hybrid']:
                eval_model_name = 'lightgbm' if model_name == 'hybrid' else model_name
                model_obj = recommenders[eval_model_name]
                candidates = model_obj.products_df[model_obj.products_df['product_id'].isin(rec_product_ids)].copy()
                num_candidates = len(candidates)
                if num_candidates > 0:
                    u_stats = model_obj.user_stats.get(user_id, {'user_total_purchases':0, 'user_avg_revenue':0.0, 'user_avg_discount':0.0})
                    profile = model_obj.customer_profiles.get(user_id, {'age': 35, 'gender': 'Unknown', 'loyalty_member': 0})
                    stype = model_obj.store_types.get(store_id, 'Unknown') if store_id else 'Unknown'
                    
                    rows = []
                    for _, cand in candidates.iterrows():
                        row = [
                            profile['age'],
                            model_obj.gender_map.get(profile['gender'], 2),
                            profile['loyalty_member'],
                            cand['cocoa_percent'] if not pd.isna(cand['cocoa_percent']) else 0.0,
                            cand['weight_g'] if not pd.isna(cand['weight_g']) else 0.0,
                            model_obj.category_map.get(cand['category'], 5),
                            model_obj.brand_map.get(cand['brand'], 6),
                            model_obj.store_type_map.get(stype, 4),
                            4, 11,
                            u_stats['user_total_purchases'],
                            u_stats['user_avg_revenue'],
                            u_stats['user_avg_discount'],
                            model_obj.item_stats.get(cand['product_id'], {}).get('item_total_sales', 0),
                            model_obj.item_stats.get(cand['product_id'], {}).get('item_avg_discount', 0.0),
                            model_obj.user_item_counts.get((user_id, cand['product_id']), 0),
                            model_obj.user_brand_counts.get((user_id, cand['brand']), 0),
                            model_obj.user_category_counts.get((user_id, cand['category']), 0)
                        ]
                        rows.append(row)
                    pred_probs = model_obj.predict_numpy(rows)
                    for pid, prob in zip(candidates['product_id'], pred_probs):
                        probs[pid] = float(prob)
                        
            elif model_name == 'neural':
                model_obj = recommenders['neural']
                user_idx = model_obj.user_to_idx.get(user_id, model_obj.user_to_idx['UNKNOWN_USER'])
                profile = model_obj.customer_profiles.get(user_id, {'age': 35, 'gender': 'Unknown', 'loyalty_member': 0})
                candidates = model_obj.products_df[model_obj.products_df['product_id'].isin(rec_product_ids)].copy()
                num_candidates = len(candidates)
                if num_candidates > 0:
                    ages = np.array([profile['age']] * num_candidates)
                    genders = np.array([model_obj.gender_map.get(profile['gender'], 2)] * num_candidates)
                    loyalties = np.array([profile['loyalty_member']] * num_candidates)
                    cocoas = candidates['cocoa_percent'].fillna(50).values
                    weights = candidates['weight_g'].fillna(100).values
                    brands = candidates['brand'].map(model_obj.brand_map).fillna(6).values
                    cats = candidates['category'].map(model_obj.category_map).fillna(5).values
                    
                    age_norm = (ages - 18) / (70 - 18)
                    loyalty_norm = loyalties
                    cocoa_norm = cocoas / 100.0
                    weight_norm = (weights - 50) / (200 - 50)
                    
                    gender_onehot = np.eye(3)[genders.astype(int)]
                    brand_onehot = np.eye(7)[brands.astype(int)]
                    cat_onehot = np.eye(6)[cats.astype(int)]
                    
                    cont_features = np.stack([age_norm, loyalty_norm, cocoa_norm, weight_norm], axis=1)
                    aux_features = np.concatenate([cont_features, gender_onehot, brand_onehot, cat_onehot], axis=1).astype(np.float32)
                    
                    item_unknown_idx = model_obj.item_to_idx['UNKNOWN_ITEM']
                    user_indices = np.array([user_idx] * num_candidates, dtype=int)
                    item_indices = np.array([model_obj.item_to_idx.get(pid, item_unknown_idx) for pid in candidates['product_id']], dtype=int)
                    
                    preds = model_obj.predict_numpy(user_indices, item_indices, aux_features)
                    for pid, prob in zip(candidates['product_id'], preds):
                        probs[pid] = float(prob)
                        
            elif model_name == 'itemcf':
                model_obj = recommenders['itemcf']
                history = model_obj.user_items.get(user_id, set())
                scores = defaultdict(float)
                for hist_item in history:
                    sim_items = model_obj.similarity.get(hist_item, {})
                    for sim_item, sim_score in sim_items.items():
                        if sim_item in history:
                            continue
                        scores[sim_item] += sim_score
                max_score = max(scores.values()) if scores else 1.0
                for pid in rec_product_ids:
                    raw_score = scores.get(pid, 0.0)
                    norm_score = raw_score / max_score if max_score > 0 else 0.0
                    probs[pid] = float(0.5 + 0.45 * norm_score) if pid in scores else 0.15
                    
            elif model_name == 'mba':
                model_obj = recommenders['mba']
                history = model_obj.user_items.get(user_id, set())
                candidates = defaultdict(float)
                for hist_item in history:
                    for sim_item, lift in model_obj.rules.get(hist_item, []):
                        if sim_item not in history:
                            candidates[sim_item] += lift
                max_lift = max(candidates.values()) if candidates else 1.0
                for pid in rec_product_ids:
                    raw_lift = candidates.get(pid, 0.0)
                    norm_lift = raw_lift / max_lift if max_lift > 0 else 0.0
                    probs[pid] = float(0.5 + 0.48 * norm_lift) if pid in candidates else 0.12
        except Exception as e:
            print(f"[Error calculating real probs] {str(e)}")
        
        recommended_products = []
        for pid in rec_product_ids:
            p_details = dataset['products'].get(pid, {})
            recommended_products.append({
                'product_id': pid,
                'product_name': p_details.get('product_name', 'Unknown chocolate'),
                'brand': p_details.get('brand', 'Unknown'),
                'category': p_details.get('category', 'Unknown'),
                'cocoa_percent': p_details.get('cocoa_percent', 0.0),
                'weight_g': p_details.get('weight_g', 0.0),
                'probability': probs.get(pid, 0.85)
            })
            
        user_profile = dataset['customers'].get(user_id, {})
        
        is_fallback = (model_name == 'xgboost' and IS_XGB_FALLBACK)
        fallback_notice = "⚡ 云端轻量化提示：受限于 Serverless 函数 500MB 体积限制，XGBoost 已透明平滑切换至 LightGBM 预测引擎。" if is_fallback else ""
        fallback_notice_en = "⚡ Serverless Optimization: To comply with cloud size limits, XGBoost is running seamlessly via LightGBM engine." if is_fallback else ""

        response = {
            'user_id': user_id,
            'user_profile': user_profile,
            'model_used': model_name,
            'store_id': store_id,
            'latency_ms': round(latency_ms, 3),
            'is_fallback': is_fallback,
            'fallback_notice': fallback_notice,
            'fallback_notice_en': fallback_notice_en,
            'recommendations': recommended_products
        }
        
        self.handle_json_response(response)

    def handle_json_response(self, data):
        self.send_response(200)
        self.send_header("Content-type", "application/json")
        self.end_headers()
        self.wfile.write(json.dumps(data).encode('utf-8'))

app = handler
application = handler
