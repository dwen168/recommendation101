import os
# Configure Matplotlib config directory to avoid /private/tmp sandbox violations
os.environ["MPLCONFIGDIR"] = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".matplotlib")
# Force single-threaded execution for OpenMP and MKL to prevent library conflict deadlocks on Mac
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import http.server
import socketserver
import json
import urllib.parse
import sys
import time
import pandas as pd
import numpy as np
from collections import defaultdict

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
import pickle

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

PORT = 5001
DATA_DIR = os.path.join(base_path, "rawdata")

# Global references for models and datasets
recommenders = {}
dataset = {}

def load_saved_models():
    print("\n[Server] Loading pre-trained models from models directory...")
    t0 = time.time()
    
    models_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
    
    # Check if models exist
    required_files = ["itemcf.pkl", "xgboost.pkl", "lightgbm.pkl", "neural.pkl", "mba.pkl", "metadata.pkl"]
    missing = [f for f in required_files if not os.path.exists(os.path.join(models_dir, f))]
    
    if missing:
        print(f"\n❌ Error: Pre-trained models are missing: {missing}")
        print("👉 Please run the offline training script first:")
        print("   python simulation/train_and_save.py\n")
        sys.exit(1)
        
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
            
    print(f"[Server] All models loaded successfully in {time.time() - t0:.2f}s!\n")

IS_XGB_FALLBACK = False

class RecommendationAPIHandler(http.server.SimpleHTTPRequestHandler):
    def end_headers(self):
        # Allow CORS
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'X-Requested-With, Content-Type')
        super().end_headers()

    def do_OPTIONS(self):
        self.send_response(200)
        self.end_headers()

    def do_GET(self):
        parsed_url = urllib.parse.urlparse(self.path)
        path = parsed_url.path
        query = urllib.parse.parse_qs(parsed_url.query)
        
        # API Route: Get recommendation
        if path == "/api/recommend":
            self.handle_recommend(query)
        # API Route: Get sample customers
        elif path == "/api/customers":
            self.handle_json_response(dataset['sample_customers'])
        # API Route: Get all stores
        elif path == "/api/stores":
            self.handle_json_response(dataset['sample_stores'])
        # API Route: Get all products
        elif path == "/api/products":
            self.handle_json_response(dataset['products'])
        # API Route: Download Raw CSV Data
        elif path == "/api/download":
            self.handle_download(query)
        # API Route: Get Algorithm README Markdown
        elif path == "/api/readme":
            self.handle_readme(query)
        elif path == "/methodology_en" or path == "/methodology_en.html":
            self.serve_html("methodology_en.html")
        elif path == "/methodology" or path == "/methodology.html":
            self.serve_html("methodology.html")
        elif path == "/zh" or path == "/index_zh.html":
            self.serve_html("index.html")
        elif path == "/" or path == "/index.html" or path == "/en" or path == "/index_en.html":
            self.serve_html("index_en.html")
        else:
            self.send_error(404, "Page Not Found")

    def handle_readme(self, query):
        model = query.get('model', [None])[0]
        mapping = {
            'itemcf': os.path.join(base_path, 'itemcf', 'README.md'),
            'lightgbm': os.path.join(base_path, 'lightgbm', 'README.md'),
            'xgboost': os.path.join(base_path, 'xg_boost', 'README.md'),
            'neural': os.path.join(base_path, 'neural', 'README.md'),
            'mba': os.path.join(base_path, 'mba', 'README.md')
        }
        if not model or model not in mapping:
            self.send_error(400, "Invalid algorithm model requested")
            return
        file_path = mapping[model]
        if not os.path.exists(file_path):
            self.send_error(404, "README file not found")
            return
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            self.handle_json_response({'model': model, 'markdown': content})
        except Exception as e:
            self.send_error(500, f"Error reading README: {str(e)}")

    def handle_download(self, query):
        filename = query.get('file', [None])[0]
        allowed = ['products.csv', 'customers.csv', 'stores.csv', 'sales.csv', 'calendar.csv']
        if not filename or filename not in allowed:
            self.send_error(400, "Invalid file requested")
            return
        file_path = os.path.join(base_path, "rawdata", filename)
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
            
    def serve_html(self, filename="index.html"):
        html_path = os.path.join(os.path.dirname(__file__), filename)
        try:
            with open(html_path, 'r', encoding='utf-8') as f:
                content = f.read()
            self.send_response(200)
            self.send_header("Content-type", "text/html; charset=utf-8")
            self.send_header("Cache-Control", "no-cache, no-store, must-revalidate")
            self.send_header("Pragma", "no-cache")
            self.send_header("Expires", "0")
            self.end_headers()
            self.wfile.write(content.encode('utf-8'))
        except Exception as e:
            self.send_error(500, f"Error reading {filename}: {str(e)}")

    def handle_recommend(self, query):
        user_id = query.get('user_id', [None])[0]
        store_id = query.get('store_id', [None])[0]
        model_name = query.get('model', ['lightgbm'])[0]
        gender = query.get('gender', [None])[0]
        age = query.get('age', [None])[0]
        
        if not user_id:
            self.send_error(400, "Missing user_id parameter")
            return
            
        # Dynamically register Guest/Cold-Start User Profile
        if user_id not in dataset['customers']:
            parsed_age = int(age) if age else 35
            parsed_gender = gender if gender else 'Unknown'
            
            guest_profile = {
                'age': parsed_age,
                'gender': parsed_gender,
                'loyalty_member': 0
            }
            dataset['customers'][user_id] = guest_profile
            
            # Inject profile into models' in-memory dictionaries so they can score natively
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
        
        # Realize Hybrid Mode or individual models
        if model_name == 'hybrid':
            # 🔥 INDUSTRIAL TWO-STAGE PIPELINE 🔥
            # Stage 1: Recall 10 items using ItemCF
            cf_recall_ids = recommenders['itemcf'].recommend(user_id, k=10, store_id=store_id)
            
            # Stage 2: Rank candidates using LightGBM
            # Score all products via LightGBM and filter down to the 10 recalled candidates
            lgb_ranked_ids = recommenders['lightgbm'].recommend(user_id, k=200, store_id=store_id)
            rec_product_ids = [pid for pid in lgb_ranked_ids if pid in cf_recall_ids][:5]
        else:
            rec_model = recommenders.get(model_name)
            if not rec_model:
                self.send_error(400, f"Invalid model name: {model_name}")
                return
            rec_product_ids = rec_model.recommend(user_id, k=5, store_id=store_id)
            
        latency_ms = (time.time() - t0) * 1000
        
        # Compute real prediction probabilities for recommended items based on active model
        probs = {}
        try:
            if model_name in ['lightgbm', 'xgboost', 'hybrid']:
                eval_model_name = 'lightgbm' if model_name == 'hybrid' else model_name
                model_obj = recommenders[eval_model_name]
                candidates = model_obj.products_df[model_obj.products_df['product_id'].isin(rec_product_ids)].copy()
                num_candidates = len(candidates)
                if num_candidates > 0:
                    X_cand = pd.DataFrame()
                    u_stats = model_obj.user_stats.get(user_id, {'user_total_purchases':0, 'user_avg_revenue':0.0, 'user_avg_discount':0.0})
                    profile = model_obj.customer_profiles.get(user_id, {'age': 35, 'gender': 'Unknown', 'loyalty_member': 0})
                    X_cand['user_age'] = [profile['age']] * num_candidates
                    X_cand['user_gender'] = pd.Categorical([model_obj.gender_map.get(profile['gender'], 2)] * num_candidates, categories=model_obj.gender_categories)
                    X_cand['user_loyalty'] = pd.Categorical([profile['loyalty_member']] * num_candidates, categories=model_obj.loyalty_categories)
                    X_cand['item_cocoa'] = candidates['cocoa_percent'].fillna(0.0).values
                    X_cand['item_weight'] = candidates['weight_g'].fillna(0.0).values
                    X_cand['item_category'] = pd.Categorical(model_obj._encode_series(candidates['category'], model_obj.category_map, 5), categories=model_obj.category_categories)
                    X_cand['item_brand'] = pd.Categorical(model_obj._encode_series(candidates['brand'], model_obj.brand_map, 6), categories=model_obj.brand_categories)
                    stype = model_obj.store_types.get(store_id, 'Unknown') if store_id else 'Unknown'
                    X_cand['store_type'] = pd.Categorical([model_obj.store_type_map.get(stype, 4)] * num_candidates, categories=model_obj.store_type_categories)
                    odate = pd.to_datetime('2024-11-15')
                    X_cand['day_of_week'] = pd.Categorical([odate.dayofweek] * num_candidates, categories=model_obj.day_of_week_categories)
                    X_cand['month'] = pd.Categorical([odate.month] * num_candidates, categories=model_obj.month_categories)
                    X_cand['user_total_purchases'] = [u_stats['user_total_purchases']] * num_candidates
                    X_cand['user_avg_revenue'] = [u_stats['user_avg_revenue']] * num_candidates
                    X_cand['user_avg_discount'] = [u_stats['user_avg_discount']] * num_candidates
                    X_cand['item_total_sales'] = candidates['product_id'].map(lambda pid: model_obj.item_stats.get(pid, {}).get('item_total_sales', 0))
                    X_cand['item_avg_discount'] = candidates['product_id'].map(lambda pid: model_obj.item_stats.get(pid, {}).get('item_avg_discount', 0.0))
                    X_cand['user_item_purchase_count'] = candidates['product_id'].map(lambda pid: model_obj.user_item_counts.get((user_id, pid), 0))
                    X_cand['user_brand_purchase_count'] = candidates['brand'].map(lambda brand: model_obj.user_brand_counts.get((user_id, brand), 0))
                    X_cand['user_category_purchase_count'] = candidates['category'].map(lambda cat: model_obj.user_category_counts.get((user_id, cat), 0))
                    
                    cols_order = [
                        'user_age', 'user_gender', 'user_loyalty',
                        'item_cocoa', 'item_weight', 'item_category', 'item_brand',
                        'store_type', 'day_of_week', 'month',
                        'user_total_purchases', 'user_avg_revenue', 'user_avg_discount',
                        'item_total_sales', 'item_avg_discount',
                        'user_item_purchase_count', 'user_brand_purchase_count', 'user_category_purchase_count'
                    ]
                    X_cand = X_cand[cols_order]
                    pred_probs = model_obj.model.predict_proba(X_cand)[:, 1]
                    for pid, prob in zip(candidates['product_id'], pred_probs):
                        probs[pid] = float(prob)
                        
            elif model_name == 'neural':
                import torch
                model_obj = recommenders['neural']
                model_obj.model.eval()
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
                    
                    u_tensor = torch.tensor([user_idx] * num_candidates, dtype=torch.long).to(model_obj.device)
                    item_unknown_idx = model_obj.item_to_idx['UNKNOWN_ITEM']
                    i_tensor = torch.tensor([model_obj.item_to_idx.get(pid, item_unknown_idx) for pid in candidates['product_id']], dtype=torch.long).to(model_obj.device)
                    f_tensor = torch.tensor(aux_features, dtype=torch.float32).to(model_obj.device)
                    
                    with torch.no_grad():
                        preds = model_obj.model(u_tensor, i_tensor, f_tensor)
                        if preds.dim() == 0:
                            preds = preds.unsqueeze(0)
                        preds = preds.cpu().numpy()
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
        
        # Populate rich product details
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
            
        # Get customer profile details
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

def run_server():
    load_saved_models()
    
    # Start server
    socketserver.TCPServer.allow_reuse_address = True
    handler = RecommendationAPIHandler
    with socketserver.TCPServer(("127.0.0.1", PORT), handler) as httpd:
        print(f"[Server] Premium Chocolate Recommendation Demo running at:")
        print(f"👉 http://localhost:{PORT}/")
        print("[Server] Press Ctrl+C to stop.")
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\n[Server] Shutting down.")

if __name__ == '__main__':
    run_server()

if __name__ == '__main__':
    run_server()
