import http.server
import socketserver
import json
import urllib.parse
import os
import sys
import time
import pandas as pd
import numpy as np

# Add parent directory to path to import models
base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(base_path)
sys.path.append(os.path.join(base_path, "itemcf"))
sys.path.append(os.path.join(base_path, "xg_boost"))
sys.path.append(os.path.join(base_path, "lightgbm"))

from evaluate import load_data
from itemcf_recommender import ItemCFRecommender
from xgboost_recommender import XGBoostRecommender
from lightgbm_recommender import LightGBMRecommender

PORT = 5001
DATA_DIR = os.path.join(base_path, "rawdata")

# Global references for models and datasets
recommenders = {}
dataset = {}

def pretrain_models():
    print("\n[Server] Initializing and pre-training models...")
    t0 = time.time()
    
    # Load all data
    train_sales, test_sales, products, customers, stores = load_data()
    
    # Store globally
    dataset['products'] = products.set_index('product_id').to_dict(orient='index')
    dataset['customers'] = customers.set_index('customer_id').to_dict(orient='index')
    dataset['stores'] = stores.set_index('store_id').to_dict(orient='index')
    
    # Pre-select 50 customers to show in UI dropdown
    np.random.seed(42)
    sample_cids = np.random.choice(customers['customer_id'].tolist(), size=100, replace=False)
    dataset['sample_customers'] = [
        {**{'customer_id': cid}, **dataset['customers'][cid]} for cid in sample_cids
    ]
    
    dataset['sample_stores'] = [
        {**{'store_id': sid}, **dataset['stores'][sid]} for sid in stores['store_id'].tolist()
    ]
    
    # For instant startup (< 2s), we train on a subsampled transaction set
    print("[Server] Subsampling transactions for lightning-fast training...")
    sub_train = train_sales.sample(n=30000, random_state=42).copy()
    
    # 1. ItemCF
    print("[Server] Training ItemCF Recommender...")
    itemcf = ItemCFRecommender()
    itemcf.fit(sub_train, products, customers, stores)
    recommenders['itemcf'] = itemcf
    
    # 2. XGBoost
    print("[Server] Training XGBoost Recommender...")
    xgb_rec = XGBoostRecommender()
    xgb_rec.fit(sub_train, products, customers, stores)
    recommenders['xgboost'] = xgb_rec
    
    # 3. LightGBM
    print("[Server] Training LightGBM Recommender...")
    lgb_rec = LightGBMRecommender()
    lgb_rec.fit(sub_train, products, customers, stores)
    recommenders['lightgbm'] = lgb_rec
    
    print(f"[Server] All models initialized successfully in {time.time() - t0:.2f}s!\n")

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
        # Serve Frontend Static HTML (English or Chinese)
        elif path == "/en" or path == "/index_en.html":
            self.serve_html("index_en.html")
        elif path == "/" or path == "/index.html":
            self.serve_html("index.html")
        else:
            self.send_error(404, "Page Not Found")
            
    def serve_html(self, filename="index.html"):
        html_path = os.path.join(os.path.dirname(__file__), filename)
        try:
            with open(html_path, 'r', encoding='utf-8') as f:
                content = f.read()
            self.send_response(200)
            self.send_header("Content-type", "text/html; charset=utf-8")
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
            for r_name in ['lightgbm', 'xgboost']:
                if r_name in recommenders:
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
                'weight_g': p_details.get('weight_g', 0.0)
            })
            
        # Get customer profile details
        user_profile = dataset['customers'].get(user_id, {})
        
        response = {
            'user_id': user_id,
            'user_profile': user_profile,
            'model_used': model_name,
            'store_id': store_id,
            'latency_ms': round(latency_ms, 3),
            'recommendations': recommended_products
        }
        
        self.handle_json_response(response)
        
    def handle_json_response(self, data):
        self.send_response(200)
        self.send_header("Content-type", "application/json")
        self.end_headers()
        self.wfile.write(json.dumps(data).encode('utf-8'))

def run_server():
    pretrain_models()
    
    # Start server
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
