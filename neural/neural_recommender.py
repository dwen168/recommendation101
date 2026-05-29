import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import os
import sys

# Add parent dir to path for BaseRecommender
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from evaluate import BaseRecommender

class ChocolateDataset(Dataset):
    def __init__(self, users, items, features, labels):
        self.users = torch.tensor(users, dtype=torch.long)
        self.items = torch.tensor(items, dtype=torch.long)
        self.features = torch.tensor(features, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.float32)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.users[idx], self.items[idx], self.features[idx], self.labels[idx]

class NCFModel(nn.Module):
    def __init__(self, num_users, num_items, aux_feat_dim, embed_dim=16):
        super(NCFModel, self).__init__()
        self.user_embed = nn.Embedding(num_users, embed_dim)
        self.item_embed = nn.Embedding(num_items, embed_dim)
        
        # MLP Layers
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim * 2 + aux_feat_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )

    def forward(self, user_indices, item_indices, aux_features):
        user_vec = self.user_embed(user_indices)
        item_vec = self.item_embed(item_indices)
        
        combined = torch.cat([user_vec, item_vec, aux_features], dim=1)
        # Use view(-1) instead of squeeze() to avoid batch dimension squeezing bugs when batch size = 1
        return self.mlp(combined).view(-1)

class NeuralRecommender(BaseRecommender):
    def __init__(self, embed_dim=16, epochs=5, batch_size=256, lr=0.001):
        self.embed_dim = embed_dim
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        
        self.user_to_idx = {}
        self.item_to_idx = {}
        
        self.model = None
        self.products_df = None
        
        # Enable GPU: CUDA for Nvidia. Avoid MPS on macOS as it is known to deadlock/hang forever in subprocesses.
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")
        
        # Mappings for categorical features
        self.gender_map = {'Male': 0, 'Female': 1, 'Unknown': 2}
        self.brand_map = {'Ferrero': 0, 'Cadbury': 1, 'Lindt': 2, 'Mars': 3, 'Godiva': 4, 'Hershey': 5, 'Unknown': 6}
        self.category_map = {'Praline': 0, 'White': 1, 'Dark': 2, 'Truffle': 3, 'Milk': 4, 'Unknown': 5}

    def fit(self, train_sales, products, customers, stores):
        print("Pre-processing data for Neural Network...")
        self.products_df = products.copy()
        self.fit_extra(train_sales, products, customers, stores)
        
        # Fit mapping dicts with a catch-all for unknown IDs
        all_customers = list(customers['customer_id'].unique()) + ['UNKNOWN_USER']
        all_products = list(products['product_id'].unique()) + ['UNKNOWN_ITEM']
        
        self.user_to_idx = {user: idx for idx, user in enumerate(all_customers)}
        self.item_to_idx = {item: idx for idx, item in enumerate(all_products)}
        
        user_unknown_idx = self.user_to_idx['UNKNOWN_USER']
        item_unknown_idx = self.item_to_idx['UNKNOWN_ITEM']
        
        # Highly optimized dict maps for fast feature mapping (avoids Pandas merge OOM explosion)
        customer_age_map = customers.set_index('customer_id')['age'].to_dict()
        customer_gender_map = customers.set_index('customer_id')['gender'].to_dict()
        customer_loyalty_map = customers.set_index('customer_id')['loyalty_member'].to_dict()
        
        product_cocoa_map = products.set_index('product_id')['cocoa_percent'].to_dict()
        product_weight_map = products.set_index('product_id')['weight_g'].to_dict()
        product_brand_map = products.set_index('product_id')['brand'].to_dict()
        product_category_map = products.set_index('product_id')['category'].to_dict()
        
        # Positive samples
        pos_sample_size = min(100000, len(train_sales))
        pos_df = train_sales.sample(n=pos_sample_size, random_state=42)
        pos_cids = pos_df['customer_id'].values
        pos_pids = pos_df['product_id'].values
        pos_labels = np.ones(pos_sample_size, dtype=np.float32)
        
        # Negative samples
        all_pids = products['product_id'].values
        neg_cids = np.random.choice(customers['customer_id'].values, size=pos_sample_size)
        neg_pids = np.random.choice(all_pids, size=pos_sample_size)
        neg_labels = np.zeros(pos_sample_size, dtype=np.float32)
        
        users_cids = np.concatenate([pos_cids, neg_cids])
        items_pids = np.concatenate([pos_pids, neg_pids])
        labels = np.concatenate([pos_labels, neg_labels])
        
        # Map IDs using high-performance dictionary lookup
        users = np.array([self.user_to_idx.get(cid, user_unknown_idx) for cid in users_cids], dtype=np.int64)
        items = np.array([self.item_to_idx.get(pid, item_unknown_idx) for pid in items_pids], dtype=np.int64)
        
        # Map auxiliary features directly from ID maps without expensive Pandas Merges
        ages = np.array([customer_age_map.get(cid, 35) for cid in users_cids])
        genders = np.array([self.gender_map.get(customer_gender_map.get(cid, 'Unknown'), 2) for cid in users_cids])
        loyalties = np.array([customer_loyalty_map.get(cid, 0) for cid in users_cids])
        
        cocoas = np.array([product_cocoa_map.get(pid, 50) for pid in items_pids])
        weights = np.array([product_weight_map.get(pid, 100) for pid in items_pids])
        brands = np.array([self.brand_map.get(product_brand_map.get(pid, 'Unknown'), 6) for pid in items_pids])
        cats = np.array([self.category_map.get(product_category_map.get(pid, 'Unknown'), 5) for pid in items_pids])
        
        # Continuous Normalization
        age_norm = (np.nan_to_num(ages, nan=35) - 18) / (70 - 18)
        loyalty_norm = np.nan_to_num(loyalties, nan=0)
        cocoa_norm = np.nan_to_num(cocoas, nan=50) / 100.0
        weight_norm = (np.nan_to_num(weights, nan=100) - 50) / (200 - 50)
        
        # Proper Categorical One-Hot Encoding
        gender_onehot = np.eye(3)[np.nan_to_num(genders, nan=2).astype(int)]
        brand_onehot = np.eye(7)[np.nan_to_num(brands, nan=6).astype(int)]
        cat_onehot = np.eye(6)[np.nan_to_num(cats, nan=5).astype(int)]
        
        cont_features = np.stack([age_norm, loyalty_norm, cocoa_norm, weight_norm], axis=1)
        aux_features = np.concatenate([cont_features, gender_onehot, brand_onehot, cat_onehot], axis=1).astype(np.float32)
        
        dataset = ChocolateDataset(users, items, aux_features, labels)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        
        print(f"Initializing model on {self.device}...")
        self.model = NCFModel(len(self.user_to_idx), 
                             len(self.item_to_idx), 
                             aux_features.shape[1], 
                             self.embed_dim).to(self.device)
        
        optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        criterion = nn.BCELoss()
        
        print(f"Training for {self.epochs} epochs...")
        self.model.train()
        for epoch in range(self.epochs):
            total_loss = 0
            for u, i, f, l in dataloader:
                u, i, f, l = u.to(self.device), i.to(self.device), f.to(self.device), l.to(self.device)
                
                optimizer.zero_grad()
                outputs = self.model(u, i, f)
                loss = criterion(outputs, l)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            
            print(f"Epoch {epoch+1}/{self.epochs}, Loss: {total_loss/len(dataloader):.4f}")

    def recommend(self, user_id, k=5, store_id=None, order_date=None):
        if self.model is None:
            return []
            
        self.model.eval()
        
        user_idx = self.user_to_idx.get(user_id, self.user_to_idx['UNKNOWN_USER'])
        profile = self.customer_profiles.get(user_id, {'age': 35, 'gender': 'Unknown', 'loyalty_member': 0})
        
        cand_df = self.products_df.copy()
        num_candidates = len(cand_df)
        
        # Build features directly for high-speed predictions
        ages = np.array([profile['age']] * num_candidates)
        genders = np.array([self.gender_map.get(profile['gender'], 2)] * num_candidates)
        loyalties = np.array([profile['loyalty_member']] * num_candidates)
        
        cocoas = cand_df['cocoa_percent'].fillna(50).values
        weights = cand_df['weight_g'].fillna(100).values
        brands = cand_df['brand'].map(self.brand_map).fillna(6).values
        cats = cand_df['category'].map(self.category_map).fillna(5).values
        
        age_norm = (ages - 18) / (70 - 18)
        loyalty_norm = loyalties
        cocoa_norm = cocoas / 100.0
        weight_norm = (weights - 50) / (200 - 50)
        
        gender_onehot = np.eye(3)[genders.astype(int)]
        brand_onehot = np.eye(7)[brands.astype(int)]
        cat_onehot = np.eye(6)[cats.astype(int)]
        
        cont_features = np.stack([age_norm, loyalty_norm, cocoa_norm, weight_norm], axis=1)
        aux_features = np.concatenate([cont_features, gender_onehot, brand_onehot, cat_onehot], axis=1).astype(np.float32)
        
        u_tensor = torch.tensor([user_idx] * num_candidates, dtype=torch.long).to(self.device)
        item_unknown_idx = self.item_to_idx['UNKNOWN_ITEM']
        i_tensor = torch.tensor([self.item_to_idx.get(pid, item_unknown_idx) for pid in cand_df['product_id']], dtype=torch.long).to(self.device)
        f_tensor = torch.tensor(aux_features, dtype=torch.float32).to(self.device)
        
        with torch.no_grad():
            preds = self.model(u_tensor, i_tensor, f_tensor)
            if preds.dim() == 0:
                preds = preds.unsqueeze(0)
            preds = preds.cpu().numpy()
            
        cand_df['score'] = preds
        top_k = cand_df.sort_values('score', ascending=False).head(k)['product_id'].tolist()
        return top_k

    def fit_extra(self, train_sales, products, customers, stores):
        self.customer_profiles = {}
        for row in customers.itertuples(index=False):
            self.customer_profiles[row.customer_id] = {
                'age': row.age if not pd.isna(row.age) else 35,
                'gender': row.gender if not pd.isna(row.gender) else 'Unknown',
                'loyalty_member': row.loyalty_member if not pd.isna(row.loyalty_member) else 0
            }
