import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import LabelEncoder
import os
import sys
from collections import Counter

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
        return self.mlp(combined).squeeze()

class NeuralRecommender(BaseRecommender):
    def __init__(self, embed_dim=16, epochs=5, batch_size=256, lr=0.001):
        self.embed_dim = embed_dim
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        
        self.user_encoder = LabelEncoder()
        self.item_encoder = LabelEncoder()
        
        self.model = None
        self.products_df = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Mappings for categorical features
        self.gender_map = {'Male': 0, 'Female': 1, 'Unknown': 2}
        self.brand_map = {'Ferrero': 0, 'Cadbury': 1, 'Lindt': 2, 'Mars': 3, 'Godiva': 4, 'Hershey': 5, 'Unknown': 6}
        self.category_map = {'Praline': 0, 'White': 1, 'Dark': 2, 'Truffle': 3, 'Milk': 4, 'Unknown': 5}

    def _prepare_aux_features(self, df):
        # Age normalized (approx 18-70)
        age = (df['age'].fillna(35) - 18) / (70 - 18)
        gender = df['gender'].map(self.gender_map).fillna(2) / 2.0
        loyalty = df['loyalty_member'].fillna(0)
        
        cocoa = df['cocoa_percent'].fillna(50) / 100.0
        weight = (df['weight_g'].fillna(100) - 50) / (200 - 50)
        brand = df['brand'].map(self.brand_map).fillna(6) / 6.0
        cat = df['category'].map(self.category_map).fillna(5) / 5.0
        
        return np.stack([age, gender, loyalty, cocoa, weight, brand, cat], axis=1).astype(np.float32)

    def fit(self, train_sales, products, customers, stores):
        print("Pre-processing data for Neural Network...")
        self.products_df = products.copy()
        self.fit_extra(train_sales, products, customers, stores)
        
        # Fit encoders with a catch-all for unknown IDs
        all_customers = list(customers['customer_id'].unique()) + ['UNKNOWN_USER']
        all_products = list(products['product_id'].unique()) + ['UNKNOWN_ITEM']
        
        self.user_encoder.fit(all_customers)
        self.item_encoder.fit(all_products)
        
        # Map unseen IDs to UNKNOWN
        def safe_encode(encoder, values, unknown_label):
            classes = set(encoder.classes_)
            return encoder.transform([v if v in classes else unknown_label for v in values])
            
        # Positive samples
        pos_df = train_sales.sample(n=min(100000, len(train_sales)), random_state=42).copy()
        pos_df = pos_df.merge(customers, on='customer_id', how='left')
        pos_df = pos_df.merge(products, on='product_id', how='left')
        pos_df['label'] = 1
        
        # Negative samples
        all_pids = products['product_id'].values
        neg_cids = np.random.choice(customers['customer_id'].values, size=len(pos_df))
        neg_pids = np.random.choice(all_pids, size=len(pos_df))
        
        neg_df = pd.DataFrame({'customer_id': neg_cids, 'product_id': neg_pids})
        neg_df = neg_df.merge(customers, on='customer_id', how='left')
        neg_df = neg_df.merge(products, on='product_id', how='left')
        neg_df['label'] = 0
        
        full_df = pd.concat([pos_df, neg_df], ignore_index=True)
        
        users = safe_encode(self.user_encoder, full_df['customer_id'], 'UNKNOWN_USER')
        items = safe_encode(self.item_encoder, full_df['product_id'], 'UNKNOWN_ITEM')
        aux_features = self._prepare_aux_features(full_df)
        labels = full_df['label'].values
        
        dataset = ChocolateDataset(users, items, aux_features, labels)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        
        print(f"Initializing model on {self.device}...")
        self.model = NCFModel(len(self.user_encoder.classes_), 
                             len(self.item_encoder.classes_), 
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
        
        # Map unseen IDs to UNKNOWN
        def safe_encode(encoder, values, unknown_label):
            classes = set(encoder.classes_)
            return encoder.transform([v if v in classes else unknown_label for v in values])
            
        # Get user profile
        user_idx = safe_encode(self.user_encoder, [user_id], 'UNKNOWN_USER')[0]
        profile = self.customer_profiles.get(user_id, {'age': 35, 'gender': 'Unknown', 'loyalty_member': 0})
        
        # Create candidates (all products)
        cand_df = self.products_df.copy()
        
        # Repeat user features for all items
        user_features_df = pd.DataFrame({
            'age': [profile['age']] * len(cand_df),
            'gender': [profile['gender']] * len(cand_df),
            'loyalty_member': [profile['loyalty_member']] * len(cand_df)
        })
        input_df = pd.concat([user_features_df, cand_df.reset_index(drop=True)], axis=1)
        
        aux_features = self._prepare_aux_features(input_df)
        
        u_tensor = torch.tensor([user_idx] * len(cand_df), dtype=torch.long).to(self.device)
        i_tensor = torch.tensor(safe_encode(self.item_encoder, cand_df['product_id'], 'UNKNOWN_ITEM'), dtype=torch.long).to(self.device)
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
