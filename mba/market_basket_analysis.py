import pandas as pd
from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules
import os

def run_mba():
    # Load data
    print("Loading data...")
    sales = pd.read_csv('rawdata/sales.csv')
    products = pd.read_csv('rawdata/products.csv')

    # Merge sales with products to get product names
    df = pd.merge(sales, products[['product_id', 'product_name']], on='product_id')

    # Since each order_id has only one product, we group by customer_id 
    # to find products often bought by the same customer.
    print("Preparing transactions (grouping by customer_id)...")
    
    # Create a list of products per customer
    # We use unique products per customer for standard Apriori
    basket = (df.groupby(['customer_id', 'product_name'])['quantity']
              .sum().unstack().reset_index().fillna(0)
              .set_index('customer_id'))

    # Convert to boolean (True/False)
    basket_sets = basket.map(lambda x: x > 0)

    print(f"Basket matrix shape: {basket_sets.shape}")
    print(f"Number of items in basket: {len(basket_sets.columns)}")

    # Build frequent itemsets
    # We use a low min_support because there are 200+ products and many customers
    print("Finding frequent itemsets (min_support=0.05)...")
    frequent_itemsets = apriori(basket_sets, min_support=0.05, use_colnames=True)

    if frequent_itemsets.empty:
        print("No frequent itemsets found with min_support=0.05. Trying 0.01...")
        frequent_itemsets = apriori(basket_sets, min_support=0.01, use_colnames=True)

    # Generate association rules
    print("Generating association rules...")
    if not frequent_itemsets.empty:
        rules = association_rules(frequent_itemsets, metric="lift", min_threshold=1)
        
        # Sort rules by lift
        rules = rules.sort_values('lift', ascending=False)
        
        # Save results
        output_file = 'mba/mba_results.md'
        with open(output_file, 'w') as f:
            f.write("# Market Basket Analysis Results\n\n")
            f.write("Analysis performed by grouping products by `customer_id`.\n\n")
            f.write("## Top 20 Association Rules (Sorted by Lift)\n\n")
            f.write(rules.head(20)[['antecedents', 'consequents', 'support', 'confidence', 'lift']].to_markdown())
            f.write("\n")
        
        print(f"Results saved to {output_file}")
        print("\nTop 5 Rules:")
        print(rules.head(5)[['antecedents', 'consequents', 'support', 'confidence', 'lift']])
    else:
        print("No frequent itemsets found even with lower support.")

class MBARecommender:
    def __init__(self):
        self.rules = {}
        self.popular_items = []
        self.user_items = {}

    def recommend(self, user_id, k=5, store_id=None, order_date=None):
        history = self.user_items.get(user_id, set())
        candidates = {}
        for hist_item in history:
            for sim_item, lift in self.rules.get(hist_item, []):
                if sim_item not in history:
                    candidates[sim_item] = candidates.get(sim_item, 0.0) + lift
        if not candidates:
            return self.popular_items[:k]
        sorted_cands = sorted(candidates.items(), key=lambda x: x[1], reverse=True)
        return [item for item, _ in sorted_cands[:k]]

if __name__ == "__main__":
    run_mba()
