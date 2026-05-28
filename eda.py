import csv
import os
from collections import defaultdict, Counter

data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "rawdata")

# Load products
products = {}
with open(os.path.join(data_dir, "products.csv"), 'r', encoding='utf-8') as f:
    reader = csv.DictReader(f)
    for row in reader:
        products[row['product_id']] = {
            'product_name': row['product_name'],
            'brand': row['brand'],
            'category': row['category'],
            'cocoa_percent': float(row['cocoa_percent']) if row['cocoa_percent'] else 0.0,
            'weight_g': float(row['weight_g']) if row['weight_g'] else 0.0
        }

# Load customers
customers = {}
with open(os.path.join(data_dir, "customers.csv"), 'r', encoding='utf-8') as f:
    reader = csv.DictReader(f)
    for row in reader:
        customers[row['customer_id']] = {
            'age': int(row['age']) if row['age'] else None,
            'gender': row['gender'],
            'loyalty_member': int(row['loyalty_member']) if row['loyalty_member'] else 0,
            'join_date': row['join_date']
        }

# Load stores
stores = {}
with open(os.path.join(data_dir, "stores.csv"), 'r', encoding='utf-8') as f:
    reader = csv.DictReader(f)
    for row in reader:
        stores[row['store_id']] = {
            'store_name': row['store_name'],
            'city': row['city'],
            'country': row['country'],
            'store_type': row['store_type']
        }

# Process sales row-by-row
total_orders = 0
total_revenue = 0.0
total_cost = 0.0
total_profit = 0.0
total_quantity = 0

product_revenue = defaultdict(float)
product_quantity = defaultdict(int)
product_profit = defaultdict(float)

category_revenue = defaultdict(float)
brand_revenue = defaultdict(float)
brand_profit = defaultdict(float)
brand_quantity = defaultdict(int)
brand_orders = defaultdict(int)

customer_order_count = defaultdict(int)
unique_customers_in_sales = set()
unique_products_in_sales = set()

store_type_revenue = defaultdict(float)
store_type_profit = defaultdict(float)
store_type_orders = defaultdict(int)

gender_loyalty_revenue = defaultdict(float)
gender_loyalty_orders = defaultdict(int)

cocoa_bins = {
    '40-50%': 0.0,
    '50-60%': 0.0,
    '60-70%': 0.0,
    '70-80%': 0.0,
    '80-90%': 0.0,
    '90-100%': 0.0
}

with open(os.path.join(data_dir, "sales.csv"), 'r', encoding='utf-8') as f:
    reader = csv.DictReader(f)
    for row in reader:
        total_orders += 1
        qty = int(row['quantity'])
        rev = float(row['revenue'])
        cost = float(row['cost'])
        profit = float(row['profit'])
        pid = row['product_id']
        cid = row['customer_id']
        sid = row['store_id']
        
        total_revenue += rev
        total_cost += cost
        total_profit += profit
        total_quantity += qty
        
        unique_customers_in_sales.add(cid)
        unique_products_in_sales.add(pid)
        customer_order_count[cid] += 1
        
        p = products.get(pid, {})
        pname = p.get('product_name', 'Unknown')
        pcat = p.get('category', 'Unknown')
        pbrand = p.get('brand', 'Unknown')
        cocoa = p.get('cocoa_percent', 0.0)
        
        product_revenue[pname] += rev
        product_quantity[pname] += qty
        product_profit[pname] += profit
        
        category_revenue[pcat] += rev
        brand_revenue[pbrand] += rev
        brand_profit[pbrand] += profit
        brand_quantity[pbrand] += qty
        brand_orders[pbrand] += 1
        
        if 40 <= cocoa < 50:
            cocoa_bins['40-50%'] += rev
        elif 50 <= cocoa < 60:
            cocoa_bins['50-60%'] += rev
        elif 60 <= cocoa < 70:
            cocoa_bins['60-70%'] += rev
        elif 70 <= cocoa < 80:
            cocoa_bins['70-80%'] += rev
        elif 80 <= cocoa < 90:
            cocoa_bins['80-90%'] += rev
        elif 90 <= cocoa <= 100:
            cocoa_bins['90-100%'] += rev
            
        s = stores.get(sid, {})
        stype = s.get('store_type', 'Unknown')
        store_type_revenue[stype] += rev
        store_type_profit[stype] += profit
        store_type_orders[stype] += 1
        
        cust = customers.get(cid, {})
        gender = cust.get('gender', 'Unknown')
        loyalty = cust.get('loyalty_member', 0)
        loyalty_str = 'Loyalty' if loyalty == 1 else 'Regular'
        key = f"{gender}_{loyalty_str}"
        gender_loyalty_revenue[key] += rev
        gender_loyalty_orders[key] += 1

print("=== Raw Data Size ===")
print(f"Total Transactions: {total_orders:,}")
print(f"Total Unique Customers in Sales: {len(unique_customers_in_sales):,}")
print(f"Total Unique Products in Sales: {len(unique_products_in_sales):,}")
print(f"Total Products in DB: {len(products):,}")
print(f"Total Customers in DB: {len(customers):,}")
print(f"Total Stores in DB: {len(stores):,}")

print("\n=== Aggregated sales ===")
print(f"Total Revenue: ${total_revenue:,.2f}")
print(f"Total Cost: ${total_cost:,.2f}")
print(f"Total Profit: ${total_profit:,.2f}")
print(f"Average Profit Margin: {total_profit / total_revenue * 100:.2f}%")
print(f"Total Quantity Sold: {total_quantity:,}")
print(f"Average Order Value (AOV): ${total_revenue / total_orders:.2f}")

# Top products by revenue
sorted_products = sorted(product_revenue.items(), key=lambda x: x[1], reverse=True)
print("\n=== Top 10 Products by Revenue ===")
for i, (name, rev) in enumerate(sorted_products[:10], 1):
    qty = product_quantity[name]
    prof = product_profit[name]
    print(f"{i}. {name}: Revenue: ${rev:,.2f} | Quantity: {qty:,} | Profit: ${prof:,.2f}")

# Top categories
sorted_categories = sorted(category_revenue.items(), key=lambda x: x[1], reverse=True)
print("\n=== Category Performance ===")
for cat, rev in sorted_categories:
    print(f"- {cat}: ${rev:,.2f} ({rev / total_revenue * 100:.2f}%)")

# Brand performance
sorted_brands = sorted(brand_revenue.items(), key=lambda x: x[1], reverse=True)
print("\n=== Brand Performance ===")
for brand, rev in sorted_brands:
    prof = brand_profit[brand]
    qty = brand_quantity[brand]
    orders = brand_orders[brand]
    print(f"- {brand}: Revenue: ${rev:,.2f} | Profit: ${prof:,.2f} | Orders: {orders:,} | Quantity: {qty:,}")

# Customer repeat behavior
order_counts = list(customer_order_count.values())
mean_orders = sum(order_counts) / len(order_counts)
repeat_customers = sum(1 for c in order_counts if c > 1)
print("\n=== Customer Order Frequency ===")
print(f"Avg Orders per Customer: {mean_orders:.2f}")
print(f"Repeat Customers (>1 order): {repeat_customers:,} ({repeat_customers / len(unique_customers_in_sales) * 100:.2f}%)")

# Sparsity
num_users = len(unique_customers_in_sales)
num_items = len(unique_products_in_sales)
sparsity = 1.0 - (total_orders / (num_users * num_items))
print(f"User-Item Matrix Sparsity: {sparsity * 100:.4f}%")

# Store types
sorted_stores = sorted(store_type_revenue.items(), key=lambda x: x[1], reverse=True)
print("\n=== Performance by Store Type ===")
for stype, rev in sorted_stores:
    prof = store_type_profit[stype]
    orders = store_type_orders[stype]
    print(f"- {stype}: Revenue: ${rev:,.2f} | Profit: ${prof:,.2f} | Orders: {orders:,}")

# Gender and loyalty member
print("\n=== Sales by Demographics (Gender & Loyalty) ===")
for key, rev in gender_loyalty_revenue.items():
    orders = gender_loyalty_orders[key]
    print(f"- {key}: Revenue: ${rev:,.2f} | Orders: {orders:,} | Avg Ticket: ${rev / orders:.2f}")

# Cocoa bins
print("\n=== Sales by Cocoa Percent Bins ===")
for bin_name, rev in cocoa_bins.items():
    print(f"- {bin_name}: ${rev:,.2f} ({rev / total_revenue * 100:.2f}%)")
