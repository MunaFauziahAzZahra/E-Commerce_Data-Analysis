import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Load data dari URL
customers_df = pd.read_csv('https://raw.githubusercontent.com/MunaFauziahAzZahra/E-Commerce_Data-Analysis/refs/heads/main/dataset/customers_dataset.csv')
geolocation_df = pd.read_csv('https://raw.githubusercontent.com/MunaFauziahAzZahra/E-Commerce_Data-Analysis/refs/heads/main/dataset/geolocation_dataset.csv')
order_items_df = pd.read_csv('https://raw.githubusercontent.com/MunaFauziahAzZahra/E-Commerce_Data-Analysis/refs/heads/main/dataset/order_items_dataset.csv')
order_payments_df = pd.read_csv('https://raw.githubusercontent.com/MunaFauziahAzZahra/E-Commerce_Data-Analysis/refs/heads/main/dataset/order_payments_dataset.csv')
order_reviews_df = pd.read_csv('https://raw.githubusercontent.com/MunaFauziahAzZahra/E-Commerce_Data-Analysis/refs/heads/main/dataset/order_reviews_dataset.csv')
orders_df = pd.read_csv('https://raw.githubusercontent.com/MunaFauziahAzZahra/E-Commerce_Data-Analysis/refs/heads/main/dataset/orders_dataset.csv')
product_category_name_translation_df = pd.read_csv('https://raw.githubusercontent.com/MunaFauziahAzZahra/E-Commerce_Data-Analysis/refs/heads/main/dataset/product_category_name_translation.csv')
products_df = pd.read_csv('https://raw.githubusercontent.com/MunaFauziahAzZahra/E-Commerce_Data-Analysis/refs/heads/main/dataset/products_dataset.csv')
sellers_df = pd.read_csv('https://raw.githubusercontent.com/MunaFauziahAzZahra/E-Commerce_Data-Analysis/refs/heads/main/dataset/sellers_dataset.csv')

# Konversi 'order_purchase_timestamp'
if 'order_purchase_timestamp' in order_items_df.columns:
    order_items_df['order_purchase_timestamp'] = pd.to_datetime(order_items_df['order_purchase_timestamp'])
else:
    st.error("Column 'order_purchase_timestamp' not found in order_items_df!")
    st.stop()

# Filter berdasarkan order_purchase_timestamp
st.sidebar.header("Filter Data by Date Range")
start_date = st.sidebar.date_input("Start Date", value=order_items_df['order_purchase_timestamp'].min().date())
end_date = st.sidebar.date_input("End Date", value=order_items_df['order_purchase_timestamp'].max().date())

# Validasi input tanggal
if start_date > end_date:
    st.error("Start date must be before end date!")
    st.stop()

# Filter data berdasarkan tanggal yang dipilih
filtered_data = order_items_df[(order_items_df['order_purchase_timestamp'] >= pd.to_datetime(start_date)) &
                               (order_items_df['order_purchase_timestamp'] <= pd.to_datetime(end_date))]
st.success(f"Data filtered from {start_date} to {end_date}. Total rows: {filtered_data.shape[0]}")

# Dashboard 1: Total Payment Value dan Order Count by Month
st.title("Dashboard Online Commerce")
# Resample data by month (on the 'order_purchase_timestamp') for order count and revenue
monthly_orders_df = filtered_data.resample(rule='M', on='order_purchase_timestamp').agg({
    "order_id": "nunique",
    "payment_value": "sum"
})

# Ubah format tanggal menjadi tahun-bulan dan assign ke kolom baru
monthly_orders_df['year_month'] = monthly_orders_df.index.strftime('%Y-%m')

# Reset index dan rename kolom
monthly_orders_df = monthly_orders_df.reset_index()
monthly_orders_df.rename(columns={
    "order_id": "order_count",
    "payment_value": "revenue"
}, inplace=True)

# Plot line chart untuk total payment value (revenue)
st.subheader("Total Payment Value by Month with Trendline")
fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(monthly_orders_df["year_month"], monthly_orders_df["revenue"], marker='o', linewidth=2, color="#72BCD4", label="Payment Value")

# Menambahkan trendline
x = np.arange(len(monthly_orders_df))
y = monthly_orders_df["revenue"].values
coefficients = np.polyfit(x, y, 1)
poly = np.poly1d(coefficients)
trendline = poly(x)

ax.plot(monthly_orders_df["year_month"], trendline, color='orange', linestyle='--', label="Trendline")

# Menambah judul dan label
ax.set_title("Total Payment Value by Month with Trendline", fontsize=20)
ax.set_xticklabels(monthly_orders_df["year_month"], rotation=45, fontsize=10)
ax.set_yticklabels(ax.get_yticks(), fontsize=10)
ax.set_xlabel("Month", fontsize=12)
ax.set_ylabel("Payment Value (Revenue)", fontsize=12)

# Menampilkan legenda
ax.legend()

# Menampilkan plot
st.pyplot(fig)

# Plot line chart untuk total order count
st.subheader("Total Number of Monthly Orders with Trendline")
fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(monthly_orders_df["year_month"], monthly_orders_df["order_count"], marker='o', linewidth=2, color="#72BCD4", label="Order Count")

# Menambahkan trendline
x = np.arange(len(monthly_orders_df))
y = monthly_orders_df["order_count"].values
coefficients = np.polyfit(x, y, 1)
poly = np.poly1d(coefficients)
trendline = poly(x)

ax.plot(monthly_orders_df["year_month"], trendline, color='orange', linestyle='--', label="Trendline")

# Menambah judul dan label
ax.set_title("Total Number of Monthly Orders with Trendline", fontsize=20)
ax.set_xticklabels(monthly_orders_df["year_month"], rotation=45, fontsize=10)
ax.set_yticklabels(ax.get_yticks(), fontsize=10)
ax.set_xlabel("Month", fontsize=12)
ax.set_ylabel("Order Count", fontsize=12)

# Menampilkan legenda
ax.legend()

# Menampilkan plot
st.pyplot(fig)

# Dashboard 2: Monthly Revenue Heatmap
st.subheader("Monthly Revenue Heatmap")
monthly_df = filtered_data.resample('M', on='order_purchase_timestamp').agg({
    "order_id": "nunique",
    "payment_value": "sum"
})
monthly_df['Year'] = monthly_df.index.year
monthly_df['Month'] = monthly_df.index.month

# Membuat pivot tabel untuk Revenue
pivot_revenue = monthly_df.pivot(index="Year", columns="Month", values="payment_value")

# Plot Heatmap
fig2, ax = plt.subplots(figsize=(18, 6))
sns.heatmap(pivot_revenue, cmap='Blues', annot=True, fmt='.2f', linewidths=0.5, cbar_kws={'label': 'Total Revenue'})
ax.set_title("Monthly Revenue Heatmap", fontsize=16)
ax.set_xlabel("Month", fontsize=12)
ax.set_ylabel("Year", fontsize=12)
ax.set_xticklabels(['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'], rotation=45)
st.pyplot(fig2)

# Dashboard 3: Monthly Order Heatmap
st.subheader("Monthly Order Heatmap")
# Creating pivot table for Orders
pivot_orders = monthly_df.pivot(index="Year", columns="Month", values="order_id")

# Plot Heatmap
fig3, ax = plt.subplots(figsize=(18, 6))
sns.heatmap(pivot_orders, cmap='Blues', annot=True, fmt='.2f', linewidths=0.5, cbar_kws={'label': 'Total Orders'})
ax.set_title("Monthly Order Heatmap", fontsize=16)
ax.set_xlabel("Month", fontsize=12)
ax.set_ylabel("Year", fontsize=12)
ax.set_xticklabels(['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'], rotation=45)
st.pyplot(fig3)

# Dashboard 4: Best and Worst Performing Products by Number of Sales
st.subheader("Best and Worst Performing Products by Number of Sales")

# Membuat plot
fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(24, 6))

# Kalkulasi payment_value for each product
sum_order_items_df = filtered_data.groupby("product")["payment_value"].sum().sort_values(ascending=False).reset_index()

# Best Performing Products Sorted by Payment Value
sns.barplot(x="payment_value", y="product", data=sum_order_items_df.head(5), palette="Blues_d", ax=ax[0])
ax[0].set_title("Best Performing Product", loc="center", fontsize=15)
ax[0].tick_params(axis='y', labelsize=12)

# Worst Performing Products Sorted by Payment Value
sns.barplot(x="payment_value", y="product", data=sum_order_items_df.sort_values(by="payment_value", ascending=True).head(5), palette="Blues_d", ax=ax[1])
ax[1].set_title("Worst Performing Product", loc="center", fontsize=15)
ax[1].invert_xaxis()  # Invert X axis for the Worst performing product
ax[1].yaxis.set_label_position("right")
ax[1].yaxis.tick_right()
ax[1].tick_params(axis='y', labelsize=12)

plt.suptitle("Best and Worst Performing Product by Payment Value", fontsize=20)

# Display the plot in Streamlit
st.pyplot(fig)

# 5 RFM Analysis
# Konversi 'order_purchase_timestamp' to datetime
filtered_data["order_purchase_timestamp"] = pd.to_datetime(filtered_data["order_purchase_timestamp"])
current_date = pd.to_datetime("2024-12-28")  # Tanggal referensi

# Hitung RFM (Recency, Frequency, Monetary)
rfm = filtered_data.groupby('customer_id').agg({
    'order_purchase_timestamp': lambda x: (current_date - x.max()).days,  # Recency
    'order_id': 'nunique',  # Frequency
    'payment_value': 'sum'  # Monetary
})

# Ubah nama kolom
rfm.rename(columns={
    'order_purchase_timestamp': 'Recency',
    'order_id': 'Frequency',
    'payment_value': 'Monetary'
}, inplace=True)

# Streamlit interface
st.subheader("RFM Analysis")

# Create a figure with 3 subplots
fig, axes = plt.subplots(1, 3, figsize=(20, 6))  # 1 row, 3 columns

# Histogram Recency
sns.histplot(rfm['Recency'], bins=20, kde=True, color="#72BCD4", ax=axes[0])
axes[0].set_title('Distribution of Recency')
axes[0].set_xlabel('Recency (days)')
axes[0].set_ylabel('Frequency')

# Histogram Frequency
sns.histplot(rfm['Frequency'], bins=20, kde=True, color="#72BCD4", ax=axes[1])
axes[1].set_title('Distribution of Frequency')
axes[1].set_xlabel('Frequency')
axes[1].set_ylabel('Frequency')

# Histogram Monetary
sns.histplot(rfm['Monetary'], bins=20, kde=True, color="#72BCD4", ax=axes[2])
axes[2].set_title('Distribution of Monetary')
axes[2].set_xlabel('Monetary Value')
axes[2].set_ylabel('Frequency')

# Layout adjustment for Streamlit
plt.tight_layout()  # To avoid overlap of plots

# Show plots in Streamlit
st.pyplot(fig)

# Display the first few rows of RFM dataframe
st.write("RFM Data:", rfm.head())
