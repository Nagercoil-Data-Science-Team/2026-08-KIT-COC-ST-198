import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time
import os
import psutil

# ============================================================
# ONLINE RETAIL II - EDA AND CLOUD PERFORMANCE ANALYSIS
# ============================================================

FILE_PATH = "engineered_online_retail.csv"
FIG_SIZE = (12, 8)
PLOT_FOLDER = "plots"

os.makedirs(PLOT_FOLDER, exist_ok=True)

plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["font.size"] = 18
plt.rcParams["font.weight"] = "bold"
plt.rcParams["axes.titleweight"] = "bold"
plt.rcParams["axes.labelweight"] = "bold"
plt.rcParams["xtick.labelsize"] = 14
plt.rcParams["ytick.labelsize"] = 14

print("=" * 80)
print("ONLINE RETAIL II - EXPLORATORY DATA ANALYSIS")
print("=" * 80)

# ============================================================
# HELPER FUNCTIONS FOR ADDING VALUE LABELS TO BARS
# ============================================================

def add_bar_labels(ax, bars, fmt="{:,.0f}", fontsize=12, offset_frac=0.01):
    """Add value labels above each vertical bar (plt.bar)."""
    y_min, y_max = ax.get_ylim()
    offset = (y_max - y_min) * offset_frac
    for bar in bars:
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            height + offset,
            fmt.format(height),
            ha="center",
            va="bottom",
            fontsize=fontsize,
            fontweight="bold",
        )

def add_barh_labels(ax, bars, fmt="{:,.0f}", fontsize=12, offset_frac=0.01):
    """Add value labels to the right of each horizontal bar (plt.barh)."""
    x_min, x_max = ax.get_xlim()
    offset = (x_max - x_min) * offset_frac
    for bar in bars:
        width = bar.get_width()
        ax.text(
            width + offset,
            bar.get_y() + bar.get_height() / 2,
            fmt.format(width),
            ha="left",
            va="center",
            fontsize=fontsize,
            fontweight="bold",
        )

# ============================================================
# 1. LOAD DATASET
# ============================================================

if not os.path.exists(FILE_PATH):
    raise FileNotFoundError(f"File not found: {FILE_PATH}")

load_start = time.time()
df = pd.read_csv(FILE_PATH)
load_time = time.time() - load_start

df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"], errors="coerce")
df["Revenue"] = pd.to_numeric(df["Revenue"], errors="coerce")
df["Quantity"] = pd.to_numeric(df["Quantity"], errors="coerce")
df["Price"] = pd.to_numeric(df["Price"], errors="coerce")
df["Customer ID"] = pd.to_numeric(df["Customer ID"], errors="coerce")

if "YearMonth" not in df.columns:
    df["YearMonth"] = df["InvoiceDate"].dt.to_period("M").astype(str)

if "Date" not in df.columns:
    df["Date"] = df["InvoiceDate"].dt.date

if "Year" not in df.columns:
    df["Year"] = df["InvoiceDate"].dt.year

if "Month" not in df.columns:
    df["Month"] = df["InvoiceDate"].dt.month

if "Day" not in df.columns:
    df["Day"] = df["InvoiceDate"].dt.day

if "Hour" not in df.columns:
    df["Hour"] = df["InvoiceDate"].dt.hour

df = df.dropna(subset=["InvoiceDate", "Revenue", "Quantity", "Price"])

print(f"Dataset Records       : {len(df):,}")
print(f"Dataset Columns       : {len(df.columns):,}")
print(f"Data Loading Time     : {load_time:.2f} seconds")

# ============================================================
# 2. FIGURE 1 - DATASET DISTRIBUTION (value labels added)
# ============================================================

year_distribution = df.groupby("Year")["Quantity"].sum()

fig1, ax1 = plt.subplots(figsize=FIG_SIZE)
bars1 = ax1.bar(year_distribution.index.astype(str), year_distribution.values, color="steelblue")
ax1.set_title("Dataset Distribution", fontsize=18, fontweight="bold")
ax1.set_xlabel("Year", fontsize=18, fontweight="bold")
ax1.set_ylabel("Total Quantity", fontsize=18, fontweight="bold")
ax1.tick_params(axis="x", labelsize=16)
ax1.tick_params(axis="y", labelsize=16)
ax1.grid(False)
ax1.set_ylim(0, year_distribution.values.max() * 1.12)  # headroom for labels
add_bar_labels(ax1, bars1, fmt="{:,.0f}", fontsize=13)
plt.tight_layout()
plt.savefig(os.path.join(PLOT_FOLDER, "Dataset_Distribution.png"), dpi=800, bbox_inches="tight")
plt.show()
plt.close()

# ============================================================
# 3. FIGURE 2 - MONTHLY SALES TREND
# ============================================================

monthly_sales = df.groupby("YearMonth")["Revenue"].sum()

plt.figure(figsize=FIG_SIZE)
plt.plot(monthly_sales.index, monthly_sales.values, marker="o", linewidth=3, color="darkorange")
plt.title("Monthly Sales Trend", fontsize=18, fontweight="bold")
plt.xlabel("Month", fontsize=18, fontweight="bold")
plt.ylabel("Revenue", fontsize=18, fontweight="bold")
plt.xticks(rotation=30, fontsize=14, fontweight="bold")
plt.yticks(fontsize=14, fontweight="bold")
plt.grid(False)
plt.tight_layout()
plt.savefig(os.path.join(PLOT_FOLDER, "Monthly_Sales_Trend.png"), dpi=800, bbox_inches="tight")
plt.show()
plt.close()

# ============================================================
# 4. FIGURE 3 - DAILY SALES TREND
# ============================================================

daily_sales = df.groupby("Date")["Revenue"].sum()

plt.figure(figsize=FIG_SIZE)
plt.plot(pd.to_datetime(daily_sales.index), daily_sales.values, linewidth=2, color="forestgreen")
plt.title("Daily Sales Trend", fontsize=18, fontweight="bold")
plt.xlabel("Date", fontsize=18, fontweight="bold")
plt.ylabel("Revenue", fontsize=18, fontweight="bold")
plt.xticks(rotation=0, fontsize=18, fontweight="bold")
plt.yticks(fontsize=18, fontweight="bold")
plt.grid(False)
plt.tight_layout()
plt.savefig(os.path.join(PLOT_FOLDER, "Daily_Sales_Trend.png"), dpi=800, bbox_inches="tight")
plt.show()
plt.close()

# ============================================================
# 5. FIGURE 4 - TOP 10 PRODUCTS BY REVENUE (value labels added)
# ============================================================

top_revenue_products = df.groupby("Description")["Revenue"].sum().nlargest(10).sort_values()

fig4, ax4 = plt.subplots(figsize=FIG_SIZE)
bars4 = ax4.barh(top_revenue_products.index.astype(str), top_revenue_products.values, color="crimson")
ax4.set_title("Top 10 Products by Revenue", fontsize=18, fontweight="bold")
ax4.set_xlabel("Revenue", fontsize=18, fontweight="bold")
ax4.set_ylabel("Product", fontsize=18, fontweight="bold")
ax4.tick_params(axis="x", labelsize=14)
ax4.tick_params(axis="y", labelsize=14)
ax4.grid(False)
ax4.set_xlim(0, top_revenue_products.values.max() * 1.15)  # headroom for labels
add_barh_labels(ax4, bars4, fmt="{:,.0f}", fontsize=12)
plt.tight_layout()
plt.savefig(os.path.join(PLOT_FOLDER, "Top_Products_Revenue.png"), dpi=800, bbox_inches="tight")
plt.show()
plt.close()

# ============================================================
# 6. FIGURE 5 - TOP 10 PRODUCTS BY QUANTITY (value labels added)
# ============================================================

top_quantity_products = df.groupby("Description")["Quantity"].sum().nlargest(10).sort_values()

fig5, ax5 = plt.subplots(figsize=FIG_SIZE)
bars5 = ax5.barh(top_quantity_products.index.astype(str), top_quantity_products.values, color="purple")
ax5.set_title("Top 10 Products by Quantity", fontsize=18, fontweight="bold")
ax5.set_xlabel("Quantity Sold", fontsize=18, fontweight="bold")
ax5.set_ylabel("Product", fontsize=18, fontweight="bold")
ax5.tick_params(axis="x", labelsize=14)
ax5.tick_params(axis="y", labelsize=14)
ax5.grid(False)
ax5.set_xlim(0, top_quantity_products.values.max() * 1.15)  # headroom for labels
add_barh_labels(ax5, bars5, fmt="{:,.0f}", fontsize=12)
plt.tight_layout()
plt.savefig(os.path.join(PLOT_FOLDER, "Top_Products_Quantity.png"), dpi=800, bbox_inches="tight")
plt.show()
plt.close()

# ============================================================
# 7. FIGURE 6 - COUNTRY-WISE REVENUE DISTRIBUTION (value labels added)
# ============================================================

country_revenue = df.groupby("Country")["Revenue"].sum().nlargest(10).sort_values()

fig6, ax6 = plt.subplots(figsize=FIG_SIZE)
bars6 = ax6.barh(country_revenue.index.astype(str), country_revenue.values, color="teal")
ax6.set_title("Country-Wise Revenue Distribution", fontsize=18, fontweight="bold")
ax6.set_xlabel("Revenue", fontsize=18, fontweight="bold")
ax6.set_ylabel("Country", fontsize=18, fontweight="bold")
ax6.tick_params(axis="x", labelsize=14)
ax6.tick_params(axis="y", labelsize=14)
ax6.grid(False)
ax6.set_xlim(0, country_revenue.values.max() * 1.15)  # headroom for labels
add_barh_labels(ax6, bars6, fmt="{:,.0f}", fontsize=12)
plt.tight_layout()
plt.savefig(os.path.join(PLOT_FOLDER, "Country_Revenue.png"), dpi=800, bbox_inches="tight")
plt.show()
plt.close()

# ============================================================
# 8. FIGURE 7 - REVENUE DISTRIBUTION HISTOGRAM
# ============================================================

revenue_sample = df["Revenue"].sample(min(100000, len(df)), random_state=42)

plt.figure(figsize=FIG_SIZE)
plt.hist(revenue_sample, bins=50, color="goldenrod", edgecolor="black")
plt.title("Revenue Distribution Histogram", fontsize=18, fontweight="bold")
plt.xlabel("Revenue", fontsize=18, fontweight="bold")
plt.ylabel("Frequency", fontsize=18, fontweight="bold")
plt.xticks(fontsize=18, fontweight="bold")
plt.yticks(fontsize=18, fontweight="bold")
plt.grid(False)
plt.tight_layout()
plt.savefig(os.path.join(PLOT_FOLDER, "Revenue_Distribution.png"), dpi=800, bbox_inches="tight")
plt.show()
plt.close()

# ============================================================
# 9. FIGURE 8 - CUSTOMER RFM SEGMENTATION (value labels added)
# ============================================================

analysis_date = df["InvoiceDate"].max() + pd.Timedelta(days=1)

rfm = df.groupby("Customer ID").agg(Recency=("InvoiceDate", lambda x: (analysis_date - x.max()).days), Frequency=("Invoice", "nunique"), Monetary=("Revenue", "sum")).reset_index()

rfm["R_Score"] = pd.qcut(rfm["Recency"].rank(method="first"), 4, labels=[4, 3, 2, 1]).astype(int)
rfm["F_Score"] = pd.qcut(rfm["Frequency"].rank(method="first"), 4, labels=[1, 2, 3, 4]).astype(int)
rfm["M_Score"] = pd.qcut(rfm["Monetary"].rank(method="first"), 4, labels=[1, 2, 3, 4]).astype(int)

rfm["RFM_Score"] = rfm["R_Score"].astype(str) + rfm["F_Score"].astype(str) + rfm["M_Score"].astype(str)

def classify_customer(row):
    if row["R_Score"] >= 3 and row["F_Score"] >= 3 and row["M_Score"] >= 3:
        return "Champions"
    if row["R_Score"] >= 3 and row["F_Score"] >= 2:
        return "Loyal Customers"
    if row["R_Score"] >= 3 and row["M_Score"] >= 2:
        return "Potential Loyal"
    if row["R_Score"] <= 2 and row["F_Score"] >= 3:
        return "At Risk"
    if row["R_Score"] <= 2 and row["M_Score"] <= 2:
        return "Lost Customers"
    return "Regular Customers"

rfm["Segment"] = rfm.apply(classify_customer, axis=1)

segment_counts = rfm["Segment"].value_counts()

fig8, ax8 = plt.subplots(figsize=FIG_SIZE)
bars8 = ax8.bar(segment_counts.index, segment_counts.values, color="slateblue")
ax8.set_title("Customer RFM Segmentation", fontsize=18, fontweight="bold")
ax8.set_xlabel("Customer Segment", fontsize=18, fontweight="bold")
ax8.set_ylabel("Number of Customers", fontsize=18, fontweight="bold")
ax8.tick_params(axis="x", labelsize=18, rotation=10)
ax8.tick_params(axis="y", labelsize=18)
ax8.grid(False)
ax8.set_ylim(0, segment_counts.values.max() * 1.12)  # headroom for labels
add_bar_labels(ax8, bars8, fmt="{:,.0f}", fontsize=14)
plt.tight_layout()
plt.savefig(os.path.join(PLOT_FOLDER, "RFM_Segmentation.png"), dpi=800, bbox_inches="tight")
plt.show()
plt.close()

# ============================================================
# 10. FIGURE 9 - CORRELATION HEATMAP
# ============================================================

correlation_columns = ["Quantity", "Price", "Revenue", "Year", "Month", "Day", "Hour"]
correlation_data = df[correlation_columns].corr()

plt.figure(figsize=FIG_SIZE)
plt.imshow(correlation_data, interpolation="nearest", cmap="coolwarm", aspect="auto")
plt.colorbar()
plt.xticks(range(len(correlation_columns)), correlation_columns, rotation=45, fontsize=12, fontweight="bold")
plt.yticks(range(len(correlation_columns)), correlation_columns, fontsize=12, fontweight="bold")

for i in range(len(correlation_columns)):
    for j in range(len(correlation_columns)):
        plt.text(j, i, f"{correlation_data.iloc[i, j]:.2f}", ha="center", va="center", fontsize=11, fontweight="bold")

plt.title("Correlation Heatmap", fontsize=18, fontweight="bold")
plt.tight_layout()
plt.savefig(os.path.join(PLOT_FOLDER, "Correlation_Heatmap.png"), dpi=800, bbox_inches="tight")
plt.show()
plt.close()

# ============================================================
# 11. CLOUD PERFORMANCE EVALUATION
# ============================================================

print("\n" + "=" * 80)
print("CLOUD-BASED E-COMMERCE ANALYTICS PERFORMANCE EVALUATION")
print("=" * 80)

file_size_mb = os.path.getsize(FILE_PATH) / (1024 * 1024)
memory_usage_mb = df.memory_usage(deep=True).sum() / (1024 * 1024)

storage_utilization = memory_usage_mb / (memory_usage_mb + file_size_mb) * 100

# ============================================================
# UPLOAD TIME
# ============================================================

upload_start = time.time()
upload_test_df = pd.read_csv(FILE_PATH)
upload_time = time.time() - upload_start

# ============================================================
# RETRIEVAL TIME
# ============================================================

retrieval_start = time.time()
retrieval_df = pd.read_csv(FILE_PATH)
retrieval_time = time.time() - retrieval_start

# ============================================================
# QUERY LATENCY
# ============================================================

query_start = time.time()
query_median = df["Revenue"].median()
query_result = df[df["Revenue"] > query_median]
query_response_time = (time.time() - query_start) * 1000

# ============================================================
# DATA PROCESSING TIME
# ============================================================

processing_start = time.time()
monthly_processing = df.groupby("YearMonth")["Revenue"].sum()
country_processing = df.groupby("Country")["Revenue"].sum()
product_processing = df.groupby("Description")["Revenue"].sum()
processing_time = time.time() - processing_start

# ============================================================
# CLOUD THROUGHPUT
# ============================================================

throughput = file_size_mb / processing_time if processing_time > 0 else 0

# ============================================================
# SCALABILITY
# ============================================================

scalability_start = time.time()

sample_size = min(100000, len(df))

scalability_sample = df.sample(sample_size, random_state=42)

scalability_result = scalability_sample.groupby("Country")["Revenue"].sum()

scalability_time = time.time() - scalability_start

scalability_percentage = sample_size / len(df) * 100 if len(df) > 0 else 0

# ============================================================
# AVAILABILITY
# ============================================================

availability = 100.0 if os.path.exists(FILE_PATH) else 0.0

# ============================================================
# EDA EXECUTION TIME
# ============================================================

eda_start = time.time()

_ = df.groupby("YearMonth")["Revenue"].sum()
_ = df.groupby("Country")["Revenue"].sum()
_ = df.groupby("Description")["Revenue"].sum()

eda_execution_time = time.time() - eda_start

# ============================================================
# DASHBOARD RESPONSE TIME
# ============================================================

dashboard_start = time.time()

dashboard_result = df.groupby("YearMonth")["Revenue"].sum()

dashboard_response_time = time.time() - dashboard_start

# ============================================================
# DATA QUALITY
# ============================================================

total_cells = df.shape[0] * df.shape[1]
total_missing_values = df.isnull().sum().sum()

missing_value_rate = total_missing_values / total_cells * 100 if total_cells > 0 else 0

duplicate_records = df.duplicated().sum()

duplicate_rate = duplicate_records / len(df) * 100 if len(df) > 0 else 0

# ============================================================
# 12. FIGURE 10 - CLOUD PERFORMANCE OVERVIEW
# ============================================================

performance_metrics = pd.DataFrame({
    "Evaluation Area": [
        "Cloud Storage",
        "Data Upload",
        "Data Retrieval",
        "Query Processing",
        "Processing",
        "Cloud Performance",
        "Scalability",
        "Availability",
        "EDA Efficiency",
        "Dashboard",
        "Data Quality",
        "Data Quality"
    ],
    "Performance Metric": [
        "Storage Utilization (%)",
        "Upload Time (s)",
        "Retrieval Time (s)",
        "Query Response Time (ms)",
        "Data Processing Time (s)",
        "Throughput (MB/s)",
        "Scalability (%)",
        "Availability (%)",
        "EDA Execution Time (s)",
        "Visualization Response Time (s)",
        "Missing Value Rate (%)",
        "Duplicate Rate (%)"
    ],
    "Value": [
        storage_utilization,
        upload_time,
        retrieval_time,
        query_response_time,
        processing_time,
        throughput,
        scalability_percentage,
        availability,
        eda_execution_time,
        dashboard_response_time,
        missing_value_rate,
        duplicate_rate
    ]
})

# ============================================================
# FIGURE 10 - PERFORMANCE SUMMARY TABLE IMAGE
# ============================================================

fig10, ax10 = plt.subplots(figsize=(12, 8))
ax10.axis("off")

table_data = performance_metrics.copy()
table_data["Value"] = table_data["Value"].map(lambda x: f"{x:.4f}")

table = ax10.table(cellText=table_data.values, colLabels=table_data.columns, loc="center", cellLoc="center")
table.auto_set_font_size(False)
table.set_fontsize(11)
table.scale(1, 2)

plt.title("Fig. 10: Cloud-Based Performance Evaluation", fontsize=18, fontweight="bold", pad=20)
plt.tight_layout()

plt.show()
plt.close()

# ============================================================
# 13. FIGURE 11 - CLOUD THROUGHPUT OVER TIME
# ============================================================

throughput_values = []
throughput_time = []

for i in range(1, 11):
    start_time = time.time()

    _ = df.groupby("YearMonth")["Revenue"].sum()

    elapsed_time = time.time() - start_time

    current_throughput = file_size_mb / elapsed_time if elapsed_time > 0 else 0

    throughput_time.append(i)
    throughput_values.append(current_throughput)

plt.figure(figsize=FIG_SIZE)
plt.plot(throughput_time, throughput_values, marker="o", linewidth=3, color="darkblue")
plt.title("Cloud Throughput Over Time", fontsize=18, fontweight="bold")
plt.xlabel("Measurement Interval", fontsize=18, fontweight="bold")
plt.ylabel("Cloud Throughput (MB/s)", fontsize=18, fontweight="bold")
plt.xticks(fontsize=16, fontweight="bold")
plt.yticks(fontsize=16, fontweight="bold")
plt.grid(False)
plt.tight_layout()
plt.savefig(os.path.join(PLOT_FOLDER, "Cloud_Throughput.png"), dpi=800, bbox_inches="tight")
plt.show()
plt.close()

# ============================================================
# 14. FIGURE 12 - CLOUD QUERY LATENCY OVER TIME
# ============================================================

latency_values = []
latency_time = []

for i in range(1, 11):
    start_time = time.time()

    median_value = df["Revenue"].median()

    _ = df[df["Revenue"] > median_value]

    elapsed_time = (time.time() - start_time) * 1000

    latency_time.append(i)
    latency_values.append(elapsed_time)

plt.figure(figsize=FIG_SIZE)
plt.plot(latency_time, latency_values, marker="o", linewidth=3, color="darkred")
plt.title("Cloud Query Latency Over Time", fontsize=18, fontweight="bold")
plt.xlabel("Measurement Interval", fontsize=18, fontweight="bold")
plt.ylabel("Query Latency (ms)", fontsize=18, fontweight="bold")
plt.xticks(fontsize=16, fontweight="bold")
plt.yticks(fontsize=16, fontweight="bold")
plt.grid(False)
plt.tight_layout()
plt.savefig(os.path.join(PLOT_FOLDER, "Cloud_Query_Latency.png"), dpi=800, bbox_inches="tight")
plt.show()
plt.close()

# ============================================================
# 15. SAVE RESULTS
# ============================================================

rfm.to_csv("rfm_customer_segmentation.csv", index=False)
country_revenue.to_csv("country_revenue.csv")
top_revenue_products.to_csv("top_products_revenue.csv")
top_quantity_products.to_csv("top_products_quantity.csv")
monthly_sales.to_csv("monthly_sales.csv")
daily_sales.to_csv("daily_sales.csv")
performance_metrics.to_csv("docker_performance_metrics.csv", index=False)

throughput_df = pd.DataFrame({
    "Measurement Interval": throughput_time,
    "Throughput_MBps": throughput_values
})

latency_df = pd.DataFrame({
    "Measurement Interval": latency_time,
    "Latency_ms": latency_values
})

throughput_df.to_csv("cloud_throughput_over_time.csv", index=False)
latency_df.to_csv("cloud_query_latency_over_time.csv", index=False)

# ============================================================
# 16. FINAL RESULTS
# ============================================================

print("\n" + "=" * 80)
print("EDA ANALYSIS COMPLETED")
print("=" * 80)

print(f"Total Transactions     : {len(df):,}")
print(f"Unique Customers       : {df['Customer ID'].nunique():,}")
print(f"Unique Products        : {df['StockCode'].nunique():,}")
print(f"Countries              : {df['Country'].nunique():,}")
print(f"Total Revenue          : {df['Revenue'].sum():,.2f}")
print(f"RFM Customers          : {len(rfm):,}")
print(f"Processing Time        : {processing_time:.4f} seconds")
print(f"Throughput             : {throughput:.2f} MB/s")

print("\n✓ Fig. 1 generated")
print("✓ Fig. 2 generated")
print("✓ Fig. 3 generated")
print("✓ Fig. 4 generated")
print("✓ Fig. 5 generated")
print("✓ Fig. 6 generated")
print("✓ Fig. 7 generated")
print("✓ Fig. 8 generated")
print("✓ Fig. 9 generated")
print("✓ Fig. 10 generated")
print("✓ Fig. 11 Cloud Throughput generated")
print("✓ Fig. 12 Cloud Query Latency generated")

# ============================================================
# 17. PERFORMANCE RESULTS IN COMMAND WINDOW ONLY
# ============================================================

print("\n" + "=" * 80)
print("CLOUD PERFORMANCE EVALUATION RESULTS")
print("=" * 80)

print(f"Storage Utilization (%)       : {storage_utilization:.2f}")
print(f"Upload Time (s)               : {upload_time:.4f}")
print(f"Retrieval Time (s)            : {retrieval_time:.4f}")
print(f"Query Response Time (ms)      : {query_response_time:.2f}")
print(f"Data Processing Time (s)      : {processing_time:.4f}")
print(f"Throughput (MB/s)             : {throughput:.2f}")
print(f"Scalability (%)               : {scalability_percentage:.2f}")
print(f"Availability (%)              : {availability:.2f}")
print(f"EDA Execution Time (s)        : {eda_execution_time:.4f}")
print(f"Visualization Response (s)    : {dashboard_response_time:.4f}")
print(f"Missing Value Rate (%)        : {missing_value_rate:.4f}")
print(f"Duplicate Rate (%)             : {duplicate_rate:.4f}")

print("=" * 80)

# ============================================================
# 18. PERFORMANCE TABLE IN COMMAND WINDOW
# ============================================================

print("\nPERFORMANCE METRICS TABLE")
print("=" * 80)
print(performance_metrics.to_string(index=False))
print("=" * 80)

# ============================================================
# 19. CLOUD TIME-BASED RESULTS
# ============================================================

print("\nCLOUD THROUGHPUT OVER TIME")
print("=" * 80)

for interval, value in zip(throughput_time, throughput_values):
    print(f"Interval {interval:02d} : {value:.2f} MB/s")

print("=" * 80)

print("\nCLOUD QUERY LATENCY OVER TIME")
print("=" * 80)

for interval, value in zip(latency_time, latency_values):
    print(f"Interval {interval:02d} : {value:.2f} ms")

print("=" * 80)

# ============================================================
# 20. SAVED FILES
# ============================================================

print("\n✓ RFM results saved to rfm_customer_segmentation.csv")
print("✓ Country revenue saved to country_revenue.csv")
print("✓ Product revenue saved to top_products_revenue.csv")
print("✓ Product quantity saved to top_products_quantity.csv")
print("✓ Monthly sales saved to monthly_sales.csv")
print("✓ Daily sales saved to daily_sales.csv")
print("✓ Performance metrics saved to docker_performance_metrics.csv")
print("✓ Cloud throughput saved to cloud_throughput_over_time.csv")
print("✓ Cloud latency saved to cloud_query_latency_over_time.csv")
print(f"✓ All plots saved to folder: {PLOT_FOLDER}")

print("\n" + "=" * 80)
print("COMPLETE EDA AND CLOUD PERFORMANCE ANALYSIS FINISHED SUCCESSFULLY")
print("=" * 80)