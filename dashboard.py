import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(page_title="Online Retail II EDA Dashboard", page_icon="📊", layout="wide")

# ============================================================
# THEME / COLOR PALETTE  (matches the reference dark-purple dashboard)
# ============================================================

BG_DARK       = "#1a1130"   # page background
PANEL_DARK    = "#241a3d"   # card / panel background
PANEL_BORDER  = "#4a3b7a"
TEXT_LIGHT    = "#f5f3fb"
TEXT_MUTED    = "#b9aed6"

PURPLE        = "#7c5cff"   # primary bars
PURPLE_LIGHT  = "#a78bfa"
ORANGE        = "#ffab40"   # secondary bars (customers)
PINK          = "#ff5da2"
INDIGO        = "#5b4fd6"
DONUT_COLORS  = ["#7c5cff", "#5b4fd6", "#ff5da2", "#ffab40", "#4dd0e1"]

plt.rcParams["font.family"] = "DejaVu Sans"
plt.rcParams["font.weight"] = "bold"
plt.rcParams["axes.titleweight"] = "bold"
plt.rcParams["axes.labelweight"] = "bold"
plt.rcParams["text.color"] = TEXT_LIGHT
plt.rcParams["axes.labelcolor"] = TEXT_LIGHT
plt.rcParams["xtick.color"] = TEXT_LIGHT
plt.rcParams["ytick.color"] = TEXT_LIGHT
plt.rcParams["axes.edgecolor"] = PANEL_BORDER
plt.rcParams["figure.facecolor"] = "none"
plt.rcParams["axes.facecolor"] = "none"
plt.rcParams["savefig.facecolor"] = "none"

FIG_SIZE_WIDE = (7, 3.2)
FIG_SIZE_TALL = (7, 4.5)
FIG_SIZE_SQ = (5.5, 4.2)

# ============================================================
# CUSTOM CSS - dark purple "card" theme
# ============================================================

st.markdown(f"""
<style>
    .stApp {{
        background: radial-gradient(circle at top left, #2a1d4d 0%, {BG_DARK} 55%);
        color: {TEXT_LIGHT};
    }}
    section[data-testid="stSidebar"] {{
        background-color: {PANEL_DARK};
        border-right: 1px solid {PANEL_BORDER};
    }}
    h1, h2, h3, h4, h5, h6, p, label, span, div {{
        color: {TEXT_LIGHT};
    }}
    /* Title banner */
    .dash-title {{
        background: linear-gradient(135deg, #3a2a66 0%, #241a3d 100%);
        border: 1px solid {PANEL_BORDER};
        border-radius: 14px;
        padding: 18px 26px;
        margin-bottom: 18px;
    }}
    .dash-title h1 {{
        margin: 0;
        font-size: 30px;
        font-weight: 800;
    }}
    .dash-title p {{
        margin: 4px 0 0 0;
        color: {TEXT_MUTED};
        font-size: 14px;
    }}
    /* KPI cards */
    .kpi-card {{
        background: linear-gradient(160deg, #2f2256 0%, #221836 100%);
        border: 1px solid {PANEL_BORDER};
        border-radius: 14px;
        padding: 16px 18px;
        text-align: left;
        height: 100%;
    }}
    .kpi-value {{
        font-size: 30px;
        font-weight: 800;
        color: {TEXT_LIGHT};
        line-height: 1.1;
    }}
    .kpi-label {{
        font-size: 13px;
        color: {TEXT_MUTED};
        margin-top: 4px;
        letter-spacing: 0.3px;
    }}
    /* Chart panel wrapper */
    .chart-panel {{
        background: {PANEL_DARK};
        border: 1px solid {PANEL_BORDER};
        border-radius: 14px;
        padding: 14px 18px 6px 18px;
        margin-bottom: 18px;
    }}
    .chart-panel h4 {{
        margin: 0 0 6px 0;
        font-size: 15px;
        font-weight: 700;
        color: {TEXT_LIGHT};
        text-align: center;
    }}
    div[data-testid="stDataFrame"] {{
        border-radius: 10px;
        overflow: hidden;
    }}
</style>
""", unsafe_allow_html=True)

# ============================================================
# LOAD DATA
# ============================================================

DATA_FILE = "engineered_online_retail.csv"

if not os.path.exists(DATA_FILE):
    st.error(f"{DATA_FILE} not found. Run the preprocessing and EDA code first.")
    st.stop()

df = pd.read_csv(DATA_FILE)

df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"], errors="coerce")
df["Revenue"] = pd.to_numeric(df["Revenue"], errors="coerce")
df["Quantity"] = pd.to_numeric(df["Quantity"], errors="coerce")
df["Price"] = pd.to_numeric(df["Price"], errors="coerce")
df["Customer ID"] = pd.to_numeric(df["Customer ID"], errors="coerce")

df = df.dropna(subset=["InvoiceDate", "Revenue", "Quantity", "Price"])

if "YearMonth" not in df.columns:
    df["YearMonth"] = df["InvoiceDate"].dt.to_period("M").astype(str)
if "Year" not in df.columns:
    df["Year"] = df["InvoiceDate"].dt.year

# ============================================================
# SIDEBAR FILTERS
# ============================================================

st.sidebar.header("Dashboard Filters")

countries = sorted(df["Country"].dropna().unique())
selected_country = st.sidebar.selectbox("Select Country", ["All"] + countries)

years = sorted(df["Year"].dropna().unique())
selected_year = st.sidebar.multiselect("Select Year", years, default=years)

filtered_df = df.copy()

if selected_country != "All":
    filtered_df = filtered_df[filtered_df["Country"] == selected_country]

if selected_year:
    filtered_df = filtered_df[filtered_df["Year"].isin(selected_year)]

# ============================================================
# HELPER: compact number formatting (438K style, like the reference)
# ============================================================

def compact_number(value):
    if pd.isna(value):
        return "0"
    abs_val = abs(value)
    if abs_val >= 1_000_000:
        return f"{value/1_000_000:.1f}M"
    if abs_val >= 1_000:
        return f"{value/1_000:.0f}K"
    return f"{value:,.0f}"

def kpi_card(col, value_str, label):
    col.markdown(f"""
    <div class="kpi-card">
        <div class="kpi-value">{value_str}</div>
        <div class="kpi-label">{label}</div>
    </div>
    """, unsafe_allow_html=True)

def start_panel(title):
    st.markdown(f'<div class="chart-panel"><h4>{title}</h4>', unsafe_allow_html=True)

def end_panel():
    st.markdown('</div>', unsafe_allow_html=True)

# ============================================================
# TITLE BANNER
# ============================================================

st.markdown("""
<div class="dash-title">
    <h1>📊 Online Retail II Ecommerce Sales Dashboard</h1>
    <p>Interactive Exploratory Data Analysis and Sales Intelligence Dashboard</p>
</div>
""", unsafe_allow_html=True)

# ============================================================
# KPI ROW  (mirrors the 4 big number cards in the reference)
# ============================================================

total_revenue = filtered_df["Revenue"].sum()
total_quantity = filtered_df["Quantity"].sum()
total_transactions = len(filtered_df)  # matches EDA script's "Total Transactions : len(df)" (e.g. 779,425)
unique_invoices = filtered_df["Invoice"].nunique()
avg_order_value = filtered_df.groupby("Invoice")["Revenue"].sum().mean() if unique_invoices > 0 else 0
total_customers = filtered_df["Customer ID"].nunique()

k1, k2, k3, k4, k5 = st.columns(5)
kpi_card(k1, compact_number(total_revenue), "Sum of Revenue")
kpi_card(k2, compact_number(total_quantity), "Sum of Quantity")
kpi_card(k3, f"{total_transactions:,}", "Transactions")
kpi_card(k4, compact_number(avg_order_value), "Avg Order Value")
kpi_card(k5, f"{total_customers:,}", "Customers")

st.write("")

# ============================================================
# ROW 1 : Top Countries (bar)  |  RFM Segment (donut)  |  Monthly Revenue (bar)
# ============================================================

row1_left, row1_mid, row1_right = st.columns([1.1, 1, 1.1])

with row1_left:
    start_panel("Top 6 Countries by Revenue")
    country_revenue = filtered_df.groupby("Country")["Revenue"].sum().nlargest(6).sort_values()
    fig, ax = plt.subplots(figsize=FIG_SIZE_TALL)
    ax.barh(country_revenue.index.astype(str), country_revenue.values, color=PURPLE)
    ax.set_xlabel("Sum of Revenue")
    ax.grid(False)
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    plt.tight_layout()
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)
    end_panel()

with row1_mid:
    start_panel("Quantity by RFM Segment")
    rfm_file = "rfm_customer_segmentation.csv"
    if os.path.exists(rfm_file):
        rfm_all = pd.read_csv(rfm_file)
        segment_qty = rfm_all["Segment"].value_counts()
        fig, ax = plt.subplots(figsize=FIG_SIZE_SQ)
        wedges, _, autotexts = ax.pie(
            segment_qty.values,
            labels=None,
            autopct=lambda p: f"{p:.1f}%",
            pctdistance=0.8,
            colors=DONUT_COLORS,
            wedgeprops=dict(width=0.42, edgecolor=BG_DARK, linewidth=2),
            textprops={"fontsize": 9, "color": TEXT_LIGHT, "fontweight": "bold"},
        )
        ax.legend(wedges, segment_qty.index, loc="center", bbox_to_anchor=(0.5, -0.15),
                   ncol=2, fontsize=8, frameon=False, labelcolor=TEXT_LIGHT)
        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)
    else:
        st.info("Run the EDA script first to generate rfm_customer_segmentation.csv")
    end_panel()

with row1_right:
    start_panel("Revenue by Month")
    monthly_sales = filtered_df.groupby("YearMonth")["Revenue"].sum()
    fig, ax = plt.subplots(figsize=FIG_SIZE_TALL)
    ax.bar(monthly_sales.index.astype(str), monthly_sales.values, color=PURPLE_LIGHT)
    ax.set_ylabel("Revenue")
    ax.tick_params(axis="x", rotation=60, labelsize=8)
    ax.grid(False)
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    plt.tight_layout()
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)
    end_panel()

# ============================================================
# ROW 2 : Top Products (bar)  |  Revenue Share Top Countries (donut)  |  Top Customers (bar)
# ============================================================

row2_left, row2_mid, row2_right = st.columns([1.1, 1, 1.1])

with row2_left:
    start_panel("Top 5 Products by Quantity")
    top_products_quantity = filtered_df.groupby("Description")["Quantity"].sum().nlargest(5).sort_values()
    fig, ax = plt.subplots(figsize=FIG_SIZE_TALL)
    ax.barh([str(p)[:22] for p in top_products_quantity.index], top_products_quantity.values, color=PURPLE)
    ax.set_xlabel("Sum of Quantity")
    ax.grid(False)
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    plt.tight_layout()
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)
    end_panel()

with row2_mid:
    start_panel("Revenue Share - Top 5 Countries")
    top5_country_rev = filtered_df.groupby("Country")["Revenue"].sum().nlargest(5)
    fig, ax = plt.subplots(figsize=FIG_SIZE_SQ)
    wedges, _, autotexts = ax.pie(
        top5_country_rev.values,
        labels=None,
        autopct=lambda p: f"{p:.1f}%",
        pctdistance=0.8,
        colors=DONUT_COLORS,
        wedgeprops=dict(width=0.42, edgecolor=BG_DARK, linewidth=2),
        textprops={"fontsize": 9, "color": TEXT_LIGHT, "fontweight": "bold"},
    )
    ax.legend(wedges, top5_country_rev.index, loc="center", bbox_to_anchor=(0.5, -0.15),
               ncol=2, fontsize=8, frameon=False, labelcolor=TEXT_LIGHT)
    plt.tight_layout()
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)
    end_panel()

with row2_right:
    start_panel("Top 4 Customers by Revenue")
    top_customers = filtered_df.groupby("Customer ID")["Revenue"].sum().nlargest(4)
    fig, ax = plt.subplots(figsize=FIG_SIZE_TALL)
    ax.bar([f"{int(c)}" for c in top_customers.index], top_customers.values, color=ORANGE)
    ax.set_ylabel("Sum of Revenue")
    ax.grid(False)
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    plt.tight_layout()
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)
    end_panel()

# ============================================================
# ROW 3 : Revenue Distribution Histogram | Correlation Heatmap
# ============================================================

row3_left, row3_right = st.columns(2)

with row3_left:
    start_panel("Revenue Distribution")
    if len(filtered_df) > 0:
        revenue_sample = filtered_df["Revenue"].sample(min(100000, len(filtered_df)), random_state=42)
        fig, ax = plt.subplots(figsize=FIG_SIZE_TALL)
        ax.hist(revenue_sample, bins=40, color=PURPLE, edgecolor=BG_DARK)
        ax.set_xlabel("Revenue")
        ax.set_ylabel("Frequency")
        ax.grid(False)
        for spine in ["top", "right"]:
            ax.spines[spine].set_visible(False)
        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)
    end_panel()

with row3_right:
    start_panel("Correlation Heatmap")
    correlation_columns = ["Quantity", "Price", "Revenue", "Year", "Month", "Day", "Hour"]
    available_correlation_columns = [c for c in correlation_columns if c in filtered_df.columns]
    if len(available_correlation_columns) >= 2:
        correlation_data = filtered_df[available_correlation_columns].corr()
        fig, ax = plt.subplots(figsize=FIG_SIZE_TALL)
        image = ax.imshow(correlation_data, cmap="coolwarm", aspect="auto")
        cbar = fig.colorbar(image)
        cbar.ax.yaxis.set_tick_params(color=TEXT_LIGHT)
        plt.setp(plt.getp(cbar.ax.axes, "yticklabels"), color=TEXT_LIGHT)
        ax.set_xticks(range(len(available_correlation_columns)))
        ax.set_xticklabels(available_correlation_columns, rotation=45, fontsize=9)
        ax.set_yticks(range(len(available_correlation_columns)))
        ax.set_yticklabels(available_correlation_columns, fontsize=9)
        for i in range(len(available_correlation_columns)):
            for j in range(len(available_correlation_columns)):
                ax.text(j, i, f"{correlation_data.iloc[i, j]:.2f}", ha="center", va="center",
                        fontsize=8, fontweight="bold", color=TEXT_LIGHT)
        ax.grid(False)
        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)
    end_panel()

# ============================================================
# ADAPTIVE SALES INSIGHTS
# ============================================================

start_panel("Adaptive Sales Insights")
if len(filtered_df) > 0:
    best_product = filtered_df.groupby("Description")["Revenue"].sum().idxmax()
    best_country = filtered_df.groupby("Country")["Revenue"].sum().idxmax()
    best_month = filtered_df.groupby("YearMonth")["Revenue"].sum().idxmax()

    c1, c2 = st.columns(2)
    with c1:
        st.success(f"Highest Revenue Product: {best_product}")
        st.info(f"Highest Revenue Country: {best_country}")
    with c2:
        st.success(f"Best Sales Month: {best_month}")
        st.info(f"Average Order Value: {avg_order_value:,.2f}")
end_panel()

# ============================================================
# SALES SUMMARY TABLE
# ============================================================

start_panel("Sales Summary")
summary_df = pd.DataFrame({
    "Metric": ["Total Revenue", "Average Order Value", "Total Transactions", "Unique Invoices", "Customers", "Products", "Countries"],
    "Value": [
        filtered_df["Revenue"].sum(),
        avg_order_value,
        total_transactions,
        filtered_df["Invoice"].nunique(),
        filtered_df["Customer ID"].nunique(),
        filtered_df["StockCode"].nunique(),
        filtered_df["Country"].nunique(),
    ]
})
st.dataframe(summary_df, use_container_width=True)
end_panel()

# ============================================================
# FILTERED DATASET PREVIEW
# ============================================================

start_panel("Filtered Dataset Preview")
st.dataframe(filtered_df.head(100), use_container_width=True)
end_panel()

# ============================================================
# COMPLETION MESSAGE
# ============================================================

st.success("Online Retail II EDA Dashboard Completed Successfully")