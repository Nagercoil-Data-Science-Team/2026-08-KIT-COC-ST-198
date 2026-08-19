import pandas as pd

def engineer_features(df):

    print("\n" + "=" * 75)
    print("STEP 3 - FEATURE ENGINEERING")
    print("=" * 75)

    df = df.copy()

    df["Revenue"] = df["Quantity"] * df["Price"]

    df["Year"] = df["InvoiceDate"].dt.year

    df["Month"] = df["InvoiceDate"].dt.month

    df["MonthName"] = df["InvoiceDate"].dt.month_name()

    df["Day"] = df["InvoiceDate"].dt.day

    df["DayOfWeek"] = df["InvoiceDate"].dt.day_name()

    df["Hour"] = df["InvoiceDate"].dt.hour

    df["Date"] = df["InvoiceDate"].dt.date

    df["YearMonth"] = df["InvoiceDate"].dt.to_period("M").astype(str)

    customer_summary = df.groupby("Customer ID").agg(TotalRevenue=("Revenue", "sum"), TotalQuantity=("Quantity", "sum"), TotalOrders=("Invoice", "nunique"), AverageOrderValue=("Revenue", "mean"), UniqueProducts=("StockCode", "nunique")).reset_index()

    product_summary = df.groupby(["StockCode", "Description"]).agg(ProductRevenue=("Revenue", "sum"), ProductQuantity=("Quantity", "sum"), ProductOrders=("Invoice", "nunique"), UniqueCustomers=("Customer ID", "nunique")).reset_index()

    country_summary = df.groupby("Country").agg(CountryRevenue=("Revenue", "sum"), CountryQuantity=("Quantity", "sum"), CountryOrders=("Invoice", "nunique"), UniqueCustomers=("Customer ID", "nunique")).reset_index()

    print("Revenue Feature       : Created")
    print("Date Features         : Created")
    print(f"Customer Features     : {len(customer_summary):,}")
    print(f"Product Features      : {len(product_summary):,}")
    print(f"Country Features      : {len(country_summary):,}")
    print(f"Final Feature Rows    : {len(df):,}")
    print(f"Final Feature Columns : {len(df.columns):,}")

    print("=" * 75)

    return df, customer_summary, product_summary, country_summary