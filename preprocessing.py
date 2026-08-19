import pandas as pd

def preprocess_data(df):

    print("\n" + "=" * 75)
    print("STEP 2 - DATA CLEANING & PREPROCESSING")
    print("=" * 75)

    original_records = len(df)

    df = df.copy()

    required_columns = ["Invoice", "StockCode", "Description", "Quantity", "InvoiceDate", "Price", "Customer ID", "Country"]

    df = df[required_columns]

    df["Invoice"] = df["Invoice"].astype(str).str.strip()

    df["StockCode"] = df["StockCode"].astype(str).str.strip()

    df["Description"] = df["Description"].astype(str).str.strip()

    df["Country"] = df["Country"].astype(str).str.strip()

    df["Quantity"] = pd.to_numeric(df["Quantity"], errors="coerce")

    df["Price"] = pd.to_numeric(df["Price"], errors="coerce")

    df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"], errors="coerce")

    missing_values = df.isnull().sum().sum()

    duplicate_records = df.duplicated().sum()

    cancellation_records = df["Invoice"].str.startswith("C").sum()

    invalid_quantity = (df["Quantity"] <= 0).sum()

    invalid_price = (df["Price"] <= 0).sum()

    missing_customer = df["Customer ID"].isnull().sum()

    df = df.drop_duplicates()

    df = df.dropna(subset=["Invoice", "StockCode", "Quantity", "Price", "InvoiceDate", "Customer ID", "Country"])

    df = df[df["Quantity"] > 0]

    df = df[df["Price"] > 0]

    df = df[~df["Invoice"].str.startswith("C")]

    df["Customer ID"] = pd.to_numeric(df["Customer ID"], errors="coerce")

    df = df.dropna(subset=["Customer ID"])

    df["Customer ID"] = df["Customer ID"].astype(int)

    print(f"Original Records      : {original_records:,}")
    print(f"Missing Values        : {missing_values:,}")
    print(f"Duplicate Records     : {duplicate_records:,}")
    print(f"Cancellation Records  : {cancellation_records:,}")
    print(f"Invalid Quantity      : {invalid_quantity:,}")
    print(f"Invalid Price         : {invalid_price:,}")
    print(f"Missing Customer ID   : {missing_customer:,}")
    print(f"Clean Records         : {len(df):,}")
    print(f"Removed Records       : {original_records - len(df):,}")

    print("=" * 75)

    return df