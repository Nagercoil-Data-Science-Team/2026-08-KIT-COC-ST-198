import pandas as pd
import os

def validate_dataset(file_path):

    print("=" * 75)
    print("STEP 1 - DATASET VALIDATION")
    print("=" * 75)

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Dataset not found: {file_path}")

    excel_file = pd.ExcelFile(file_path)

    expected_sheets = ["Year 2009-2010", "Year 2010-2011"]

    required_columns = ["Invoice", "StockCode", "Description", "Quantity", "InvoiceDate", "Price", "Customer ID", "Country"]

    print(f"Dataset File          : {file_path}")
    print(f"Available Sheets      : {', '.join(excel_file.sheet_names)}")

    missing_sheets = [sheet for sheet in expected_sheets if sheet not in excel_file.sheet_names]

    sheet_status = "VALID" if not missing_sheets else "INVALID"

    print(f"Sheet Validation      : {sheet_status}")

    all_data = []

    for sheet in expected_sheets:

        df = pd.read_excel(file_path, sheet_name=sheet)

        rows = len(df)

        missing_columns = [col for col in required_columns if col not in df.columns]

        attribute_status = "VALID" if not missing_columns else "INVALID"

        missing_values = df[required_columns].isnull().sum().sum()

        duplicate_records = df[required_columns].duplicated().sum()

        quantity_numeric = pd.to_numeric(df["Quantity"], errors="coerce")

        invalid_quantity = (quantity_numeric <= 0).sum()

        price_numeric = pd.to_numeric(df["Price"], errors="coerce")

        invalid_price = (price_numeric <= 0).sum()

        cancellation_records = df["Invoice"].astype(str).str.startswith("C").sum()

        print(f"{sheet} | Rows: {rows:,} | Columns: {len(required_columns)} | Attributes: {attribute_status} | Missing: {missing_values:,} | Duplicates: {duplicate_records:,} | Invalid Quantity: {invalid_quantity:,} | Invalid Price: {invalid_price:,} | Cancellations: {cancellation_records:,}")

        df["SourceSheet"] = sheet

        all_data.append(df)

    combined_df = pd.concat(all_data, ignore_index=True)

    combined_missing = combined_df[required_columns].isnull().sum().sum()

    combined_duplicates = combined_df[required_columns].duplicated().sum()

    combined_df["Quantity"] = pd.to_numeric(combined_df["Quantity"], errors="coerce")

    combined_df["Price"] = pd.to_numeric(combined_df["Price"], errors="coerce")

    invalid_quantity_total = (combined_df["Quantity"] <= 0).sum()

    invalid_price_total = (combined_df["Price"] <= 0).sum()

    cancellation_total = combined_df["Invoice"].astype(str).str.startswith("C").sum()

    actual_columns = [col for col in combined_df.columns if col != "SourceSheet"]

    schema_valid = set(required_columns).issubset(set(actual_columns))

    schema_status = "VALID" if schema_valid else "INVALID"

    overall_status = "VALID" if schema_valid and not missing_sheets else "INVALID"

    print("-" * 75)
    print(f"Total Records          : {len(combined_df):,}")
    print(f"Total Columns          : {len(required_columns)}")
    print(f"Missing Values         : {combined_missing:,}")
    print(f"Duplicate Records      : {combined_duplicates:,}")
    print(f"Invalid Quantity       : {invalid_quantity_total:,}")
    print(f"Invalid Price          : {invalid_price_total:,}")
    print(f"Cancellation Records   : {cancellation_total:,}")
    print(f"Schema Status          : {schema_status}")
    print(f"Overall Validation     : {overall_status}")
    print("=" * 75)

    return combined_df