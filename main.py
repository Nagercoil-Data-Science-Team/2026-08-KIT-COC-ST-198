from Validation import validate_dataset
from preprocessing import preprocess_data
from feature_engineering import engineer_features

FILE_PATH = "online_retail_II.xlsx"

CLEANED_FILE = "cleaned_online_retail.csv"

FEATURE_FILE = "engineered_online_retail.csv"

CUSTOMER_FILE = "customer_summary.csv"

PRODUCT_FILE = "product_summary.csv"

COUNTRY_FILE = "country_summary.csv"

print("\n" + "=" * 80)
print("ONLINE RETAIL II - COMPLETE DATA PROCESSING PIPELINE")
print("=" * 80)

validated_data = validate_dataset(FILE_PATH)

cleaned_data = preprocess_data(validated_data)

engineered_data, customer_summary, product_summary, country_summary = engineer_features(cleaned_data)

cleaned_data.to_csv(CLEANED_FILE, index=False)

engineered_data.to_csv(FEATURE_FILE, index=False)

customer_summary.to_csv(CUSTOMER_FILE, index=False)

product_summary.to_csv(PRODUCT_FILE, index=False)

country_summary.to_csv(COUNTRY_FILE, index=False)

print("\n" + "=" * 80)
print("PIPELINE OUTPUT")
print("=" * 80)

print(f"Cleaned Dataset       : {CLEANED_FILE}")

print(f"Engineered Dataset    : {FEATURE_FILE}")

print(f"Customer Summary      : {CUSTOMER_FILE}")

print(f"Product Summary       : {PRODUCT_FILE}")

print(f"Country Summary       : {COUNTRY_FILE}")

print("\n✓ Dataset validation completed")

print("✓ Data preprocessing completed")

print("✓ Feature engineering completed")

print("✓ Complete pipeline executed successfully")

print("=" * 80)