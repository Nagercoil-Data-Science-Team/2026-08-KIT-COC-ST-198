import pandas as pd

file_path = "online_retail_II.xlsx"

print("=" * 60)
print("DOCKER ONLINE RETAIL TEST")
print("=" * 60)

excel_file = pd.ExcelFile(file_path)

print("\nAvailable Excel Sheets:")

for sheet in excel_file.sheet_names:
    print("-", sheet)

for sheet in excel_file.sheet_names:

    df = pd.read_excel(
        file_path,
        sheet_name=sheet
    )

    print("\n" + "=" * 60)
    print("Sheet:", sheet)
    print("Shape:", df.shape)
    print("=" * 60)

    print(df.head())

print("\n✓ Online Retail Excel file loaded successfully")
print("✓ Docker Python environment is working")