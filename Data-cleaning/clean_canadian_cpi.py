import pandas as pd
import os
import re

# Setup paths
base_dir = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(base_dir, "canadian-cpi", "cpi_20160101.csv")

# 1. Load CSV: Skipping 9 lines lands us exactly on the header: "Products and product groups 3 4"
df = pd.read_csv(file_path, skiprows=9, encoding="utf-8-sig")

# 2. Clean Product Names (Remove trailing footnote numbers like ' 5' or ' 3 4')
df.iloc[:, 0] = df.iloc[:, 0].astype(str).str.replace(r'\s+\d+(\s+\d+)?$', '', regex=True).str.strip()
df = df.rename(columns={df.columns[0]: "Product"})

# 3. Filter for specific products: use the cleaned names here
target_products = ["Food", "Gasoline"]
df = df[df["Product"].isin(target_products)].copy()

# 4. Generate proper YYYYMM column names from the original columns are "January 2016", etc. 
new_col_names = ["Product"]
for col in df.columns[1:]:
    try:
        # Convert "January 2016" -> Timestamp(2016-01-01) -> "201601"
        clean_date = pd.to_datetime(col)
        new_col_names.append(clean_date.strftime('%Y%m'))
    except:
        new_col_names.append(col) # Fallback if parsing fails

df.columns = new_col_names

# 5. Convert numeric columns safely: apply pd.to_numeric to all columns except the first one ("Product")
cols_to_fix = df.columns[1:]
df[cols_to_fix] = df[cols_to_fix].apply(pd.to_numeric, errors='coerce')

# 6. Save cleaned CSV
cleaned_file_path = os.path.join(base_dir, "canadian-cpi", "cpi_20160101_cleaned.csv")
# Ensure directory exists
os.makedirs(os.path.dirname(cleaned_file_path), exist_ok=True)
df.to_csv(cleaned_file_path, index=False)

print("Cleaned dataset saved to:", cleaned_file_path)
print(df.head())