import pandas as pd

# Load the CSV files
file1 = r"C:\Users\Seai_\Documents\seai_pro\ml_algo\place_timing.csv"  
file2 = r"C:\Users\Seai_\Documents\seai_pro\ml_algo\route_timing.csv" 
# Read the CSV files into DataFrames
df1 = pd.read_csv(file1)
df2 = pd.read_csv(file2)

# Ensure both files have the 'endpoint' column
if 'endpoint' not in df1.columns or 'endpoint' not in df2.columns:
    raise ValueError("Both CSV files must contain the 'endpoint' column.")

# Sort the DataFrames by 'endpoint'
df1_sorted = df1.sort_values(by='endpoint')
df2_sorted = df2.sort_values(by='endpoint')

# Filter rows in the first DataFrame whose 'endpoint' exists in the second DataFrame
df1_filtered = df1_sorted[df1_sorted['endpoint'].isin(df2_sorted['endpoint'])]

# Filter rows in the second DataFrame whose 'endpoint' exists in the first DataFrame
df2_filtered = df2_sorted[df2_sorted['endpoint'].isin(df1_sorted['endpoint'])]

# Function to clean data: remove rows with NaN or "na" (case-insensitive) in any column
def clean_data(df):
    # Drop rows with NaN values
    df_cleaned = df.dropna()
    # Drop rows with the string "na" (case-insensitive) in any column
    df_cleaned = df_cleaned[~df_cleaned.apply(lambda row: row.astype(str).str.contains(r'\bna\b', case=False)).any(axis=1)]
    return df_cleaned

# Clean both DataFrames
df1_cleaned = clean_data(df1_filtered)
df2_cleaned = clean_data(df2_filtered)

# Save the cleaned DataFrames to new CSV files
df1_cleaned.to_csv("filtered_place_csv_file_cleaned.csv", index=False)
df2_cleaned.to_csv("filtered_route_csv_file_cleaned.csv", index=False)

print("Filtered and cleaned CSV files have been created:")
print("- Filtered first CSV saved as 'filtered_place_csv_file_cleaned.csv'")
print("- Filtered second CSV saved as 'filtered_route_csv_file_cleaned.csv'")