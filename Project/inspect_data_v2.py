import pandas as pd

file_path = r'c:\Users\AGhantasala\Downloads\Project\Forecasting Case- Study.xlsx'
try:
    df = pd.read_excel(file_path)
    print("Columns:", df.columns.tolist())
    print("\nFirst 5 rows:")
    print(df.head().to_string())
    print("\nDate range:")
    if 'Date' in df.columns:
        print(f"Min Date: {df['Date'].min()}")
        print(f"Max Date: {df['Date'].max()}")
    elif 'Order Date' in df.columns:
         print(f"Min Date: {df['Order Date'].min()}")
         print(f"Max Date: {df['Order Date'].max()}")
    
    print("\nMissing values:")
    print(df.isnull().sum())
    
except Exception as e:
    print(f"Error: {e}")
