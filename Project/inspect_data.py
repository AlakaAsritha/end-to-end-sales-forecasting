import pandas as pd

file_path = r'c:\Users\AGhantasala\Downloads\Project\Forecasting Case- Study.xlsx'
try:
    df = pd.read_excel(file_path)
    print("Columns:", df.columns.tolist())
    print("\nFirst 5 rows:")
    print(df.head())
    print("\nData Info:")
    print(df.info())
    print("\nUnique States:", df['State'].unique() if 'State' in df.columns else "State column not found")
except Exception as e:
    print(f"Error: {e}")
