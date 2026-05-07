import pandas as pd

file_path = r'c:\Users\AGhantasala\Downloads\Project\Forecasting Case- Study.xlsx'
try:
    df = pd.read_excel(file_path)
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
    df = df.dropna(subset=['Date'])
    
    print("Unique Categories:", df['Category'].unique().tolist())
    
    # Check if we have multiple records for same State/Date
    counts = df.groupby(['State', 'Date']).size()
    print("\nMax records per State-Date combination:", counts.max())
    
    # Check average frequency
    df_alabama = df[df['State'] == 'Alabama'].sort_values('Date')
    df_alabama['diff'] = df_alabama['Date'].diff().dt.days
    print("\nAlabama Date Diffs (value counts):")
    print(df_alabama['diff'].value_counts().head())
    
except Exception as e:
    print(f"Error: {e}")
