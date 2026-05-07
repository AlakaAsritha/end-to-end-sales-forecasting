import pickle
import pandas as pd
import os

def export_metrics(models_dir='models_storage', output_file='evaluation_metrics.csv'):
    summary_path = os.path.join(models_dir, 'summary.pkl')
    if not os.path.exists(summary_path):
        print("Summary file not found. Run training first.")
        return
        
    with open(summary_path, 'rb') as f:
        summary = pickle.load(f)
        
    data = []
    for state, info in summary.items():
        data.append({
            'State': state,
            'Best Model': info['name'],
            'RMSE': info['rmse']
        })
        
    df = pd.DataFrame(data)
    df.to_csv(output_file, index=False)
    print(f"Metrics exported to {output_file}")
    
    # Also print as markdown table for terminal/readme
    print("\n--- Model Performance Summary ---")
    print(df.to_markdown(index=False))

if __name__ == "__main__":
    export_metrics()
