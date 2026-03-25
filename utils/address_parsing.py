import pandas as pd

def parse_addresses(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        print("Input file is empty or invalid")
        return 
    
    split_data = df['response'].str.split('; ', expand=True)
    df[['street', 'city', 'state', 'zip']] = split_data
    return df