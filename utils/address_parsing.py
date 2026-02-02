import pandas as pd

def parse_addresses(df: pd.DataFrame) -> pd.DataFrame:
    split_data = df['response'].str.split('; ', expand=True)
    df[['street', 'city', 'state', 'zip']] = split_data
    return df