from typing import Any
import pandas as pd
import time
import random
from tqdm import tqdm
from openai import OpenAI
import AUTH

client = OpenAI(api_key=AUTH.OAI_API_KEY)

req_cols: list[str] = ['role', 'prompt']

def load_data(file_path: str, required_columns: list[str]) -> pd.DataFrame | None:
    try:
        data = pd.read_excel(file_path)
        if all(col in data.columns for col in required_columns):
            print(f"'{file_path}' loaded successfully...")
            return data
        else:
            print(f"'{file_path}' is missing required columns: {required_columns}")
            return None
    except FileNotFoundError:
        print(f"Error: File '{file_path}' not found.")
        return None
    except Exception as e:
        print(f"Unexpected error loading '{file_path}': {e}")
        return None

def get_response(prompt: str) -> str:
    """Send prompt to OpenAI model and return the response."""
    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",  # 'gpt-3.5-turbo' and 'gpt-4'
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Error: {e}"

def throttle() -> None:
    """Sleep for a short random interval to avoid rate limits."""
    time.sleep(random.uniform(0.3, 0.7))

def main(input_filepath: str, output_filepath: str) -> None:
    """Main execution flow."""
    df = load_data(input_filepath, req_cols)

    if df is not None:
        responses = []
        for _, row in tqdm(df.iterrows(), total=len(df), desc="Generating responses"):
            prompt = f"{row['role']}. {row['prompt']}"
            response = get_response(prompt)
            responses.append(response)
            throttle()

        df['response'] = responses
        df.to_excel(output_filepath, index=False)
        print(f"Responses saved to '{output_filepath}'")
    else:
        print("Data loading failed. Exiting.")

if __name__ == "__main__":
    input_filepath: str = 'prompts.xlsx'
    oai_output_filepath: str = 'openai_responses.xlsx'
    
    main(input_filepath, oai_output_filepath)