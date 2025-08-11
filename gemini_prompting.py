from typing import Any
import time
import random
from tqdm import tqdm
import google.generativeai as genai
import pandas as pd
import AUTH

genai.configure(api_key=AUTH.GEM_API_KEY)
model = genai.GenerativeModel('gemini-2.5-flash-lite')  # gemini-1.5-flash and gemini-2.5-flash-lite

req_cols: list[str] = ['role', 'prompt']

def load_data(file_path: str, required_columns: list[str]) -> pd.DataFrame | None:
    try:
        data = pd.read_excel(file_path)
        if all(col in data.columns for col in required_columns):
            print(f"'{file_path}' loaded successfully...")
            return data
        else:
            print(f"'{file_path}' does not contain the required columns: {required_columns}")
            return None
    except FileNotFoundError:
        print(f"Error: File '{file_path}' not found.")
        return None
    except Exception as e:
        print(f"Unexpected error loading '{file_path}': {e}")
        return None

def get_response(prompt: str) -> Any:
    try:
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"Error: {e}"
    
def throttle() -> None:
    time.sleep(random.uniform(.3, .7))

def main() -> None:
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
    output_filepath: str = 'gemini_responses.xlsx'

    main()
