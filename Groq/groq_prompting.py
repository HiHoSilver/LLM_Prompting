from typing import Any
import time
import random
from tqdm import tqdm
from groq import Groq
import pandas as pd
from config import GROQ_CONFIG

client = Groq(api_key=GROQ_CONFIG.api_key)
req_cols: list[str] = ['prompt']

def load_data(filepath: str, required_columns: list[str], sheetname: str=None) -> pd.DataFrame | None:
    try:
        df = pd.read_excel(filepath, sheet_name=sheetname) if sheetname else pd.read_excel(filepath)
        missing = set(required_columns) - set(df.columns)
        if missing:
            print(f"'{filepath}' is missing required columns: {missing}")
            return None
        print(f"'{filepath}' loaded successfully...")
        return df
    except FileNotFoundError:
        print(f"Error: File '{filepath}' not found.")
    except Exception as e:
        print(f"Unexpected error loading '{filepath}': {e}")
        return None

def get_response(prompt: str, max_retries: int = 6) -> str:
    """
    Calls Groq with exponential backoff + jitter.
    Retries on transient server errors.
    """

    for attempt in range(1, max_retries + 1):
        try:
            completion = client.chat.completions.create(
                model=GROQ_CONFIG.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0
            )
            return completion.choices[0].message.content

        except Exception as e:
            # Groq uses generic exceptions; retry on anything transient
            wait = (2 ** attempt) + random.uniform(0, 2)
            print(f"Groq error: {e}. Retrying in {wait:.2f} seconds...")
            time.sleep(wait)

    return "Error: Maximum retries exceeded"

def throttle():
    time.sleep(random.uniform(0.3, 0.8))

def process_prompts(df: pd.DataFrame) -> pd.DataFrame | None:
    if df is None or df.empty:
        print("Input file is empty or invalid")
        return None

    try:
        df_unique = df.drop_duplicates()
        duplicates = len(df) - len(df_unique)
        if duplicates:
            print(f"{duplicates} duplicate prompts found.")

        responses = []

        for row in tqdm(df_unique.itertuples(), total=len(df_unique), desc="Generating responses"):
            # Optional cooldown for very large batches
            if row.Index % 50 == 0 and row.Index > 0:
                print("Cooling down for 5 seconds...")
                time.sleep(5)

            parts = [GROQ_CONFIG.role, GROQ_CONFIG.base_prompt, row.prompt]
            prompt = ". ".join(p for p in parts if p)

            response = get_response(prompt)
            responses.append(response)
            throttle()

        df_unique["response"] = responses

        df = df.merge(df_unique[["prompt", "response"]], how="left", on="prompt")
        df["role"] = GROQ_CONFIG.role
        df["base_prompt"] = GROQ_CONFIG.base_prompt

        df = df[["role", "base_prompt", "prompt", "response"]]
        return df

    except Exception as e:
        print(f"Fatal error during response generation: {e}")
        raise

def main() -> None:
    df = load_data(GROQ_CONFIG.input_filepath, req_cols)
    if df is None:
        print("Failed to load input file. Exiting.")
        return

    df_processed = process_prompts(df)
    if df_processed is None:
        print("Processing failed. Exiting.")
        return

    df_processed.to_excel(GROQ_CONFIG.output_filepath, index=False)
    print(f"Successfully processed {len(df)} rows")
    print(f"Output saved to {GROQ_CONFIG.output_filepath}")


if __name__ == "__main__":
    main()
