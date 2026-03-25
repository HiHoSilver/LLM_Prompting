import time
import random
from typing import Any
from tqdm import tqdm
from groq import Groq
import pandas as pd
from Groq import GROQ_CONFIG
from utils.address_parsing import parse_addresses
from utils.timing import Timer

client = Groq(api_key=GROQ_CONFIG.api_key)
req_cols: list[str] = ['prompt']

def load_data(filepath: str, required_columns: list[str], sheetname: str=None) -> pd.DataFrame | None:
    try:
        df: pd.DataFrame = pd.read_excel(filepath, sheet_name=sheetname) if sheetname else pd.read_excel(filepath)
        missing: set[str] = set(required_columns) - set(df.columns)
        if missing:
            print(f"'{filepath}' is missing required columns: {missing}")
            return
        print(f"'{filepath}' loaded successfully...")
        return df
    except FileNotFoundError:
        print(f"Error: File '{filepath}' not found.")
        return
    except Exception as e:
        print(f"Unexpected error loading '{filepath}': {e}")
        return

def get_response(prompt: str, max_retries: int = 6) -> str:
    """
    Calls Groq with exponential backoff + jitter.
    Retries on transient server errors.
    """
    for attempt in range(1, max_retries + 1):
        try:
            completion: Any = client.chat.completions.create(
                model=GROQ_CONFIG.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0
            )
            return completion.choices[0].message.content

        # Groq uses generic exceptions; retry on anything transient
        except Exception as e:
            wait = (2 ** attempt) + random.uniform(0, 2)
            print(f"Groq error: {e}. Retrying in {wait:.2f} seconds...")
            time.sleep(wait)

    return "Error: Maximum retries exceeded"

def throttle() -> None:
    time.sleep(random.uniform(0.3, 0.8))

def process_prompts(df: pd.DataFrame) -> pd.DataFrame | None:
    if df is None or df.empty:
        print("Input file is empty or invalid")
        return

    try:
        df_unique: pd.DataFrame = df.drop_duplicates()
        duplicates: int = len(df) - len(df_unique)
        if duplicates:
            print(f"{duplicates} duplicate prompts found.")

        responses: list[str] = []
        timer = Timer()
        total_rows = len(df_unique)

        for row in tqdm(df_unique.itertuples(), total=total_rows, desc="Generating responses"):
            if row.Index % 50 == 0 and row.Index > 0:
                print("Cooling down for 5 seconds...")
                time.sleep(5)

            parts = [GROQ_CONFIG.role, GROQ_CONFIG.base_prompt, row.prompt]
            prompt = ". ".join(p for p in parts if p)

            response = get_response(prompt)
            responses.append(response)
            throttle()

            if row.Index % 2 == 0 and row.Index > 0:
                elapsed = timer.elapsed()
                avg_per_row = elapsed / row.Index
                remaining = total_rows - row.Index
                eta_seconds = avg_per_row * remaining

                print(
                    f"Elapsed: {timer.format(elapsed)} | "
                    f"ETA: {timer.format(eta_seconds)} remaining"
                )

            print(f"Total processing time: {timer.format(timer.elapsed())}")

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
    overall = Timer()

    df = load_data(GROQ_CONFIG.input_filepath, req_cols)
    if df is None:
        print("Failed to load input file. Exiting.")
        return

    df_processed = process_prompts(df)
    if df_processed is None:
        print("Processing failed. Exiting.")
        return

    if GROQ_CONFIG.parse_add_flag:
        parse_addresses(df_processed)

    df_processed.to_excel(GROQ_CONFIG.output_filepath, index=False)
    print(f"Successfully processed {len(df)} rows")
    print(f"Output saved to {GROQ_CONFIG.output_filepath}")
    print(f"Total runtime: {overall.elapsed()}")
    

if __name__ == "__main__":
    main()
