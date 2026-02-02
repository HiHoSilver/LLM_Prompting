import time
import random
from tqdm import tqdm
import google.generativeai as genai
from google.api_core.exceptions import ResourceExhausted, InternalServerError, ServiceUnavailable
import pandas as pd
from config import GEM_CONFIG

genai.configure(api_key=GEM_CONFIG.api_key)
model = genai.GenerativeModel(GEM_CONFIG.model)

req_cols: list[str] = ['prompt']

def load_data(filepath: str, required_columns: list[str], sheetname: str=None) -> pd.DataFrame | None:
    try: 
        df = pd.read_excel(filepath, sheet_name=sheetname) if sheetname else pd.read_excel(filepath)
        missing = set(required_columns) - set(df.columns)
        if missing:
            print(f"'{filepath}' is missing required columns: {missing}")
            return
        print(f"'{filepath}' loaded successfully...") 
        return df
    except FileNotFoundError:
        print(f"Error: File '{filepath}' not found.")
    except Exception as e:
        print(f"Unexpected error loading '{filepath}': {e}")

def get_response(prompt: str, max_retries: int = 6) -> str:
    for attempt in range(1, max_retries + 1):
        try:
            response = model.generate_content(prompt)
            return response.text

        except genai.types.BlockedPromptException:
            print("Prompt was blocked by safety filters")
            return "Error: Prompt blocked by safety filters"

        except ResourceExhausted:
            wait = (2 ** attempt) + random.uniform(0, 2)
            print(f"Rate limit hit. Retrying in {wait:.2f} seconds...")
            time.sleep(wait)

        except (InternalServerError, ServiceUnavailable):
            wait = (2 ** attempt) + random.uniform(0, 2)
            print(f"Server error. Retrying in {wait:.2f} seconds...")
            time.sleep(wait)

        except Exception as e:
            print(f"Unexpected error from model.generate_content: {e}")
            return f"Error: {e}"

    return "Error: Maximum retries exceeded"
    
def throttle():
    time.sleep(random.uniform(1.0, 2.5))

def process_prompts(df: pd.DataFrame) -> None:
    if df is None or df.empty:
        print("Input file is empty or invalid")
        return
    
    try:
        df_unique = df.drop_duplicates()
        duplicates: int = len(df) - len(df_unique)
        if duplicates:
            print(f"{duplicates} duplicate prompts found.")

        responses = []
        for row in tqdm(df_unique.itertuples(), total=len(df_unique), desc="Generating responses"):
            if row.Index % 25 == 0 and row.Index > 0:
                print("Cooling down for 10 seconds...")
                time.sleep(10)
            
            parts = [GEM_CONFIG.role, GEM_CONFIG.base_prompt, row.prompt]
            prompt = ". ".join(p for p in parts if p)
            
            response = get_response(prompt)
            responses.append(response)
            throttle()

        df_unique['response'] = responses
        df = df.merge(df_unique[['prompt', 'response']], how='left', on='prompt')

        df['role'] = GEM_CONFIG.role
        df['base_prompt'] = GEM_CONFIG.base_prompt
        df = df[['role', 'base_prompt', 'prompt', 'response']]
        
        return df

    except Exception:
        print("Fatal error during response generation")
        raise

def main() -> None:
    """
    Requires `GEM_CONFIG` dataclass or related object with:
        'api_key': str = field(repr=False), 'model': str, 'input_filepath': str, 
        'output_filepath': str, 'role': str, 'base_prompt': str
    """
    # Load
    df = load_data(GEM_CONFIG.input_filepath, req_cols)
    if df is None:
        print("Failed to load input file. Exiting.")
        return

    # Process / get responses
    df_processed = process_prompts(df)
    if df_processed is None:
        print("Processing failed. Exiting.")
        return

    # Export
    df_processed.to_excel(GEM_CONFIG.output_filepath, index=False)
    print(f"Successfully processed {len(df)} rows")
    print(f"Output saved to {GEM_CONFIG.output_filepath}")


if __name__ == "__main__":
    main()
