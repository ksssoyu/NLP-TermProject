import pandas as pd

# Path to your ORIGINAL dataset
EXISTING_DATASET_PATH = "merged_nlp_papers_restored.csv"

print(f"Checking file: {EXISTING_DATASET_PATH}")

try:
    df = pd.read_csv(EXISTING_DATASET_PATH)

    total_rows = len(df)
    empty_reference_rows = df['citations_data'].isna().sum()

    print(f"Total papers in original dataset: {total_rows}")
    print(f"Papers with EMPTY 'references_data': {empty_reference_rows}")

    if empty_reference_rows > 0:
        print("\nConclusion: Your original dataset has incomplete rows.")
        print("The fetch_connections.py script is correctly trying to backfill this missing data.")
    else:
        print("\nConclusion: Your original dataset appears to be complete.")

except FileNotFoundError:
    print(f"Error: Could not find the file {EXISTING_DATASET_PATH}")   