import pandas as pd
import logging

# --- Configuration ---
# The dataset you want to filter (e.g., the output from the venue filter)
INPUT_CSV_FILE = "filtered_nlp_papers.csv" 
# The final output file after applying the citation filter
OUTPUT_CSV_FILE = "final_papers_min_citations.csv"

# Set the minimum number of citations a paper must have to be kept
MINIMUM_CITATION_COUNT = 5

# --- Logging Configuration ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def filter_by_citations(input_path, output_path, min_citations):
    """
    Loads a dataset, filters out papers with fewer citations than the minimum,
    and saves the result.
    """
    logging.info(f"Loading dataset from {input_path}...")
    try:
        df = pd.read_csv(input_path)
    except FileNotFoundError:
        logging.error(f"File not found: {input_path}. Please provide a valid input file.")
        return

    logging.info(f"Original dataset has {len(df)} papers.")
    logging.info(f"Filtering to keep papers with at least {min_citations} citations...")

    # Ensure the 'citationCount' column is numeric, coercing errors to NaN
    df['citationCount'] = pd.to_numeric(df['citationCount'], errors='coerce')

    # Drop rows where citationCount might be NaN after coercion
    df.dropna(subset=['citationCount'], inplace=True)

    # Convert the column to integer type for clean comparison
    df['citationCount'] = df['citationCount'].astype(int)

    # Perform the boolean filtering
    filtered_df = df[df['citationCount'] >= min_citations].copy()

    logging.info(f"Filtering complete. Kept {len(filtered_df)} of {len(df)} papers.")

    try:
        filtered_df.to_csv(output_path, index=False)
        logging.info(f"Successfully saved filtered dataset to {output_path}")
    except IOError as e:
        logging.error(f"Failed to save filtered file: {e}")


if __name__ == "__main__":
    filter_by_citations(INPUT_CSV_FILE, OUTPUT_CSV_FILE, MINIMUM_CITATION_COUNT)