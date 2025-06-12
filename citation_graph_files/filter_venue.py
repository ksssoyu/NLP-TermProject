import pandas as pd
import logging
from tqdm import tqdm

# --- Configuration ---
# The dataset you want to filter (e.g., the output from the collection or the previous filter)
INPUT_CSV_FILE = "merged_nlp_papers_for_enrichment.csv" 
# The final, strictly filtered output file
OUTPUT_CSV_FILE = "merged_nlp_papers.csv"

# --- Top Venue List (from your original script) ---
# Only papers from these venues will be kept.
TOP_NLP_VENUES = [
    'acl', 'emnlp', 'naacl', 'coling', 'eacl',
    'transactions of the association for computational linguistics', 
    'computational linguistics', 
    'lrec',
    'neurips', 'icml', 'iclr', 'aaai', 'ijcai'
]

# --- Logging and Progress Bar Setup ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
tqdm.pandas()

def is_in_top_venue(row):
    """
    A strict filter that returns True only if the paper's venue is in the TOP_NLP_VENUES list.
    """
    # Get the publication venue name from the row, convert to string and lower case
    venue_name = str(row.get('publicationVenueName', '')).lower()

    # Check if any of the top venue names are a substring of the paper's venue name
    if any(top_venue in venue_name for top_venue in TOP_NLP_VENUES):
        return True

    return False

def filter_dataset_by_venue(input_path, output_path):
    """
    Loads a dataset, applies the strict top-venue filter, and saves the result.
    """
    logging.info(f"Loading dataset from {input_path}...")
    try:
        df = pd.read_csv(input_path)
    except FileNotFoundError:
        logging.error(f"File not found: {input_path}. Please provide a valid input file.")
        return

    logging.info(f"Applying strict top-venue filter to {len(df)} papers...")

    # Apply the function row-by-row to create a boolean mask
    is_relevant_mask = df.progress_apply(is_in_top_venue, axis=1)

    # Use the boolean mask to filter the DataFrame
    filtered_df = df[is_relevant_mask].copy()

    logging.info(f"Filtering complete. Kept {len(filtered_df)} of {len(df)} papers from top venues.")

    try:
        filtered_df.to_csv(output_path, index=False)
        logging.info(f"Successfully saved top-venue dataset to {output_path}")
    except IOError as e:
        logging.error(f"Failed to save filtered file: {e}")


if __name__ == "__main__":
    filter_dataset_by_venue(INPUT_CSV_FILE, OUTPUT_CSV_FILE)