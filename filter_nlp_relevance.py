import pandas as pd
import json
import logging
from tqdm import tqdm

# --- Configuration ---
# The dataset collected by the strategic fetching script
INPUT_CSV_FILE = "merged_nlp_papers_for_enrichment.csv" 
# The final, filtered output file
OUTPUT_CSV_FILE = "filtered_nlp_papers.csv"

# --- NLP Relevance Constants (from your original script) ---
CORE_S2_NLP_FOS = ['natural language processing', 'computational linguistics']
TOP_NLP_VENUES = [
    'acl', 'emnlp', 'naacl', 'coling', 'eacl',
    'transactions of the association for computational linguistics', 'computational linguistics', 'lrec',
    'neurips', 'icml', 'iclr', 'aaai', 'ijcai'
]
BROADER_S2_NLP_FOS = ['artificial intelligence', 'linguistics', 'information retrieval', 'speech recognition', 'machine learning']
CS_FOS_CATEGORY = "computer science"

# --- Logging and Progress Bar Setup ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
tqdm.pandas()

def is_nlp_related_from_csv(row):
    """
    An adapted version of the is_nlp_related_enhanced filter that works on a 
    pandas DataFrame row from the collected CSV.
    """
    # Check 1: Paper has an ACL ID
    if pd.notna(row.get('externalIdsACL')):
        return True

    # Check 2: Paper is published in a top NLP-related venue
    venue_name = str(row.get('publicationVenueName', '')).lower()
    if any(top_venue in venue_name for top_venue in TOP_NLP_VENUES):
        return True

    # Check 3: Paper's fields of study (FOS) are relevant
    s2_categories = []
    try:
        # The FOS data in the CSV is a JSON string, so it must be parsed
        if pd.notna(row['s2FieldsOfStudy']):
            s2_categories = [cat.lower() for cat in json.loads(row['s2FieldsOfStudy'])]
    except (json.JSONDecodeError, TypeError):
        pass # Handle cases where the cell is empty or malformed

    if CS_FOS_CATEGORY in s2_categories:
        return True
    
    if any(core_fos in s2_categories for core_fos in CORE_S2_NLP_FOS):
        return True

    # Check 4: Paper has broader NLP signals (e.g., ArXiv ID + broader FOS)
    nlp_signal_count = sum(1 for broader_fos in BROADER_S2_NLP_FOS if broader_fos in s2_categories)
    
    has_arxiv_id = pd.notna(row.get('externalIdsArXiv'))
    if has_arxiv_id and nlp_signal_count > 0:
        return True
    
    # A final, more lenient check if it has any of the broader FOS
    if nlp_signal_count > 0:
        return True

    return False

def filter_dataset(input_path, output_path):
    """
    Loads a dataset, applies the NLP relevance filter, and saves the result.
    """
    logging.info(f"Loading dataset from {input_path}...")
    try:
        df = pd.read_csv(input_path)
    except FileNotFoundError:
        logging.error(f"File not found: {input_path}. Please run the collection script first.")
        return

    logging.info(f"Applying NLP relevance filter to {len(df)} papers...")

    # Apply the function row-by-row and create a boolean mask
    is_relevant_mask = df.progress_apply(is_nlp_related_from_csv, axis=1)

    # Use the boolean mask to filter the DataFrame
    filtered_df = df[is_relevant_mask].copy()

    logging.info(f"Filtering complete. Kept {len(filtered_df)} of {len(df)} papers.")

    try:
        filtered_df.to_csv(output_path, index=False)
        logging.info(f"Successfully saved filtered dataset to {output_path}")
    except IOError as e:
        logging.error(f"Failed to save filtered file: {e}")


if __name__ == "__main__":
    filter_dataset(INPUT_CSV_FILE, OUTPUT_CSV_FILE)