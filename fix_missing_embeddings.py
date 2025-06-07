import pandas as pd
import requests
import time
import json
import logging
from tqdm import tqdm

# --- Configuration ---
# The script will read from and save back to this file, filling in the blanks.
INPUT_OUTPUT_CSV_FILE = "papers_with_embeddings.csv"

# --- API Configuration ---
# Your Semantic Scholar API Key (optional but highly recommended)
API_KEY = "KRzThuhqxK42MZRgsnQJR5NnLLO17hKx8flTuDnx"
API_DELAY_SECONDS = 1.7
# The specific field we want to fetch
FIELDS_TO_FETCH = "embedding"

# --- Progress Saving ---
# Save the file after every N papers to avoid data loss.
SAVE_PROGRESS_INTERVAL = 200

# --- Logging Configuration ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Main Functions ---

def make_embedding_api_request(paper_id):
    """Makes a single API request to fetch the embedding for one paper."""
    url = f"https://api.semanticscholar.org/graph/v1/paper/{paper_id}"
    params = {'fields': FIELDS_TO_FETCH}
    headers = {}
    if API_KEY and API_KEY != "YOUR_SEMANTIC_SCHOLAR_API_KEY":
        headers['X-API-KEY'] = API_KEY
    
    try:
        time.sleep(API_DELAY_SECONDS)
        response = requests.get(url, params=params, headers=headers)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.HTTPError as e:
        if e.response.status_code == 404:
            logging.warning(f"Paper {paper_id} not found or has no embedding (404). Will not retry.")
            # Return a special value to mark as "checked but not found"
            return "NO_EMBEDDING_FOUND"
        else:
            logging.error(f"HTTP error for paper {paper_id}: {e}")
    except Exception as e:
        logging.error(f"An unexpected error occurred for paper {paper_id}: {e}")
    # Return None for transient errors so we can retry later
    return None

def fill_missing_embeddings(csv_path): 
    """
    Reads a CSV, finds rows with missing embeddings, and attempts to fetch them.
    """
    logging.info(f"Loading data from {csv_path} to find and fill missing embeddings...")
    try:
        # Load the CSV, ensuring empty cells are read as NaN/None
        df = pd.read_csv(csv_path)
    except FileNotFoundError:
        logging.error(f"File not found: {csv_path}. Cannot proceed.")
        return

    # --- Identify papers with missing embeddings ---
    # We look for cells that are null/NaN.
    missing_df = df[df['embedding'].isnull()]
    
    if missing_df.empty:
        logging.info("No missing embeddings found. Dataset is complete!")
        return
        
    logging.info(f"Found {len(missing_df)} papers with missing embeddings. Starting fetch process...")
    
    papers_processed = 0
    # Use tqdm for a progress bar on the missing items
    for index, row in tqdm(missing_df.iterrows(), total=missing_df.shape[0], desc="Filling Embeddings"):
        paper_id = row['paperId']
        
        api_response = make_embedding_api_request(paper_id)
        
        if api_response:
            if api_response == "NO_EMBEDDING_FOUND":
                # Mark it so we don't try again in the future
                df.at[index, 'embedding'] = "FETCHED_NONE"
            elif 'embedding' in api_response and api_response['embedding']:
                embedding_vector = api_response['embedding'].get('vector')
                if embedding_vector:
                    # Store the vector as a JSON string
                    df.at[index, 'embedding'] = json.dumps(embedding_vector)
        
        # Save progress periodically
        papers_processed += 1
        if papers_processed % SAVE_PROGRESS_INTERVAL == 0:
            try:
                logging.info(f"Saving progress after processing {papers_processed} missing entries...")
                df.to_csv(csv_path, index=False)
            except Exception as e:
                logging.error(f"Failed to save progress to {csv_path}: {e}")
    
    # Final save at the end
    try:
        logging.info("Embedding fill process complete. Saving final dataset...")
        df.to_csv(csv_path, index=False)
        logging.info(f"Successfully updated dataset at '{csv_path}'.")
    except Exception as e:
        logging.error(f"Failed to save final CSV file: {e}")

# --- Main Execution ---
if __name__ == "__main__":
    fill_missing_embeddings(INPUT_OUTPUT_CSV_FILE)
